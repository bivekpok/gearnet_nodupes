import os
import sys
import logging
from itertools import islice

import torch
from torch import distributed as dist
from torch import nn
from torch.utils import data as torch_data

from torchdrug import data, core, utils
from torchdrug.core import Registry as R
from torchdrug.utils import comm, pretty
from torch.nn import functional as F

import numpy as np
import wandb


module = sys.modules[__name__]
logger = logging.getLogger(__name__)


# best_model_path = '/ocean/projects/bio230029p/bpokhrel/data/checkpoints/best_r5k10s2m5-gcn_full_sum_slr5.pth' ## to_save the best model


# class EarlyStopping:
#     """Early stops the training if validation loss doesn't improve after a given patience."""
#     def __init__(self, patience=40, verbose=False, delta=0):
#         """
#         Args:
#             patience (int): How long to wait after last time validation loss improved.
#                             Default: 10
#             verbose (bool): If True, prints a message for each validation loss improvement. 
#                             Default: False
#             delta (float): Minimum change in the monitored quantity to qualify as an improvement.
#                             Default: 0
#         """
#         self.patience = patience
#         self.verbose = verbose
#         self.counter = 0
#         self.best_loss = np.Inf
#         self.early_stop = False
#         self.val_loss_min = np.Inf
#         self.delta = delta

#     def __call__(self, val_loss):
#         if self.best_loss == np.Inf:
#             self.best_loss = val_loss
#         elif val_loss > self.best_loss + self.delta:
#             self.counter += 1
#             if self.verbose:
#                 print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_loss = val_loss
#             self.counter = 0




@R.register("core.Engine")
class Engine(core.Configurable):
    """
    General class that handles everything about training and test of a task.

    This class can perform synchronous distributed parallel training over multiple CPUs or GPUs.
    To invoke parallel training, launch with one of the following commands.

    1. Single-node multi-process case.

    .. code-block:: bash

        python -m torch.distributed.launch --nproc_per_node={number_of_gpus} {your_script.py} {your_arguments...}

    2. Multi-node multi-process case.

    .. code-block:: bash

        python -m torch.distributed.launch --nnodes={number_of_nodes} --node_rank={rank_of_this_node}
        --nproc_per_node={number_of_gpus} {your_script.py} {your_arguments...}

    If :meth:`preprocess` is defined by the task, it will be applied to ``train_set``, ``valid_set`` and ``test_set``.

    Parameters:
        task (nn.Module): task
        train_set (data.Dataset): training set
        valid_set (data.Dataset): validation set
        test_set (data.Dataset): test set
        optimizer (optim.Optimizer): optimizer
        scheduler (lr_scheduler._LRScheduler, optional): scheduler
        gpus (list of int, optional): GPU ids. By default, CPUs will be used.
            For multi-node multi-process case, repeat the GPU ids for each node.
        batch_size (int, optional): batch size of a single CPU / GPU
        gradient_interval (int, optional): perform a gradient update every n batches.
            This creates an equivalent batch size of ``batch_size * gradient_interval`` for optimization.
        num_worker (int, optional): number of CPU workers per GPU
        logger (str or core.LoggerBase, optional): logger type or logger instance.
            Available types are ``logging`` and ``wandb``.
        log_interval (int, optional): log every n gradient updates
    """

    def __init__(self, task, train_set, valid_set, test_set, optimizer, best_model_path, early_stopping=None, scheduler=None, gpus=None, batch_size=1,
                 gradient_interval=1, num_worker=0, logger="logging", log_interval=100):
        self.rank = comm.get_rank()
        self.world_size = comm.get_world_size()
        self.gpus = gpus
        self.batch_size = batch_size
        self.gradient_interval = gradient_interval
        self.num_worker = num_worker

        if gpus is None:
            self.device = torch.device("cpu")
        else:
            if len(gpus) != self.world_size:
                error_msg = "World size is %d but found %d GPUs in the argument"
                if self.world_size == 1:
                    error_msg += ". Did you launch with `python -m torch.distributed.launch`?"
                raise ValueError(error_msg % (self.world_size, len(gpus)))
            self.device = torch.device(gpus[self.rank % len(gpus)])

        if self.world_size > 1 and not dist.is_initialized():
            if self.rank == 0:
                module.logger.info("Initializing distributed process group")
            backend = "gloo" if gpus is None else "nccl"
            comm.init_process_group(backend, init_method="env://")

        if hasattr(task, "preprocess"):
            if self.rank == 0:
                module.logger.warning("Preprocess training set")
            # TODO: more elegant implementation
            # handle dynamic parameters in optimizer
            old_params = list(task.parameters())
            result = task.preprocess(train_set, valid_set, test_set)
            if result is not None:
                train_set, valid_set, test_set = result
            new_params = list(task.parameters())
            if len(new_params) != len(old_params):
                optimizer.add_param_group({"params": new_params[len(old_params):]})
        if self.world_size > 1:
            task = nn.SyncBatchNorm.convert_sync_batchnorm(task)
            buffers_to_ignore = []
            for name, buffer in task.named_buffers():
                if not isinstance(buffer, torch.Tensor):
                    buffers_to_ignore.append(name)
            task._ddp_params_and_buffers_to_ignore = set(buffers_to_ignore)
        if self.device.type == "cuda":
            task = task.cuda(self.device)

        # torch.save(train_set, "traindata.pth")
        # torch.save(valid_set, "validdata.pth")
        # train_set = torch.load("traindata.pth")
        # valid_set = torch.load("validdata.pth")

        self.model = task
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_model_path = best_model_path
        self.early_stopping = early_stopping

        

        if isinstance(logger, str):
            if logger == "logging":
                logger = core.LoggingLogger()
            elif logger == "wandb":
                logger = core.WandbLogger(project=task.__class__.__name__)
            else:
                raise ValueError("Unknown logger `%s`" % logger)
        self.meter = core.Meter(log_interval=log_interval, silent=self.rank > 0, logger=logger)
        self.meter.log_config(self.config_dict())
        
       

    def train(self, num_epoch=1, batch_per_epoch=None):
        """
        Train the model.

        If ``batch_per_epoch`` is specified, randomly draw a subset of the training set for each epoch.
        Otherwise, the whole training set is used for each epoch.

        Parameters:
            num_epoch (int, optional): number of epochs
            batch_per_epoch (int, optional): number of batches per epoch
        """
        sampler = torch_data.DistributedSampler(self.train_set, self.world_size, self.rank)
        dataloader = data.DataLoader(self.train_set, self.batch_size, sampler=sampler, num_workers=self.num_worker)
        # torch.save(self.train_set, "traindata.pth")
        # print("DATA SAVED")
        best_val_loss = float('inf')
        batch_per_epoch = batch_per_epoch or len(dataloader)
        model = self.model
        model.split = "train"
        loss_train = []
        loss_val = []
        acc_train = []
        acc_val = []
        if self.world_size > 1:
            if self.device.type == "cuda":
                model = nn.parallel.DistributedDataParallel(model, device_ids=[self.device],
                                                            find_unused_parameters=True)
            else:
                model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        model.train()
        wandb.watch(model, log = 'all', log_freq = 3)

        for epoch in self.meter(num_epoch):
            sampler.set_epoch(epoch)

            metrics = []
            start_id = 0
            # the last gradient update may contain less than gradient_interval batches
            gradient_interval = min(batch_per_epoch - start_id, self.gradient_interval)
            total_loss_train = 0
            total_samples_train = 0
            total_correct_train = 0
            num_batches = 0
            for batch_id, batch in enumerate(islice(dataloader, batch_per_epoch)):
                if self.device.type == "cuda":
                    batch = utils.cuda(batch, device=self.device)

                loss, metric = model(batch)
                if not loss.requires_grad:
                    raise RuntimeError("Loss doesn't require grad. Did you define any loss in the task?")
                total_loss_train += loss.item()
                loss = loss / gradient_interval
                loss.backward()
                metrics.append(metric)
                num_batches += 1

                # pred, target = model.module.predict_and_target(batch)
                if isinstance(model, nn.parallel.DistributedDataParallel):
                    actual_model = model.module
                else:
                    actual_model = model

                pred, target = actual_model.predict_and_target(batch)
                # pred, target = model.predict_and_target(batch)
                # print(f"pred: {pred}")
                # print(f"target: {target}")
                pred = torch.argmax(pred, dim=1)
                target = target.squeeze(1).long()
                total_correct_train += (pred == target).sum().item()
                total_samples_train += target.size(0)

                if batch_id - start_id + 1 == gradient_interval:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    metric = utils.stack(metrics, dim=0)
                    metric = utils.mean(metric, dim=0)
                    if self.world_size > 1:
                        metric = comm.reduce(metric, op="mean")
                    self.meter.update(metric)

                    metrics = []
                    start_id = batch_id + 1
                    gradient_interval = min(batch_per_epoch - start_id, self.gradient_interval)

           

            avg_loss_train = total_loss_train / num_batches
            loss_train.append(avg_loss_train)

            acc_train_epoch = total_correct_train / total_samples_train
            acc_train.append(acc_train_epoch)

            vloss, vacc = self.evaluate_loss('valid', False)
            if vloss < (best_val_loss):
                
                print(f'Validation loss decreased ( {best_val_loss:.6f} ------> {vloss:.6f} ).  Saving model ...')
                best_val_loss = vloss

                print('saving the best model in  --->>>>>>>>>>') ### check the path
                self.save(self.best_model_path) ## match the path here and in the main script
            if self.scheduler:
                self.scheduler.step(vloss)

            loss_val.append(vloss)
            acc_val.append(vacc)
            wandb.log({'wandb_val_loss_epoch':vloss, 'wandb_train_loss_epoch':avg_loss_train, 'wandbmy_val_accuracy':vacc, 'wandbmy_train_accuracy':acc_train_epoch, 'my_epoch': epoch})
            
            self.early_stopping(val_loss= vloss)
            if self.early_stopping.early_stop:
                print("early stopping")
                
                break
            

        loss_train_floats = [round(loss, 2) for loss in loss_train]
        loss_val_floats = [round(loss.cpu().item(), 2) for loss in loss_val]
        acc_train_floats = [round(acc*100, 2) for acc in acc_train]
        acc_val_floats = [round(acc*100, 2) for acc in acc_val]
        # self.meter.log({'loss_train_floats':loss_train_floats})
        # self.meter.log({'loss_val_floats':loss_val_floats})
        # wandb.log({"loss_train_floats": loss_train_floats})

        if self.rank == 0:  
            print("END OF TRAINING")
            print(f"tloss={loss_train_floats}")
            print(f"vloss={loss_val_floats}")
            print(f"tacc={acc_train_floats}")
            print(f"vacc={acc_val_floats}")
        

    @torch.no_grad()
    def evaluate_loss(self, split, log=False):
        """
        Evaluate the model.

        Parameters:
            split (str): split to evaluate. Can be ``train``, ``valid`` or ``test``.
            log (bool, optional): log metrics or not

        Returns:
            dict: metrics
        """
        if comm.get_rank() == 0:
            logger.warning(pretty.separator)
            logger.warning("Evaluate on %s" % split)
        test_set = getattr(self, "%s_set" % split)
        sampler = torch_data.DistributedSampler(test_set, self.world_size, self.rank)
        dataloader = data.DataLoader(test_set, self.batch_size, sampler=sampler, num_workers=self.num_worker)
        model = self.model
        model.split = 'valid'

        preds = []
        targets = []
        
        count = 0
        total_loss = 0
        correct = 0
        total = 0
        model.eval()
        #added
        # my_pred = []
        # my_target = []
        # eval_names = []
        for batch in dataloader:
            if self.device.type == "cuda":
                batch = utils.cuda(batch, device=self.device)

            loss, _ = model(batch)
            # pred, target = model.module.predict_and_target(batch)
            if isinstance(model, nn.parallel.DistributedDataParallel):
                actual_model = model.module
            else:
                actual_model = model

            pred, target = actual_model.predict_and_target(batch)
            # pred, target = model.predict_and_target(batch)
            total_loss += loss
            preds.append(pred)
            targets.append(target)
            # my_pred.extend(pred.cpu().numpy().astype(float))
            # my_target.extend(target.cpu().numpy().astype(int))
            # eval_names.extend(batch['name'])
            count +=1 
            pred = torch.argmax(pred, dim=1)
            target = target.squeeze(1).long()
            correct += (pred == target).sum().item()
            total += target.size(0)
            
        # print('engine_eval_name = ', eval_names)
        # print('engine_preds = ', my_pred)
        # print('engine_targets = ', my_target)
        # my_predictions = [np.argmax(i) for i in my_pred]
        # my_targets = [ i.item() for i in my_target]


        
            
            # count +=1 
            # pred = torch.argmax(pred, dim=1)
            # target = target.squeeze(1).long()
            # correct += (pred == target).sum().item()
            # total += target.size(0)
        vacc = correct / total
        avg_loss = total_loss / count   
    
        model.train()

       
        return avg_loss, vacc

    @torch.no_grad()
    def evaluate(self, split, log=True):
        """
        Evaluate the model.

        Parameters:
            split (str): split to evaluate. Can be ``train``, ``valid`` or ``test``.
            log (bool, optional): log metrics or not

        Returns:
            dict: metrics
        """
        if comm.get_rank() == 0:
            logger.warning(pretty.separator)
            logger.warning("Evaluate on %s" % split)
        test_set = getattr(self, "%s_set" % split)
        sampler = torch_data.DistributedSampler(test_set, self.world_size, self.rank)
        dataloader = data.DataLoader(test_set, self.batch_size, sampler=sampler, num_workers=self.num_worker)
        model = self.model
        model.split = split

        model.eval()
        preds = []
        targets = []
        
        #added
        my_pred = []
        my_target = []
        eval_names = []
        for batch in dataloader:
            if self.device.type == "cuda":
                batch = utils.cuda(batch, device=self.device)

            # pred, target = model.predict_and_target(batch)
            if isinstance(model, nn.parallel.DistributedDataParallel):
                actual_model = model.module
            else:
                actual_model = model

            pred, target = actual_model.predict_and_target(batch)
            preds.append(pred)
            targets.append(target)
            
            my_pred.extend(pred.cpu().numpy().astype(float))
            my_target.extend(target.cpu().numpy().astype(int))
            eval_names.extend(batch['name'])
        print('engine_eval_name = ', eval_names)
        print('engine_preds = ', my_pred)
        print('engine_targets = ', my_target)
        # my_predictions = [np.argmax(i) for i in my_pred]
        # my_targets = [ i.item() for i in my_target]


        
        # def get_acc(prediction,target):
        #     assert len(prediction) == len(target), "Lists must be of the same length."

        #     # Count the number of matches
        #     matches = sum(1 for p, t in zip(prediction, target) if p == t)

        #     # Calculate accuracy
        #     accuracy = matches / len(prediction)
        #     return accuracy

            

        # my_val_accuracy = get_acc(my_predictions,my_targets)
        
        pred = utils.cat(preds)
        target = utils.cat(targets)
        if self.world_size > 1:
            pred = comm.cat(pred)
            target = comm.cat(target)
        metric = model.evaluate(pred, target)
        if log:
            self.meter.log(metric, category="%s/epoch" % split)
            # self.meter.log({'my_val_accuracy':my_val_accuracy})

        return metric

    def load(self, checkpoint, load_optimizer=True, strict=True):
        """
        Load a checkpoint from file.

        Parameters:
            checkpoint (file-like): checkpoint file
            load_optimizer (bool, optional): load optimizer state or not
            strict (bool, optional): whether to strictly check the checkpoint matches the model parameters
        """
        if comm.get_rank() == 0:
            logger.warning("Load checkpoint from %s" % checkpoint)
        checkpoint = os.path.expanduser(checkpoint)
        state = torch.load(checkpoint, map_location=self.device)

        self.model.load_state_dict(state["model"], strict=strict)

        if load_optimizer:
            self.optimizer.load_state_dict(state["optimizer"])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
                        
        print("chk loaded>>> at engine")

        comm.synchronize()

    def save(self, checkpoint):
        """
        Save checkpoint to file.

        Parameters:
            checkpoint (file-like): checkpoint file
        """
        if comm.get_rank() == 0:
            logger.warning("Save checkpoint to %s" % checkpoint)
        checkpoint = os.path.expanduser(checkpoint)
        if self.rank == 0:
            state = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict()
            }
            torch.save(state, checkpoint)

        comm.synchronize()

    @classmethod
    def load_config_dict(cls, config):
        """
        Construct an instance from the configuration dict.
        """
        if getattr(cls, "_registry_key", cls.__name__) != config["class"]:
            raise ValueError("Expect config class to be `%s`, but found `%s`" % (cls.__name__, config["class"]))

        optimizer_config = config.pop("optimizer")
        new_config = {}
        for k, v in config.items():
            if isinstance(v, dict) and "class" in v:
                v = core.Configurable.load_config_dict(v)
            if k != "class":
                new_config[k] = v
        optimizer_config["params"] = new_config["task"].parameters()
        new_config["optimizer"] = core.Configurable.load_config_dict(optimizer_config)

        return cls(**new_config)

    @property
    def epoch(self):
        """Current epoch."""
        return self.meter.epoch_id
