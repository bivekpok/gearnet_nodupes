import math
from collections import defaultdict
import os


import torch
from torch import nn
from torch.nn import functional as F

from torchdrug import core, layers, tasks, metrics, utils
from torchdrug.core import Registry as R
from torchdrug.layers import functional
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

@R.register("tasks.PropertyPrediction")
class PropertyPrediction(tasks.Task, core.Configurable):
    """
    Graph / molecule / protein property prediction task.

    This class is also compatible with semi-supervised learning.

    Parameters:
        model (nn.Module): graph representation model
        task (str, list or dict, optional): training task(s).
            For dict, the keys are tasks and the values are the corresponding weights.
        criterion (str, list or dict, optional): training criterion(s). For dict, the keys are criterions and the values
            are the corresponding weights. Available criterions are ``mse``, ``bce`` and ``ce``.
        metric (str or list of str, optional): metric(s).
            Available metrics are ``mae``, ``rmse``, ``auprc`` and ``auroc``.
        num_mlp_layer (int, optional): number of layers in mlp prediction head
        normalization (bool, optional): whether to normalize the target
        num_class (int, optional): number of classes
        mlp_batch_norm (bool, optional): apply batch normalization in mlp or not
        mlp_dropout (float, optional): dropout in mlp
        graph_construction_model (nn.Module, optional): graph construction model
        class_weights (list, torch.Tensor, optional): class weights for imbalanced cross-entropy loss.
        verbose (int, optional): output verbose level
    """

    eps = 1e-10
    _option_members = {"task", "criterion", "metric"}
 # fold_idx _to save diffrent fold outputs
    def __init__(self, model, task=(), criterion="mse", metric=("mae", "rmse"), num_mlp_layer=1,
                 normalization=True, num_class=None, mlp_batch_norm=False, mlp_dropout=0,
                 graph_construction_model=None, class_weights=None, verbose=0):
        super(PropertyPrediction, self).__init__()
        self.model = model
        self.task = task
        self.criterion = criterion
        self.metric = metric
        self.num_mlp_layer = num_mlp_layer
        # self.fold_idx=fold_idx
        # For classification tasks, we disable normalization tricks.
        self.normalization = normalization and ("ce" not in criterion) and ("bce" not in criterion)
        self.num_class = (num_class,) if isinstance(num_class, int) else num_class
        self.mlp_batch_norm = mlp_batch_norm
        self.mlp_dropout = mlp_dropout
        self.graph_construction_model = graph_construction_model
        self.class_weights = class_weights
        self.verbose = verbose
        self.all_name =[]
        

    def preprocess(self, train_set, valid_set, test_set):
        """
        Compute the mean and derivation for each task on the training set.
        """
        values = defaultdict(list)
        for sample in train_set:
            if not sample.get("labeled", True):
                continue
            # print(self.task)
            for task in self.task:
                if not math.isnan(sample[task]):


                    
                    values[task].append(sample[task])
        mean = []
        std = []
        weight = []
        num_class = []
        for task, w in self.task.items():
            value = torch.tensor(values[task])
            mean.append(value.float().mean())
            std.append(value.float().std())
            weight.append(w)
            if value.ndim > 1:
                num_class.append(value.shape[1])
            elif value.dtype == torch.long:
                task_class = value.max().item()
                if task_class == 1 and "bce" in self.criterion:
                    num_class.append(1)
                else:
                    num_class.append(task_class + 1)
            else:
                num_class.append(1)

        

        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float))
        self.register_buffer("weight", torch.as_tensor(weight, dtype=torch.float))
        self.num_class = self.num_class or num_class

        hidden_dims = [self.model.output_dim] * (self.num_mlp_layer - 1)
        self.mlp = layers.MLP(self.model.output_dim, hidden_dims + [sum(self.num_class)],
                            batch_norm=self.mlp_batch_norm, dropout=self.mlp_dropout)


    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred = self.predict(batch, all_loss, metric)
        

        # print(f'inside forward { pred}')
        



        if all([t not in batch for t in self.task]):
            # unlabeled data
            return all_loss, metric

        target = self.target(batch)
        labeled = ~torch.isnan(target)
        target[~labeled] = 0
        def compute_focal_loss(pred, target, weights, gamma=2.0):
            """Computes Focal Loss to address class imbalance."""
            ce_loss = F.cross_entropy(pred, target.long().squeeze(-1), reduction="none")  # No class weights here
            pt = torch.exp(-ce_loss)  # Probability of the true class
            focal_loss = (1 - pt) ** gamma * ce_loss
            focal_loss = focal_loss * weights[target.long().squeeze(-1)]  # Apply class weights after focal term
            return focal_loss.unsqueeze(-1)

        for criterion, weight in self.criterion.items():
            if criterion == "mse":
                if self.normalization:
                    loss = F.mse_loss((pred - self.mean) / self.std, (target - self.mean) / self.std, reduction="none")
                else:
                    loss = F.mse_loss(pred, target, reduction="none")
            elif criterion == "bce":
                loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            elif criterion == "ce":
                if self.class_weights is not None:
                    # 1. Convert to tensor and move to the correct GPU
                    if not isinstance(self.class_weights, torch.Tensor):
                        weight_tensor = torch.tensor(self.class_weights, dtype=torch.float32, device=pred.device)
                    else:
                        # If it's already a tensor, just make sure it's on the right GPU
                        weight_tensor = self.class_weights.to(pred.device)
                    loss = F.cross_entropy(pred, target.long().squeeze(-1), weight=weight_tensor, reduction="mean").unsqueeze(-1)
                ### seq_homo_0.3
                # weights = torch.tensor([
                #                         2.2662,   # Archaebac.
                #                         1.4066,   # Endoplasm. reticulum
                #                         0.2549,   # Eykaryo. plasma
                #                         7.4167,   # Golgi
                #                         0.3413,   # Gram-neg. inner
                #                         0.7156,   # Gram-neg. outer
                #                         0.9378,   # Gram-pos. inner
                #                         9.0648,   # Lysosome
                #                         1.5689,   # Mitochon. inner
                #                         6.2756,   # Mitochon. outer
                #                         3.2633,   # Thylakoid
                #                         5.4389    # Vacuole
                #                     ], device=pred.device) # <-- DYNAMIC DEVICE FIX
                # weights = weights / weights.mean()
                ##seq_homo_0.7
                # weights = torch.tensor([0.6175, 0.4460, 0.0673, 2.4699, 0.1026, 0.2032, 0.2817, 2.4699, 0.4522,
                #             2.4699, 0.7298, 1.6900], device = pred.device) # already normalized
                # weights = weights / weights.mean()
                else: 
                    print("!!! not using weights for the Loss")
                    loss = F.cross_entropy(pred, target.long().squeeze(-1), reduction="mean").unsqueeze(-1)
                # loss = F.cross_entropy(pred, target.long().squeeze(-1), reduction="mean").unsqueeze(-1)
            elif criterion == "focal":  # Implement Focal Loss
                    # Define class weights (inverse class frequency)
                if self.class_weights is not None:
                    loss = compute_focal_loss(pred, target, self.class_weights)
                else:
                    loss = compute_focal_loss(pred, target)
            else:
                raise ValueError("Unknown criterion `%s`" % criterion)
            loss = functional.masked_mean(loss, labeled, dim=0)

            name = tasks._get_criterion_name(criterion)
            if self.verbose > 0:
                for t, l in zip(self.task, loss):
                    metric["%s [%s]" % (name, t)] = l
            loss = (loss * self.weight).sum() / self.weight.sum()
            metric[name] = loss
            all_loss += loss * weight

        return all_loss, metric

    def predict(self, batch, all_loss=None, metric=None):
        # self.all_name =[]
        graph = batch["graph"]
        name_batch = batch['name']
        self.all_name.append(name_batch)
        
     
        if self.graph_construction_model:
            graph = self.graph_construction_model(graph)
          

        output = self.model(graph, graph.node_feature.float(), all_loss=all_loss, metric=metric)
        pred = self.mlp(output["graph_feature"])
        # print('Inside the task/proterty_prediction/predict')
        # num_params_m22 = sum(p.numel() for p in self.mlp.parameters() if p.requires_grad)
        # print(f'num_params_m2 {num_params_m22}')
        if self.normalization:
            pred = pred * self.std + self.mean
        return pred

    def target(self, batch):
        target = torch.stack([batch[t].float() for t in self.task], dim=-1)
        labeled = batch.get("labeled", torch.ones(len(target), dtype=torch.bool, device=target.device))
        target[~labeled] = math.nan
        return target

    def evaluate(self, pred, target):
        labeled = ~torch.isnan(target)
        metric = {}
        for _metric in self.metric:
            if _metric == "mae":
                score = F.l1_loss(pred, target, reduction="none")
                score = functional.masked_mean(score, labeled, dim=0)
            elif _metric == "rmse":
                score = F.mse_loss(pred, target, reduction="none")
                score = functional.masked_mean(score, labeled, dim=0).sqrt()
            elif _metric == "acc":
                # print('inside acc pred')
                # print(pred)
                # print('inside acc target')
                # print(target)
                score = []
                num_class = 0
                print(f'self.num_class {self.num_class}')
                for i, cur_num_class in enumerate(self.num_class):
                    _pred = pred[:, num_class:num_class + cur_num_class]
                    _target = target[:, i]
                    _labeled = labeled[:, i]
                    _score = metrics.accuracy(_pred[_labeled], _target[_labeled].long())
                    score.append(_score)
                    num_class += cur_num_class ## uncommnet for multigpu test

                score = torch.stack(score)
  
               
              
                
            elif _metric == "mcc":
                score = []
                num_class = 0
                for i, cur_num_class in enumerate(self.num_class):
                    _pred = pred[:, num_class:num_class + cur_num_class]
                    _target = target[:, i]
                    _labeled = labeled[:, i]
                    _score = metrics.matthews_corrcoef(_pred[_labeled], _target[_labeled].long())
                    score.append(_score)
                    num_class += cur_num_class
                score = torch.stack(score)
            elif _metric == "auroc":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.long().t(), labeled.t()):
                    _score = metrics.area_under_roc(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            elif _metric == "auprc":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.long().t(), labeled.t()):
                    _score = metrics.area_under_prc(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            elif _metric == "r2":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.t(), labeled.t()):
                    _score = metrics.r2(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            elif _metric == "spearmanr":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.t(), labeled.t()):
                    _score = metrics.spearmanr(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            elif _metric == "pearsonr":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.t(), labeled.t()):
                    _score = metrics.pearsonr(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            else:
                raise ValueError("Unknown metric `%s`" % _metric)

            name = tasks._get_metric_name(_metric)
            for t, s in zip(self.task, score):
                metric["%s [%s]" % (name, t)] = s

        return metric


@R.register("tasks.MultipleBinaryClassification")
class MultipleBinaryClassification(tasks.Task, core.Configurable):
    """
    Multiple binary classification task for graphs / molecules / proteins.

    Parameters:
        model (nn.Module): graph representation model
        task (list of int, optional): training task id(s).
        criterion (list or dict, optional): training criterion(s). For dict, the keys are criterions and the values
            are the corresponding weights. Available criterions are ``bce``.
        metric (str or list of str, optional): metric(s).
            Available metrics are ``auroc@macro``, ``auprc@macro``, ``auroc@micro``, ``auprc@micro`` and ``f1_max``.
        num_mlp_layer (int, optional): number of layers in the MLP prediction head
        normalization (bool, optional): whether to normalize the target
        reweight (bool, optional): whether to re-weight tasks according to the number of positive samples
        graph_construction_model (nn.Module, optional): graph construction model
        verbose (int, optional): output verbose level
    """

    eps = 1e-10
    _option_members = {"criterion", "metric"}

    def __init__(self, model, task=(), criterion="bce", metric=("auprc@micro", "f1_max"), num_mlp_layer=1,
                 normalization=True, reweight=False, graph_construction_model=None, verbose=0):
        super(MultipleBinaryClassification, self).__init__()
        self.model = model
        self.task = task
        self.register_buffer("task_indices", torch.LongTensor(task))
        self.criterion = criterion
        self.metric = metric
        self.num_mlp_layer = num_mlp_layer
        self.normalization = normalization
        self.reweight = reweight
        self.graph_construction_model = graph_construction_model
        self.verbose = verbose

        hidden_dims = [self.model.output_dim] * (self.num_mlp_layer - 1)
        self.mlp = layers.MLP(self.model.output_dim, hidden_dims + [len(task)])

    def preprocess(self, train_set, valid_set, test_set):
        """
        Compute the weight for each task on the training set.
        """
        values = []
        for data in train_set:
            values.append(data["targets"][self.task_indices])
        values = torch.stack(values, dim=0)    
        
        if self.reweight:
            num_positive = values.sum(dim=0)
            weight = (num_positive.mean() / num_positive).clamp(1, 10)
        else:
            weight = torch.ones(len(self.task), dtype=torch.float)

        self.register_buffer("weight", torch.as_tensor(weight, dtype=torch.float))

    def forward(self, batch):
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred = self.predict(batch, all_loss, metric)
        target = self.target(batch)

        for criterion, weight in self.criterion.items():
            if criterion == "bce":
                loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            else:
                raise ValueError("Unknown criterion `%s`" % criterion)
            loss = loss.mean(dim=0)
            loss = (loss * self.weight).sum() / self.weight.sum()
            
            name = tasks._get_criterion_name(criterion)
            metric[name] = loss
            all_loss += loss * weight

        return all_loss, metric

    def predict(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        if self.graph_construction_model:
            graph = self.graph_construction_model(graph)
        output = self.model(graph, graph.node_feature.float(), all_loss=all_loss, metric=metric)
        pred = self.mlp(output["graph_feature"])
        return pred

    def target(self, batch):
        target = batch["targets"][:, self.task_indices]
        return target

    def evaluate(self, pred, target):
        metric = {}
        for _metric in self.metric:
            if _metric == "auroc@micro":
                score = metrics.area_under_roc(pred.flatten(), target.long().flatten())
            elif _metric == "auroc@macro":
                score = metrics.variadic_area_under_roc(pred, target.long(), dim=0).mean()
            elif _metric == "auprc@micro":
                score = metrics.area_under_prc(pred.flatten(), target.long().flatten())
            elif _metric == "auprc@macro":
                score = metrics.variadic_area_under_prc(pred, target.long(), dim=0).mean()
            elif _metric == "f1_max":
                score = metrics.f1_max(pred, target)
            else:
                raise ValueError("Unknown criterion `%s`" % _metric)

            name = tasks._get_metric_name(_metric)
            metric[name] = score

        return metric


@R.register("tasks.NodePropertyPrediction")
class NodePropertyPrediction(tasks.Task, core.Configurable):
    """
    Node / atom / residue property prediction task.

    Parameters:
        model (nn.Module): graph representation model
        criterion (str, list or dict, optional): training criterion(s). For dict, the keys are criterions and the values
            are the corresponding weights. Available criterions are ``mse``, ``bce`` and ``ce``.
        metric (str or list of str, optional): metric(s).
            Available metrics are ``mae``, ``rmse``, ``auprc`` and ``auroc``.
        num_mlp_layer (int, optional): number of layers in mlp prediction head
        normalization (bool, optional): whether to normalize the target
            Available entities are ``node``, ``atom`` and ``residue``.
        num_class (int, optional): number of classes
        verbose (int, optional): output verbose level
    """

    _option_members = {"criterion", "metric"}

    def __init__(self, model, criterion="bce", metric=("macro_auprc", "macro_auroc"), num_mlp_layer=1,
                 normalization=True, num_class=None, verbose=0):
        super(NodePropertyPrediction, self).__init__()
        self.model = model
        self.criterion = criterion
        self.metric = metric
        # For classification tasks, we disable normalization tricks.
        self.normalization = normalization and ("ce" not in criterion) and ("bce" not in criterion)
        self.num_mlp_layer = num_mlp_layer
        self.num_class = num_class
        self.verbose = verbose

    def preprocess(self, train_set, valid_set, test_set):
        """
        Compute the mean and derivation on the training set.
        """
        self.view = getattr(train_set[0]["graph"], "view", "atom")
        values = torch.cat([data["graph"].target for data in train_set])
        mean = values.float().mean()
        std = values.float().std()
        if values.dtype == torch.long:
            num_class = values.max().item()
            if num_class > 1 or "bce" not in self.criterion:
                num_class += 1
        else:
            num_class = 1

        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float))
        self.num_class = self.num_class or num_class

        if hasattr(self.model, "node_output_dim"):
            model_output_dim = self.model.node_output_dim
        else:
            model_output_dim = self.model.output_dim
        hidden_dims = [model_output_dim] * (self.num_mlp_layer - 1)
        self.mlp = layers.MLP(model_output_dim, hidden_dims + [self.num_class])

    def predict(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        output = self.model(graph, graph.node_feature.float(), all_loss=all_loss, metric=metric)
        if self.view in ["node", "atom"]:
            output_feature = output["node_feature"]
        else:
            output_feature = output.get("residue_feature", output.get("node_feature"))
        pred = self.mlp(output_feature)
        if self.normalization:
            pred = pred * self.std + self.mean
        return pred

    def target(self, batch):
        size = batch["graph"].num_nodes if self.view in ["node", "atom"] else batch["graph"].num_residues
        return {
            "label": batch["graph"].target,
            "mask": batch["graph"].mask,
            "size": size
        }

    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred, target = self.predict_and_target(batch, all_loss, metric)
        labeled = ~torch.isnan(target["label"]) & target["mask"]

        for criterion, weight in self.criterion.items():
            if criterion == "mse":
                if self.normalization:
                    loss = F.mse_loss((pred - self.mean) / self.std, (target - self.mean) / self.std, reduction="none")
                else:
                    loss = F.mse_loss(pred, target, reduction="none")
            elif criterion == "bce":
                loss = F.binary_cross_entropy_with_logits(pred, target["label"].float(), reduction="none")
            elif criterion == "ce":
                loss = F.cross_entropy(pred, target["label"], reduction="none")
            else:
                raise ValueError("Unknown criterion `%s`" % criterion)
            loss = functional.masked_mean(loss, labeled, dim=0)

            name = tasks._get_criterion_name(criterion)
            metric[name] = loss
            all_loss += loss * weight

        all_loss += loss

        return all_loss, metric

    def evaluate(self, pred, target):
        metric = {}
        _target = target["label"]
        _labeled = ~torch.isnan(_target) & target["mask"]
        _size = functional.variadic_sum(_labeled.long(), target["size"]) 
        for _metric in self.metric:
            if _metric == "micro_acc":
                score = metrics.accuracy(pred[_labeled], _target[_labeled].long())
            elif metric == "micro_auroc":
                score = metrics.area_under_roc(pred[_labeled], _target[_labeled])
            elif metric == "micro_auprc":
                score = metrics.area_under_prc(pred[_labeled], _target[_labeled])
            elif _metric == "macro_auroc":
                score = metrics.variadic_area_under_roc(pred[_labeled], _target[_labeled], _size).mean()
            elif _metric == "macro_auprc":
                score = metrics.variadic_area_under_prc(pred[_labeled], _target[_labeled], _size).mean()
            elif _metric == "macro_acc":
                score = pred[_labeled].argmax(-1) == _target[_labeled]
                score = functional.variadic_mean(score.float(), _size).mean()
            else:
                raise ValueError("Unknown criterion `%s`" % _metric)

            name = tasks._get_metric_name(_metric)
            metric[name] = score

        return metric


@R.register("tasks.InteractionPrediction")
@utils.copy_args(PropertyPrediction, ignore=("graph_construction_model",))
class InteractionPrediction(PropertyPrediction):
    """
    Predict the interaction property of graph pairs.

    Parameters:
        model (nn.Module): graph representation model
        model2 (nn.Module, optional): graph representation model for the second item. If ``None``, use tied-weight
            model for the second item.
        **kwargs
    """

    def __init__(self, model, model2=None, **kwargs):
        super(InteractionPrediction, self).__init__(model, **kwargs)
        self.model2 = model2 or model

    def preprocess(self, train_set, valid_set, test_set):
        """
        Compute the mean and derivation for each task on the training set.
        """
        values = defaultdict(list)
        for sample in train_set:
            if not sample.get("labeled", True):
                continue
            for task in self.task:
                if not math.isnan(sample[task]):
                    values[task].append(sample[task])
        mean = []
        std = []
        weight = []
        num_class = []
        for task, w in self.task.items():
            value = torch.tensor(values[task])
            mean.append(value.float().mean())
            std.append(value.float().std())
            weight.append(w)
            if value.ndim > 1:
                num_class.append(value.shape[1])
            elif value.dtype == torch.long:
                task_class = value.max().item()
                if task_class == 1 and "bce" in self.criterion:
                    num_class.append(1)
                else:
                    num_class.append(task_class + 1)
            else:
                num_class.append(1)

        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float))
        self.register_buffer("weight", torch.as_tensor(weight, dtype=torch.float))
        self.num_class = self.num_class or num_class

        hidden_dims = [self.model.output_dim] * (self.num_mlp_layer - 1)
        self.mlp = layers.MLP(self.model.output_dim + self.model2.output_dim, hidden_dims + [sum(self.num_class)])

    def predict(self, batch, all_loss=None, metric=None):
        graph1 = batch["graph1"]
        output1 = self.model(graph1, graph1.node_feature.float(), all_loss=all_loss, metric=metric)
        graph2 = batch["graph2"]
        output2 = self.model2(graph2, graph2.node_feature.float(), all_loss=all_loss, metric=metric)
        pred = self.mlp(torch.cat([output1["graph_feature"], output2["graph_feature"]], dim=-1))
        if self.normalization:
            pred = pred * self.std + self.mean
        return pred


@R.register("tasks.Unsupervised")
class Unsupervised(nn.Module, core.Configurable):
    """
    Wrapper task for unsupervised learning.

    The unsupervised loss should be computed by the model.

    Parameters:
        model (nn.Module): any model
    """

    def __init__(self, model, graph_construction_model=None):
        super(Unsupervised, self).__init__()
        self.model = model
        self.graph_construction_model = graph_construction_model

    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred = self.predict(batch, all_loss, metric)

        return all_loss, metric

    def predict(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        if self.graph_construction_model:
            graph = self.graph_construction_model(graph)
        pred = self.model(graph, graph.node_feature.float(), all_loss=all_loss, metric=metric)
        return pred
