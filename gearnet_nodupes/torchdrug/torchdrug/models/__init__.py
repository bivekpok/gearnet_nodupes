from .chebnet import ChebyshevConvolutionalNetwork
from .gcn import GraphConvolutionalNetwork, RelationalGraphConvolutionalNetwork
from .gat import GraphAttentionNetwork
from .gin import GraphIsomorphismNetwork
from .schnet import SchNet
from .mpnn import MessagePassingNeuralNetwork
from .neuralfp import NeuralFingerprint
from .infograph import InfoGraph, MultiviewContrast
from .flow import GraphAutoregressiveFlow

# ESM depends on the `fair-esm` package, which is large and optional for this
# project. Import it lazily so users who only need GearNet can run without it.
try:
    from .esm import EvolutionaryScaleModeling
    _ESM_AVAILABLE = True
    _ESM_IMPORT_ERROR = None
except ImportError as _e:
    EvolutionaryScaleModeling = None
    _ESM_AVAILABLE = False
    _ESM_IMPORT_ERROR = _e

from .embedding import TransE, DistMult, ComplEx, RotatE, SimplE
from .neurallp import NeuralLogicProgramming
from .kbgat import KnowledgeBaseGraphAttentionNetwork
from .cnn import ProteinConvolutionalNetwork, ProteinResNet
from .lstm import ProteinLSTM
from .bert import ProteinBERT
from .statistic import Statistic
from .physicochemical import Physicochemical
from .gearnet import GeometryAwareRelationalGraphNeuralNetwork

# alias
ChebNet = ChebyshevConvolutionalNetwork
GCN = GraphConvolutionalNetwork
GAT = GraphAttentionNetwork
RGCN = RelationalGraphConvolutionalNetwork
GIN = GraphIsomorphismNetwork
MPNN = MessagePassingNeuralNetwork
NFP = NeuralFingerprint
GraphAF = GraphAutoregressiveFlow
ESM = EvolutionaryScaleModeling
NeuralLP = NeuralLogicProgramming
KBGAT = KnowledgeBaseGraphAttentionNetwork
ProteinCNN = ProteinConvolutionalNetwork
GearNet = GeometryAwareRelationalGraphNeuralNetwork

__all__ = [
    "ChebyshevConvolutionalNetwork", "GraphConvolutionalNetwork", "RelationalGraphConvolutionalNetwork",
    "GraphAttentionNetwork", "GraphIsomorphismNetwork", "SchNet", "MessagePassingNeuralNetwork",
    "NeuralFingerprint",
    "InfoGraph", "MultiviewContrast",
    "GraphAutoregressiveFlow",
    "EvolutionaryScaleModeling", "ProteinConvolutionalNetwork", "GeometryAwareRelationalGraphNeuralNetwork",
    "Statistic", "Physicochemical",
    "TransE", "DistMult", "ComplEx", "RotatE", "SimplE",
    "NeuralLogicProgramming", "KnowledgeBaseGraphAttentionNetwork",
    "ChebNet", "GCN", "GAT", "RGCN", "GIN", "MPNN", "NFP",
    "GraphAF", "ESM", "NeuralLP", "KBGAT",
    "ProteinCNN", "ProteinResNet", "ProteinLSTM", "ProteinBERT", "GearNet",
]