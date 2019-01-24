from collagen.core import Callback
from ._visualizers import ProgressbarVisualizer, TensorboardSynthesisVisualizer
from ._gan_callbacks import OnDiscriminatorBatchFreezer, OnGeneratorBatchFreezer, GeneratorLoss


class BackwardCallback(Callback):
    def __init__(self, retain_graph=True, create_graph=False):
        super().__init__()
        self.__retain_graph = retain_graph
        self.__create_graph = create_graph

    def on_backward_begin(self, session, **kwargs):
        session.set_backward_param(retain_graph=self.__retain_graph, create_graph=self.__create_graph)
