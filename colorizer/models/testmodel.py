from typing import Callable, Dict, Tuple

from colorizer.models.base import Model
from colorizer.datasets import FlowersDataset
from colorizer.networks import simple_cnn

class TestModel(Model):
    def  __init__(self, dataset_cls: type=FlowersDataset, network_fn: Callable=simple_cnn, dataset_args: Dict=None, network_args: Dict=None):
        """Define the default dataset and network values for this model."""
        super().__init__(dataset_cls, network_fn, dataset_args, network_args)