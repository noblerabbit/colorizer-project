"""Base model class. 
The boilerplate code for compiling and traning the model is present here """

from typing import Callable, Dict
import pathlib

import numpy as np
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.optimizers import Adam

from colorizer.datasets.base import Dataset


DIRNAME = pathlib.Path(__file__).parents[1].resolve() / 'weights/'
print("[INFO] DIRNAME is: {}".format(DIRNAME))

class Model:
    """Base class, to be subclassed by predictors for specific type of data."""
    def __init__(self, dataset_cls: type, network_fn: Callable, dataset_args: Dict=None, network_args: Dict=None):
        self.name = f'{self.__class__.__name__}_{dataset_cls.__name__}_{network_fn.__name__}'

        if dataset_args is None:
            dataset_args = {}
        self.data = dataset_cls(**dataset_args)

        if network_args is None:
            network_args = {}
            
        self.network = network_fn(self.data.input_shape, self.data.output_shape, **network_args)
        # self.network.summary()
        
    def weight_filename(self):
        DIRNAME.mkdir(parents=True, exists_ok=True)
        return str(DIRNAME/ f'{self.name}_weights.h5')
        
    def fit(self, dataset, batch_size=32, epochs=10, learning_rate = 0.001, callbacks=[]):

        self.network.compile(loss=self.loss(), optimizer=self.optimizer(learning_rate), metrics=self.metrics())
        
        self.network.fit(dataset.Xdata_train, dataset.Ydata_train, batch_size = batch_size, epochs = epochs,
                        validation_data=(dataset.Xdata_val, dataset.Ydata_val), callbacks=callbacks)
        
        ## TODO add option for fit_generator (in case dataset is too big to fit in ram or we need to augmnet it)

    def evaluate(self, x, y):
        ## TODO
        pass

    def loss(self):
        return 'mean_squared_error'

    def optimizer(self, learning_rate):
        return Adam(lr=learning_rate)

    def metrics(self):
        return ['accuracy']

    def load_weights(self):
        self.network.load_weights(self.weights_filename)

    def save_weights(self):
        self.network.save_weights(self.weights_filename)