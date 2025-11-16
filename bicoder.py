from tensorflow.keras import layers
from abc import ABC, abstractmethod

class BiCoder(layers.Layer, ABC):
    '''
    abstract base class containing call method for de/encoder classes
    '''
    def __init__(self, network, latent_dim, name='bicoder'):
        '''
        @args: 
            network: sequential model (encoder_mlp, decoder_mlp, encoder_conv, decoder_conv)
            latent_dim: dim of latent space 
            name: name of the layer 
        '''
        super().__init__(name=name)
        self.network = network
        self.latent_dim = latent_dim 

    @abstractmethod
    def call(self, inputs):
        '''
        abstract method that must be implemented by encoder and decoder
        '''
        pass
