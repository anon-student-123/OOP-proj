import numpy as np
from losses import kl_divergence, log_diag_mvn
import tensorflow as tf

class VAE(tf.keras.Model):
    '''
    VAE class that combines encoder and decoder
    performs the full forward pass, exposes a train step and 
    offers helpers for sampling from the prior/posterior and inspecting latent codes 
    '''
    def __init__(self, encoder, decoder, name='vae'):
        '''
        Args: 
            encoder: encoder object
            decoder: decoder object
        '''
        super().__init__(name=name)
        self.encoder = encoder
        self.decoder = decoder
        self.vae_loss = None #stores the loss

    def call(self, x):
        '''
        overrides Model.call() to implement forward pass and compute loss
        Args: 
            x: input data 
        Returns: 
            total loss
        '''
        #encode: get latent code and distribution params
        z, mu, log_var = self.encoder(x)

        #decode: get reconstruction
        x_recon = self.decoder(z)

        log_sig = tf.math.log(self.decoder.std)
        recon_loss = log_diag_mvn(x, x_recon, log_sig)

        kl_loss = kl_divergence(mu, log_var)
        L = recon_loss - kl_loss 
        loss = tf.reduce_mean(L) #avg over the batch

        self.vae_loss = -loss #negate since all optimizers minimize

        return -loss

    #stolen from utils
    @tf.function
    def train(self, x, optimizer):
        '''
        training step that updates model params

        Args: 
            x: input batch 
            optimizer: tf optimizer 
        
        Returns: 
            loss: computed loss value 
        '''
        with tf.GradientTape() as tape:
            loss = self.call(x)
        # self.vae_los = - ELBO
        # where ELBO is given in Equation 1
        gradients = tape.gradient(self.vae_loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss

    def generate_from_prior(self, num_samples=10):
        '''
        generate samples by sampling z from prior
        '''
        z = self.encoder.sample_prior(num_samples)
        x_gen = self.decoder(z)
        return x_gen 

    def generate_from_posterior(self, x):
        '''
        generate samples by encoding x and then decoding
        '''
        z, _, _ = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

    def get_latent_codes(self, x):
        '''
        get latent representations for visualization
        '''
        z, mu, log_var = self.encoder(x)
        return z, mu, log_var 