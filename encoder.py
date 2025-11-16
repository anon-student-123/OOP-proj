from bicoder import BiCoder
import tensorflow as tf

class Encoder(BiCoder):
    '''
    maps input to 2 times latent dim outputs, getting the first half as 
    posterior mean and the second half as log variance
    during call, it splits those two and applies the reparameterization
    trick to sample z, returning z, mu, log_var 
    
    encoder class that learns the posterior distribution
    '''
    def __init__(self, network, latent_dim, name='encoder'):
        super().__init__(network, latent_dim, name=name)
    
    def call(self, x):
        '''
        overrides BiCoder.call() to implement encoding logic

        Returns: 
            z: sampled latent code
            mu: mean of posterior
            log_var: log var of posterior
        '''
        #pass thru network
        out = self.network(x)

        #split output into mean anb log var
        mu = out[:, :self.latent_dim]
        log_var = out[:, self.latent_dim:]

        #reparameterization trick (stolen from neural_networks.py)
        std = tf.math.exp(0.5*log_var)
        eps = tf.random.normal(mu.shape)
        z = mu + eps*std

        return z, mu, log_var 

    def sample_prior(self, num_samples=10):
        '''
        sample from prior
        useful for generating new images 
        '''
        return tf.random.normal(shape=(num_samples, self.latent_dim))
