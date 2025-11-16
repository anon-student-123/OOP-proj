from bicoder import BiCoder

class Decoder(BiCoder):
    '''
    maps latent vectors to reconstruction and assumes a fixed STD.
    during call, the latent batch z is passed through the network
    to produce the mean mu

    decoder class that learns the likelihood
    '''
    def __init__(self, network, latent_dim, std=0.75, name='decoder'):
        super().__init__(network, latent_dim, name=name)
        self.std = std #fixed std

    def call(self, z):
        '''
        overrides BiCoder.call() to implement decoding logic

        Returns:
            mu: mean of the likelihood, this is the reconstructed output
        '''
        #pass latent code through network
        mu = self.network(z)
        return mu