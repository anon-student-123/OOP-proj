import argparse 
from vae import VAE 
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt 
from utils import plot_grid

def parse_args():
    '''
    sets up the arg parser
    '''
    parser = argparse.ArgumentParser(description="Train a bootleg VAE")

    parser.add_argument(
        "--dset",
        type=str,
        required=True,
        choices=["mnist_bw", "mnist_color"],
        help="Define which dataset to use",
    )
    parser.add_argument(
        "--epochs", type=int, required=True, help="Number of training epochs"
    )

    parser.add_argument(
        "--visualize_latent",
        action="store_true",
        help="Visualize the latent space with TSNE",
    )
    parser.add_argument(
        "--generate_from_prior",
        action="store_true",
        help="Generate images from prior p(z)",
    )
    parser.add_argument(
        "--generate_from_posterior",
        action="store_true",
        help="Generate images from posterior q(z | x)",
    )

    return parser.parse_args()

def visualize_latent_space(vae: VAE, test_data, test_labels, is_color, random_state):
    print(f'visualizing from latent space...')
    # get latent codes for test data
    z, _, _ = vae.get_latent_codes(test_data)

    # apply tsne
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=30)
    print(f'TNSE loaded...\nfitting model')
    z_2d = tsne.fit_transform(z.numpy())
    print('tsne model fitted')

    # create scatter plot colored by digit labels
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        z_2d[:, 0], z_2d[:, 1], c=test_labels, cmap="tab10", alpha=0.6, s=10
    )
    plt.colorbar(scatter, label="digit")
    plt.xlabel("TSNE component 1")
    plt.ylabel("TSNE component 2")
    plt.title("latent space visualization (TSNE)")
    filename = "latent_space_bw.pdf" if not is_color else "latent_space_color.pdf"
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"saved viz to {filename}")

def generate_from_prior(vae: VAE, is_color, num_samples=10):
    x_gen = vae.generate_from_prior(num_samples=num_samples).numpy()

    if is_color:
        images = x_gen
    else:
        images = x_gen.reshape(-1, 28, 28, 1)  # bw images need reshaping from flat

    # clip to valid range 0-1
    images = np.clip(
        images, 0, 1
    )  # since decoder output is not guaranteed to be in [0, 1]

    # plot grid
    prefix = 'xhat_color_' if is_color else 'xhat_bw_'
    plot_grid(images, N=10, C=10, name='prior')
    print(
        f"generated {num_samples} images from prior and saved to ./{prefix}prior.pdf"
    )

def generate_from_posterior(vae: VAE, test_data, is_color, num_samples=10):
    # take first num_samples from test set
    x_in = test_data[:num_samples]

    # encode and decode
    x_recon = vae.generate_from_posterior(x_in).numpy()

    # reshape for viz
    if is_color:
        images = x_recon
    else:
        images = x_recon.reshape(-1, 28, 28, 1)

    images = np.clip(images, 0, 1)

    prefix = 'xhat_color_' if is_color else 'xhat_bw_'
    plot_grid(images, N=10, C=10, name='posterior')
    print(
        f"generated {num_samples} images from posterior and saved to ./{prefix}posterior.pdf"
    )