import tensorflow as tf
import numpy as np
# import pickle
# from mpl_toolkits.axes_grid1 import ImageGrid

from utils import optimizer
from vae import VAE
from encoder import Encoder
from decoder import Decoder
from neural_networks import encoder_conv, encoder_mlp, decoder_conv, decoder_mlp
from dataloader import MNISTBWDataLoader, MNISTColorDataLoader
import random
from train_helpers import parse_args, visualize_latent_space, generate_from_posterior, generate_from_prior


'''
load_data func has been commented out after implementation of dataloader classes
'''
# def load_data(dset_name):
#     if dset_name == "mnist_bw":
#         # load training data
#         train_data = np.load("data/mnist_bw.npy")
#         train_data = (
#             train_data.astype(np.float32) / 255.0  # cast to correct type and normalize
#         )
#         train_data = train_data.reshape(train_data.shape[0], -1)  # flatten

#         # load test data
#         test_data = np.load("data/mnist_bw_te.npy")
#         test_data = (
#             test_data.astype(np.float32) / 255.0  # cast to correct type and normalize
#         )
#         test_data = test_data.reshape(test_data.shape[0], -1)  # flatten

#         # load labels
#         test_labels = np.load("data/mnist_bw_y_te.npy")
#         print(f"loaded mnist_bw\ntrain: {train_data.shape}\ntest: {test_data.shape}")
#         return (
#             train_data,
#             test_data,
#             test_labels,
#             False,  # bool to indicate not in color
#         )

#     # color
#     # load training data
#     with open("data/mnist_color.pkl", "rb") as f:
#         train_dict = pickle.load(f)
#     train_data = train_dict["m0"].astype(
#         np.float32
#     )  # already normalized, we can also change the color version

#     # test data
#     with open("data/mnist_color_te.pkl", "rb") as f:
#         test_dict = pickle.load(f)
#     test_data = test_dict["m0"].astype(
#         np.float32
#     )  # already normalized, we can also change the color version

#     # labels
#     test_labels = np.load("data/mnist_color_y_te.npy")
#     print(f"loaded mnist_color\ntrain: {train_data.shape}\ntest: {test_data.shape}")
#     return train_data, test_data, test_labels, True



def main():
    RANDOM_STATE = 42
    BATCH_SIZE = 256
    SHUFFLE_BUFFER = 10_000
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)
    args = parse_args()

    # load data
    #train_data, test_data, test_labels, is_color = load_data(args.dset)


    if args.dset == "mnist_bw":
        print(f"building VAE for bw mnist")
        dataloader = MNISTBWDataLoader()
        is_color = False
        encoder = Encoder(encoder_mlp, latent_dim=20, name="encoder_bw")
        decoder = Decoder(decoder_mlp, latent_dim=20, std=0.75, name="decoder_bw")
    else:
        print(f"building VAE for colorful mnist")
        dataloader = MNISTColorDataLoader()
        is_color=True
        encoder = Encoder(
            encoder_conv, latent_dim=50, name="encoder_color"
        )  # use conv network for colorful mnist
        decoder = Decoder(decoder_conv, latent_dim=50, std=0.75, name="decoder_color")

    test_data, test_labels = dataloader.get_test_data()
    train_dataset = dataloader.get_training_data(batch_size=BATCH_SIZE, shuffle_buffer=SHUFFLE_BUFFER, seed=RANDOM_STATE)

    vae = VAE(encoder, decoder)

    

    print(f"training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_dataset:
            loss = vae.train(batch, optimizer)
            epoch_loss += loss.numpy()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"epoch {epoch + 1} / {args.epochs}\nloss: {avg_loss:.4f}")

    print(f"training complete")

    if args.visualize_latent:
        print(f"\nvisualizing latent space...")
        visualize_latent_space(vae, test_data, test_labels, is_color, RANDOM_STATE)
    if args.generate_from_prior:
        print(f"\ngenerating images from prior")
        generate_from_prior(vae, is_color, num_samples=100)
    if args.generate_from_posterior:
        print(f"\ngenerating images from posterior")
        generate_from_posterior(vae, test_data, is_color, num_samples=100)


if __name__ == "__main__":
    main()
