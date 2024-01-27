from templates import *
from templates_latent import *
import wandb
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Diffusion AE training")

    # Add the login option
    parser.add_argument('--login', type=str, help='Weights & Biases API key')

    # Add the datatype option with choices real, sim, or mixed
    parser.add_argument('--datatype', type=str, choices=['real', 'sim', 'mixed'],
                        help='Type of dataset: real, sim, or mixed')

    # Add the dataset_path option
    parser.add_argument('--dataset_path', type=str, help='Path to the dataset')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    wandb.login(args.login)
    gpus = [0]
    if args.datatype == 'sim':
        conf = sim_us_training()
        conf.custom_dataset_path = args.dataset_path
    elif args.datatype == 'real':
        conf = real_us_training()
        conf.custom_dataset_path = args.dataset_path
    elif args.datatype == 'mixed':
        raise NotImplementedError()
    
    train(conf, gpus=gpus)


