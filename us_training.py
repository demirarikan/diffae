from templates import *
from templates_latent import *
import wandb
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Diffusion AE training")

    # Add the login option
    parser.add_argument('--login', type=str, help='Weights & Biases API key')

    # first the simulated path, then the real path to the datasets
    parser.add_argument('--dataset_path', type=str, nargs=2, metavar=('path1', 'path2'),
                            help='Simulated and real paths to the datasets')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    if args.login:
        wandb.login(key=args.login)
    # change this depending on the number of gpus
    gpus = [0]
    conf = mixed_us_training()
    conf.custom_dataset_path = args.dataset_path
    
    train(conf, gpus=gpus)


