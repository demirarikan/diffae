from templates import *
from templates_latent import *

if __name__ == '__main__':
    gpus = [0]
    conf = sim_us_training()
    train(conf, gpus=gpus)