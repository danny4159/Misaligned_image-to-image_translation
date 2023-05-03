import sys
import os
import subprocess
import argparse
from options.train_options import TrainOptions
from pGAN import train as pGAN_train
from cGAN import train as cGAN_train
# from ourGAN import train as ourGAN_train


if __name__ == "__main__":

    opt = TrainOptions().parse()
    model = opt.model
 
    if model == 'pGAN':
        pGAN_train(opt)
    elif model == 'cGAN':
        cGAN_train(opt)
    # elif model == 'ourGAN':
    #     ourGAN_train(opt)
    else:
        sys.exit(1)