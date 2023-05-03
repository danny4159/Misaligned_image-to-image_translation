import sys
import os
import subprocess
import argparse
from options.test_options import TestOptions
from pGAN import test as pGAN_test
from cGAN import test as cGAN_test
# from ourGAN import test as ourGAN_test

if __name__ == "__main__":

    opt = TestOptions().parse()
    model = opt.model
 
    if model == 'pGAN':
        pGAN_test(opt)
    elif model == 'cGAN':
        cGAN_test(opt)
    # elif model == 'ourGAN':
    #     ourGAN_test(opt)
    else:
        sys.exit(1)