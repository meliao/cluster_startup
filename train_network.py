import os
import argparse
import logging
import torch
import numpy as np

from torch.optim.lr_scheduler import MultiStepLR


from src.train_loop import train_L_layers
from src.data_gen import gen_data
from src.network import Llayers

def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-L_vals', nargs='+', default=[2, 4])
    parser.add_argument('-weight_decay_vals', nargs='+', default=[1e-03, ])
    parser.add_argument('-R_val', type=int, default=2)
    parser.add_argument('-dataset_size', default=64)
    parser.add_argument('-n_epochs', type=int, default=10)
    parser.add_argument('-n_epochs_per_log', type=int, default=1)
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('-seed', default=1234, type=int)
    parser.add_argument('-train_results_dir')
    parser.add_argument('-models_dir')

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:

    #######################################################################
    # Make sure output paths exist & are writeable

    for x in [args.train_results_dir, args.models_dir]:
        if not os.path.isdir(x):
            os.mkdir(x)

    #######################################################################
    # Set random seeds

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    #######################################################################
    # Figure out CUDA

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info("Computing on device: %s", device)

    #######################################################################
    # Generate data

    logging.info("Generating data")

    trainX, trainY, testX, testY = gen_data(args.dataset_size, 
                                            args.R_val)


    for L in args.L_vals:
        for wd in args.weight_decay_vals:
            logging.info("L=%i and wd=%f", L, wd)

            ##############################################################
            # Generate model

            model = Llayers(L, d=20, width=1_000)

            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            num_params = sum([np.prod(p.size()) for p in model_parameters])
            logging.debug("Model has %i trainable parameters", num_params)

            train_results_fp = os.path.join(args.train_results_dir, f'L_{L}_wd_{wd}.txt')
            models_dir = os.path.join(args.models_dir, f'L_{L}_wd_{wd}')
            os.mkdir(models_dir)

            
            train_L_layers(model=model,
                            trainX=trainX,
                            trainY=trainY,
                            testX=testX,
                            testY=testY,
                            train_results_fp=train_results_fp,
                            models_dir=models_dir,
                            weight_decay=wd,
                            epochs=args.n_epochs,
                            n_epochs_per_log=args.n_epochs_per_log,
                            scheduler=MultiStepLR,
                            milestones=[30_000], 
                            gamma=0.1)

    logging.info("Finished")

if __name__ == "__main__":


    args = setup_args()

    FMT = "%(asctime)s:cluster_startup: %(levelname)s - %(message)s"
    TIMEFMT = '%Y-%m-%d %H:%M:%S'

    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    
    logging.basicConfig(level=level,
                    format=FMT,
                    datefmt=TIMEFMT)

    main(args)