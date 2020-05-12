import argparse
import os

import torch

from argparse import Namespace


def get_default_args(desired_hparams, desired_params):
    
    hparams = {
        'batch_size': 4,
        'lr': 0.0005,
        'lr_factor': 1,
        'epochs': 10,
        'max_seq_len': 250,
        'train_pct': 1.0,
        'val_pct': 1.0,
        'seed': 404,
    }

    params = {
        'sst_train_path': os.path.join("..", "..", "..", "..", "data", "SST-2", "sentence_splits", "train_cat.tsv"),
        'sst_val_path': os.path.join("..", "..", "..", "..", "data", "SST-2", "sentence_splits", "dev_cat.tsv"),
        'disable_cuda': False,
        'run_name': 'default',
        'debug': False,
        'log_level': 'info',
        'num_saved_models': 1,
        'early_stopping': None,
    }
    if torch.cuda.is_available() and not params['disable_cuda']:
        params['device'] = torch.device('cuda')
    else:
        params['device'] = torch.device('cpu')

    hparams = Namespace(**dict(hparams, **desired_hparams))
    params = Namespace(**dict(params, **desired_params))
    return hparams, params


def parse_args():
    parser = argparse.ArgumentParser()

    #############################
    ## Hyperparameter arguments
    #############################
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--lr", default=0.0005, type=float)
    parser.add_argument("--lr_factor", default=1.0, type=float, help="Factor by which to reduce the learning rate at each milestone")
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--max_seq_len", default=250, type=int)
    parser.add_argument("--train_pct", default=1.0, type=float)
    parser.add_argument("--val_pct", default=1.0, type=float)
    parser.add_argument("--seed", type=int, default=404)

    #############################
    ## Dataset arguments
    #############################
    parser.add_argument("--sst_train_path", default=os.path.join("..", "..", "..", "..", "data", "SST-2", "sentence_splits", "train_cat.tsv"), type=str, help="Path to sst-train.tsv for sst dataset")
    parser.add_argument("--sst_val_path", default=os.path.join("..", "..", "..", "..", "data", "SST-2", "sentence_splits", "dev_cat.tsv"), type=str, help="Path to sst-dev.tsv for sst dataset")

    #############################
    ## Misc and settings
    #############################
    parser.add_argument("--disable_cuda", action="store_true", help="disable cuda for training")
    parser.add_argument("--run_name", default="default", type=str, help="Name for this particular run. Used to write .txt file displaying arguments and results, and .pth file for model checkpoint")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--log_level", default="info", choices=["debug", "info", "warn"])
    parser.add_argument("--num_saved_models", default=1, type=int, help="The number of models to checkpoint for this run")
    parser.add_argument("--early_stopping", default=None, type=int, help="The number of epochs to wait before early stopping")


    args = parser.parse_args()
    # Set device to cuda if available unless explicitly disabled
    if torch.cuda.is_available() and not args.disable_cuda:
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    # Output debug logs in debug mode
    if args.debug:
        args.log_level = "debug"

    hparam_keys = 'lr lr_factor epochs max_seq_len train_pct val_pct seed batch_size'.split()
    hparams = {'lr': args.lr, 'lr_factor': args.lr_factor, 'epochs': args.epochs, 'max_seq_len': args.max_seq_len, 'train_pct': args.train_pct, 'val_pct': args.val_pct, 'seed': args.seed, 'batch_size': args.batch_size}
    hparams = Namespace(**hparams)
    return hparams, args


def parse_eval_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_path', required=True)
    parser.add_argument('--mean_teacher', action='store_true', help="Load the mean teacher model")
    # Load replacement model for testing
    parser.add_argument("--replacement", action='store_true', help="Load the replacement model")
    # Load frozen BERT model for testing
    parser.add_argument("--no_tuning", action='store_true', help="Run the baseline of frozen BERT with no tuning")


    parser.add_argument("--adv_test_path", default=os.path.join("datasets", "msr", "repl_test_hard_flip.tsv"),
        help="This is the path to the adversarial test you want to run against")
    
    return parser.parse_args()


def get_comet_key():
    try:
        with open(os.path.join("utils", "comet_key.txt"), 'r') as fp:
            return fp.read().strip()
    except FileNotFoundError as e:
        raise FileNotFoundError("{}\nPath to comet_key.txt is incorrect".format(e))
