import sys
sys.path.append('../structural-probes/')
import torch
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.comet import CometLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from utils.setup_runs import parse_args, get_comet_key
from finetune_bert_module import SST_Test

import os
import random

if __name__ == "__main__":
    args = parse_args()
    # Set device to cuda if available unless explicitly disabled
    if args.device == torch.device('cuda'):
        num_gpus = 1
    else:
        num_gpus = 0

    # Output debug logs in debug mode
    if args.debug:
        args.log_level = "debug"

    # Set all seeds manually for consistent results
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


    ###############
    # CometLogger configuration
    ###############

    comet_key = get_comet_key()
    comet_logger = CometLogger(
        api_key = comet_key,
        workspace = "mykobob",
        project_name = "structural-probes-extension",
        experiment_name = args.run_name
    )
    # Log args to comet
    comet_logger.log_hyperparams(args)
    comet_logger.experiment.set_name(args.run_name)
    comet_logger.experiment.add_tag("sst-tests")

    ###############
    # Other callback configuration
    ###############

    checkpoint_callback = ModelCheckpoint(
        filepath= os.path.join("lightning_logs", args.run_name, "checkpoints"),
        save_top_k=args.num_saved_models,
        verbose=True,
        monitor="val_acc",
        mode="max",
    )

    if args.early_stopping:
        early_stopping = EarlyStopping(
            #monitor='val_loss',
            monitor='val_acc',
            min_delta=0.00,
            patience=args.early_stopping,
            verbose=False,
            #mode='min'
            mode='max'
        )
    else:
        early_stopping = None

    ###############
    # Model creation
    ###############

    model = SST_Test(args).to(args.device)
    comet_logger.experiment.add_tag("SST")
    

    ###############
    # Create Trainer with specified attributes
    ###############

    trainer = Trainer(
        fast_dev_run=args.debug,
        max_nb_epochs=args.epochs,
        gpus=num_gpus,
        train_percent_check=args.train_pct,
        val_percent_check=args.val_pct,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stopping,
        logger=comet_logger,
    )
    
    trainer.fit(model)
