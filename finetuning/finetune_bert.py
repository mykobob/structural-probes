#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../structural-probes/')
import torch
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.comet import CometLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from utils.setup_runs import parse_args, get_default_args, get_comet_key
from finetune_bert_module import SST_Test

import os
import random


# In[2]:


#desired_hparams = {
#    'lr': 7e-3,
#    'batch_size': 48,
#}
#desired_params = {}
#hparams, args = get_default_args(desired_hparams, desired_params)
hparams, args = parse_args() 


# In[3]:


# Set device to cuda if available unless explicitly disabled
if args.device == torch.device('cuda'):
    num_gpus = 1
else:
    num_gpus = 0

# Output debug logs in debug mode
if args.debug:
    args.log_level = "debug"


# In[4]:


# Set all seeds manually for consistent results
torch.manual_seed(hparams.seed)
np.random.seed(hparams.seed)
random.seed(hparams.seed)


# In[5]:


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
comet_logger.log_hyperparams(hparams)
comet_logger.experiment.set_name(args.run_name)
comet_logger.experiment.add_tag("sst-tests")


# In[6]:


###############
# Other callback configuration
###############

checkpoint_callback = ModelCheckpoint(
    filepath= os.path.join("lightning_logs", args.run_name, "checkpoints"),
    save_top_k=args.num_saved_models,
    verbose=True,
    monitor="val_loss",
    mode="min",
)


# In[7]:


if args.early_stopping:
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=args.early_stopping,
        verbose=False,
        mode='min'
        #mode='max'
    )
else:
    early_stopping = None


# In[8]:


###############
# Model creation
###############

testing = False
if testing:
    model = SST_Test.load_from_metrics(
            weights_path='lightning_logs/regression_training/_ckpt_epoch_19',
            map_location=None)
else:
    model = SST_Test(args, hparams).to(args.device)

comet_logger.experiment.add_tag("SST")


# In[9]:


###############
# Create Trainer with specified attributes
###############

trainer = Trainer(
    fast_dev_run=args.debug,
    max_nb_epochs=hparams.epochs,
    gpus=num_gpus,
    train_percent_check=hparams.train_pct,
    val_percent_check=hparams.val_pct,
    checkpoint_callback=checkpoint_callback,
    early_stop_callback=early_stopping,
    logger=comet_logger
)


# In[10]:


if testing:
    trainer.test(model)
else:
    trainer.fit(model)


# In[ ]:


try:
    model_path = f'../best_models/{args.run_name}'
    os.mkdir(model_path)
except:
    pass

model.bert.save_pretrained(model_path)

