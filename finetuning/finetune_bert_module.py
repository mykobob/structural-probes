import os
import json
from collections import OrderedDict
import logging

from transformers import BertModel, BertTokenizer, BertForSequenceClassification, BertConfig
import torch.nn as nn
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
import pytorch_lightning as pl

from loaders import SST


sigmoid = nn.Sigmoid()
def determine_correctness(pred, label):
    """Takes in pred and label (both tensors) and returns average correctness (float)"""
    with torch.no_grad():
        return torch.mean((label - pred) ** 2)


class SST_Test(pl.LightningModule):
    def __init__(self, args, hparams):
        super().__init__()
        self.args = args
        self.log = self._config_logger()
        self.hparams = hparams

        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.train_dataset = SST(args, hparams, args.sst_train_path, tokenizer)
        self.val_dataset =   SST(args, hparams, args.sst_val_path, tokenizer)

        self.bert = self.create_model(args.device)
        self.training_acc = []
        self.val_acc = []

        #self.loss = nn.MSELoss()
        self.loss = nn.CrossEntropyLoss()
        self.do_sigmoid = 'MSELoss' in type(self.loss)


    def create_model(self, device):
        config = BertConfig.from_pretrained('bert-base-cased')
        config.output_hidden_states=True
        config.num_labels = 1
        bert = BertForSequenceClassification.from_pretrained('bert-base-cased', config=config).to(device)
        bert = self.freeze_bert(bert)

        return bert

    def freeze_bert(self, bert):
        unfreeze_layers = '11'.split()
        # Freeze layers of bert
        for name, param in bert.named_parameters():
            for layer in unfreeze_layers:
                if layer in name:
                    param.requires_grad = True
        to_unfreeze = '11'.split()
        # Freeze layers of bert
        for name, param in bert.named_parameters():
            for layer_num in to_unfreeze:
                if layer_num in name:
                    print(f"{name} is unfrozen")
                    param.requires_grad = True

            if not "classifier" in name:
                param.requires_grad = False
            else:
                print(f"{name} is unfrozen")
                param.requires_grad = True
        return bert

    def forward(self, x, attn_mask):
        bert_out, encoded_layers = self.bert(x, attention_mask=attn_mask)

        #output = bert_out[0].squeeze(1)
        if self.do_sigmoid:
            output = sigmoid(bert_out)
        
#         import pdb; pdb.set_trace()
        return output

    def training_step(self, batch, batch_nb):
        sent_tensor = batch['sentence_tensor']
        attn_mask = batch['attn_mask']
        label = batch['label']

        pred = self.forward(sent_tensor, attn_mask)

        correct = determine_correctness(pred, label)
        self.training_acc.append(correct.clone().detach().float())
        loss = self.loss(pred, label, reduction='mean')  # , pos_weight=self.pos_weight)
        #loss = torch.exp(log_loss)

        try:
            # If there is a scheduler set up, log the learning rate, otherwise skip it.
            tensorboard_logs = {'train_loss': loss, 'lr': torch.tensor(self.sched.get_lr()).squeeze()}
        except AttributeError:
            tensorboard_logs = {'train_loss': loss}
#         import pdb; pdb.set_trace()
#         self.logger.log_metrics(tensorboard_logs, step=self.global_step)

        return {'loss': loss, 'log': tensorboard_logs}

    def on_epoch_end(self):
        training_acc_tensor = torch.stack(self.training_acc)
        avg_loss = {"loss": training_acc_tensor.mean()}
#         self.logger.log_metrics(avg_mse, step=self.global_step)

        # clear training_acc
        self.training_acc = []
        return avg_loss

    def validation_step(self, batch, batch_nb):
        sent_tensor = batch['sentence_tensor']
        attn_mask = batch['attn_mask']
        label = batch['label']

        pred = self.forward(sent_tensor, attn_mask)

        loss = self.loss(pred, label, reduction='mean')

        return {'val_loss': loss.clone().detach(),
                'predicted': pred.detach()}

    def validation_epoch_end(self, outputs):
        # Combine validation results
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        predictions = torch.stack([x['predicted'] for x in outputs])

        tensorboard_logs = {
            'val_loss': avg_loss,
            'mean_pred': predictions.mean(), 'std_pred': predictions.std()
        }

#         if self.current_epoch != 0:
            # We don't want the "sanity check" validation run to be logged
#             self.logger.log_metrics(tensorboard_logs, step=self.global_step)

        print("Val loss: {:.3f}".format(avg_loss))
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_end(self, outputs):
        end_dict = self.validation_epoch_end(outputs)
        # When running against the standard test set, extract loss and accuracy as normal
        test_loss = end_dict['val_loss']
        end_dict['test_loss'] = test_loss
        print(f"Test loss: {test_loss}\n")
        del end_dict['val_loss']
        return end_dict

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        # optim = torch.optim.SGD(self.parameters(), lr=self.args.lr, momentum=0.9)
        #milestones = [self.args.unfreeze_offset + (x * self.args.unfreeze_interval) for x in range(self.args.epochs)]
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim)#, gamma=self.args.lr_factor, milestones=milestones)

        return [optim], [self.sched]

    ############################
    ## Dataloaders
    ############################


    @pl.data_loader
    def train_dataloader(self):
        self.log.info(f"Training dataset has {len(self.train_dataset)} examples, and batch_size of {self.hparams.batch_size}")
        return DataLoader(self.train_dataset, self.hparams.batch_size, shuffle=True, num_workers=4)

    @pl.data_loader
    def val_dataloader(self):
        self.log.info(f"Validation dataset has {len(self.val_dataset)} examples")
        return DataLoader(self.val_dataset, 1, shuffle=False, num_workers=4)


    ############################
    ## MISCILLANEOUS
    ############################
    def _config_logger(self):
        log = logging.getLogger(__name__)
        log.addHandler(logging.StreamHandler())
        if self.args.log_level == "debug":
            level = logging.DEBUG
        elif self.args.log_level == "warn":
            level = logging.WARNING
        else:
            level = logging.INFO

        log.setLevel(level)
        return log
