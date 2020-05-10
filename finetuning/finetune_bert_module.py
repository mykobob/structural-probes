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


def determine_correctness(pred, label):
    """Takes in pred and label (both tensors) and returns average correctness (float)"""
    with torch.no_grad():
        sig_pred = F.sigmoid(pred)
        return torch.mean((label - sig_pred) ** 2)
        # tensor of 0s and 1s

        #guesses = sig_pred * 5
        #labels = label * 5

        #preds = torch.floor(guesses)
        #labels = torch.floor(labels)
        #avg_correct = (preds == labels).float().mean()
        #return avg_correct


class SST_Test(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # This is only used for testing purposes. Please don't remove or edit this
        self.adversarial_test_flag = False
        self.log = self._config_logger()
        self.hparams = args

        # For all possible test sets,

        # self.layer_pooling = nn.Linear(in_features=13, out_features=1)
        self.train_dataset = SST(args, args.sst_train_path)
        self.val_dataset =   SST(args, args.sst_val_path)
         
        #self.bert_layers_dict = self._load_bert_dict()
        self.bert = self.create_model(args.device)

        self.training_acc = []

    def create_model(self, device):
        #config = BertConfig(hidden_size=1024, num_attention_heads=16, intermediate_size=4096,
        #                    vocab_size_or_config_json_file=30522, output_hidden_states=True, num_labels=1)
        config = BertConfig.from_pretrained('bert-large-cased')
        config.num_labels = 1
        #bert = BertModel.from_pretrained('bert-large-uncased', config=config).to(device)
        bert = BertForSequenceClassification.from_pretrained('bert-large-cased', config=config).to(device)
        #bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=config).to(device)
        bert = self.freeze_bert(bert)

        return bert

    def freeze_bert(self, bert):
        # Freeze layers of bert
        for name, param in bert.named_parameters():
            if not "classifier" in name:
                param.requires_grad = False
            else:
                print(f"{name} is unfrozen")
                param.requires_grad = True
        return bert

    def forward(self, x, attn_mask):
        bert_out = self.bert(x, attention_mask=attn_mask)

        #output = bert_out[0].squeeze(1)
        output = bert_out[0]
        return output

    def training_step(self, batch, batch_nb):
        sent_tensor = batch['sentence_tensor']
        attn_mask = batch['attn_mask']
        label = batch['label']

        print('training', sent_tensor.shape)
        pred = self.forward(sent_tensor, attn_mask)

        correct = determine_correctness(pred, label)
        self.training_acc.append(correct.clone().detach().float())

        loss = F.mse_loss(pred, label, reduction='mean')  # , pos_weight=self.pos_weight)
        #loss = torch.exp(log_loss)

        try:
            # If there is a scheduler set up, log the learning rate, otherwise skip it.
            tensorboard_logs = {'train_loss': loss, 'lr': torch.tensor(self.sched.get_lr()).squeeze()}
        except AttributeError:
            tensorboard_logs = {'train_loss': loss}

        self.logger.log_metrics(tensorboard_logs, step=self.global_step)

        return {'loss': loss, 'log': tensorboard_logs}

    def on_epoch_end(self):
        training_acc_tensor = torch.stack(self.training_acc)
        avg_mse = {"mse": training_acc_tensor.mean()}
        self.logger.log_metrics(avg_mse, step=self.global_step)

        # clear training_acc
        self.training_acc = []

    def validation_step(self, batch, batch_nb):
        sent_tensor = batch['sentence_tensor']
        attn_mask = batch['attn_mask']
        label = batch['label']

        print('validation', sent_tensor.shape, attn_mask.shape)
        pred = self.forward(sent_tensor, attn_mask)

        loss = F.mse_loss(pred, label, reduction='mean')

        return {'val_loss': loss.clone().detach(),
                'predicted': pred.detach()}

    def validation_epoch_end(self, outputs):
        # Check if a layer should be unfrozen this epoch
        #if self.current_epoch >= self.args.unfreeze_offset and (
        #        self.current_epoch - self.args.unfreeze_offset) % self.args.unfreeze_interval == 0:
        #    self._unfreeze_next_layer()

        # Combine validation results
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        predictions = torch.stack([x['predicted'] for x in outputs])

        tensorboard_logs = {
            'val_loss': avg_loss,
            'mean_pred': predictions.mean(), 'std_pred': predictions.std()
        }

        if self.current_epoch != 0:
            # We don't want the "sanity check" validation run to be logged
            self.logger.log_metrics(tensorboard_logs, step=self.global_step)

        print("Val loss: {:.3f}".format(avg_loss))
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        # If we're running against the original test set, just forward batch to validation_step
        if not self.adversarial_test_flag:
            return self.validation_step(batch, batch_nb)
        # however, if we are running against the adversarial test set, we want to forward the "teacher" data from the test loader
        # because this contains the adversarial examples
        else:
            adverse_batch = {}
            adverse_batch['normal'] = batch['teacher']
            return self.validation_step(adverse_batch, batch_nb)

    def test_end(self, outputs):
        end_dict = self.validation_end(outputs)
        # When running against the standard test set, extract loss and accuracy as normal
        if not self.adversarial_test_flag:
            test_loss = end_dict['val_loss']
            end_dict['test_loss'] = test_loss
            print(f"Test loss: {test_loss}\n")
            del end_dict['val_loss']
            return end_dict
        else:
            # If we're already on the adversarial set, extract loss and accuracy but name them adversarial
            test_loss = end_dict['val_loss']
            end_dict['adverse_test_loss'] = test_loss
            print(f"Adversarial Test loss: {test_loss}\n")
            del end_dict['val_loss']
            return end_dict

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        # optim = torch.optim.SGD(self.parameters(), lr=self.args.lr, momentum=0.9)
        #milestones = [self.args.unfreeze_offset + (x * self.args.unfreeze_interval) for x in range(self.args.epochs)]
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim)#, gamma=self.args.lr_factor, milestones=milestones)

        return [optim], [self.sched]

    ############################
    ## Dataloaders
    ############################

    @pl.data_loader
    def train_dataloader(self):
        self.log.info(f"Training dataset has {len(self.train_dataset)} examples")
        return DataLoader(self.train_dataset, self.args.batch_size, shuffle=True)

    @pl.data_loader
    def val_dataloader(self):
        self.log.info(f"Validation dataset has {len(self.val_dataset)} examples")
        return DataLoader(self.val_dataset, 2, shuffle=False)
    
    ############################
    ## BERT Unfreezing Logic
    ############################

    def _unfreeze_next_layer(self):
        """This function will find the next frozen BERT layer (from top down) and unfreeze its parameters"""
        # Get the highest layer that is currently frozen, returns a tuple of form (key, {values})
        try:
            layer_type, layer_param_names = self.bert_layers_dict.popitem()
            self.log.info(f"Unfreezing group: {layer_type}")
            for name, param in self.bert.named_parameters():
                if name in layer_param_names:
                    param.requires_grad = True
        except KeyError:
            # Once all layers have been unfrozen, the dictionary will be empty
            # so popitem() will raise a KeyError
            return

    def _load_bert_dict(self):
        with open(os.path.join("utils", "bert_large_layers.json"), "r") as fp:
            layer_groups = json.load(fp, object_pairs_hook=OrderedDict)

        # Transform lists into sets for quick lookup
        for key, value in layer_groups.items():
            layer_groups[key] = set(value)

        return layer_groups

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
