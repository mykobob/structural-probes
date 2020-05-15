'''
Takes raw text and saves BERT-cased features for that text to disk

Adapted from the BERT readme (and using the corresponding package) at

https://github.com/huggingface/pytorch-pretrained-BERT

###
John Hewitt, johnhew@stanford.edu
Feb 2019

'''
import sys
import os
sys.path.append('finetuning')
sys.path.append('structural-probes')
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, WordpieceTokenizer
from argparse import ArgumentParser
import h5py
import numpy as np
import finetune_bert_module
from utils import setup_runs

argp = ArgumentParser()
argp.add_argument('input_path')
argp.add_argument('output_path')
argp.add_argument('bert_model', help='base or large')
args = argp.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
desired_params = {
    'sst_train_path': os.path.join("data", "SST-2", "sentence_splits", "train_cat.tsv"),
    'sst_val_path': os.path.join("data", "SST-2", "sentence_splits", "dev_cat.tsv"),
}
hparams, params = setup_runs.get_default_args({}, desired_params)
# Load pre-trained model tokenizer (vocabulary)
# Crucially, do not do basic tokenization; PTB is tokenized. Just do wordpiece tokenization.
#if args.bert_model == 'base':
#  tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
#  model = BertModel.from_pretrained('bert-base-cased')
#  LAYER_COUNT = 12
#  FEATURE_COUNT = 768
#elif args.bert_model == 'large':
#  tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
#  model = BertModel.from_pretrained('bert-large-cased')
#  LAYER_COUNT = 24
#  FEATURE_COUNT = 1024
#else:
#  raise ValueError("BERT model must be base or large")

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
#model = BertModel.from_pretrained('best_models/classification')

model = finetune_bert_module.SST_Test.load_from_checkpoint(
        args.bert_model, None, None, params).to(device)
        

LAYER_COUNT = 12
FEATURE_COUNT = 768


model.eval()

with h5py.File(args.output_path, 'w') as fout:
  for index, line in enumerate(open(args.input_path)):
    line = line.strip() # Remove trailing characters
    line = '[CLS] ' + line + ' [SEP]'
    tokenized_text = tokenizer.wordpiece_tokenizer.tokenize(line)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segment_ids = [1 for x in tokenized_text]
  
    if (len(tokenized_text) == 14 or len(tokenized_text) == 15) and 'Absorbing' in line:
        print(tokenized_text)
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens]).to(device)
    segments_tensors = torch.tensor([segment_ids]).to(device)
  
    with torch.no_grad():
        pred, encoded_layers = model(tokens_tensor, segments_tensors)
        encoded_layers = [encoded_layer.cpu() for encoded_layer in encoded_layers[1:]]
    dset = fout.create_dataset(str(index), (LAYER_COUNT, len(tokenized_text), FEATURE_COUNT))
    dset[:,:,:] = np.vstack([np.array(x) for x in encoded_layers])
  

