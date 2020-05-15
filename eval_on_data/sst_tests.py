#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('cd', '..')


# In[2]:


import os
import sys
sys.path.append('structural-probes')
sys.path.append('finetuning')
from pathlib import Path

from run_experiment import setup_new_experiment_dir, execute_experiment
import yaml
import torch
import pandas as pd
import eval_probes_on_dataset
import jupyter_slack
from utils import setup_runs
import finetune_bert_module

from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
from sklearn.metrics import f1_score


# In[3]:


def setup_args_and_folder(): 
    CONFIG_FILE = 'configs/bert_base_distance_ptb3.yaml'
    EXPERIMENT_NAME = ''
    SEED = 123

    class Object(object):
        pass

    cli_args = Object()
    cli_args.experiment_config = CONFIG_FILE
    cli_args.results_dir = EXPERIMENT_NAME
    cli_args.train_probe = -1
    cli_args.report_results = 1
    cli_args.seed = SEED

    yaml_args = yaml.load(open(cli_args.experiment_config))
    setup_new_experiment_dir(cli_args, yaml_args, cli_args.results_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    yaml_args['device'] = device
    yaml_args['model_type'] = 'large'
    return yaml_args

yaml_args = setup_args_and_folder()


# # Requirements before running this code
# - Corpus (conllx, for BERT)
#     - Run `convert_splits_to_depparse.sh` (to get .conllx format)
#     - Run `convert_conll_to_raw.py` (to get into raw text) 
# - BERT-layer embeddings (.bert-layers.hdf5)
#     - Run `convert_raw_to_bert.py` (Uses BERT to create bert-embeddings for ALL layers)
# - Depth & Distance Params Path
#     - Pretrained from data
# - Ground Truths
#     - Trees
#         - Run Stanford CoreNLP's `ReadSentimentDataset` to get the ground truth trees
#     - Sentiment
#         - Run `apply_splits.py` to average all phrases' sentiments in the sentences

# # Notes for generating the Tree data for SST
# - This is for reading in the Ground Truth trees that is already given to us in SST
# - We'll use Stanford's CoreNLP tools
# - Run the ReadSentimentDataset `java -mx4g edu.stanford.nlp.sentiment.ReadSentimentDataset -inputDir data/SST-2/original -outputDir tmp/`
#   - The ground truth already does subword partitions, so need to account for that

# In[4]:


from nltk.tree import Tree

from reporter import WordPairReporter, WordReporter, prims_matrix_to_edges
from tqdm import tqdm
import copy


# In[5]:


def read_trees(path):
    with open(path) as f:
        tree_lines = f.readlines()
        
    trees = [Tree.fromstring(treeline) for treeline in tree_lines]
    return trees

def read_sentiment_sentences(path):
    with open(path) as f:
        all_sentences = f.readlines()
        
        sentences, labels = [], []
        for pair in all_sentences:
            sentence, label = pair.split('\t')
            sentences.append(sentence)
            labels.append(label)
        return sentences, labels


# In[6]:


data_base = Path('data/SST-2')

train_path = data_base / 'sentence_splits' / 'train_cat.tsv'
dev_path = data_base / 'sentence_splits' / 'dev_cat.tsv'

# read in SST dataset
sst_trees_base = data_base / 'tree_format/'
gt_tree_train_path = sst_trees_base / 'train.txt'
gt_tree_dev_path = sst_trees_base / 'dev.txt'

train_sentiment, train_labels = read_sentiment_sentences(train_path)
dev_sentiment, dev_labels = read_sentiment_sentences(dev_path)

# Read into NTLK Trees
gt_train_trees = read_trees(gt_tree_train_path)
gt_dev_trees = read_trees(gt_tree_dev_path)


# In[7]:


from finetune_bert_module import SST_Test

desired_params = {
    'sst_train_path': os.path.join("data", "SST-2", "sentence_splits", "train_cat.tsv"),
    'sst_val_path': os.path.join("data", "SST-2", "sentence_splits", "dev_cat.tsv"),
}
hparams, params = setup_runs.get_default_args({}, desired_params)

config = BertConfig.from_pretrained('bert-base-cased')
config.output_hidden_states=True
config.num_labels = 1
model = finetune_bert_module.SST_Test.load_from_checkpoint(
        'finetuning/lightning_logs/proper_classification/_ckpt_epoch_0.ckpt', None, None, params)
# model = BertForSequenceClassification.from_pretrained('bert-large-cased', config=config)
model = model.to(yaml_args['device'])
model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


# In[9]:


# Calculate distance between the two trees
# one tree's format is from structural probes
# other tree's format is from the dataset (gt_*_trees)
# import importlib
# importlib.reload(eval_probes_on_dataset)

word_dists, word_depths, predicted_edges = eval_probes_on_dataset.use_probes(yaml_args, dev_sentiment, model, tokenizer)


# In[18]:


# Calculate accuracy of the model
softmax = torch.nn.Softmax(dim=1)
preds = []
labels = [] 
for idx, line in tqdm(enumerate(dev_sentiment), desc='Eval Dev Sentiment'):
    _, tokens_tensor, segments_tensors = eval_probes_on_dataset.prepare_sentence_for_bert(line, tokenizer)
    line_label = int(dev_labels[idx])

    device = yaml_args['device']
    tokens_tensor = tokens_tensor.unsqueeze(0).to(device)
    segments_tensors = segments_tensors.unsqueeze(0).to(device)
    with torch.no_grad():
        logits, encoded_layers = model(tokens_tensor, segments_tensors)
        probs = softmax(logits)
        preds.append(0 if probs[0][0] >= probs[0][1] else 1)
    labels.append(line_label)        


# In[19]:


print('f1', f1_score(labels, preds))


# # Loading in test sets

# In[26]:


import pandas as pd

# Load the dataset into a pandas dataframe.
# temp_root_path = "/home/garysnake/Desktop/structural-probes/experiments/data/cola_public"
temp_data_path = "data/SST-2/sentence_splits/dev_cat.tsv"
df = pd.read_csv(temp_data_path, delimiter='\t', header=None, names=['sentence', 'label'])

# Report the number of sentences.
print('Number of test sentences: {:,}\n'.format(df.shape[0]))

# Create sentence and label lists
sentences = df.sentence.values
labels = df.label.values

# Tokenize all of the sentences and map the tokens to thier word IDs.
input_ids = []
attention_masks = []

# For every sentence...
for sent in sentences:
    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.
    encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 64,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
    # Add the encoded sentence to the list.    
    input_ids.append(encoded_dict['input_ids'])
    
    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

# Set the batch size.  
batch_size = 32  

from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
# Create the DataLoader.
prediction_data = TensorDataset(input_ids, attention_masks, labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)


# In[38]:


import numpy as np
# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# In[39]:


from sklearn.metrics import matthews_corrcoef

# Prediction on test set


print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))

# Tracking variables 
# predictions , true_labels = [], []
batch_accuracy = []
mcc = []

# Predict 
for batch in prediction_dataloader:
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)

    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch

    # Telling the model not to compute or store gradients, saving memory and 
    # speeding up prediction
    with torch.no_grad():
        # Forward pass, calculate logit predictions
        logits, encoded_layers = model.forward(b_input_ids, attn_mask=b_input_mask)

    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    
    # Record batch accuracy
    batch_accuracy.append(flat_accuracy(logits, label_ids))

    # Record batch mcc  
    mcc.append(matthews_corrcoef(label_ids, np.argmax(logits, axis=1).flatten()))
      
    # Store predictions and true labels
#     predictions.append(logits)
#     true_labels.append(label_ids)

print('    DONE.')


# # Plotting Scatter plots of UUAS v Accuracy & Spearmanr vs Accuracy

# In[40]:


import matplotlib.pyplot as plt 
#  matplotlib inline

import seaborn as sns

def plot_scatter(x, y, x_name, y_name):
    # Use plot styling from seaborn.
    sns.set(style='darkgrid')

    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12,6)

    # Plot the learning curve.
    plt.scatter(x, y)

    # Label the plot.
    plt.title("{:} & {:}".format(x_name, y_name))
    plt.xlabel(x_name)
    plt.ylabel(y_name)

    plt.show()


#  Next step export this set, and run probe on this dev
# Next step might be running 
print(len(batch_accuracy))

batch_uuas = []
with open('probe_scores/dev.uuas','r') as f:
    for line in f.readlines():
        try:
            batch_uuas.append(float(line))
        except ValueError:
            pass


batch_spearman = []
with open('probe_scores/dev.spearmanr_batch_average','r') as f:
    for line in f.readlines():
        try:
            batch_spearman.append(float(line))
        except ValueError:
            pass
    

plot_scatter(batch_accuracy, batch_uuas, "accuracy", "uuas")

plot_scatter(batch_accuracy, batch_spearman, "accuracy", "spearman")

plot_scatter(mcc, batch_uuas, "mcc", "uuas")

plot_scatter(mcc, batch_spearman, "mcc", "spearman")



        

    


# In[ ]:




