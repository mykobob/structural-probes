#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('structural-probes')
from pathlib import Path

from run_experiment import setup_new_experiment_dir, execute_experiment
import yaml
import torch
import pandas as pd
import eval_probes_on_dataset
import jupyter_slack


# In[2]:


def setup_args_and_folder(): 
    CONFIG_FILE = 'example/config/bert_ptb3.yaml'
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

# In[3]:


from nltk.tree import Tree

from reporter import WordPairReporter, WordReporter, prims_matrix_to_edges
from tqdm import tqdm
import copy


# In[4]:


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


# In[5]:


data_base = Path('../../../data/SST-2')

train_path = data_base / 'sentence_splits' / 'train.tsv'
dev_path = data_base / 'sentence_splits' / 'dev.tsv'

# read in SST dataset
sst_trees_base = data_base / 'tree_format/'
gt_tree_train_path = sst_trees_base / 'train.txt'
gt_tree_dev_path = sst_trees_base / 'dev.txt'

train_sentiment, train_labels = read_sentiment_sentences(train_path)
dev_sentiment, dev_labels = read_sentiment_sentences(dev_path)

# Read into NTLK Trees
gt_train_trees = read_trees(gt_tree_train_path)
gt_dev_trees = read_trees(gt_tree_dev_path)


# In[6]:


# Calculate distance between the two trees
# one tree's format is from structural probes
# other tree's format is from the dataset (gt_*_trees)
# import importlib
# importlib.reload(eval_probes_on_dataset)

word_dists, word_depths, predicted_edges, model, tokenizer = eval_probes_on_dataset.report_on_stdin(yaml_args, dev_sentiment)


# In[7]:


sigmoid = torch.nn.Sigmoid()
for line in tqdm(dev_sentiment, desc='Eval Dev Sentiment'):
    _, tokens_tensor, segments_tensors = eval_probes_on_dataset.prepare_sentence_for_bert(line, tokenizer)

    device = yaml_args['device']
    tokens_tensor = tokens_tensor.to(device)
    segments_tensors = segments_tensors.to(device)
    with torch.no_grad():
        logits, encoded_layers = model(tokens_tensor, segments_tensors)
        probs = sigmoid(logits)


# In[8]:


def earliest_root(a, b):
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return -1

def get_adj_matrix(cur_tree):
    num_words = len(cur_tree.leaves())
    all_paths = [cur_tree.leaf_treeposition(i) for i in range(num_words)]
    matrix = [[0] * num_words for _ in range(num_words)]

    for a_idx, a in enumerate(all_paths):
        for b_idx, b in enumerate(all_paths):
            if a_idx < b_idx:
                root = earliest_root(a, b)
                matrix[a_idx][b_idx] = len(a) + len(b) - 2 * root
                matrix[b_idx][a_idx] = len(a) + len(b) - 2 * root
    return matrix


# In[9]:


def report_uuas(predicted_edges, trees, split_name):
    uspan_total = 0
    uspan_correct = 0
    total_sents = 0
    for predicted_edge, gt_tree in tqdm(zip(predicted_edges, trees), desc='[uuas,tikz]'):
        words = gt_tree.leaves()
        poses = gt_tree.leaves()
        gold_edges = get_adj_matrix(gt_tree)
        gold_edges = prims_matrix_to_edges(gold_edges, words, poses)
        pred_edges = predicted_edge

        gold_span = set([tuple(sorted(x)) for x in gold_edges])
        pred_span = set([tuple(sorted(x)) for x in pred_edges])
        uspan_correct += len(gold_span.intersection(pred_span))
        uspan_total += len(gold_edges)
        total_sents += 1

    uuas = uspan_correct / float(uspan_total)
    return uuas

uuas_score = report_uuas(predicted_edges, gt_dev_trees, 'dev')
print(uuas_score)


# In[10]:


def get_sentiments():
    pass    


# In[ ]:


#get_ipython().run_cell_magic(u'notify', u'"result"', u'result = f"Finished SST: {uuas_score:.4f}"\nprint(result)')


# In[ ]:

