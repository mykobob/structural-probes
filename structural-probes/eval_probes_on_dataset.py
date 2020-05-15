""" Demos a trained structural probe by making parsing predictions on stdin """

from argparse import ArgumentParser
import os
from datetime import datetime
import shutil
import yaml
from tqdm import tqdm
import torch
import numpy as np
import sys

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
mpl.rcParams['agg.path.chunksize'] = 10000

import data
import model
import probe
import regimen
import reporter
import task
import loss
import run_experiment
import pdb

from transformers import BertTokenizer, BertModel, BertForSequenceClassification, BertConfig


def get_word_depths(args, words, prediction, sent_index):
    """Writes an depth visualization image to disk.

    Args:
    args: the yaml config dictionary
    words: A list of strings representing the sentence
    prediction: A numpy matrix of shape (len(words),)
    sent_index: An index for identifying this sentence, used
        in the image filename.
    """
    return prediction * 2


def prepare_sentence_for_bert(line, tokenizer, max_seq_len=64):
    untokenized_sent = line.strip().split()
    untokenized_sent = untokenized_sent[:max_seq_len - 2]
    tokenized_sent = tokenizer.wordpiece_tokenizer.tokenize('[CLS] ' + ' '.join(untokenized_sent) + ' [SEP]')
    if len(tokenized_sent) > max_seq_len:
        tokenized_sent = tokenized_sent[:max_seq_len - 1] + ['[SEP]']
    num_proper_words = len(tokenized_sent)
    tokenized_sent = tokenized_sent + ['[PAD]'] * (max_seq_len - len(tokenized_sent))
    untok_tok_mapping = data.SubwordDataset.match_tokenized_to_untokenized(tokenized_sent, untokenized_sent)

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_sent)
    segment_ids = [1 if idx < num_proper_words else 0 for idx, x in enumerate(tokenized_sent)] 

    tokens_tensor = torch.tensor(indexed_tokens)
    segments_tensors = torch.tensor(segment_ids)
    if tokens_tensor.shape[0] > max_seq_len:
        print('='*80) 
        print('PROBLEM')
        print(tokens_tensor.shape[0])
        print(len(untokenized_sent))
        print(tokenized_sent)
    return (untokenized_sent, untok_tok_mapping), tokens_tensor, segments_tensors


def use_probes(args, sentences, model, tokenizer):
    """Runs a trained structural probe on sentences piped to stdin.

    Sentences should be space-tokenized.
    A single distance image and depth image will be printed for each line of stdin.

    Args:
    args: the yaml config dictionary
    """

    # Define the distance probe
    distance_probe = probe.TwoWordPSDProbe(args)
    distance_probe.load_state_dict(torch.load(args['probe']['distance_params_path'], map_location=args['device']))

    # Define the depth probe
    depth_probe = probe.OneWordPSDProbe(args)
    depth_probe.load_state_dict(torch.load(args['probe']['depth_params_path'], map_location=args['device']))

    all_word_dists = []
    all_word_depths = []
    all_predicted_edges = []
    for index, line in tqdm(enumerate(sentences), desc='[demoing]'):
        # Tokenize the sentence and create tensor inputs to BERT
        (untokenized_sent, untok_tok_mapping), tokens_tensor, segments_tensors = prepare_sentence_for_bert(line, tokenizer)

        tokens_tensor = tokens_tensor.unsqueeze(0).to(args['device'])
        segments_tensors = segments_tensors.unsqueeze(0).to(args['device'])


        with torch.no_grad():
            # Run sentence tensor through BERT after averaging subwords for each token

            last_hidden_states, encoded_layers = model(tokens_tensor, segments_tensors)
            single_layer_features = encoded_layers[args['model']['model_layer']]
            representation = torch.stack([torch.mean(single_layer_features[0,untok_tok_mapping[i][0]:untok_tok_mapping[i][-1]+1,:], dim=0) for i in range(len(untokenized_sent))], dim=0)
            representation = representation.view(1, *representation.size())

            # Run BERT token vectors through the trained probes
            distance_predictions = distance_probe(representation.to(args['device'])).detach().cpu()[0][:len(untokenized_sent),:len(untokenized_sent)].numpy()
            depth_predictions = depth_probe(representation).detach().cpu()[0][:len(untokenized_sent)].numpy()

            # Print results visualizations
            word_dists = distance_predictions
            word_depths = get_word_depths(args, untokenized_sent, depth_predictions, index)

            predicted_edges = reporter.prims_matrix_to_edges(distance_predictions, untokenized_sent, untokenized_sent)
            #print('prediction edges', len(predicted_edges))
            #       print_tikz(args, predicted_edges, untokenized_sent)

            all_word_dists.append(word_dists)
            all_word_depths.append(word_depths)
            all_predicted_edges.append(predicted_edges)
            
    return all_word_dists, all_word_depths, all_predicted_edges


if __name__ == '__main__':
    argp = ArgumentParser()
    argp.add_argument('experiment_config')
    argp.add_argument('--results-dir', default='',
      help='Set to reuse an old results dir; '
      'if left empty, new directory is created')
    argp.add_argument('--seed', default=0, type=int,
      help='sets all random seeds for (within-machine) reproducibility')
    cli_args = argp.parse_args()
    if cli_args.seed:
        np.random.seed(cli_args.seed)
        torch.manual_seed(cli_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    yaml_args= yaml.load(open(cli_args.experiment_config))
    run_experiment.setup_new_experiment_dir(cli_args, yaml_args, cli_args.results_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    yaml_args['device'] = device
    report_on_stdin(yaml_args)
