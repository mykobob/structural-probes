#! /bin/sh

#python scripts/convert_raw_to_bert.py my_tests/ptb3-wsj-dev.raw   my_tests/ptb3-wsj-dev.bert-base-layers.hdf5   best_models/bert_base
#python scripts/convert_raw_to_bert.py my_tests/ptb3-wsj-test.raw  my_tests/ptb3-wsj-test.bert-base-layers.hdf5  best_models/bert_base
python scripts/convert_raw_to_bert.py my_tests/ptb3-wsj-train.raw my_tests/ptb3-wsj-train.bert-base-layers.hdf5 best_models/bert_base

