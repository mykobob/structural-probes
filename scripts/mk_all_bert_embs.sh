#! /bin/sh

#python scripts/convert_raw_to_bert.py data/SST-2/new_splits/dev_cat.raw   data/embeddings/sst-dev.bert-base-layers.hdf5   finetuning/lightning_logs/proper_classification/_ckpt_epoch_0.ckpt 
python scripts/convert_raw_to_bert.py data/SST-2/new_splits/train_cat.raw data/embeddings/sst-train.bert-base-layers.hdf5 finetuning/lightning_logs/proper_classification/_ckpt_epoch_0.ckpt 

