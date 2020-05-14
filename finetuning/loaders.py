from torch.utils.data import Dataset
import torch
import pandas as pd

from transformers import BertTokenizer
from eval_probes_on_dataset import prepare_sentence_for_bert

class SST(Dataset):
    def __init__(self, args, hparams, data_path, tokenizer, debug=False):
        super().__init__()
        self.data_path = data_path
        self.dataset = self._load_data_set()
        self.tokenizer = tokenizer
        self.device = args.device
        self.max_seq_len = hparams.max_seq_len
        self.pad_token = "[PAD]"
        self.debug = debug

    def _load_data_set(self):
        df = pd.read_csv(self.data_path, sep="\t", index_col=None, names=['sentence', 'sentiment'])
        return df

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        sentence = row["sentence"]
        sentiment = row["sentiment"]

        _, tokenized_tensor, segments_tensors = prepare_sentence_for_bert(sentence, self.tokenizer, max_seq_len=self.max_seq_len)
        data = {
            'sentence_tensor': tokenized_tensor,
            'attn_mask': segments_tensors,
            'label': torch.tensor([sentiment]).float()
        }

        return data


    def __len__(self):
        return len(self.dataset)
