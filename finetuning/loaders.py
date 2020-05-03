from torch.utils.data import Dataset
import torch
import pandas as pd

from transformers import BertTokenizer

class SST(Dataset):
    def __init__(self, args, data_path, debug=False):
        super().__init__()
        self.data_path = data_path
        self.dataset = self._load_train_set()
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
        self.device = args.device
        self.max_seq_len = args.max_seq_len
        self.pad_token = "[PAD]"
        self.debug = debug

    def _load_train_set(self):
        df = pd.read_csv(self.data_path, sep="\t", index_col=None, names=['sentence', 'sentiment'])
        return df

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        sentence = row["sentence"]
        sentiment = row["sentiment"]

        # Need to do tokenization here so that it can be run in batches
        tokenized_sent = self.tokenizer.tokenize(sentence)

        sent_tokens = ["[CLS]"] + tokenized_sent + ["[SEP]"]

        # Pad the tokens on the right so that all sequences are the same length
        sent_tokens = sent_tokens + [self.pad_token for _ in range(self.max_seq_len - len(sent_tokens))]

        sent_ids = self.tokenizer.convert_tokens_to_ids(sent_tokens)
        sent_tensor = torch.tensor(sent_ids).to(self.device)

        segment_ids = [0 if x < len(tokenized_sent) else 1 for x in
                       range(len(sent_tokens))]  # + [1 for _ in range(len(q2_token))]
        segment_tensor = torch.tensor(segment_ids).to(self.device)

        attention_mask = [1 if x != '[PAD]' else 0 for x in sent_tokens]
        attn_mask_tensor = torch.tensor(attention_mask).to(self.device)

        data = {'sentence_tensor': sent_tensor, 
                'attn_mask': attn_mask_tensor,
                'label': torch.tensor(sentiment).float().to(self.device)}
        return data


    def __len__(self):
        return len(self.dataset)
