from torch.utils.data import Dataset
import torch
import pandas as pd

from transformers import BertTokenizer
from eval_probes_on_dataset import prepare_sentence_for_bert

class SST(Dataset):
    def __init__(self, args, hparams, data_path, tokenizer, debug=False):
        super().__init__()
        self.data_path = data_path
        self.categorical = 'cat' in data_path
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
#         print(tokenized_tensor.shape, segments_tensors.shape)
        data = {
            'sentence_tensor': tokenized_tensor,
            'attn_mask': segments_tensors,
            'label': torch.tensor(sentiment) if self.categorical else torch.tensor([sentiment])
        }

        # Need to do tokenization here so that it can be run in batches
        #tokenized_sent = self.tokenizer.tokenize(sentence)

        #sent_tokens = ["[CLS]"] + tokenized_sent + ["[SEP]"]

        ## Pad the tokens on the right so that all sequences are the same length
        #sent_tokens = sent_tokens + [self.pad_token for _ in range(self.max_seq_len - len(sent_tokens))]

        #sent_ids = self.tokenizer.convert_tokens_to_ids(sent_tokens)
        #sent_tensor = torch.tensor(sent_ids).to(self.device)

        #segment_ids = [0 if x < len(tokenized_sent) else 1 for x in
        #               range(len(sent_tokens))]  # + [1 for _ in range(len(q2_token))]
        #segment_tensor = torch.tensor(segment_ids).to(self.device)

        #attention_mask = [1 if x != '[PAD]' else 0 for x in sent_tokens]
        #attn_mask_tensor = torch.tensor(attention_mask).to(self.device)

        #data = {'sentence_tensor': sent_tensor, 
        #        'attn_mask': attn_mask_tensor,
        #        'label': torch.tensor(sentiment).float().to(self.device)}
        return data


    def __len__(self):
        return len(self.dataset)
