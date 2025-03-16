import torch
import pandas as pd
import sys
import numpy as np

sys.path.append("/home/k/kubik32/DISSERTATION/data/text_data")

from transformers import PreTrainedTokenizer
from typing import Tuple

from torch.utils.data import DataLoader

from sklearn import preprocessing

class NerBaseLoader(DataLoader):
    def __init__(self, dataset_name: str, text_file: str):
        super(NerBaseLoader).__init__()

        self.dataset_name = dataset_name
        self.text_file = text_file

    def load_csv(self, test=False):
        if not test:
            df = pd.read_csv(self.dataset_name, encoding="latin-1")
        else:
            df = pd.read_csv(self.dataset_name, encoding="latin-1", nrows=100)
        print(df.head(20))
        return df

    def process_data(self, df=None, test=False):
        if df is None:
            df = self.load_csv(test)
        df.loc[:, "Sentence #"] = df["Sentence #"].fillna(method="ffill")

        enc_pos = preprocessing.LabelEncoder()
        enc_tag = preprocessing.LabelEncoder()

        df.loc[:, "POS"] = enc_pos.fit_transform(df["POS"])
        df.loc[:, "Tag"] = enc_tag.fit_transform(df["Tag"])

        sentences = df.groupby("Sentence #")["Word"].apply(list).values
        pos = df.groupby("Sentence #")["POS"].apply(list).values
        tag = df.groupby("Sentence #")["Tag"].apply(list).values

        indeces = np.arange(len(sentences))

        return indeces, sentences, pos, tag, enc_pos, enc_tag

class NerBaseDataset:
    def __init__(self, texts, pos, tags, indeces, tokenizer, max_length:int =128) -> None:
        self.texts = texts
        self.pos = pos
        self.tags = tags
        self.indeces = indeces

        self.tokenizer = tokenizer
        self.max_length = max_length


    def __len__(self):
        return len(self.texts)
        
    def preprocess_data(self, text: pd.Series, tokenizer: PreTrainedTokenizer, max_length: int, special:bool =False) -> Tuple[torch.Tensor, torch.Tensor]:
        return tokenizer.encode(text.strip(), add_special_tokens=False)

    
    def __getitem__(self, item):
        text = self.texts[item]
        pos = self.pos[item]
        tags = self.tags[item]
        indeces = self.indeces[item]

        ids = []
        target_pos = []
        target_tag = []

        for i, s in enumerate(text):
            inputs = self.preprocess_data(s, self.tokenizer, self.max_length)
            # abhishek: ab ##hi ##sh ##ek
            input_len = len(inputs)
            ids.extend(inputs)
            target_pos.extend([pos[i]] * input_len)
            target_tag.extend([tags[i]] * input_len)
    

        ids = ids[:self.max_length - 2]
        target_pos = target_pos[:self.max_length - 2]
        target_tag = target_tag[:self.max_length - 2]

        ids = [0] + ids + [2]  #101 is [CLS] and 102 is [SEP] in bert, roberta has 0 and 2
        target_pos = [0] + target_pos + [0]
        target_tag = [0] + target_tag + [0]

        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)

        padding_len = self.max_length - len(ids)

        ids = ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        target_pos = target_pos + ([0] * padding_len)
        target_tag = target_tag + ([0] * padding_len)

        return {
            "indeces": torch.tensor(indeces, dtype=torch.long),  #TODO: check indeces
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target_pos": torch.tensor(target_pos, dtype=torch.long),
            "target_tag": torch.tensor(target_tag, dtype=torch.long),
        }
