from nerbaseloader import NerBaseLoader

import pandas as pd
import numpy as np

from datasets import load_dataset


class WikiGoldSkLoader(NerBaseLoader):
    def __init__(self, dataset_name: str, text_file: str):
        super().__init__(dataset_name=dataset_name, text_file=text_file)

        self.dataset = load_dataset("NaiveNeuron/wikigoldsk")

    def process_data(self, test=False):
        sentence_num = 1
        changed_now = False
        rows = []
        all_counter = 0
        with open(self.text_file, "r") as file:
            for index, line in enumerate(file):
                line = line.strip()
                if not line:
                    sentence_num += 1
                    changed_now = True
                else:
                    value, tag = line.split(" ")
                    tag = tag.strip()
                    #if '-' in tag:
                    #    tag = tag.split("-")[1]
                    if changed_now or index == 0:
                        rows.append([f"Sentence: {sentence_num}",f"{value}","O",f"{tag}"])
                        changed_now = False
                    else:
                        rows.append([np.nan,f"{value}","O",f"{tag}"])
                all_counter += 1
                if test and all_counter == 100:
                    break

        df = pd.DataFrame(rows, columns=["Sentence #", "Word", "POS", "Tag"])

        return super().process_data(df, test)

        