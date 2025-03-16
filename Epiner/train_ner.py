from activetokenenn import ActiveTokenENN
from slovakbert import SlovakBertTokenClassifier
from nerbaseloader import NerBaseDataset
from wikigoldskloader import WikiGoldSkLoader
from tokenepinet import TokenEpiNet
from seqeval.metrics import classification_report, f1_score

from transformers import RobertaTokenizer, AutoTokenizer
from dotdict import dotdict
from sklearn import model_selection
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import numpy as np

index_dim = 10
epinet_input_size = 768
epochs = 3

training_file = "/home/k/kubik32/DISSERTATION/data/text_data/ner/train.txt"
#data_loader = NerBaseLoader("/home/k/kubik32/DISSERTATION/data/csv_data/ner/ner_dataset.csv", training_file)
data_loader = WikiGoldSkLoader("WikiGoldSK", training_file)

indeces, sentences, pos, tag, enc_pos, enc_tag = data_loader.process_data(test=False)


num_pos = len(list(enc_pos.classes_))
num_tag = len(list(enc_tag.classes_))

base_network = SlovakBertTokenClassifier(num_tag=num_tag, num_pos=num_pos)
tokenizer = AutoTokenizer.from_pretrained(base_network.model_name, use_fast=True, add_prefix_space=True)

base_batch_size = 32

(
    train_sentences,
    test_sentences,
    train_pos,
    test_pos,
    train_tag,
    test_tag
) = model_selection.train_test_split(
    sentences, 
    pos, 
    tag, 
    random_state=42, 
    test_size=0.1
)


sentence_list = list(sentences)
train_sentence_list = list(train_sentences)
test_sentence_list = list(test_sentences)
indeces_list = list(indeces)


train_indeces = [indeces_list[index] for index in range(len(sentence_list)) if sentence_list[index] in train_sentence_list]
test_indeces = [indeces_list[index] for index in range(len(sentence_list)) if sentence_list[index] in test_sentence_list]

train_indeces = np.array(train_indeces, dtype="object")
test_indeces = np.array(test_indeces, dtype="object")

train_dataset = NerBaseDataset(
    texts=train_sentences, pos=train_pos, tags=train_tag, indeces=train_indeces, tokenizer=tokenizer
)

train_data_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=base_batch_size, num_workers=4
)

valid_dataset = NerBaseDataset(
    texts=test_sentences, pos=test_pos, tags=test_tag, indeces=test_indeces, tokenizer=tokenizer
)

valid_data_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=base_batch_size, num_workers=1
)

epinet = TokenEpiNet(num_tag=num_tag, num_pos=num_pos, input_size=epinet_input_size, index_size=index_dim)
loggers = dotdict(dict(log_wandb=False, log_stats=False))
whole_config = dotdict(dict(
    active_functor = 'entropy',
    al_ratio = 0.75,
    true_validation = False,
    tokenizer = tokenizer,
    warm_start_ratio = 0.0,
    index_number=10,
    index_dimension=index_dim,
    only_bert=True,
    dimension_out=None,
    epochs=epochs,
    model_name='TokenENN',
    averaging_type = 'macro',
))


device = torch.device("cuda")
model = ActiveTokenENN(base_network, epinet, whole_config, loggers)
model.to(device)

param_optimizer = list(model.named_parameters())
no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
optimizer_parameters = [
    {
        "params": [
            p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay
            )
        ],
        "weight_decay": 0.001,
    },
    {
        "params": [
            p for n, p in param_optimizer if any(
                nd in n for nd in no_decay
            )
        ],
        "weight_decay": 0.0,
    },
]

num_train_steps = int(
    len(train_sentences) / base_batch_size * epochs
)
optimizer = AdamW(optimizer_parameters, lr=3e-5)
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=0, 
    num_training_steps=num_train_steps
)

best_loss = np.inf


train_loss = model.train_model(train_data_loader, optimizer, scheduler)
test_loss = model.eval_model(valid_data_loader)
print(f"Train Loss = {train_loss} Valid Loss = {test_loss}")


TEST = True



if TEST:
    sentences = ['V roku 1944 pracoval v redakcii " Národných novín " v Banskej Bystrici .',
                 'HC Košice v sezóne 1993/1994 hrali svoju 30. sezónu v rade na najvyššej možnej úrovni v Česko - Slovensku .',
                 'V rokoch 1938 - 1945 bol súčasťou Maďarska na základe Prvej Viedenskej arbitráže .',
                 ]
    labels = [['O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-MISC', 'I-MISC', 'O', 'O', 'B-LOC', 'I-LOC', 'O'],
              ['B-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'I-LOC', 'I-LOC', 'O'],
              ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O']
              ]
    
    for index, (sentence, real) in enumerate(zip(sentences, labels)):
        tokenized_sentence = tokenizer.tokenize(sentence)

        splitted_sentence = sentence.split()

        print(splitted_sentence)
    

        test_dataset = NerBaseDataset(
            texts=[splitted_sentence], 
            pos=[[0] * len(splitted_sentence)], 
            tags=[[0] * len(splitted_sentence)], 
            indeces=[[0] * len(splitted_sentence)], 
            tokenizer=tokenizer
        )

        with torch.no_grad():
            idata = test_dataset[0]
            for k, v in idata.items():
                idata[k] = v.to(device).unsqueeze(0)
            _, ids, mask, token_type_ids, target_pos, target_tag = idata["indeces"], idata["ids"], idata["mask"], idata["token_type_ids"], idata["target_pos"], idata["target_tag"]

            ids, mask, token_type_ids, target_pos, target_tag = ids.cuda(), mask.cuda(), token_type_ids.cuda(), target_pos.cuda(), target_tag.cuda()

            #tag, pos, _ = model.forward(ids, mask, token_type_ids, target_pos, target_tag, ids.shape[0], use_index=False) 
            tag, _ = model.forward(ids, mask, token_type_ids, target_pos, target_tag, ids.shape[0], use_index=False)
            tag_result = list(enc_tag.inverse_transform(tag.argmax(2).cpu().numpy().reshape(-1))[:len(tokenized_sentence)])
            other_result = tag_result
            print(f'Result:          {other_result}')
            print(f'Label:           {real}')
            #print('F1 score: ', f1_score([real], [other_result]))
            print('\n')
