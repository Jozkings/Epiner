
from transformers import RobertaModel

import torch
import torch.nn as nn

from token_loss import token_loss_fn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SlovakBertTokenClassifier(nn.Module):
    def __init__(self, num_tag: int, num_pos: int, model_name: str='gerulata/slovakbert', dim_in: int=768, dim_out: int=50, name='slovakbert') -> None:
        super(SlovakBertTokenClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in = dim_in  # 768 for base models

        self.model = RobertaModel.from_pretrained(model_name, return_dict=False)
        self.model_name = model_name
        self.name = name

        self.only_mlm = False

        self.bert_drop_1 = nn.Dropout(0.3)
        #self.bert_drop_2 = nn.Dropout(0.3)

        self.num_tag = num_tag
        self.num_pos = num_pos

        self.out_tag = nn.Linear(D_in, self.num_tag)
        #self.out_pos = nn.Linear(D_in, self.num_pos)

        self.frozen = False

        self.to(device)
  
    def freeze(self):
        self.frozen = True
        for param in self.model.parameters():
            param.requires_grad = False

        self.to(device)  #TODO: treba?
 
    def forward_epinet(self, ids, mask, token_type_ids, target_pos, target_tag):        

        self.train()
        self.zero_grad()
        
        o1, _ = self.model(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids)
                           
        bo_tag = self.bert_drop_1(o1)
        #bo_pos = self.bert_drop_2(o1)
                           
        tag = self.out_tag(bo_tag)
        #pos = self.out_pos(bo_pos)


        loss_tag = token_loss_fn(tag, target_tag, mask, self.num_tag, "forward tag")
        #loss_pos = token_loss_fn(pos, target_pos, mask, self.num_pos, "forward pos")

        #loss = (loss_tag + loss_pos) / 2  #average of losses

        return tag, loss_tag, o1

        #return tag, pos, loss, o1