from collections import defaultdict 
import numpy as np
import json
import torch.nn as nn
import tqdm
import torch
import sys
from collections import defaultdict as dd
from copy import deepcopy

sys.path.append('../..')

from active_learning import _All_Metrics, _New_Metrics

#TODO: add docstrings


class ActiveTokenENN(nn.Module):

    def __init__(self, base_model, epinet_model, config, loggers):
        super(ActiveTokenENN, self).__init__()
        #super().__init__(base_model, epinet_model, config, loggers)

        self.al_functor = config.active_functor
        self.al_ratio = config.al_ratio
        self.true_validation = config.true_validation
        self.tokenizer = config.tokenizer
        self.warm_start_ratio = config.warm_start_ratio
        self.active_scheduler = config.active_scheduler

        self.num = config.index_number 
        self.index_dim = config.index_dimension
        self.only_bert = config.only_bert
        self.output_size = config.dimension_out
        self.epochs = config.epochs
        self.model_name = config.model_name
        self.type_averaging =  config.averaging_type

        self.epinet = epinet_model
        self.base_model = base_model 

        self.loggers = loggers

        self.name = "ActiveTokenENN"

    def forward(self, ids, mask, token_type_ids, target_pos, target_tag, batch_size, use_index=True):
        #tag, pos, loss, features = self.base_model.forward_epinet(ids, mask, token_type_ids, target_pos, target_tag)
        tag, loss, features = self.base_model.forward_epinet(ids, mask, token_type_ids, target_pos, target_tag)

        if self.only_bert:
            return tag, loss  #tag, pos, loss 

    def train_step(self, optim, scheduler, ids, mask, token_type_ids, target_pos, target_tag, indices=None):
        # zero the parameter gradients
        optim.zero_grad()
        ids, mask, token_type_ids, target_pos, target_tag = ids.cuda(), mask.cuda(), token_type_ids.cuda(), target_pos.cuda(), target_tag.cuda()

        #_, _, loss = self.forward(ids, mask, token_type_ids, target_pos, target_tag, ids.shape[0])
        _, loss = self.forward(ids, mask, token_type_ids, target_pos, target_tag, ids.shape[0])
        
        loss.backward()

        norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)  #gradient clipping :)))))

        optim.step()
        scheduler.step()

        return loss.item()
    
    def eval_step(self, ids, mask, token_type_ids, target_pos, target_tag, indices=None):
        ids, mask, token_type_ids, target_pos, target_tag = ids.cuda(), mask.cuda(), token_type_ids.cuda(), target_pos.cuda(), target_tag.cuda()
        #_, _, loss = self.forward(ids, mask, token_type_ids, target_pos, target_tag, ids.shape[0], use_index=False)
        _, loss = self.forward(ids, mask, token_type_ids, target_pos, target_tag, ids.shape[0], use_index=False)
        return loss.item()
        
    def eval_model(self, data_loader): 
        self.eval()
        final_loss = 0
        with torch.no_grad():
            for idata in data_loader:
                _, ids, mask, token_type_ids, target_pos, target_tag = idata["indeces"], idata["ids"], idata["mask"], idata["token_type_ids"], idata["target_pos"], idata["target_tag"]
                loss = self.eval_step(ids, mask, token_type_ids, target_pos, target_tag, None)            
                final_loss += loss
                
        return final_loss


    def train_model(self, train_loader, optimizer, scheduler):
        self.used_data = defaultdict(set)

        start_epoch = 0 if self.warm_start_ratio == 0.0 else 1

        training_loss = 0
        all_data_number = 0
        

        for epoch in tqdm.tqdm(range(start_epoch, self.epochs), desc="Training", position=0):
            print(f"Epoch {epoch+1}: start") 

            self.train(True)

            data_number = 0
            final_loss = 0

            # for data in train_loader:
            for idata in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}: ", leave=False, position=1):
                _, ids, mask, token_type_ids, target_pos, target_tag = idata["indeces"], idata["ids"], idata["mask"], idata["token_type_ids"], idata["target_pos"], idata["target_tag"]

                if self.al_functor not in _All_Metrics:
                    raise Exception('Chosen active learning method is not known or implemented yet!')

                data_number += len(ids)
                # get the inputs; data is a list of [inputs, labels]

                loss = self.train_step(optimizer, scheduler, ids, mask, token_type_ids, target_pos, target_tag, None)
                final_loss += loss
            print(f"Epoch loss = {final_loss / data_number}")
            training_loss += final_loss
            all_data_number += data_number

        if all_data_number == 0:
            return np.inf

        return training_loss / all_data_number
