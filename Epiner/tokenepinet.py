import torch
import torch.nn as nn

import numpy as np

from utils.token_loss import token_loss_fn, get_logits_labels

device = torch.device("cuda")

#TODO: add docstrings

class TokenEpiNet(nn.Module):

    def __init__(self, num_tag, num_pos, input_size: int, index_size: int) -> None:
        super(TokenEpiNet, self).__init__()

        self.index_size = index_size

        self.num_tag = num_tag
        self.num_pos = num_pos

        self.prior_drop_1 = nn.Dropout(0.3)
        #self.prior_drop_2 = nn.Dropout(0.3)

        self.prior_out_tag = nn.Linear(input_size + index_size, self.num_tag)
        #self.prior_out_pos = nn.Linear(input_size + index_size, self.num_pos)

        # Freeze the prior network
        for param in self.prior_out_tag.parameters():
            param.requires_grad = False

        #for param in self.prior_out_pos.parameters():
        #    param.requires_grad = False


        self.learnable_drop_1 = nn.Dropout(0.3)
        #self.learnable_drop_2 = nn.Dropout(0.3)

        self.learnable_out_tag = nn.Linear(input_size + index_size, self.num_tag)
        #self.learnable_out_pos = nn.Linear(input_size + index_size, self.num_pos)

        nn.init.xavier_uniform_(self.learnable_out_tag.weight)
        #nn.init.xavier_uniform_(self.learnable_out_pos.weight)

        self.to(device)

    def forward_logits_labels(self, x, indices, mask, target_pos, target_tag):
        len_indices = len(indices)
        indices = indices.cuda()

        x = torch.cat([x, indices], dim=-1)
        x = x.cuda()

        zeros = torch.zeros(len_indices, self.index_size)
        zeros = zeros.cuda()

        mask = mask.cuda()
        mask = torch.cat([mask, zeros], dim=1)
        #target_pos = torch.cat([target_pos, zeros], dim=1)
        #target_pos = target_pos.cuda()

        target_tag = torch.cat([target_tag, zeros], dim=1)
        target_tag = target_tag.cuda()

        pbo_tag = self.prior_drop_1(x)
        #pbo_pos = self.prior_drop_2(x)
                           
        ptag = self.prior_out_tag(pbo_tag)
        #ppos = self.prior_out_pos(pbo_pos)

        prior_tag_logits, prior_tag_labels = get_logits_labels(nn.CrossEntropyLoss(), ptag, target_tag, mask, self.num_tag, "logits labels prior tag")
        #prior_pos_logits, prior_pos_labels = get_logits_labels(nn.CrossEntropyLoss(), ppos, target_pos, mask, self.num_pos, "logits labels prior pos")

        lbo_tag = self.learnable_drop_1(x)
        #lbo_pos = self.learnable_drop_2(x)
                           
        ltag = self.learnable_out_tag(lbo_tag)
        #lpos = self.learnable_out_pos(lbo_pos)

        learnable_tag_logits, learnable_tag_labels = get_logits_labels(nn.CrossEntropyLoss(), ltag, target_tag, mask, self.num_tag, "logits labels learnable tag")
        #learnable_pos_logits, learnable_pos_labels = get_logits_labels(nn.CrossEntropyLoss(), lpos, target_pos, mask, self.num_pos, "logits labels learnable pos")

        #return prior_tag_logits, prior_tag_labels, prior_pos_logits, prior_pos_labels, learnable_tag_logits, learnable_tag_labels, learnable_pos_logits, learnable_pos_labels

        return prior_tag_logits, prior_tag_labels, learnable_tag_logits, learnable_tag_labels


    def forward(self, x, indices, mask, target_pos, target_tag, batch_size, index_dim):
        indices = indices.cuda()

        x = torch.cat([x, indices], dim=-1)
        x = x.cuda()


        zeros = torch.zeros(1, index_dim)  #TODO: niečo z toho je asi skor index_num
        zeros = zeros.repeat(batch_size, index_dim).reshape(-1, index_dim)
        zeros = zeros.cuda()

        #mask = torch.cat([mask, zeros], dim=1)
        mask = mask.cuda()


        #target_pos = torch.cat([target_pos, zeros], dim=1)
        #target_pos = target_pos.cuda()

        #target_tag = torch.cat([target_tag, zeros], dim=1)
        #target_tag = target_tag.cuda()

        #p_tags, p_pos, p_loss = self.forward_prior(x, mask, target_pos, target_tag)
        #l_tags, l_pos, l_loss = self.forward_learnable(x, mask, target_pos, target_tag)

        p_tags, p_loss = self.forward_prior(x, mask, target_pos, target_tag)
        l_tags, l_loss = self.forward_learnable(x, mask, target_pos, target_tag)

        return (p_tags + l_tags) / 2, (p_loss + l_loss) / 2

        #return (p_tags + l_tags) / 2, (p_pos + l_pos) / 2, (p_loss + l_loss) / 2
    
    def forward_prior(self, x, mask, target_pos, target_tag):
        bo_tag = self.prior_drop_1(x)
        #bo_pos = self.prior_drop_2(x)
                           
        tag = self.prior_out_tag(bo_tag)
        #pos = self.prior_out_pos(bo_pos)

        loss_tag = token_loss_fn(tag, target_tag, mask, self.num_tag, "forward prior tag")
        #loss_pos = token_loss_fn(pos, target_pos, mask, self.num_pos, "forward prior pos")

        #loss = (loss_tag + loss_pos) / 2  #average of losses

        loss = loss_tag

        return tag, loss

       #return tag, pos, loss
    
    def forward_learnable(self, x, mask, target_pos, target_tag):
        bo_tag = self.learnable_drop_1(x)
        #bo_pos = self.learnable_drop_2(x)
                           
        tag = self.learnable_out_tag(bo_tag)
        #pos = self.learnable_out_pos(bo_pos)

        loss_tag = token_loss_fn(tag, target_tag, mask, self.num_tag, "forward learnable tag")
        #loss_pos = token_loss_fn(pos, target_pos, mask, self.num_pos, "forward learnable pos")

        #loss = (loss_tag + loss_pos) / 2  #average of losses

        loss = loss_tag

        return tag, loss

        #return tag, pos, loss

        
