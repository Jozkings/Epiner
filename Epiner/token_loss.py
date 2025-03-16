import torch.nn as nn
import torch

def token_loss_fn(output, target, mask, num_labels, usage):  #https://www.youtube.com/watch?v=MqQ7rqRllIc&ab_channel=AbhishekThakur
    lfn = nn.CrossEntropyLoss()
    active_logits, active_labels = get_logits_labels(lfn, output, target, mask, num_labels, usage)
    loss = lfn(active_logits, active_labels)
    return loss

def get_logits_labels(lfn, output, target, mask, num_labels, usage):
    active_loss = mask.view(-1) == 1
    active_logits = output.view(-1, num_labels)
    active_labels = torch.where(
        active_loss, target.view(-1), torch.tensor(lfn.ignore_index).type_as(target)
    )
    return active_logits, active_labels