import numpy as np
import torch
import json

from exceptions import NotSupportedException, MissingValueException

#logits -> array of logits, shape (num_indeces, batch_size, output_size)
#probabilities -> array of probability arrays, shape (num_indeces, batch_size, output_size)   (now not used)
#labels -> array of labels, shape (output_size)

#return_value -> (func_value (in original order), suggested_order)


#classic AL functions

def uniform_per_example(logits, probabilities, labels, indeces):
    """Returns uniformly random scores per example."""
    del probabilities, logits, indeces
    labels = torch.squeeze(labels)

    res = torch.rand(labels.shape).numpy()
    
    return res, torch.tensor(np.argsort(-res))


def entropy_per_example(logits, probabilities, labels, indeces):
    """Calculates entropy per example."""
    del labels, probabilities, indeces
    _, data_size, num_classes = logits.shape
    #logits = torch.tensor(logits)
    logits = logits.clone().detach()

    sample_probs = torch.nn.functional.softmax(logits, dim=2)
    probs = torch.mean(sample_probs, dim=0)
    assert probs.shape == (data_size, num_classes)
    entropies = -torch.sum(probs * torch.log(probs), dim=1)
    assert entropies.shape == (data_size,)

    res = entropies.cpu().numpy()

    return res, torch.tensor(np.argsort(-res))

def margin_per_example(logits, probabilities, labels, indeces):
    """Calculates margin between top and second probabilities per example."""
    # See e.g. use in PLEX paper: https://arxiv.org/abs/2207.07411
    del labels, probabilities, indeces
    _, data_size, num_classes = logits.shape
    logits = np.array(logits.cpu())

    sample_probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    sample_probs /= np.sum(sample_probs, axis=-1, keepdims=True)

    probs = np.mean(sample_probs, axis=0)
    assert probs.shape == (data_size, num_classes)

    sorted_probs = np.sort(probs, axis=-1)
    margins = sorted_probs[:, -1] - sorted_probs[:, -2]
    assert margins.shape == (data_size,)

    # Return the *negative* margin

    res = -margins

    return res, torch.tensor(np.argsort(-res))


_Classic_Metrics = {
    'entropy': entropy_per_example,
    'uniform': uniform_per_example,
    'margin': margin_per_example
}


#epistemic AL functions


def bald_per_example(logits, probabilities, labels, indeces):
    """Calculates BALD mutual information per example."""
    del probabilities, labels, indeces
    num_enn_samples, data_size, num_classes = logits.shape
    #logits = torch.tensor(logits)
    logits = logits.clone().detach()
    
    # Function to compute entropy
    def compute_entropy(p):
        return -torch.sum(p * torch.log(p), dim=1)

    # Compute entropy for average probabilities
    sample_probs = torch.nn.functional.softmax(logits, dim=2)
    mean_probs = torch.mean(sample_probs, dim=0)
    assert mean_probs.shape == (data_size, num_classes)
    mean_entropy = compute_entropy(mean_probs)
    assert mean_entropy.shape == (data_size,)

    # Compute entropy for each sample probabilities
    sample_entropies = torch.stack([compute_entropy(sample_probs[i]) for i in range(num_enn_samples)])
    assert sample_entropies.shape == (num_enn_samples, data_size)

    models_disagreement = mean_entropy - torch.mean(sample_entropies, dim=0)
    assert models_disagreement.shape == (data_size,)

    res = models_disagreement.cpu().numpy()
    
    return res, torch.tensor(np.argsort(-res))

def variance_per_example(logits, probabilities, labels, indeces):
    """Calculates variance per example."""
    del labels, probabilities, indeces
    _, data_size, _ = logits.shape
    #logits_tensor = torch.tensor(logits)
    logits_tensor = logits.clone().detach()
    probs = torch.nn.functional.softmax(logits_tensor, dim=-1)
    variances = torch.sum(torch.var(probs, dim=0), dim=-1)
    assert variances.shape == (data_size,)
    res = variances.cpu().numpy()

    return res, torch.tensor(np.argsort(-res))


def nll_per_example(logits, probabilities, labels, indeces):
    """Calculates negative log-likelihood (nll) per example."""
    del probabilities, indeces
    _, data_size, unused_num_classes = logits.shape
    #logits = torch.tensor(logits)
    logits = logits.clone().detach()

    sample_probs = torch.nn.functional.softmax(logits, dim=2)
    probs = torch.mean(sample_probs, dim=0)
    assert probs.shape == (data_size, unused_num_classes)

    # Penalize with log loss
    labels = labels.cpu().numpy().astype(np.int32)
    labels = np.squeeze(labels)
    true_probs = probs[torch.arange(data_size), labels]
    losses = -torch.log(true_probs)
    assert losses.shape == (data_size,)

    res = losses.cpu().numpy()

    return res, torch.tensor(np.argsort(-res))

def joint_nll_per_example(logits, probabilities, labels, indeces):
    """Calculates joint negative log-likelihood (nll) per example."""
    del probabilities, indeces
    num_enn_samples, data_size, _ = logits.shape
    #logits = torch.tensor(logits)
    logits = logits.clone().detach()

    sample_probs = torch.nn.functional.softmax(logits, dim=2)

    # Penalize with log loss
    labels = labels.cpu().numpy().astype(np.int32)
    labels = np.squeeze(labels)
    true_probs = sample_probs[:, torch.arange(data_size), labels]
    tau = 10
    repeated_lls = tau * torch.log(true_probs)
    assert repeated_lls.shape == (num_enn_samples, data_size)

    # Take average of joint lls over num_enn_samples
    joint_lls = torch.mean(repeated_lls, dim=0)
    assert joint_lls.shape == (data_size,)

    res = -joint_lls.cpu().numpy()

    return res, torch.tensor(np.argsort(-res))


_Epistemic_Metrics = {

    'bald' : bald_per_example,
    'variance': variance_per_example,
    'nll': nll_per_example,
    'jointnll': joint_nll_per_example
}

#custom AL functions

class FurthestFromCluster:
    def __init__(self):
        self.order = None

    def load_file(self, path):
        self.distances = json.load(open(path, 'r'))
        self.order = list(map(int, sorted(self.distances.keys(), key=lambda x: self.distances[x]))) #TODO: check if from biggest to smallest
        print(len(self.order))

    def check(self):
        if self.order is None:
            raise MissingValueException('You have to load the distances file first!')

    def furthest_all(self, logits, probabilities, labels, indeces):  #TODO musel by sa vytvoriť LOADER a natchovo sa brať napr. 75% potom, to nepatri tu 
        self.check()
        del probabilities, labels, logits, indeces
        raise NotSupportedException('This function is not supported')

    def furthest_batch(self, logits, probabilities, labels, indeces): #ma zmysel to použivať bez clustrovacej enn?
        self.check()
        del probabilities, labels, logits
        vals = [x for x in self.order if x in indeces]
        order = [vals.index(val) for val in indeces]
        return None, torch.Tensor(order).int()    #no res should be used
    
f = FurthestFromCluster()	
        
_New_Metrics = {
    'furthest': f,
    'furthest-all': f.furthest_all,
    'furthest-batch': f.furthest_batch
}    

_All_Metrics = _Classic_Metrics | _Epistemic_Metrics | _New_Metrics

#test

if __name__ == '__main__':
    probs = torch.Tensor([[0.2, 0.3, 0.5], [0.0, 0.9, 0.1], [0.4, 0.4, 0.2]])

    for name, fnc in _Classic_Metrics.items():
        return_vals, suggested_order = fnc(probs)
        print(f'{name} : {return_vals} {suggested_order}')





