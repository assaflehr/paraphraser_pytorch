import numpy as np
import torch

def T(arg):
    cuda =  torch.cuda.is_available()
    if cuda:
        if type(arg) == tuple:
            arg = tuple(t.cuda() for t in arg)  # new tuple all cuda-d
        else:
            arg = arg.cuda()
    return arg


def N(arg):
    if isinstance(arg, np.ndarray) or isinstance(arg, np.float64):
        return arg  # as is
    # to numpy
    return arg.cpu().numpy()

def revers_vocab(vocab,sent,seperator):
  return seperator.join([vocab.itos[token] for token in sent])
