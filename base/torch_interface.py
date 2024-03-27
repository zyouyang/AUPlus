import numpy as np
import torch

class TorchGraphInterface(object):
    def __init__(self):
        pass

    @staticmethod
    def convert_sparse_mat_to_tensor(X, dropout=-1):
        coo = X.tocoo()
        if dropout < 0:
            i = torch.LongTensor(np.array([coo.row, coo.col]))
            # i = torch.LongTensor([coo.row, coo.col])
            v = torch.from_numpy(coo.data).float()
        else:
            assert(dropout < 1 and dropout > 0)
            mask = (torch.rand(v.shape) > dropout).bool()
            i = torch.LongTensor([coo.row[mask], coo.col[mask]])
            v = torch.from_numpy(coo.data[mask]).float()
            
        return torch.sparse.FloatTensor(i, v, coo.shape)