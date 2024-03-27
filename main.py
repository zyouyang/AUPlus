import argparse
import random
import numpy as np
import torch

from SELFRec import SELFRec
from util.conf import ModelConf

def init_seed(seed=2024):
    r"""init random seed for random functions in numpy, torch, cuda and cudnn

    Args:
        seed (int): random seed
        reproducibility (bool): Whether to require reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SELFRec')
    parser.add_argument('--model', type=str, default='AUPlus', help='model name')
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    args = parser.parse_args()

    # Register your model here
    graph_baselines = ['LightGCN','DirectAU','MF','SASRec']
    ssl_graph_models = ['AUPlus', 'SGL', 'SimGCL', 'SEPT', 'MHCN', 'BUIR', 'SelfCF', 'SSL4Rec', 'XSimGCL', 'NCL','MixGCF']
    sequential_baselines= ['SASRec']
    ssl_sequential_models = ['CL4SRec']

    model = args.model
    init_seed(args.seed)
    import time

    s = time.time()
    if model in graph_baselines or model in ssl_graph_models or model in sequential_baselines or model in ssl_sequential_models:
        conf = ModelConf('./conf/' + model + '.conf')
        print(f'Running {model}...')
    else:
        print('Wrong model name!')
        exit(-1)
    rec = SELFRec(conf)
    rec.seed = args.seed
    rec.execute()
    e = time.time()
    print("Running time: %f s" % (e - s))
