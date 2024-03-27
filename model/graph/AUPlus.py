import copy
import time
import scipy as sp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
from data.augmentor import GraphAugmentor

class AUPlus(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(AUPlus, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['AUPlus'])
        aug_type = self.aug_type = int(args['-augtype'])
        self.cl_rate = float(args['-lambda'])
        self.eps = float(args['-eps'])
        self.n_layers = int(args['-n_layer'])
        self.bpr = bool(int(args['-bpr']))
        self.mode = int(args['-mode'])
        droprate = float(args['-droprate'])
        self.gamma = float(args['-gamma'])
        self.gammap = float(args['-gammap'])
        self.model = SimGCL_Encoder(
            self.data, self.emb_size, 
            self.eps, self.n_layers, aug_type, droprate)
        self.temp = float(args['-temp'])
        self.save_path = f'./results/AUPlus-{self.dataset_name}-lamb{self.cl_rate}-eps{self.eps}-nlayer{self.n_layers}-temp{self.temp}-gamma{self.gamma}-bpr{self.bpr}-mode{self.mode}-gammap{self.gammap}.pt'

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.lRate)
        for epoch in range(self.maxEpoch):
            if self.mode == 2:
                dropped_adj1 = model.graph_reconstruction()
                dropped_adj2 = model.graph_reconstruction()
            align_losses, unif_losses = 0, 0
            batch_losses = 0
            start = time.time()
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)): 
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                if self.bpr:
                    # BPR loss + emb0 perturb
                    assert(self.mode == 0)
                    rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                    with torch.no_grad():
                        align_loss, unif_loss = self.calculate_loss(rec_user_emb[user_idx], rec_item_emb[pos_idx])
                        align_losses += align_loss.item()
                        unif_losses += unif_loss.item()
                    cl_loss = self.cl_rate * self.cal_cl_loss(
                        [user_idx, pos_idx],
                        None, None)
                    batch_loss =  rec_loss + cl_loss + \
                        l2_reg_loss(self.reg, user_emb, pos_item_emb)/self.batch_size 
                else:
                    align_loss, unif_loss = self.calculate_loss(rec_user_emb[user_idx], rec_item_emb[pos_idx])
                    rec_loss = align_loss + self.gamma * unif_loss
                    with torch.no_grad():
                        align_losses += align_loss.item()
                        unif_losses += unif_loss.item()
                    if self.mode == 2:
                        cl_loss = self.cl_rate * self.cal_cl_loss(
                            [user_idx, pos_idx],
                            dropped_adj1, dropped_adj2)
                    else:
                        cl_loss = self.cl_rate * self.cal_cl_loss(
                            [user_idx, pos_idx],
                            None, None)
                    batch_loss =  rec_loss + \
                        l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss
                with torch.no_grad():
                    batch_losses += batch_loss.item()
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 200==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item(), 'time: ', time.time()-start)
                    start = time.time()
            print('training:', epoch + 1, 'average batch loss', batch_losses / (n+1))
            with torch.no_grad():
                align_losses /= (n+1)
                unif_losses /= (n+1)
                self.align_logs += (str(align_losses)+'\n')
                self.unif_logs += (str(unif_losses)+'\n')
                self.model.eval()
                self.user_emb, self.item_emb = self.model()
                self.model.train()
            if epoch % 1 == 0:
                self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def alignment(self, x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    def uniformity(self, x):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

    def calculate_loss(self, user_e, item_e):
        # user_e, item_e = self.encoder(user, item)  # [bsz, dim]
        align = self.alignment(user_e, item_e)
        uniform = (self.uniformity(user_e) + self.uniformity(item_e)) / 2
        return align, uniform
    
    def cosine_similarity_loss(self, x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 1 - torch.sum(x * y, dim=1).mean()

    def cal_align_loss(self, x, y, alpha=2):
        return (x - y).norm(dim=1).pow(alpha).mean()

    def cal_unif_loss(self, x, t=2):
        sq_pdist = torch.pdist(x, p=2).pow(2)
        return sq_pdist.mul(-t).exp().mean().log()

    def cal_cl_loss(self, idx, perturbed_mat1, perturbed_mat2):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        
        if self.mode == 2:
            user_view_1, item_view_1 = self.model.forward(perturbed_adj=perturbed_mat1)
            user_view_2, item_view_2 = self.model.forward(perturbed_adj=perturbed_mat2)
        else:
            user_view_1, item_view_1 = self.model.forward(perturb_embs=True)
            user_view_2, item_view_2 = self.model.forward(perturb_embs=True)
        
        if self.mode == 0 or self.mode == 2:
            view1 = torch.cat((user_view_1[u_idx],item_view_1[i_idx]),0)
            view2 = torch.cat((user_view_2[u_idx],item_view_2[i_idx]),0)
            loss = InfoNCE(view1, view2, self.temp)
        elif self.mode == 1:
            user_view_1 = user_view_1[u_idx]
            user_view_2 = user_view_2[u_idx]
            item_view_1 = item_view_1[i_idx]
            item_view_2 = item_view_2[i_idx]
            align_loss_item, unif_loss_item = self.calculate_loss(item_view_1, item_view_2)
            align_loss_user, unif_loss_user = self.calculate_loss(user_view_1, user_view_2)
            align_loss = (align_loss_user + align_loss_item) / 2
            unif_loss = (unif_loss_user + unif_loss_item) / 2
            loss = align_loss + self.gammap * unif_loss
        return loss

    def save(self):
        # current is the best model
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()
        if self.save_model:
            torch.save(self.model, self.save_path)

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

class EMA:
    def __init__(self, beta, epochs):
        super().__init__()
        self.beta = beta
        self.step = 0
        self.total_steps = epochs

    def update_average(self, old, new):
        if old is None:
            return new
        beta = 1 - (1 - self.beta) * (np.cos(np.pi * self.step / self.total_steps) + 1) / 2.0
        self.step += 1
        return old * beta + (1 - beta) * new

class SimGCL_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers, aug_type, drop_rate):
        super(SimGCL_Encoder, self).__init__()
        self.data = data
        self.eps = eps
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.aug_type = aug_type
        self.drop_rate = drop_rate
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def graph_reconstruction(self):
        if self.aug_type== 0 or 1:
            dropped_adj = self.random_graph_augment()
        else:
            dropped_adj = []
            for k in range(self.n_layers):
                dropped_adj.append(self.random_graph_augment())
        return dropped_adj

    def random_graph_augment(self):
        dropped_mat = None
        if self.aug_type == 0:
            dropped_mat = GraphAugmentor.node_dropout(self.data.interaction_mat, self.drop_rate)
        elif self.aug_type == 1 or self.aug_type == 2:
            dropped_mat = GraphAugmentor.edge_dropout(self.data.interaction_mat, self.drop_rate)
        dropped_mat = self.data.convert_to_laplacian_mat(dropped_mat)
        return TorchGraphInterface.convert_sparse_mat_to_tensor(dropped_mat).cuda()

    def forward(self, perturbed=False, perturbed_adj=None, perturb_embs=False):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = []
        if perturb_embs:
            random_noise = torch.rand_like(ego_embeddings).cuda()
            ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
        
        for k in range(self.n_layers):
            if perturbed_adj is not None:
                if isinstance(perturbed_adj,list):
                    ego_embeddings = torch.sparse.mm(perturbed_adj[k], ego_embeddings)
                else:
                    ego_embeddings = torch.sparse.mm(perturbed_adj, ego_embeddings)
            else:
                ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).cuda()
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
            all_embeddings.append(ego_embeddings)
        
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])
        return user_all_embeddings, item_all_embeddings

