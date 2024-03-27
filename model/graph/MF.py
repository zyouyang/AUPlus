import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from util.loss_torch import bpr_loss,l2_reg_loss


class MF(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(MF, self).__init__(conf, training_set, test_set)
        self.model = Matrix_Factorization(self.data, self.emb_size)
        self.save_path = f'./results/MF-{self.dataset_name}.pt'

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            align_losses, unif_losses = 0, 0
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(self.reg, user_emb,pos_item_emb,neg_item_emb)/self.batch_size
                with torch.no_grad():
                    align_loss, unif_loss = self.calculate_loss(rec_user_emb[user_idx], rec_item_emb[pos_idx])
                    align_losses += align_loss.item()
                    unif_losses += unif_loss.item()
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 300==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
            align_losses /= (n+1)
            unif_losses /= (n+1)
            self.align_logs += (str(align_losses)+'\n')
            self.unif_logs += (str(unif_losses)+'\n')
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            if epoch % 5 == 0:
                self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        with torch.no_grad():
            u = self.data.get_user_id(u)
            score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
            return score.cpu().numpy()

    @torch.no_grad()
    def alignment(self, x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    @torch.no_grad()
    def uniformity(self, x):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

    @torch.no_grad()
    def calculate_loss(self, user_e, item_e):
        # user_e, item_e = self.encoder(user, item)  # [bsz, dim]
        align = self.alignment(user_e, item_e)
        uniform = (self.uniformity(user_e) + self.uniformity(item_e)) / 2
        # loss = align + uniform
        return align, uniform

class Matrix_Factorization(nn.Module):
    def __init__(self, data, emb_size):
        super(Matrix_Factorization, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.embedding_dict = self._init_model()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def forward(self, perturb):
        return self.embedding_dict['user_emb'], self.embedding_dict['item_emb']


