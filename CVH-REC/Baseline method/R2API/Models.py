import h5py
import torch
import torch.nn as nn
from utility.batch_test import *
import torch.nn as nn
import torch.sparse as sparse
import torch.nn.functional as F

#This source file is based on the GRec published by Bo Li et al.
#We would like to thank and offer our appreciation to them.
#Original algorithm can be found in paper: Embedding App-Library Graph for Neural Third Party Library Recommendation. ESEC/FSE ’21

class HCF(nn.Module):
    def __init__(self, n_users, n_items, n_mashup,n_mashup_tag,n_api,n_api_tag,embedding_dim, layer_num, dropout_list):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_mashup = n_mashup
        self.n_mashup_tag = n_mashup_tag
        self.n_api = n_api
        self.n_api_tag = n_api_tag
        self.embedding_dim = embedding_dim

        self.n_layers = layer_num
        self.dropout_list = nn.ModuleList()

        torch.manual_seed(50)
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.mashup_tag_embedding=nn.Embedding(n_mashup_tag,embedding_dim)
        self.api_tag_embedding= nn.Embedding(n_api_tag,embedding_dim)

        self._init_weight_()
        for i in range(self.n_layers):
            self.dropout_list.append(nn.Dropout(dropout_list[i]))

        # 可训练的三层加权系数
        self.u_weights = nn.Parameter(torch.ones(self.n_layers))
        self.i_weights = nn.Parameter(torch.ones(self.n_layers))
        self.m_weights = nn.Parameter(torch.ones(self.n_layers))
        self.a_weights = nn.Parameter(torch.ones(self.n_layers))

    def get_vector_by_id(self, id, file):
        with h5py.File(file, 'r') as f:
            if str(id) in f:
                return f[str(id)][:]
            else:
                return None

    def _init_weight_(self):
        list_data = []
        for i in range(self.n_users):
            list_data.append(self.get_vector_by_id(i, 'data/vectors.h5'))
        tensor = torch.FloatTensor(list_data)
        self.user_embedding = nn.Embedding.from_pretrained(tensor)

        api_data = []
        for i in range(self.n_items):
            api_data.append(self.get_vector_by_id(i, 'data/API_vectors.h5'))
        tensor = torch.FloatTensor(api_data)
        self.item_embedding = nn.Embedding.from_pretrained(tensor)

        mashup_tag_data = []
        for i in range(self.n_mashup_tag):
            a = self.get_vector_by_id(i, 'data/mashup_tag_vector.h5')
            mashup_tag_data.append(a)

        tensor = torch.FloatTensor(mashup_tag_data)
        self.mashup_tag_embedding = nn.Embedding.from_pretrained(tensor)

        api_tag_data = []
        for i in range(self.n_api_tag):
            api_tag_data.append(self.get_vector_by_id(i, 'data/api_tag_vector.h5'))
        tensor = torch.FloatTensor(api_tag_data)
        self.api_tag_embedding = nn.Embedding.from_pretrained(tensor)

        self.user_embedding.weight.requires_grad = True #计算梯度，将会更新
        self.item_embedding.weight.requires_grad = True
        self.mashup_tag_embedding.weight.requires_grad= True
        self.api_tag_embedding.weight.requires_grad =True
        # torch.manual_seed(50)
        # nn.init.xavier_uniform_(self.user_embedding.weight)
        #
        # nn.init.xavier_uniform_(self.item_embedding.weight)
        # nn.init.xavier_uniform_(self.mashup_tag_embedding.weight)
        # nn.init.xavier_uniform_(self.api_tag_embedding.weight)



    def forward(self, adj_u1,adj_u2,adj_i1,adj_i2,adj_m1,adj_m2,adj_a1,adj_a2):
        hu = self.user_embedding.weight
        embedding=[hu]
        for i in range(self.n_layers):
            t=torch.sparse.mm(adj_u2,embedding[-1])
            t =torch.sparse.mm(adj_u1,t)
            embedding.append(t)
        # u_emb=torch.stack(embedding,dim=1)
        # u_emb=torch.mean(u_emb, dim=1, keepdim=False)
        u_emb = torch.stack(embedding[:self.n_layers], dim=1)  # 取前3层
        u_w = torch.softmax(self.u_weights, dim=0)  # 归一化权重
        u_emb = torch.sum(u_emb * u_w.view(1, self.n_layers, 1), dim=1)  # 加权求和

        hi = self.item_embedding.weight
        embedding_i = [hi]
        for i in range(self.n_layers):
            t =torch.sparse.mm(adj_i2, embedding_i[-1])
            t = torch.sparse.mm(adj_i1, t)
            embedding_i.append(t)
        # i_emb = torch.stack(embedding_i, dim=1)
        # i_emb = torch.mean(i_emb, dim=1, keepdim=False)
        i_emb = torch.stack(embedding_i[:self.n_layers], dim=1)
        i_w = torch.softmax(self.i_weights, dim=0)
        i_emb = torch.sum(i_emb * i_w.view(1, self.n_layers, 1), dim=1)

        hm = self.mashup_tag_embedding.weight
        embedding_m = [hm]
        for i in range(self.n_layers):
            t = torch.sparse.mm(adj_m2, embedding_m[-1])
            t = torch.sparse.mm(adj_m1, t)
            embedding_m.append(t)
        # m_emb = torch.stack(embedding_m, dim=1)
        # m_emb = torch.mean(m_emb, dim=1, keepdim=False)
        m_emb = torch.stack(embedding_m[:self.n_layers], dim=1)
        m_w = torch.softmax(self.m_weights, dim=0)
        m_emb = torch.sum(m_emb * m_w.view(1, self.n_layers, 1), dim=1)

        ha = self.api_tag_embedding.weight
        embedding_a = [ha]
        for i in range(self.n_layers):
            t = torch.sparse.mm(adj_a2, embedding_a[-1])
            t = torch.sparse.mm(adj_a1, t)
            embedding_a.append(t)
        # a_emb = torch.stack(embedding_a, dim=1)
        # a_emb = torch.mean(a_emb, dim=1, keepdim=False)
        a_emb = torch.stack(embedding_a[:self.n_layers], dim=1)
        a_w = torch.softmax(self.a_weights, dim=0)
        a_emb = torch.sum(a_emb * a_w.view(1, self.n_layers, 1), dim=1)
        return u_emb,i_emb,m_emb,a_emb