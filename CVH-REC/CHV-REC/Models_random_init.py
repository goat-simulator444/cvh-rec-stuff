import h5py
import torch
import torch.nn as nn
import torch.nn.init as init


class ProjectionMLP(nn.Module):
    def __init__(self, dim, hidden_dim=768):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)  # 加norm
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = torch.nn.GELU()(x)  # GELU相对ELU数值更平稳
        return self.fc2(x)

class HCF(nn.Module):
    def __init__(self, n_users, n_items, n_tags,
                 embedding_dim, layer_num, dropout_list):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_tags = n_tags

        self.embedding_dim = embedding_dim
        self.n_layers = layer_num

        # torch.manual_seed(50)

        # 初始化embedding，先用占位，后面用预训练权重替换
        self.mashup_call_embedding = nn.Embedding(n_users, embedding_dim)
        self.api_call_embedding = nn.Embedding(n_items, embedding_dim)

        self.mashup_tag_embedding = nn.Embedding(n_users, embedding_dim)
        self.api_tag_embedding = nn.Embedding(n_items, embedding_dim)

        self.global_embedding = nn.Embedding(n_users + n_items + n_tags, embedding_dim)

        self.dropout_list = nn.ModuleList([nn.Dropout(p) for p in dropout_list])

        self.u_weights = nn.Parameter(torch.ones(self.n_layers))
        self.i_weights = nn.Parameter(torch.ones(self.n_layers))
        self.m_weights = nn.Parameter(torch.ones(self.n_layers))
        self.a_weights = nn.Parameter(torch.ones(self.n_layers))
        self.m_t_weights = nn.Parameter(torch.ones(self.n_layers))
        self.a_t_weights = nn.Parameter(torch.ones(self.n_layers))
        self.global_weights = nn.Parameter(torch.ones(self.n_layers))

        # 融合视角权重
        self.mashup_view_weights = nn.Parameter(torch.ones(2))  # [call, tag]
        self.api_view_weights = nn.Parameter(torch.ones(2))  # [call, tag]

        # 全局局部权重
        self.mashup_l_g_weights = nn.Parameter(torch.ones(2))  # [local, global]
        self.api_l_g_weights = nn.Parameter(torch.ones(2))  # [local, global]

        # 用于对比学习的投影头
        self.mashup_local_proj = ProjectionMLP(768)
        self.mashup_global_proj = ProjectionMLP(768)
        self.api_local_proj = ProjectionMLP(768)
        self.api_global_proj = ProjectionMLP(768)

        self.mashup_tag_predictor = nn.Linear(768, self.n_tags)  # 输入mashup_final的维度
        self.api_tag_predictor = nn.Linear(768, self.n_tags)


        self._init_weight_()

    def get_vector_by_id(self, id, file):
        with h5py.File(file, 'r') as f:
            if str(id) in f:
                return f[str(id)][:]
            else:
                return None

    def load_pretrained_embeddings(self, num, file):
        data = []
        for i in range(num):
            vec = self.get_vector_by_id(i, file)
            if vec is None:
                raise ValueError(f"Missing vector for id {i} in {file}")
            data.append(vec)
        tensor = torch.FloatTensor(data)
        return tensor

    def _init_weight_(self):
        init.xavier_uniform_(self.mashup_call_embedding.weight)
        init.xavier_uniform_(self.api_call_embedding.weight)
        init.xavier_uniform_(self.mashup_tag_embedding.weight)
        init.xavier_uniform_(self.api_tag_embedding.weight)
        init.xavier_uniform_(self.global_embedding.weight)  # 不再拼接，而是整体初始化

        # 允许训练更新
        self.mashup_call_embedding.weight.requires_grad = True
        self.api_call_embedding.weight.requires_grad = True

        self.mashup_tag_embedding.weight.requires_grad = True
        self.api_tag_embedding.weight.requires_grad = True

        self.global_embedding.weight.requires_grad = True

    def propagate_embeddings(self, adj1, adj2, init_emb, weights):
        """

        通用传播计算：使用两个稀疏矩阵adj2和adj1交替乘法多层传播，最后按权重加权求和。
        """
        embeddings = [init_emb]
        for _ in range(self.n_layers):
            t = torch.sparse.mm(adj2, embeddings[-1])
            t = torch.sparse.mm(adj1, t)
            embeddings.append(t)
        stacked = torch.stack(embeddings[:self.n_layers], dim=1)  # 取前n_layers层
        w = torch.softmax(weights, dim=0)
        out = torch.sum(stacked * w.view(1, self.n_layers, 1), dim=1)
        return out


    def forward(self, adj_m_c1, adj_m_c2, adj_a_c1, adj_a_c2, adj_m_t1, adj_m_t2, adj_a_t1, adj_a_t2, global_1, global_2):
        mashup_call_emb = self.propagate_embeddings(adj_m_c1, adj_m_c2, self.mashup_call_embedding.weight, self.u_weights)
        api_call_emb = self.propagate_embeddings(adj_a_c1, adj_a_c2, self.api_call_embedding.weight, self.i_weights)
        mashup_tag_emb = self.propagate_embeddings(adj_m_t1, adj_m_t2, self.mashup_tag_embedding.weight, self.m_t_weights)
        api_tag_emb = self.propagate_embeddings(adj_a_t1, adj_a_t2, self.api_tag_embedding.weight, self.a_t_weights)

        # 全局视角
        global_emb = self.propagate_embeddings(global_2, global_1, self.global_embedding.weight, self.global_weights)
        global_mashup = global_emb[:self.n_users]
        global_api = global_emb[self.n_users:self.n_users+self.n_items]
        global_tag = global_emb[self.n_users+self.n_items:]

        # Mashup视图融合（调用 + tag）
        mashup_view_w = torch.softmax(self.mashup_view_weights, dim=0)
        mashup_emb = mashup_view_w[0] * mashup_call_emb + mashup_view_w[1] * mashup_tag_emb
        # mashup_final = torch.cat([mashup_emb, global_mashup], dim=1)
        mashup_l_g_w = torch.softmax(self.mashup_l_g_weights, dim=0)
        mashup_final = mashup_l_g_w[0] * mashup_emb + mashup_l_g_w[1] * global_mashup

        # API视图融合（调用 + tag）
        api_view_w = torch.softmax(self.api_view_weights, dim=0)
        api_emb = api_view_w[0] * api_call_emb + api_view_w[1] * api_tag_emb
        # api_final = torch.cat([api_emb, global_api], dim=1)
        api_l_g_w = torch.softmax(self.api_l_g_weights, dim=0)
        api_final = api_l_g_w[0] * api_emb + api_l_g_w[1] * global_api

        # 对比学习使用的投影表示（Local-level）
        mashup_call_proj = self.mashup_local_proj(mashup_call_emb)
        mashup_tag_proj = self.mashup_local_proj(mashup_tag_emb)
        mashup_local_proj = self.mashup_local_proj(mashup_emb)
        mashup_global_proj = self.mashup_global_proj(global_mashup)

        api_call_proj = self.api_local_proj(api_call_emb)
        api_tag_proj = self.api_local_proj(api_tag_emb)
        api_local_proj = self.api_local_proj(api_emb)
        api_global_proj = self.api_global_proj(global_api)

        mashup_tag_logits = self.mashup_tag_predictor(mashup_final)
        api_tag_logits = self.api_tag_predictor(api_final)

        return (mashup_final, api_final,
                mashup_call_proj, mashup_tag_proj, api_call_proj, api_tag_proj,
                mashup_local_proj, api_local_proj, mashup_global_proj, api_global_proj,
                mashup_tag_logits, api_tag_logits)
