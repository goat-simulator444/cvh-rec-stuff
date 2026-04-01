import h5py
import torch
import torch.nn as nn



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
        self.global_embedding = nn.Embedding(n_users + n_items + n_tags, embedding_dim)

        self.dropout_list = nn.ModuleList([nn.Dropout(p) for p in dropout_list])

        self.global_weights = nn.Parameter(torch.ones(self.n_layers))

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
        mashup_embedding = self.load_pretrained_embeddings(self.n_users, 'data/vectors.h5')
        api_embedding = self.load_pretrained_embeddings(self.n_items, 'data/API_vectors.h5')
        tag_embedding = self.load_pretrained_embeddings(self.n_tags, 'data/tag_vectors.h5')
        self.global_embedding.weight.data = torch.cat([mashup_embedding, api_embedding, tag_embedding], dim=0)

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


    def forward(self, global_1, global_2):
        # 全局视角
        global_emb = self.propagate_embeddings(global_2, global_1, self.global_embedding.weight, self.global_weights)
        global_mashup = global_emb[:self.n_users]
        global_api = global_emb[self.n_users:self.n_users+self.n_items]
        global_tag = global_emb[self.n_users+self.n_items:]
        mashup_tag_logits = self.mashup_tag_predictor(global_mashup)
        api_tag_logits = self.api_tag_predictor(global_api)
        return (global_mashup, global_api, mashup_tag_logits, api_tag_logits)
