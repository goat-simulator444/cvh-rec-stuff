import torch.optim as optim
import sys
import math
from Models_without_local_CL import *
from utility.helper import ensureDir
from utility.batch_test import data_generator
import h5py
from scipy.spatial.distance import cosine
import numpy as np
import json
import torch
import warnings
import random
import torch.nn.functional as F
import os
import scipy.sparse as sp
from time import time
import torch.sparse as sparse
from utility.parser import parse_args
args = parse_args()

warnings.filterwarnings("ignore")



def lamb(epoch):
    epoch += 0
    return 0.95 ** (epoch / 14)

result = []
alpha1=args.alpha1
alpha2=args.alpha2
alpha3=args.alpha3
alpha4=args.alpha4

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)



def jaccard_similarity(matrix):
    intersection = np.dot(matrix, matrix.T)
    square_sum = np.diag(intersection)  # 获取对角线上的元素
    union = square_sum[:, None] + square_sum - intersection
    return np.divide(intersection, union)


class Model_Wrapper(object):
    def __init__(self, data_config, pretrain_data):
        # argument settings
        self.model_type = args.model_type
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type
        self.mess_dropout = eval(args.mess_dropout)
        self.pretrain_data = pretrain_data
        self.n_mashup = data_config['n_users']
        self.n_api = data_config['n_items']
        self.n_tag=data_config['n_tags']

        self.record_alphas = False
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.layer_num = args.layer_num
        self.model_type += '_%s_%s_layers%d' % (self.adj_type, self.alg_type, self.layer_num)
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self.verbose = args.verbose

        print('model_type is {}'.format(self.model_type))

        self.method = "Without_local_CL"

        alpha_str = f"{args.alpha1}-{args.alpha2}-{args.alpha3}-{args.alpha4}"
        reg_str = '-'.join(str(r) for r in eval(args.regs))

        self.weights_save_path = (
            f"{args.weights_path}weights/{self.method}/{args.dataset}/{args.seed}/"
            f"alpha-{alpha_str}/"
            # "github"
            f"layer_num{args.layer_num}/"
            f"mess_dropout{args.mess_dropout}/"
            f"drop_edge{args.drop_edge}/"
            f"lr{args.lr}/"
            f"reg{reg_str}"
        )

        str(args.similar_threthold) + str(args.alpha1) + str(args.alpha2) + str(args.alpha3) + str(args.alpha4)
        self.result_message = []

        print('----self.alg_type is {}----'.format(self.alg_type))

        if self.alg_type in ['hcf']:
            self.model = HCF(self.n_mashup, self.n_api,self.n_tag, self.emb_dim, self.layer_num, self.mess_dropout)
        else:
            raise Exception('Dont know which model to train')

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # norm_mashup_call_1, norm_mashup_call_2,
        # norm_api_call_1, norm_api_call_2,
        # norm_tag_mashup_1, norm_tag_mashup_2,
        # norm_tag_api_1, norm_tag_api_2,
        # norm_mashup_tag_1, norm_mashup_tag_2,
        # norm_api_tag_1, norm_api_tag_2
        (self.norm_mashup_call_1, self.norm_mashup_call_2, self.norm_api_call_1, self.norm_api_call_2,
         self.norm_mashup_tag_1, self.norm_mashup_tag_2, self.norm_api_tag_1, self.norm_api_tag_2) = self.build_hyper_edge(
            args.data_path + args.dataset + '/TE.csv',args.data_path + args.dataset +'/mashup_tags.csv',args.data_path + args.dataset +'/api_tags.csv')
        print(f"norm_mashup_call_1 shape: {self.norm_mashup_call_1.shape}, norm_mashup_call_2 shape: {self.norm_mashup_call_2.shape}")
        self.global_1, self.global_2 = self.build_global_hyperedge(
            args.data_path + args.dataset + '/TE.csv', args.data_path + args.dataset + '/mashup_tags.csv',
            args.data_path + args.dataset + '/api_tags.csv', self.n_mashup, self.n_api, self.n_tag)
        # print(f"global_1 shape: {self.global_1.shape}, global_2 shape: {self.global_2.shape}")
        # print(f'n_mashup {self.n_mashup}')
        # print(f'n_api {self.n_api}')
        # print(f'n_tag {self.n_tag}')

        self.mashup_tag_labels = self.load_tag_labels(
            args.data_path + args.dataset + '/mashup_tags.csv',
            self.n_mashup, self.n_tag
        )
        self.api_tag_labels = self.load_tag_labels(
            args.data_path + args.dataset + '/api_tags.csv',
            self.n_api, self.n_tag
        )
        print(f"mashup_tag_labels: {self.mashup_tag_labels.shape}, api_tag_labels: {self.api_tag_labels.shape}")

        self.model = self.model.cuda()
        self.norm_mashup_call_1 = self.norm_mashup_call_1.cuda()
        self.norm_mashup_call_2 = self.norm_mashup_call_2.cuda()
        self.norm_api_call_1 = self.norm_api_call_1.cuda()
        self.norm_api_call_2 = self.norm_api_call_2.cuda()
        self.norm_mashup_tag_1 = self.norm_mashup_tag_1.cuda()
        self.norm_mashup_tag_2 = self.norm_mashup_tag_2.cuda()
        self.norm_api_tag_1 = self.norm_api_tag_1.cuda()
        self.norm_api_tag_2 = self.norm_api_tag_2.cuda()
        self.global_1 = self.global_1.cuda()
        self.global_2 = self.global_2.cuda()

        self.lr_scheduler = self.set_lr_scheduler()

    def get_D_inv(self, Hadj):

        H = sp.coo_matrix(Hadj.shape)
        H.row = Hadj.row.copy()
        H.col = Hadj.col.copy()
        H.data = Hadj.data.copy()
        rowsum = np.array(H.sum(1))
        columnsum = np.array(H.sum(0))

        Dv_inv = np.power(rowsum, -1).flatten()
        De_inv = np.power(columnsum, -1).flatten()
        Dv_inv[np.isinf(Dv_inv)] = 0.
        De_inv[np.isinf(De_inv)] = 0.

        Dv_mat_inv = sp.diags(Dv_inv)
        De_mat_inv = sp.diags(De_inv)
        return Dv_mat_inv, De_mat_inv

    def build_hyper_edge(self, mashup_api_call_file, mashup_tag_call_file, api_tag_call_file):
        def read_and_fill_matrix(filename, shape, transpose=False):
            matrix = np.zeros(shape)
            with open(filename) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = [item for item in line.split(" ") if item != '']
                    main_id = int(parts[0])
                    related_ids = list(map(int, parts[1:]))
                    if transpose:
                        matrix[related_ids, main_id] = 1
                    else:
                        matrix[main_id, related_ids] = 1
            return matrix

        def build_and_normalize_hyperedge(interaction_matrix, similarity_threshold, node_count, edge_prefix):
            jaccard_sim = jaccard_similarity(interaction_matrix)
            indices = np.where(jaccard_sim >= similarity_threshold)
            values = jaccard_sim[indices]
            print(f"Number of hyperedges for {edge_prefix}: {len(values)}")

            hyperedge = sp.coo_matrix((values, indices), (node_count, node_count))
            hyperedge_transpose = hyperedge.T

            Dv, De = self.get_D_inv(hyperedge_transpose)

            Dv_torch = self.sparse_mx_to_torch_sparse_tensor(Dv)
            De_torch = self.sparse_mx_to_torch_sparse_tensor(De)
            hyperedge_torch = self.sparse_mx_to_torch_sparse_tensor(hyperedge_transpose)
            hyperedge_transpose_torch = self.sparse_mx_to_torch_sparse_tensor(hyperedge)

            temp = sparse.mm(Dv_torch, hyperedge_torch)
            normalized_1 = sparse.mm(temp, De_torch)
            normalized_2 = hyperedge_transpose_torch

            setattr(self, f'H{edge_prefix}Edge', hyperedge_torch)
            setattr(self, f'H{edge_prefix}Edge_T', hyperedge_transpose_torch)
            setattr(self, f'norm_{edge_prefix}1', normalized_1)
            setattr(self, f'norm_{edge_prefix}2', normalized_2)

            return normalized_1, normalized_2

        # mashup调用API交互矩阵
        mashup_api_call_matrix = read_and_fill_matrix(mashup_api_call_file, (self.n_mashup, self.n_api), transpose=False)
        # API调用mashup矩阵（转置）
        api_mashup_matrix = read_and_fill_matrix(mashup_api_call_file, (self.n_api, self.n_mashup), transpose=True)

        # mashup调用tag矩阵（mashup视角）
        mashup_tag_call_matrix = read_and_fill_matrix(mashup_tag_call_file, (self.n_mashup, self.n_tag), transpose=False)


        # API调用tag矩阵（API视角）
        api_tag_call_matrix = read_and_fill_matrix(api_tag_call_file, (self.n_api, self.n_tag), transpose=False)

        # 构造mashup超边
        norm_mashup_call_1, norm_mashup_call_2 = build_and_normalize_hyperedge(mashup_api_call_matrix, alpha1, self.n_mashup,
                                                                               'mashup')
        # 构造api超边
        norm_api_call_1, norm_api_call_2 = build_and_normalize_hyperedge(api_mashup_matrix, alpha2, self.n_api, 'api')

        # 构造mashup视角的超边，mashup间调用tag相似度超边
        norm_mashup_tag_1, norm_mashup_tag_2 = build_and_normalize_hyperedge(mashup_tag_call_matrix, alpha3,
                                                                                     self.n_mashup, 'mashupTagSim')
        # 构造api视角的超边，api间调用tag相似度超边
        norm_api_tag_1, norm_api_tag_2 = build_and_normalize_hyperedge(api_tag_call_matrix, alpha4, self.n_api,
                                                                               'apiTagSim')

        return (
            norm_mashup_call_1, norm_mashup_call_2,
            norm_api_call_1, norm_api_call_2,
            norm_mashup_tag_1, norm_mashup_tag_2,
            norm_api_tag_1, norm_api_tag_2
        )


    def build_global_hyperedge(self, mashup_api_call_file, mashup_tag_call_file, api_tag_call_file,
                               N_MASHUP, N_API, N_TAG):

        """
        构造global视角的超边：
        - 对每个mashup，构造一个超边：包含mashup自身、它调用的API（0.8）、它包含的tag（0.5）。
        - 对每个API，构造一个超边：包含API自身、它使用的tag（0.5）。
        """

        def read_and_fill_matrix(filename, shape):
            matrix = np.zeros(shape)
            with open(filename) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = [item for item in line.split(" ") if item != '']
                    main_id = int(parts[0])
                    related_ids = list(map(int, parts[1:]))
                    matrix[main_id, related_ids] = 1
            return matrix

        # 读取交互矩阵
        mashup_api_matrix = read_and_fill_matrix(mashup_api_call_file, (N_MASHUP, N_API))
        mashup_tag_matrix = read_and_fill_matrix(mashup_tag_call_file, (N_MASHUP, N_TAG))
        api_tag_matrix = read_and_fill_matrix(api_tag_call_file, (N_API, N_TAG))

        hyperedge_rows = []
        hyperedge_cols = []
        hyperedge_values = []

        hyperedge_idx = 0

        # ---- 对每个 mashup 构造超边 ----
        for mashup in range(N_MASHUP):
            apis = np.where(mashup_api_matrix[mashup] > 0)[0]
            tags = np.where(mashup_tag_matrix[mashup] > 0)[0]

            hyperedge_rows.append(mashup)
            hyperedge_cols.append(hyperedge_idx)
            hyperedge_values.append(1.0)

            for api in apis:
                global_api_id = N_MASHUP + api
                hyperedge_rows.append(global_api_id)
                hyperedge_cols.append(hyperedge_idx)
                hyperedge_values.append(1)

            for tag in tags:
                global_tag_id = N_MASHUP + N_API + tag
                hyperedge_rows.append(global_tag_id)
                hyperedge_cols.append(hyperedge_idx)
                hyperedge_values.append(1)

            hyperedge_idx += 1

        # ---- 对每个 API 构造超边 ----
        for api in range(N_API):
            tags = np.where(api_tag_matrix[api] > 0)[0]

            global_api_id = N_MASHUP + api
            hyperedge_rows.append(global_api_id)
            hyperedge_cols.append(hyperedge_idx)
            hyperedge_values.append(1.0)

            for tag in tags:
                global_tag_id = N_MASHUP + N_API + tag
                hyperedge_rows.append(global_tag_id)
                hyperedge_cols.append(hyperedge_idx)
                hyperedge_values.append(1.0)

            hyperedge_idx += 1

        N_total = N_MASHUP + N_API + N_TAG
        N_edges = hyperedge_idx

        hyperedge_matrix = sp.coo_matrix(
            (hyperedge_values, (hyperedge_rows, hyperedge_cols)),
            shape=(N_total, N_edges)
        )

        hyperedge_T = hyperedge_matrix.T
        Dv, De = self.get_D_inv(hyperedge_T)

        Dv_torch = self.sparse_mx_to_torch_sparse_tensor(Dv)
        De_torch = self.sparse_mx_to_torch_sparse_tensor(De)
        hyperedge_torch = self.sparse_mx_to_torch_sparse_tensor(hyperedge_T)
        hyperedge_T_torch = self.sparse_mx_to_torch_sparse_tensor(hyperedge_matrix)

        temp = sparse.mm(Dv_torch, hyperedge_torch)
        normalized_1 = sparse.mm(temp, De_torch)
        normalized_2 = hyperedge_T_torch

        setattr(self, 'HGlobalEdge', hyperedge_torch)
        setattr(self, 'HGlobalEdge_T', hyperedge_T_torch)
        setattr(self, 'norm_global1', normalized_1)
        setattr(self, 'norm_global2', normalized_2)

        print(f"Number of global hyperedges: {N_edges}")

        return normalized_1, normalized_2

    def load_tag_labels(self, file_path, num_entity, num_tags):
        """
        Args:
            file_path: mashup_tags.csv 或 api_tags.csv
            num_entity: N_MASHUP 或 N_API
            num_tags: N_TAG
        Returns:
            labels: FloatTensor shape [num_entity, num_tags], 0/1表示该entity是否包含该tag
        """
        labels = np.zeros((num_entity, num_tags), dtype=np.float32)
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [item for item in line.split(" ") if item != '']
                entity_id = int(parts[0])
                tag_ids = [int(t) for t in parts[1:]]
                for tag_id in tag_ids:
                    if 0 <= tag_id < num_tags:
                        labels[entity_id, tag_id] = 1.0
        return torch.FloatTensor(labels)

    def set_lr_scheduler(self):  # lr_scheduler：学习率调度器
        fac = lamb
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        # 每次的lr值：来自优化器的初始lr乘上一个lambda
        return scheduler

    def save_model(self):
        ensureDir(self.weights_save_path)
        torch.save(self.model.state_dict(), self.weights_save_path)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.weights_save_path))


    def train(self):
        n_batch = data_generator.n_train // args.batch_size + 1
        patience = args.patience  # 容忍多少个 epoch 没提升
        no_improve_count = 0
        best_loss = float('inf')  # 初始化最优 loss
        best_epoch = -1  # 记录最佳 epoch
        for epoch in range(args.epoch):
            t1 = time()

            loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.
            mashup_global_cl_loss, api_global_cl_loss = 0.0, 0.0
            tag_mashup_loss, tag_api_loss = 0.0, 0.0
            n_batch = data_generator.n_train // args.batch_size + 1
            f_time, b_time, loss_time, opt_time, clip_time, emb_time = 0., 0., 0., 0., 0., 0.
            sample_time = 0.

            for idx in range(n_batch):
                self.model.train()  # 模型为训练模式，像Dropout，Normalize这些层就会起作用，测试模式不会

                self.optimizer.zero_grad()
                sample_t1 = time()
                users, pos_items, neg_items = data_generator.sample()  # 采样正相关与负相关的物品id
                sample_time += time() - sample_t1

                (m_embeddings, a_embeddings,
                 m_local_proj, a_local_proj, m_global_proj, a_global_proj,
                 mashup_tag_logits, api_tag_logits)   = (
                    self.model(self.norm_mashup_call_1, self.norm_mashup_call_2,
                               self.norm_api_call_1, self.norm_api_call_2,
                               self.norm_mashup_tag_1, self.norm_mashup_tag_2,
                               self.norm_api_tag_1, self.norm_api_tag_2, self.global_1, self.global_2))

                bce_loss_fn = torch.nn.BCEWithLogitsLoss()

                batch_mashup_tag_loss = bce_loss_fn(mashup_tag_logits,
                                                    self.mashup_tag_labels.to(mashup_tag_logits.device))
                batch_api_tag_loss = bce_loss_fn(api_tag_logits, self.api_tag_labels.to(api_tag_logits.device))

                u_g_embeddings = m_embeddings[users]
                pos_i_g_embeddings = a_embeddings[pos_items]
                neg_i_g_embeddings = a_embeddings[neg_items]

                batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                                                              neg_i_g_embeddings)



                # contrastive loss for mashup: local vs global
                batch_mashup_global_cl_loss = self.contrastive_loss(m_local_proj, m_global_proj, temperature=0.5)

                # contrastive loss for api: local vs global
                batch_api_global_cl_loss = self.contrastive_loss(a_local_proj, a_global_proj, temperature=0.5)

                # 可选加权
                cl_loss_weight = args.beta  # 可以通过 args 传入更灵活

                batch_loss = (batch_mf_loss + batch_emb_loss + batch_reg_loss)
                batch_loss += cl_loss_weight * batch_mashup_global_cl_loss + cl_loss_weight * batch_api_global_cl_loss
                batch_loss += batch_mashup_tag_loss + batch_api_tag_loss


                batch_loss.backward()
                self.optimizer.step()

                loss += float(batch_loss)
                mf_loss += float(batch_mf_loss)
                emb_loss += float(batch_emb_loss)
                reg_loss += float(batch_reg_loss)



                mashup_global_cl_loss += float(batch_mashup_global_cl_loss)
                api_global_cl_loss += float(batch_api_global_cl_loss)

                tag_mashup_loss += float(batch_mashup_tag_loss)
                tag_api_loss += float(batch_api_tag_loss)

            avg_loss = loss / n_batch

            if avg_loss < best_loss:
                best_loss = avg_loss
                no_improve_count = 0
                best_epoch = epoch
                if args.save_flag == 1:
                    self.save_model()
            else:
                no_improve_count += 1

            if no_improve_count >= patience:
                break

            self.lr_scheduler.step()  # 学习率更新

            # print the test evaluation metrics each 10 epochs; pos:neg = 1:10.
            if (epoch + 1) % 10 != 0:
                if args.verbose > 0 and epoch % args.verbose == 0:
                    perf_str = (
                        f"Epoch {epoch} [{time() - t1:.1f}s]: "
                        f"loss={loss:.5f}, mf_loss={mf_loss:.5f}, "
                        f"mashup_global_cl_loss={mashup_global_cl_loss:.5f}, api_global_cl_loss={api_global_cl_loss:.5f}, "
                        f"tag_mashup_loss={tag_mashup_loss:.5f}, tag_api_loss={tag_api_loss:.5f}"
                    )
                    print(perf_str)

            if math.isnan(loss) == True:
                print('ERROR: loss is nan.')
                sys.exit()




        # *********************************************************
        # save the user & item embeddings for pretraining.
        if args.save_flag == 1:
            self.save_model()
            print('save the weights in path: ', self.weights_save_path)


    def norm(self, adj):
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()


    def cosine_similarity_scipy(self, vec1, vec2):
        """
        计算两个向量的余弦相似度

        参数:
        vec1 -- 第一个向量，NumPy 数组
        vec2 -- 第二个向量，NumPy 数组

        返回值:
        余弦相似度，浮点数
        """
        return 1 - cosine(vec1, vec2)

    def get_vector_by_id(self, id):
        with h5py.File('data/vectors.h5', 'r') as f:
            if str(id) in f:
                return f[str(id)][:]
            else:
                return None

    def weighted_average(self, u_embeddings, similar):
        # 将 similar 转换为 PyTorch 张量
        similar = torch.tensor(similar, dtype=torch.float32).cuda()

        # 计算权重的总和
        total_weight = torch.sum(similar)

        # 将每个向量乘以其对应的权重
        weighted_embeddings = u_embeddings * similar.unsqueeze(1)

        # 对所有加权后的向量求和
        weighted_sum = torch.sum(weighted_embeddings, dim=0)

        # 将加权和除以权重的总和
        u_avg = weighted_sum / total_weight

        return u_avg


    def save_recResult(self):
        self.model.eval()
        with torch.no_grad():
            (m_embeddings, a_embeddings,
             m_local_proj, a_local_proj, m_global_proj, a_global_proj, _, _) = (
                self.model(self.norm_mashup_call_1, self.norm_mashup_call_2,
                           self.norm_api_call_1, self.norm_api_call_2,
                           self.norm_mashup_tag_1, self.norm_mashup_tag_2,
                           self.norm_api_tag_1, self.norm_api_tag_2, self.global_1, self.global_2))

        # Caching user vectors
        user_vectors = {user: self.get_vector_by_id(user) for user in data_generator.train_items.keys()}
        users_to_test = list(data_generator.test_set.keys())

        # Open file for writing recommendations
        os.makedirs('output', exist_ok=True)
        output_file = f'output/{self.method}_{args.alpha1}_{args.alpha2}_{args.alpha3}_{args.alpha4}_{args.dataset}.json'
        with open(output_file, mode='w') as write_recommend_fp:
            for user in users_to_test:
                import time
                # s_time = time.time()
                user_vector = self.get_vector_by_id(user)

                # Calculate cosine similarities between user and all users in train set
                similar_values = [
                    (self.cosine_similarity_scipy(user_vector, user_vectors[other_user]), other_user)
                    for other_user in data_generator.train_items.keys() if other_user != user
                ]

                # Sort by similarity and get top 10 similar users
                top_similar_users = sorted(similar_values, key=lambda x: x[0], reverse=True)[:int(args.similar_threthold)]
                similar_users = [i[1] for i in top_similar_users]
                similar = [i[0] for i in top_similar_users]

                # Get embeddings for the top similar users
                u_similar_embeddings = m_embeddings[similar_users]

                # Compute weighted average of similar users' embeddings
                average_embedding = self.weighted_average(u_similar_embeddings, similar)

                # Calculate item ratings
                i_embeddings = a_embeddings  # No need to recompute item embeddings each time
                rate_batch = torch.matmul(average_embedding, i_embeddings.T)

                rate_batch = rate_batch.detach().cpu().numpy()

                # Get top 20 items based on ratings
                top_20_indices = np.argsort(rate_batch)[-20:][::-1]

                # Prepare data for writing
                write_data = {
                    'mashup_id': user,
                    'recommend_api': top_20_indices.tolist(),
                }
                # e_time = time.time()
                # print('recommend time:', e_time - s_time)

                # Write recommendation results to file
                write_recommend_fp.write(json.dumps(write_data) + '\n')


    def print_final_results(self, rec_loger, pre_loger, ndcg_loger, hit_loger, map_loger, mrr_loger, fone_loger,
                            training_time_list):
        recs = np.array(rec_loger)
        pres = np.array(pre_loger)
        map = np.array(map_loger)
        mrr = np.array(mrr_loger)
        fone = np.array(fone_loger)

        best_rec_0 = max(recs[:, 0])
        idx = list(recs[:, 0]).index(best_rec_0)

        final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s], map=[%s],mrr=[%s], f1=[%s]" % \
                     (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                      '\t'.join(['%.5f' % r for r in pres[idx]]),
                      '\t'.join(['%.5f' % r for r in hit_loger[idx]]),
                      '\t'.join(['%.5f' % r for r in ndcg_loger[idx]]),
                      '\t'.join(['%.5f' % r for r in map[idx]]),
                      '\t'.join(['%.5f' % r for r in mrr[idx]]),
                      '\t'.join(['%.5f' % r for r in fone[idx]]))
        result.append(final_perf + "\n")
        print(final_perf)

    # pos_items：正相关物品的id
    # neg_items：负相关物品的id
    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)  # torch.mul():对应元素相乘
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)  # torch.mul():对应元素相乘

        regularizer = 1. / 2 * (users ** 2).sum() + 1. / 2 * (pos_items ** 2).sum() + 1. / 2 * (neg_items ** 2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.decay * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss



    def contrastive_loss(self, z1, z2, temperature=0.5):
        # 保留原始向量，不归一化
        N = z1.size(0)

        # 正样本距离
        pos_dist = torch.norm(z1 - z2, p=2, dim=1)  # [N]
        sim_pos = -pos_dist / temperature

        # 计算 pairwise 距离的平方（高效）
        z1_sq = (z1 ** 2).sum(dim=1).unsqueeze(1)  # [N, 1]
        z2_sq = (z2 ** 2).sum(dim=1).unsqueeze(0)  # [1, N]
        dot_product = torch.matmul(z1, z2.T)  # [N, N]

        # 欧几里得距离平方
        dist_squared = z1_sq + z2_sq - 2 * dot_product
        dist_squared = torch.clamp(dist_squared, min=1e-8)  # 避免负数误差

        # 取平方根，得到距离矩阵
        dist_matrix = torch.sqrt(dist_squared)  # [N, N]

        # 转为负相似度
        logits_zw = -dist_matrix / temperature

        # 屏蔽正样本
        mask = torch.eye(N, dtype=torch.bool).to(z1.device)
        logits_zw.masked_fill_(mask, -1e9)

        # 分母：所有负样本
        denom = torch.exp(logits_zw).sum(dim=1)

        # 最终损失
        loss = - torch.log(torch.exp(sim_pos) / (torch.exp(sim_pos) + denom + 1e-8))
        return loss.mean()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def get_sparse_tensor_value(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return indices, coo.data, coo.shape


def main():
    # 调用 parse_args 获取命令行参数
    # args = parse_args()
    # args.set_defaults(dataset='fold_2')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    config['n_tags'] = data_generator.n_tags

    t0 = time()

    pretrain_data = None

    Engine = Model_Wrapper(data_config=config, pretrain_data=pretrain_data)
    if args.pretrain:
        print('pretrain path: ', Engine.weights_save_path)
        if os.path.exists(Engine.weights_save_path):
            Engine.load_model()
            Engine.save_recResult()
        else:
            print('Cannot load pretrained model. Start training from stratch')
    else:
        print('without pretraining')
        Engine.train()


if __name__ == '__main__':
    main()

