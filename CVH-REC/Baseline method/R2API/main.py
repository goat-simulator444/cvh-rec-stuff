import torch.optim as optim
import sys
import math
from Models import *
from utility.helper import *
from utility.batch_test import *
import h5py
from scipy.spatial.distance import cosine
import numpy as np
import json
import torch
import warnings
warnings.filterwarnings("ignore")



def lamb(epoch):
    epoch += 0
    return 0.95 ** (epoch / 14)

result = []
alpha1=args.alpha1
alpha2=args.alpha2
alpha3=args.alpha3
alpha4=args.alpha4


def jaccard_similarity(matrix):
    intersection = np.dot(matrix, matrix.T)
    square_sum = np.diag(intersection)  # 获取对角线上的元素
    union = square_sum[:, None] + square_sum - intersection
    return np.divide(intersection, union)


class Model_Wrapper(object):
    def __init__(self, data_config,config_mashup,config_api, pretrain_data):
        # argument settings
        self.model_type = args.model_type
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type
        self.mess_dropout = eval(args.mess_dropout)
        self.pretrain_data = pretrain_data
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_mashup=config_mashup['n_mashup']
        self.n_mashup_tag=config_mashup['n_mashup_tag']
        self.n_api=config_api['n_api']
        self.n_api_tag=config_api['n_api_tag']
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

        self.weights_save_path = '%sweights/%s/%s/emb_size%s/layer_num%s/mess_dropout%s/drop_edge%s/lr%s/reg%s' % (
            args.weights_path, args.dataset, self.model_type,
            str(args.embed_size), str(args.layer_num), str(args.mess_dropout), str(args.drop_edge), str(args.lr),
            '-'.join([str(r) for r in eval(args.regs)]))

        str(args.similar_threthold) + str(args.alpha1) + str(args.alpha2) + str(args.alpha3) + str(args.alpha4)
        self.result_message = []

        print('----self.alg_type is {}----'.format(self.alg_type))

        if self.alg_type in ['hcf']:
            self.model = HCF(self.n_users, self.n_items,self.n_mashup,self.n_mashup_tag,self.n_api,self.n_api_tag,self.emb_dim, self.layer_num, self.mess_dropout)
        else:
            raise Exception('Dont know which model to train')

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.norm_u1, self.norm_u2, self.norm_i1, self.norm_i2,self.norm_m1,self.norm_m2,self.norm_a1,self.norm_a2 = self.build_hyper_edge(
            args.data_path + args.dataset + '/TE.csv',args.data_path + args.dataset +'/mashup_tags.csv',args.data_path + args.dataset +'/api_tags.csv')

        self.model = self.model.cuda()
        self.norm_u1 = self.norm_u1.cuda()
        self.norm_u2 = self.norm_u2.cuda()
        self.norm_i1 = self.norm_i1.cuda()
        self.norm_i2 = self.norm_i2.cuda()
        self.norm_m1 = self.norm_m1.cuda()
        self.norm_m2 = self.norm_m2.cuda()
        self.norm_a1 = self.norm_a1.cuda()
        self.norm_a2 = self.norm_a2.cuda()

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

    def build_hyper_edge(self, file,file_mashup,file_api):
        user_inter = np.zeros((USR_NUM, ITEM_NUM))
        items_inter = np.zeros((ITEM_NUM, USR_NUM))
        mashup_inter = np.zeros((N_MASHUP_TAG,N_MASHUP))
        api_inter= np.zeros((N_API_TAG,N_API))
        with open(file) as f:
            for l in f.readlines():
                if len(l) == 0: break
                l = l.strip("\n").split(" ")
                l = [item for item in l if item != '']
                uid = int(l[0])
                items = [int(j) for j in l[1:]]
                user_inter[uid, items] = 1
                items_inter[items, uid] = 1
        with open(file_mashup) as f:
            for l in f.readlines():
                if len(l) == 0: break
                l = l.strip("\n").split(" ")
                uid = int(l[0])
                items = [int(j) for j in l[1:]]
                mashup_inter[items, uid] = 1
        with open(file_api) as f:
            for l in f.readlines():
                if len(l) == 0: break
                l = l.strip("\n").split(" ")
                uid = int(l[0])
                items = [int(j) for j in l[1:]]
                api_inter[items, uid] = 1

        # 用户相似度矩阵
        J_u = jaccard_similarity(user_inter)
        # 每条超边的下标
        indices = np.where(J_u >= alpha1)
        print(f"Number of hyperedges for users: {len(indices[0])}")
        # 每条超边节点的权重
        values = J_u[indices]
        # 生成超边矩阵
        HEdge = sp.coo_matrix((values, indices), (USR_NUM, USR_NUM))
        self.HuEdge = (HEdge).T
        Dv_1, De_1 = self.get_D_inv(self.HuEdge)

        Dv_1 = self.sparse_mx_to_torch_sparse_tensor(Dv_1)
        De_1 = self.sparse_mx_to_torch_sparse_tensor(De_1)
        self.HuEdge = self.sparse_mx_to_torch_sparse_tensor(self.HuEdge)
        self.HuEdge_T = self.sparse_mx_to_torch_sparse_tensor(HEdge)

        spm1 = sparse.mm(Dv_1, self.HuEdge)
        self.norm_u1 = sparse.mm(spm1, De_1)
        self.norm_u2 = self.HuEdge_T

        J_i = jaccard_similarity(items_inter)
        # 每条超边的下标
        indices = np.where(J_i >=alpha2)

        # 每条超边节点的权重
        values = J_i[indices]
        print(f"Number of hyperedges for items: {len(values)}")
        # 生成超边矩阵
        HEdge = sp.coo_matrix((values, indices), (ITEM_NUM, ITEM_NUM))
        self.HiEdge = (HEdge).T
        Dv_1, De_1 = self.get_D_inv(self.HiEdge)

        Dv_1 = self.sparse_mx_to_torch_sparse_tensor(Dv_1)
        De_1 = self.sparse_mx_to_torch_sparse_tensor(De_1)
        self.HiEdge = self.sparse_mx_to_torch_sparse_tensor(self.HiEdge)
        self.HiEdge_T = self.sparse_mx_to_torch_sparse_tensor(HEdge)

        spm1 = sparse.mm(Dv_1, self.HiEdge)
        self.norm_i1 = sparse.mm(spm1, De_1)
        self.norm_i2 = self.HiEdge_T

        J_m = jaccard_similarity(mashup_inter)
        # 每条超边的下标
        indices = np.where(J_m >=alpha3)
        # 每条超边节点的权重
        values = J_m[indices]
        print(f"Number of hyperedges for mashup_tag: {len(values)}")
        # 生成超边矩阵
        HEdge = sp.coo_matrix((values, indices), (N_MASHUP_TAG, N_MASHUP_TAG))
        self.HiEdge = (HEdge).T
        Dv_1, De_1 = self.get_D_inv(self.HiEdge)

        Dv_1 = self.sparse_mx_to_torch_sparse_tensor(Dv_1)
        De_1 = self.sparse_mx_to_torch_sparse_tensor(De_1)
        self.HiEdge = self.sparse_mx_to_torch_sparse_tensor(self.HiEdge)
        self.HiEdge_T = self.sparse_mx_to_torch_sparse_tensor(HEdge)

        spm1 = sparse.mm(Dv_1, self.HiEdge)
        self.norm_m1 = sparse.mm(spm1, De_1)
        self.norm_m2 = self.HiEdge_T

        J_a = jaccard_similarity(api_inter)
        # 每条超边的下标
        indices = np.where(J_a >= alpha4)
        # 每条超边节点的权重
        values = J_a[indices]
        print(f"Number of hyperedges for api tag: {len(values)}")
        # 生成超边矩阵
        HEdge = sp.coo_matrix((values, indices), (N_API_TAG, N_API_TAG))
        self.HiEdge = (HEdge).T
        Dv_1, De_1 = self.get_D_inv(self.HiEdge)

        Dv_1 = self.sparse_mx_to_torch_sparse_tensor(Dv_1)
        De_1 = self.sparse_mx_to_torch_sparse_tensor(De_1)
        self.HiEdge = self.sparse_mx_to_torch_sparse_tensor(self.HiEdge)
        self.HiEdge_T = self.sparse_mx_to_torch_sparse_tensor(HEdge)

        spm1 = sparse.mm(Dv_1, self.HiEdge)
        self.norm_a1 = sparse.mm(spm1, De_1)
        self.norm_a2 = self.HiEdge_T

        return self.norm_u1, self.norm_u2, self.norm_i1, self.norm_i2,self.norm_m1,self.norm_m2,self.norm_a1,self.norm_a2

    def build_graph_edges(self, file, file_mashup, file_api):
        user_inter = np.zeros((USR_NUM, ITEM_NUM))
        items_inter = np.zeros((ITEM_NUM, USR_NUM))
        mashup_inter = np.zeros((N_MASHUP_TAG, N_MASHUP))
        api_inter = np.zeros((N_API_TAG, N_API))

        with open(file) as f:
            for l in f.readlines():
                if len(l) == 0:
                    break
                l = l.strip("\n").split(" ")
                l = [item for item in l if item != '']
                uid = int(l[0])
                items = [int(j) for j in l[1:]]
                user_inter[uid, items] = 1
                items_inter[items, uid] = 1

        with open(file_mashup) as f:
            for l in f.readlines():
                if len(l) == 0:
                    break
                l = l.strip("\n").split(" ")
                uid = int(l[0])
                items = [int(j) for j in l[1:]]
                mashup_inter[items, uid] = 1

        with open(file_api) as f:
            for l in f.readlines():
                if len(l) == 0:
                    break
                l = l.strip("\n").split(" ")
                uid = int(l[0])
                items = [int(j) for j in l[1:]]
                api_inter[items, uid] = 1

        # 用户邻接图
        J_u = jaccard_similarity(user_inter)
        indices = np.where(J_u >= alpha1)
        values = J_u[indices]
        A_u = sp.coo_matrix((values, indices), (USR_NUM, USR_NUM))
        A_u = A_u + A_u.T  # 对称化
        print(f"Number of user edges: {A_u.nnz}")
        self.Adj_u = self.sparse_mx_to_torch_sparse_tensor(A_u)

        # API 邻接图
        J_i = jaccard_similarity(items_inter)
        indices = np.where(J_i >= alpha2)
        values = J_i[indices]
        A_i = sp.coo_matrix((values, indices), (ITEM_NUM, ITEM_NUM))
        A_i = A_i + A_i.T
        print(f"Number of item edges: {A_i.nnz}")
        self.Adj_i = self.sparse_mx_to_torch_sparse_tensor(A_i)

        # Mashup 标签邻接图
        J_m = jaccard_similarity(mashup_inter)
        indices = np.where(J_m >= alpha3)
        values = J_m[indices]
        A_m = sp.coo_matrix((values, indices), (N_MASHUP_TAG, N_MASHUP_TAG))
        A_m = A_m + A_m.T
        print(f"Number of mashup tag edges: {A_m.nnz}")
        self.Adj_m = self.sparse_mx_to_torch_sparse_tensor(A_m)

        # API 标签邻接图
        J_a = jaccard_similarity(api_inter)
        indices = np.where(J_a >= alpha4)
        values = J_a[indices]
        A_a = sp.coo_matrix((values, indices), (N_API_TAG, N_API_TAG))
        A_a = A_a + A_a.T
        print(f"Number of API tag edges: {A_a.nnz}")
        self.Adj_a = self.sparse_mx_to_torch_sparse_tensor(A_a)
        return self.Adj_u, self.Adj_i, self.Adj_m, self.Adj_a


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

    def test(self, users_to_test, drop_flag=False, batch_test_flag=False):
        self.model.eval()  # 评估模式，batchnorm和Drop层不起作用，相当于self.model.train(False)
        with torch.no_grad():
            ua_embeddings, ia_embeddings,ma_embeddings,aa_embeddings = self.model(self.norm_u1, self.norm_u2, self.norm_i1, self.norm_i2,self.norm_m1,self.norm_m2,self.norm_a1,self.norm_a2)
        result = test_torch(ua_embeddings, ia_embeddings, users_to_test)
        return result



    def train(self):
        training_time_list = []
        loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger, map_loger, mrr_loger, fone_loger = [], [], [], [], [], [], [], []
        stopping_step = 10
        should_stop = False
        cur_best_pre_0 = 0.
        n_batch = data_generator.n_train // args.batch_size + 1
        for epoch in range(args.epoch):
            t1 = time()

            loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.
            mashup_tag_mf_loss, mashup_tag_emb_loss, mashup_tag_reg_loss = 0.0, 0.0, 0.0
            api_tag_mf_loss, api_tag_emb_loss, api_tag_reg_loss = 0.0, 0.0, 0.0

            n_batch = data_generator.n_train // args.batch_size + 1
            f_time, b_time, loss_time, opt_time, clip_time, emb_time = 0., 0., 0., 0., 0., 0.
            sample_time = 0.

            for idx in range(n_batch):
                self.model.train()  # 模型为训练模式，像Dropout，Normalize这些层就会起作用，测试模式不会

                self.optimizer.zero_grad()
                sample_t1 = time()
                users, pos_items, neg_items,pos_mashup_tags,neg_mashup_tags,pos_api_tags,neg_api_tags = data_generator.sample()  # 采样正相关与负相关的物品id
                sample_time += time() - sample_t1

                ua_embeddings, ia_embeddings,ma_embeddings,aa_embeddings = self.model(self.norm_u1, self.norm_u2, self.norm_i1,
                                                          self.norm_i2,self.norm_m1,self.norm_m2,self.norm_a1,self.norm_a2)

                u_g_embeddings = ua_embeddings[users]
                pos_i_g_embeddings = ia_embeddings[pos_items]
                neg_i_g_embeddings = ia_embeddings[neg_items]

                pos_m_g_embeddings = ma_embeddings[pos_mashup_tags]
                neg_m_g_embeddings = ma_embeddings[neg_mashup_tags]
                pos_a_g_embeddings = aa_embeddings[pos_api_tags]
                neg_a_g_embeddings = aa_embeddings[neg_api_tags]

                batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                                                              neg_i_g_embeddings)
                batch_mashup_tag_mf_loss,batch_mashup_tag_emb_loss,batch_mashup_tag_reg_loss=self.tag_loss(u_g_embeddings,pos_m_g_embeddings,neg_m_g_embeddings)
                batch_api_tag_mf_loss, batch_api_tag_emb_loss, batch_api_tag_reg_loss=self.tag_loss(pos_i_g_embeddings,pos_a_g_embeddings,neg_a_g_embeddings)

                batch_loss = (batch_mf_loss + batch_emb_loss + batch_reg_loss
                              + batch_mashup_tag_mf_loss + batch_mashup_tag_emb_loss + batch_mashup_tag_reg_loss
                              + batch_api_tag_mf_loss + batch_api_tag_emb_loss + batch_api_tag_reg_loss )


                batch_loss.backward()
                self.optimizer.step()

                loss += float(batch_loss)
                mf_loss += float(batch_mf_loss)
                emb_loss += float(batch_emb_loss)
                reg_loss += float(batch_reg_loss)
                mashup_tag_mf_loss += float(batch_mashup_tag_mf_loss)
                mashup_tag_emb_loss += float(batch_mashup_tag_emb_loss)
                mashup_tag_reg_loss += float(batch_mashup_tag_reg_loss)
                api_tag_mf_loss += float(batch_api_tag_mf_loss)
                api_tag_emb_loss += float(batch_api_tag_emb_loss)
                api_tag_reg_loss += float(batch_api_tag_reg_loss)

            self.lr_scheduler.step()  # 学习率更新

            del ua_embeddings, ia_embeddings, u_g_embeddings, neg_i_g_embeddings, pos_i_g_embeddings
            del ma_embeddings, pos_m_g_embeddings, neg_m_g_embeddings
            del aa_embeddings, pos_a_g_embeddings, neg_a_g_embeddings

            if math.isnan(loss) == True:
                print('ERROR: loss is nan.')
                sys.exit()

            # print the test evaluation metrics each 10 epochs; pos:neg = 1:10.
            if (epoch + 1) % 10 != 0:
                if args.verbose > 0 and epoch % args.verbose == 0:
                    #perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                     #   epoch, time() - t1, loss, mf_loss+mashup_tag_mf_loss+api_tag_mf_loss, emb_loss+mashup_tag_emb_loss+api_tag_emb_loss)
                    perf_str = (
                        f"Epoch {epoch} [{time() - t1:.1f}s]: "
                        f"loss={loss:.5f}, mf_loss={mf_loss:.5f}, emb_loss={emb_loss:.5f}, "
                        f"mashup_tag_mf_loss={mashup_tag_mf_loss:.5f}, api_tag_mf_loss={api_tag_mf_loss:.5f},"
                        f"mashup_tag_emb_loss={mashup_tag_emb_loss:.5f}, api_tag_emb_loss={api_tag_emb_loss:.5f}"
                    )
                    print(perf_str)


        # *********************************************************
        # save the user & item embeddings for pretraining.
        if args.save_flag == 1:
            self.save_model()
            print('save the weights in path: ', self.weights_save_path)

        # print the final recommendation results to csv files.
        if args.save_recom:
            self.save_recResult()

        if rec_loger != []:
            self.print_final_results(rec_loger, pre_loger, ndcg_loger, hit_loger, map_loger, mrr_loger, fone_loger,
                                     training_time_list)

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
            ua_embeddings, ia_embeddings, ma_embeddings, aa_embeddings = self.model(
                self.norm_u1, self.norm_u2, self.norm_i1, self.norm_i2,
                self.norm_m1, self.norm_m2, self.norm_a1, self.norm_a2
            )

        # Caching user vectors
        user_vectors = {user: self.get_vector_by_id(user) for user in data_generator.train_items.keys()}
        users_to_test = list(data_generator.test_set.keys())

        # Open file for writing recommendations
        os.makedirs('output', exist_ok=True)
        output_file = f'output/R2API_{args.dataset}.json'
        with open(output_file, mode='w') as write_recommend_fp:
            for user in users_to_test:
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
                u_similar_embeddings = ua_embeddings[similar_users]

                # Compute weighted average of similar users' embeddings
                average_embedding = self.weighted_average(u_similar_embeddings, similar)

                # Calculate item ratings
                i_embeddings = ia_embeddings  # No need to recompute item embeddings each time
                rate_batch = torch.matmul(average_embedding, i_embeddings.T)

                rate_batch = rate_batch.detach().cpu().numpy()

                # Get top 20 items based on ratings
                top_20_indices = np.argsort(rate_batch)[-20:][::-1]

                # Prepare data for writing
                write_data = {
                    'mashup_id': user,
                    'recommend_api': top_20_indices.tolist(),
                }

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

    def tag_loss(self, embeddings, pos_tags, neg_tags):
        """
        计算标签相关的损失
        """
        # 正负标签得分
        pos_scores = torch.sum(torch.mul(embeddings, pos_tags), dim=1)
        neg_scores = torch.sum(torch.mul(embeddings, neg_tags), dim=1)


        # 正则化损失
        regularizer = (
                1. / 2 * (embeddings ** 2).sum()
                + 1. / 2 * (pos_tags ** 2).sum()
                + 1. / 2 * (neg_tags ** 2).sum()
        )
        regularizer = regularizer / self.batch_size

        # 最大化正负标签得分间隔
        maxi = F.logsigmoid(pos_scores - neg_scores)
        tag_mf_loss = -torch.mean(maxi)

        tag_emb_loss = self.decay * regularizer
        reg_loss = 0.0

        return tag_mf_loss, tag_emb_loss,reg_loss

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

    config_mashup = dict()
    config_mashup['n_mashup'] = data_generator.n_mashup
    config_mashup['n_mashup_tag'] = data_generator.n_mashup_tag

    config_api = dict()
    config_api['n_api'] = data_generator.n_api
    config_api['n_api_tag'] = data_generator.n_api_tag

    t0 = time()

    pretrain_data = None

    Engine = Model_Wrapper(data_config=config, config_mashup=config_mashup, config_api=config_api,
                           pretrain_data=pretrain_data)
    if args.pretrain:
        print('pretrain path: ', Engine.weights_save_path)
        if os.path.exists(Engine.weights_save_path):
            Engine.load_model()
            # Engine.train()
            Engine.save_recResult()
        else:
            print('Cannot load pretrained model. Start training from stratch')
    else:
        print('without pretraining')
        Engine.train()


if __name__ == '__main__':
    main()