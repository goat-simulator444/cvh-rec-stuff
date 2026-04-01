import random
from collections import defaultdict
import numpy as np


def preprocess_transition_probs(G, phi=0.5, omega=0.5, p=1, q=1):
    transition_probs = dict()
    for node in G.nodes:
        neighbors = list(G.neighbors(node))
        probs = []
        for nbr in neighbors:
            edge_type = G[node][nbr].get('type')
            weight = G[node][nbr].get('weight', 1.0)

            # 根据边类型调整过渡概率
            if edge_type == 'similar':
                alpha = 1.0 / phi  # 相似性基于的游走
            elif edge_type == 'call':
                alpha = 1.0 / omega  # 调用基于的游走
            else:  # 标签链接
                alpha = 1.0  # 默认的标签链接偏置

            # 引入 node2vec 中的 p 和 q 参数，控制游走的偏向性
            if p < 1:  # 越小越倾向于深度优先
                alpha *= p
            if q < 1:  # 越小越倾向于广度优先
                alpha *= q

            probs.append(alpha * weight)

        # 归一化概率
        norm_probs = [p / sum(probs) for p in probs] if sum(probs) > 0 else [0 for _ in probs]
        transition_probs[node] = (neighbors, norm_probs)
    return transition_probs


def alias_sample(neighbors, probs):
    return random.choices(neighbors, weights=probs, k=1)[0]


def generate_random_walks(G, walk_length=20, num_walks=80, phi=0.4, omega=0.6, p=1, q=1):
    transition_probs = preprocess_transition_probs(G, phi, omega, p, q)
    walks = []

    # nodes = list(G.nodes)
    # 只从 mashup 节点开始游走
    mashup_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'api']

    for _ in range(num_walks):
        random.shuffle(mashup_nodes)
        for node in mashup_nodes:
            walk = [node]
            while len(walk) < walk_length:
                curr = walk[-1]
                if curr not in transition_probs:
                    break
                neighbors, probs = transition_probs[curr]
                if not neighbors:
                    break
                next_node = alias_sample(neighbors, probs)
                walk.append(next_node)
            walks.append(walk)
    return walks
