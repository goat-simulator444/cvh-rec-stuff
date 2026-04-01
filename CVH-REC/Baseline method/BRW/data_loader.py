import json
from collections import defaultdict

def load_data(mashup_path, api_path):
    with open(mashup_path, 'r', encoding='utf-8') as f:
        mashups = json.load(f)
    with open(api_path, 'r', encoding='utf-8') as f:
        apis = json.load(f)
    return mashups, apis


def build_graph(mashups, apis, test_ids=set()):
    import networkx as nx
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    G = nx.Graph()
    mashup_nodes, api_nodes, tag_nodes = set(), set(), set()

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    api_dict = {api["id"]: api for api in apis}

    mashup_texts, mashup_ids = [], []
    api_texts, api_ids = [], []

    # Add nodes and tag edges
    for m in mashups:
        mid = f"m_{m['id']}"
        G.add_node(mid, type='mashup', text=m['description'])
        mashup_nodes.add(mid)
        mashup_texts.append(m['description'])
        mashup_ids.append(mid)

        for tag in m['categories'].split(','):
            tag_clean = tag.strip().lower()
            tag_id = f"tag_{tag_clean}"
            if tag_id not in G:
                G.add_node(tag_id, type='tag', text=tag_clean)
            G.add_edge(mid, tag_id, type='tag')
            tag_nodes.add(tag_id)

        for aid in m['api_info']:
            if aid in api_dict:
                aid_str = f"a_{aid}"
                G.add_edge(mid, aid_str, type='call')

    for a in apis:
        aid = f"a_{a['id']}"
        desc = a['details']['description']
        G.add_node(aid, type='api', text=desc)
        api_nodes.add(aid)
        api_texts.append(desc)
        api_ids.append(aid)

        for tag in a['details'].get('tags', '').split(','):
            tag_clean = tag.strip().lower()
            tag_id = f"tag_{tag_clean}"
            if tag_id not in G:
                G.add_node(tag_id, type='tag', text=tag_clean)
            G.add_edge(aid, tag_id, type='tag')
            tag_nodes.add(tag_id)

    # Semantic similarity with MiniLM
    def add_similarity_edges(nodes, texts, threshold=0.6):
        num = 0
        embeddings = model.encode(texts, convert_to_tensor=True)
        sim_matrix = cosine_similarity(embeddings.cpu().numpy())
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if sim_matrix[i][j] > threshold:
                    G.add_edge(nodes[i], nodes[j], weight=float(sim_matrix[i][j]), type='similar')
                    num += 1
        print(f"Added {num} similarity edges")

    def add_topk_similarity_edges(nodes, texts, top_k=5):
        num = 0
        embeddings = model.encode(texts, convert_to_tensor=True)
        sim_matrix = cosine_similarity(embeddings.cpu().numpy())
        for i in range(len(nodes)):
            sims = [(j, sim_matrix[i][j]) for j in range(len(nodes)) if i != j]
            sims.sort(key=lambda x: -x[1])  # 按相似度降序排序
            for j, score in sims[:top_k]:
                G.add_edge(nodes[i], nodes[j], weight=float(score), type='similar')
                num += 1
        print(f"Added {num} similarity edges")

    # add_similarity_edges(mashup_ids, mashup_texts)
    # add_similarity_edges(api_ids, api_texts)
    add_topk_similarity_edges(mashup_ids, mashup_texts)
    add_topk_similarity_edges(api_ids, api_texts)


    return G, mashup_ids, api_ids
