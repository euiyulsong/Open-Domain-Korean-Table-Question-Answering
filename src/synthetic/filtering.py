import json
import argparse
import faiss
from FlagEmbedding import BGEM3FlagModel
import numpy as np
from tqdm import tqdm
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--corpus_dir", help="Corpus path", type=str, default="/mnt/c/Users/thddm/Documents/dataset/cleaned_corpus.jsonl", required=False)
    parser.add_argument("-t", "--train_dir", help="Train path", type=str, default="/mnt/c/Users/thddm/Documents/dataset/train.jsonl", required=False)

    parser.add_argument("-s", "--synthetic", help="Synthetic path", type=str, default="/mnt/c/Users/thddm/Documents/dataset/synthetic_qa_v2_rlaif_refine_step2.jsonl", required=False)
    parser.add_argument("-o", "--output_dir", help="Output filename to load", type=str, default="/mnt/c/Users/thddm/Documents/dataset/synthetic_v2_rlaif_nns_500.jsonl", required=False)
    parser.add_argument("-n", "--neighbors", help="Number of nearest neighbor", type=int, default=1000, required=False)

    args = parser.parse_args()
    args.output_dir = args.output_dir.replace("500", f"{args.neighbors}")
    dense_corpus = []
    existing_data = set()
    question_data = []
    for i in tqdm(open(args.train_dir, "r")):
        i = json.loads(i)
        existing_data.add(i['pos'][0])
        question_data.append(i['question'])
    lookup_table = {}

    for idx, i in enumerate(open(args.corpus_dir, "r")):
        i = json.loads(i)['content']
        dense_corpus.append(i)
        lookup_table[idx] = i
    w = open(args.output_dir, "w", encoding="utf-8")
    dense_model = BGEM3FlagModel("/mnt/c/Users/thddm/Documents/model/kkt-bge-m3-dense")
    embeddings = dense_model.encode(dense_corpus, batch_size=48, max_length=8192)['dense_vecs']
    index = faiss.IndexFlatL2(len(embeddings[0]))
    for e in embeddings:
        index.add(np.array([e]))

    question_embddings = dense_model.encode(question_data, batch_size=48, max_length=8192)['dense_vecs']
    question_index = faiss.IndexFlatL2(len(question_embddings[0]))

    for e in question_embddings:
        question_index.add(np.array([e]))
    map_queries = {}
    count_existing = 0
    for idx, i in enumerate(open(args.synthetic, "r")):
        i = json.loads(i)
        
        if i['pos'][0] not in existing_data:
            map_queries[i['question']] = i
        else:
            count_existing += 1
    
    print(f"Existing in train: {count_existing}")
    queries = list(map_queries.keys())
    embeddings = dense_model.encode(queries, batch_size=48, max_length=8192)['dense_vecs']
    indices = []
    scores = []
    for i in embeddings:
        _, indice = index.search(np.array([i]), 2)
        distances, _ = question_index.search(np.array([i]), 1)
        scores.append(distances[0][0]) # L2: Lower the better
        indices.append(indice[0])
    filtered = []
    for q, i, s in zip(queries, indices, scores):
        m = map_queries[q]
        retrieved = [lookup_table[j] for j in i]
        if m['pos'][0] in retrieved:
            m['l2_distance'] = float(s)
            filtered.append(m)
    
    filtered.sort(key= lambda x: x['l2_distance'])
    for idx, m in enumerate(filtered):
        if idx == args.neighbors:
            break
        w.write(json.dumps(m, ensure_ascii=False) + "\n")
