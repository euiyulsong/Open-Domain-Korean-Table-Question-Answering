import faiss
from rank_bm25 import BM25Okapi
from konlpy.tag import Mecab

import argparse
from tqdm import tqdm
import json
import os
from FlagEmbedding import BGEM3FlagModel, FlagReranker
import numpy as np
from src.metrics.preprocess import normalize_answer
import torch
import gc
import time

def bm25_retrieval(queries, bm25_index, corpus):
    return [bm25_index.get_top_n(mecab.morphs(normalize_answer(q)), corpus, n=12) for q in queries]

def dense_retrieval(queries, dense_index, dense_model, lookup_table):
    embeddings = dense_model.encode(queries, batch_size=12, max_length=8192)['dense_vecs']
    indices = []
    for i in embeddings:
        _, indice = dense_index.search(np.array([i]), 12)
        indices.append(indice[0])

    return [[lookup_table[j] for j in i] for i in indices]

def ce_reranker(q, contexts):
    reranker = FlagReranker('/mnt/c/Users/thddm/Documents/model/kkt-bge-reranker-v2', use_fp16=True)
    scores = reranker.compute_score([[q, c] for c in contexts], normalize=True)
    paired_data = list(zip(scores, contexts))
    sorted_data = sorted(paired_data, key=lambda x: x[0], reverse=True)
    sorted_names = [name for _, name in sorted_data]
    return sorted_names[:5]

def orchestrator(file_name, dense_index, bm25_index, dense_model, lookup_table, corpus):
    f = open(file_name, "r")
    w = open(os.path.join(os.path.dirname(file_name), f"retrieved_{os.path.basename(file_name)}"), "w", encoding="utf-8")
    queries = []
    answers = []
    poses = []
    for i in tqdm(f):
        i = json.loads(i)
        q, a = i['question'], i['answer']
        poses.append(i['pos'][0])
        queries.append(q)
        answers.append(a)
    bm25_retrieved = bm25_retrieval(queries, bm25_index, corpus)
    start = time.time()
    dense_retrieved = dense_retrieval(queries, dense_index, dense_model, lookup_table)

    
    # del dense_model, lookup_table, dense_index, bm25_index, corpus
    # torch.cuda.empty_cache()
    # gc.collect()
    ce_reranked = []
    for q, a, d, b in tqdm(zip(queries, answers, dense_retrieved, bm25_retrieved)):
        ce = ce_reranker(q, list(set(d + b)))
        ce_reranked.append(ce)
        t = "\n\n".join(ce)
        w.write(json.dumps({'question': q, 'table': t,'answer': a, 'chosen': i['chosen'], 'rejected': i['rejected']}, ensure_ascii=False) + "\n")
    end = time.time()
    r1 = 0
    r5 = 0
    for r, p in zip(dense_retrieved, poses):
        p = p
        if p in r:
            r5+=1
            if r[0] == p:
                r1+=1
    print(f"TPS: {(end - start)/ len(queries)}")
    print(f"R1: {round(r1 / len(poses) * 100, 3)}")
    print(f"R5: {round(r5 / len(poses) * 100, 3)}")

    w.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--input_dir", help="Input filename to load", type=str, default="/mnt/c/Users/thddm/Documents/dataset/cleaned_corpus.jsonl", required=False)
    parser.add_argument("-t", "--train_dir", help="Input filename to load", type=str, default="/mnt/c/Users/thddm/Documents/dataset/train.jsonl", required=False)
    parser.add_argument("-e", "--test_dir", help="Input filename to load", type=str, default="/mnt/c/Users/thddm/Documents/dataset/test.jsonl", required=False)
    parser.add_argument("-i", "--index_path", help="Path of the index", type=str, default="/mnt/c/Users/thddm/Documents/dataset/faiss.index", required=False)
    parser.add_argument("-o", "--output_name", help="Output huggingface repo name to save", type=str, default="kkt_sft", required=False)
    parser.add_argument("-v", "--view", help="View dataset", default=False, action="store_true")
    parser.add_argument("-s", "--split", help="Dataset split", type=str, default="train", required=False)
    args = parser.parse_args()

    corpus = set()
    f = open(args.input_dir, "r")
    for i in tqdm(f):
        i = json.loads(i)
        corpus.add(i['content'])
    corpus = list(corpus)
    mecab = Mecab()
    dense_model = BGEM3FlagModel('/mnt/c/Users/thddm/Documents/model/kkt-bge-m3-dense')#'/mnt/c/Users/thddm/Documents/model/kkt-bge-m3-unified')
    lookup_table = {}
    tokenized_corpus = [mecab.morphs(normalize_answer(doc)) for doc in corpus]
    dense_corpus = [doc for doc in corpus]

    for idx, i in enumerate(dense_corpus):
        lookup_table[idx] = i
    if os.path.exists(args.index_path):
        index = faiss.read_index(args.index_path)
    else:
        embeddings = dense_model.encode(dense_corpus, batch_size=12, max_length=8192)['dense_vecs']
        index = faiss.IndexFlatL2(len(embeddings[0]))
        for e in embeddings:
            index.add(np.array([e]))
        faiss.write_index(index, args.index_path)
    
    bm25 = BM25Okapi(tokenized_corpus)

    print("Test")
    orchestrator(args.test_dir, index, bm25, dense_model, lookup_table, corpus)
    #print("Train")
    #orchestrator(args.train_dir, index, bm25, dense_model, lookup_table, corpus)
    

