import faiss
from FlagEmbedding import BGEM3FlagModel, FlagReranker

from rank_bm25 import BM25Okapi
from konlpy.tag import Mecab

import argparse
from tqdm import tqdm
import json
import os
import numpy as np
from src.metrics.preprocess import normalize_answer
import time
import wandb
import datetime


def bm25_retrieval(queries, bm25_index, corpus):
    return [bm25_index.get_top_n(mecab.morphs(normalize_answer(q)), corpus, n=12) for q in queries]

def dense_retrieval(queries, dense_index, dense_model, lookup_table):
    embeddings = dense_model.encode(queries, batch_size=12, max_length=8192)['dense_vecs']
    indices = []
    for i in embeddings:
        _, indice = dense_index.search(np.array([i]), 12)
        indices.append(indice[0])

    return [[lookup_table[j] for j in i] for i in indices]

def ce_reranker(q, contexts, args):
    reranker = FlagReranker(args.reranker)
    scores = reranker.compute_score([[q, c] for c in contexts], normalize=True)
    paired_data = list(zip(scores, contexts))
    sorted_data = sorted(paired_data, key=lambda x: x[0], reverse=True)
    sorted_names = [name for _, name in sorted_data]
    return sorted_names[:5]

def compute_metric(retrieved, poses, method):
    r1 = 0
    r5 = 0
    for r, p in zip(retrieved, poses):
        i_r1 = 0
        i_r5 = 0
        for i in p:
            if i in r:
                i_r5+=1
                if i == r[0]:
                    i_r1+=1
        i_r1 /= len(p)
        i_r5 /= len(p)
        r1 += i_r1
        r5 += i_r5
    print(f"R1: {round(r1 / len(poses) * 100, 3)}")
    print(f"R5: {round(r5 / len(poses) * 100, 3)}")
    wandb.log({f"{method}_rp": f"{round(r1 / len(poses) * 100, 3)}", f"{method}_r5": f"{round(r5 / len(poses) * 100, 3)}"})
def orchestrator(file_name, dense_index, bm25_index, dense_model, lookup_table, corpus, args):
    f = open(file_name, "r")
    # w = open(os.path.join(os.path.dirname(file_name), f"retrieved_{os.path.basename(file_name)}"), "w", encoding="utf-8")
    queries = []
    answers = []
    poses = []
    for i in tqdm(f):
        i = json.loads(i)
        q, a = i['question'], i['answer']
        poses.append(i['pos'])
        queries.append(q)
        answers.append(a)
    bm25_retrieved = bm25_retrieval(queries, bm25_index, corpus)
    print(f"Lexical")
    compute_metric(bm25_retrieved, poses, "bm")
    print()
    start = time.time()
    dense_retrieved = dense_retrieval(queries, dense_index, dense_model, lookup_table)
    end = time.time()

    print(f"Dense")
    compute_metric(dense_retrieved, poses, "dense")
    print()
             
    start2 = time.time()

    ce_reranked = []
    for q, a, d, b in tqdm(zip(queries, answers, dense_retrieved, bm25_retrieved)):
        ce = ce_reranker(q, list(set(d + b)), args)
        ce_reranked.append(ce)
        t = "\n\n".join(ce)
        # w.write(json.dumps({'question': q, 'table': t,'answer': a, 'chosen': i['chosen'], 'rejected': i['rejected']}, ensure_ascii=False) + "\n")
    end2 = time.time()

    print(f"CE")
    compute_metric(ce_reranked, poses, "ce")
    print()
    print(f"TPS: {(end - start + end2 - start2)/ len(queries)}")
    # w.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--input_dir", help="Input filename to load", type=str, default="/mnt/c/Users/thddm/Documents/dataset/cleaned_corpus.jsonl", required=False)
    parser.add_argument("-t", "--train_dir", help="Input filename to load", type=str, default="/mnt/c/Users/thddm/Documents/dataset/train.jsonl", required=False)
    parser.add_argument("-e", "--test_dir", help="Input filename to load", type=str, default="/mnt/c/Users/thddm/Documents/dataset/test.jsonl", required=False)
    parser.add_argument("-i", "--index_path", help="Path of the index", type=str, default="/mnt/c/Users/thddm/Documents/dataset/faiss.index", required=False)
    parser.add_argument("-r", "--retrieval", help="Path of the retrieval", type=str, default="/mnt/c/Users/thddm/Documents/model/kkt-bge-m3-stochastic-dense", required=False)
    parser.add_argument("-c", "--reranker", help="Path of the reranker", type=str, default="/home/euiyul/kkt-bge-reranker-v2", required=False)
    parser.add_argument("-o", "--output_name", help="Output huggingface repo name to save", type=str, default="kkt_sft", required=False)
    parser.add_argument("-v", "--view", help="View dataset", default=False, action="store_true")
    parser.add_argument("-s", "--split", help="Dataset split", type=str, default="train", required=False)
    args = parser.parse_args()
    wandb.login(key=os.getenv("WANDB_TOKEN"), relogin=True)
    wandb.init(project=os.getenv("WANDB_PROJECT"), entity=os.getenv("WANDB_ID"), name=f"retrieval_performance_{str(datetime.datetime.now()).replace(" ", "")}")

    corpus = set()
    f = open(args.input_dir, "r")
    for i in tqdm(f):
        i = json.loads(i)
        corpus.add(i['content'])
    corpus = list(corpus)
    mecab = Mecab()
    dense_model = BGEM3FlagModel(args.retrieval)
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
    
    bm25 = BM25Okapi(tokenized_corpus)

    print("Test")
    orchestrator(args.test_dir, index, bm25, dense_model, lookup_table, corpus, args)
    print("Train")
    orchestrator(args.train_dir, index, bm25, dense_model, lookup_table, corpus, args)
    

