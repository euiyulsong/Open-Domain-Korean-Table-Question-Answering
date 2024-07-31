import json
from tqdm import tqdm
f = open("/mnt/c/Users/thddm/Documents/dataset/train.jsonl", "r")
w = open("/mnt/c/Users/thddm/Documents/dataset/retrieval/train.jsonl", "w", encoding="utf-8")

for i in tqdm(f):
    i = json.loads(i)
    
    w.write(json.dumps({"query": i["query"], "pos": i["pos"], "neg": i["neg"]}, ensure_ascii=False) + "\n")

w.close()

    
