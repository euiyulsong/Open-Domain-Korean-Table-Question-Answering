import json
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO
import re
from tqdm import tqdm
if __name__ in "__main__":
    read_filename = "/mnt/c/Users/thddm/Documents/dataset/cleaned_korquad.jsonl"
    f = open(read_filename, "r")
    w = open("/mnt/c/Users/thddm/Documents/dataset/cleaned_filtered_mapped_korquad.jsonl", "w", encoding="utf-8")
    c = open("/mnt/c/Users/thddm/Documents/dataset/corpus_korquad.jsonl", "w", encoding="utf-8")

    corpus = set()
    missing = []
    caches = []
    inaccurate = 0
    chosens = 0
    is_korquad = "korquad" in read_filename
    if is_korquad:
        count_none = 0
        cache = set()
    for i in tqdm(f):
        i = json.loads(i)
        pattern = r'(\{[^}]*\})'

        matches = re.findall(pattern, i['table'])
        chosen = [] 
        rejected = []
        temp_table = i['table']
        if is_korquad:
            title = i['table'].split("\n")[0]
        if not matches:
            if is_korquad:
                count_none += 1
                continue
            else:
                raise()
        found = False
        for m in matches:
            if i['answer'] in m:
                found = True
                chosen.append(m)
            else:
                rejected.append(m)
            if not is_korquad:
                temp_table = temp_table.replace(m, "")
        if is_korquad:
            if not found:
                continue
            temp_table = title
        temp_table = temp_table.strip()
        i['pos'] = [f"{temp_table}\n{c}" for c in chosen]
        if is_korquad and i['pos'][0] in cache:
            continue
        i['neg'] = [f"{temp_table}\n{r}" for r in rejected]
        i['query'] = i['question']
        del i['table']
        for j in [i['pos'], i['neg']]:
            for k in j:
                corpus.add(k)
        if len(i['pos']) != 1:
            chosens += len(i['pos'])
            inaccurate += 1
            continue
        if i['pos'][0] in i['neg']:
            raise()
        w.write(json.dumps(i, ensure_ascii=False) + "\n")
        cache.add(i['pos'][0])
    
    for i in list(corpus):
        c.write(json.dumps({"content": i}, ensure_ascii=False) + "\n")
    w.close
    c.close()
    if is_korquad:
        print(f"Count None: {count_none}")
    print(f"Inaccurate: {inaccurate}")
    print(f"Mean Wrong Chosens: {chosens / inaccurate}")


