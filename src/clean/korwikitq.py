import json
from tqdm import tqdm
if __name__ in "__main__":
    w = open("/mnt/c/Users/thddm/Documents/dataset/korwiki.jsonl", "w", encoding="utf-8")
    c = open("/mnt/c/Users/thddm/Documents/dataset/korwiki_corpus.jsonl", "w", encoding="utf-8")
    cache_title = set()
    count = 0
    corpus = set()
    for f in ["/mnt/c/Users/thddm/Documents/dataset/KorWikiTQ_ko_train.json", "/mnt/c/Users/thddm/Documents/dataset/KorWikiTQ_ko_dev.json"]:
        f = json.load(open(f, "r"))
        for i in tqdm(f['data']):
            title = i['T']
            if title in cache_title:
                continue
            cache_title.add(title)
            question = i['QAS']['question']
            answer = i['QAS']['answer']
            table = i['TBL']
            keys = i['TBL'][0]
            pos = []
            neg = []
            found = False
            for j in range(1, len(i['TBL'])):
                e = {k: v for k, v in zip(keys, i['TBL'][j])}
                e_str = f"{title}\n{json.dumps(e, ensure_ascii=False)}"
                corpus.add(e_str)
                if answer in list(e.values()):
                    pos.append(e_str)
                    found = True
                else:
                    neg.append(e_str)

            if not found:
                continue
            if len(pos) > 1:
                count += 1
                continue
            else:
                if len(neg) <= 5:
                    continue
            w.write(json.dumps({"question": question, "query": question, "answer": answer, "pos": pos, "neg": neg}, ensure_ascii=False) + "\n")
    for cor in list(corpus):
        c.write(json.dumps({"content": cor}, ensure_ascii=False) + "\n")
    print(count)


            

