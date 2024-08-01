import json
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from konlpy.tag import Mecab
from src.metrics.preprocess import normalize_answer
import re
import random
import unicodedata
if __name__ in "__main__":
    """
    For Synthetic kkt dataset
    f = open("/mnt/c/Users/thddm/Documents/dataset/synthetic_qa_v2_wrong_refine.jsonl", "r", encoding="utf-8")
    w = open("/mnt/c/Users/thddm/Documents/dataset/synthetic_qa_v2_rlaif_wrong_refine.jsonl", "w", encoding="utf-8")
    ww = open("/mnt/c/Users/thddm/Documents/dataset/synthetic_qa_v2_wrong_refine_wrong.jsonl", "w", encoding="utf-8")
    c = open("/mnt/c/Users/thddm/Documents/dataset/corpus.jsonl", "r", encoding="utf-8")
    """


    """
    For SFT kkt dataset
    f = open("/mnt/c/Users/thddm/Documents/dataset/train.jsonl", "r", encoding="utf-8")
    w = open("/mnt/c/Users/thddm/Documents/dataset/train_rlaif.jsonl", "w", encoding="utf-8")
    ww = open("/mnt/c/Users/thddm/Documents/dataset/train_wrong.jsonl", "w", encoding="utf-8")
    c = open("/mnt/c/Users/thddm/Documents/dataset/cleaned_corpus.jsonl", "r", encoding="utf-8")   
    """

    """
    For SFT Korquad v2.0 dataet
    """
    # is_korquad = True
    # f = open("/mnt/c/Users/thddm/Documents/dataset/cleaned_filtered_mapped_korquad.jsonl", "r", encoding="utf-8")
    # w = open("/mnt/c/Users/thddm/Documents/dataset/korquad_sft.jsonl", "w", encoding="utf-8")
    # ww = open("/mnt/c/Users/thddm/Documents/dataset/korquad_sft_wrong.jsonl", "w", encoding="utf-8")
    # c = open("/mnt/c/Users/thddm/Documents/dataset/corpus_korquad.jsonl", "r", encoding="utf-8")   
    

    """
    For SFT Korwiki dataet
    """
    # is_korquad = True
    # f = open("/mnt/c/Users/thddm/Documents/dataset/korwiki.jsonl", "r", encoding="utf-8")
    # w = open("/mnt/c/Users/thddm/Documents/dataset/korwiki_sft.jsonl", "w", encoding="utf-8")
    # ww = open("/mnt/c/Users/thddm/Documents/dataset/korwiki_sft_wrong.jsonl", "w", encoding="utf-8")
    # c = open("/mnt/c/Users/thddm/Documents/dataset/korwiki_corpus.jsonl", "r", encoding="utf-8")   
    
    """
    For kkt concatenation of filtered answer generation, refine step1 and refine step2
    Concat three dataset: cat synthetic_qa_v2_wrong_refine.jsonl synthetic_qa_v2_wrong_refine_error_refine.jsonl synthetic_qa_v2_rlaif.jsonl | shuf > synthetic_qa_v2_concat_refine_step2.jsonl
    Remaining Errors: synthetic_qa_v2_wrong_refine_error_refine_error.jsonl
    """
    is_korquad = False
    f = open("/mnt/c/Users/thddm/Documents/dataset/synthetic_qa_v2_concat_refine_step2.jsonl", "r", encoding="utf-8")
    w = open("/mnt/c/Users/thddm/Documents/dataset/synthetic_qa_v2_rlaif_refine_step2.jsonl", "w", encoding="utf-8")
    ww = open("/mnt/c/Users/thddm/Documents/dataset/synthetic_qa_v2_rlaif_refine_step2_wrong.jsonl", "w", encoding="utf-8")
    c = open("/mnt/c/Users/thddm/Documents/dataset/cleaned_corpus.jsonl", "r", encoding="utf-8")   
    pattern = r'(\{[^}]*\})'
    mecab = Mecab()
    corpus = []
    for e in tqdm(c):
        e = json.loads(e)
        corpus.append(e['content'])
    if 0:
        corpus = random.sample(corpus, 100000)
    tokenized_corpus = [mecab.morphs(normalize_answer(doc)) for doc in tqdm(corpus)]
    bm25 = BM25Okapi(tokenized_corpus)
    count_wrong = 0
    for i in tqdm(f):
        i = json.loads(i)
        for j in ['question', 'query']:
            i[j] = unicodedata.normalize("NFC", i[j])
        i['neg'] = [] if 'neg' not in i else i['neg']
        if is_korquad:
            for j in ['pos', 'neg']:
                i[j] = [unicodedata.normalize("NFC", k) for k in i[j]]
            if len(i['neg']) < 5:
                i['neg'] = i['neg'] + [li for li in list(set(bm25.get_top_n(i['question'], corpus, 10 - len(i['neg']))) - set(i['pos'])) if "{}" not in li] 
            i['tables'] = i['pos'] + i['neg'][:4]
            if len(i['tables']) != 5:
                raise()
            random.shuffle(i['tables'])
            i['table'] = "\n\n".join(i['tables'])


            w.write(json.dumps(i, ensure_ascii=False) + "\n")
            continue
        for j in [i['pos'], i['neg']]:
            for k in j:
                if k not in corpus:
                    raise()
            
        corpus_neg = i['neg']
        matches_pos  = None
        i['chosen'] = str(i['answer'].strip())
        i['answer'] = i['chosen']
        i['question'] = i['query']
        if 'rejected' in i:
            del i['rejected']
        i['chosen'] = str(i['chosen'])
        keys = None
        key_set = []
        matches_pos = json.loads(re.findall(pattern, i['pos'][0])[0])
        for k, v in matches_pos.items():
            v = str(v)
            if keys is None and (v == i['answer'] or i['answer'] in v or v in i['answer']):
                keys = k
            key_set.append(k)
        if keys is None:
            ww.write(json.dumps(i, ensure_ascii=False) + "\n")
            count_wrong +=1
            continue
        if len(i['neg']) > 0:
            tokenized_neg = [mecab.morphs(normalize_answer(doc)) for doc in i['neg']]
            bm25_neg = BM25Okapi(tokenized_neg)
            top8_neg = bm25_neg.get_top_n(mecab.morphs(normalize_answer(i['query'])), corpus_neg, n=8)
            for j in range(len(top8_neg)):
                matches_neg = json.loads(re.findall(pattern, top8_neg[j])[0])
                if keys in matches_neg:
                    matches_neg[keys] = str(matches_neg[keys])
                    if matches_neg[keys] != i['chosen'] and matches_neg[keys] not in i['chosen'] and matches_neg[keys] != i['chosen']:
                        i['rejected'] = matches_neg[keys]
                        break
                

            if 'rejected' not in i and matches_pos is not None:
                temp = str(matches_pos[random.sample(key_set, 1)[0]])
                found = False
                counter = 0
                while (i['chosen'] in temp or temp in i['chosen'] or i['chosen'] == temp) and counter < 5:
                    temp = str(matches_pos[random.sample(key_set, 1)[0]])
                    counter +=1

                i['rejected'] = temp
        i['chosen'] = i['answer'].strip()

        if len(i['neg']) < 8:
            out = bm25.get_top_n(mecab.morphs(normalize_answer(i['query'])), corpus, n=64)
            for o in out:
                if o not in i['neg'] and o not in i['pos']:
                    i['neg'].append(o)
                if len(i['neg']) >= 8:
                    break
        i['tables'] = i['pos'] + i['neg'][:4]
        if len(i['tables']) != 5:
            raise()
        random.shuffle(i['tables'])
        i['table'] = "\n\n".join(i['tables'])
        if len(i['neg']) < 8:
            raise Exception(f"{len(i['neg'])}")
        corpus_neg = i['neg']
        tokenized_neg = [mecab.morphs(normalize_answer(doc)) for doc in i['neg']]
        bm25_neg = BM25Okapi(tokenized_neg)
        top8_neg = bm25_neg.get_top_n(mecab.morphs(normalize_answer(i['pos'][0])), corpus_neg, n=8)
        if len(top8_neg) < 8:
            raise()
        if 'rejected' not in i or i['rejected'] == i['chosen'] or i['rejected'] in i['chosen'] or i['chosen'] in i['rejected']:
            sampled = random.sample(top8_neg, 1)[0]
            key_set = []
            json_sampled = json.loads(re.findall(pattern, sampled)[0])
            for k, v in json_sampled.items():
                key_set.append(k)   
            temp = str(json_sampled[random.sample(key_set, 1)[0]])
            i['pos'][0] = i['pos'][0]
            counter = 0
            while (i['chosen'] in temp or temp in i['chosen'] or i['chosen'] == temp) and counter < 5:
                temp = str(json_sampled[random.sample(key_set, 1)[0]])
                counter += 1


            i['rejected'] = temp

        if 'rejected' not in i or i['rejected'] == i['chosen'] or i['rejected'] in i['chosen'] or i['chosen'] in i['rejected']:
            sampled = random.sample(corpus, 1)[0]
            key_set = []
            json_sampled = json.loads(re.findall(pattern, sampled)[0])
            for k, v in json_sampled.items():
                key_set.append(k)   
            temp = str(json_sampled[random.sample(key_set, 1)[0]])
            i['pos'][0] = i['pos'][0]
            
            while i['chosen'] in temp or temp in i['chosen'] or i['chosen'] == temp:
                temp = str(json_sampled[random.sample(key_set, 1)[0]])
            i['rejected'] = temp

        assert 'rejected' in i and i['rejected'] not in i['chosen'] and i['chosen'] not in i['rejected'] and i['rejected'] != i['chosen'], f"{i['rejected']}\n{i['chosen']}"
        i['neg'] = top8_neg
        assert len(i['pos']) == 1 and len(i['neg']) == 8, f"{len(i['pos'])}\t{len(i['neg'])}"
        if i['pos'][0] in i['neg']:
            raise()
        for j in ['neg', 'pos']:
            i[j] = [k for k in i[j]]

        w.write(json.dumps(i, ensure_ascii=False) + "\n")
    w.close()
    print(f"Number of data without correct answer: {count_wrong}")
