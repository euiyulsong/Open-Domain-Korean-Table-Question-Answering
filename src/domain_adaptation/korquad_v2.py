from datasets import load_dataset
import json
from bs4 import BeautifulSoup
import warnings
import emoji
import unicodedata
import re
from tqdm import tqdm
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

if __name__ == "__main__":
    dataset = load_dataset("KorQuAD/squad_kor_v2")
    w = open("/mnt/c/Users/thddm/Documents/dataset/korquad.jsonl", "w", encoding="utf-8")
    cache = set()
    duplicates = 0
    extract_sets = set()
    for split in dataset:
        split = dataset[split]
        for current in tqdm(split):
            j = unicodedata.normalize("NFC", emoji.replace_emoji(current['context'], "").strip())
            soup = BeautifulSoup(j, "html.parser")
            if current['context'] not in extract_sets:
                extract_sets.update(set([tag.name for tag in soup.find_all()]))
                row = {"question": current['question'], "table": current['context'], "answer": current['answer']['text']}
                w.write(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                duplicates +=1

        
    w.close()
    print(f"Duplicates: {duplicates}")
    print(extract_sets)