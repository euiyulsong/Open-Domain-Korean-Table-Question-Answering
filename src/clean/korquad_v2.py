from datasets import load_dataset
import json
from bs4 import BeautifulSoup
import warnings
import emoji
import unicodedata
import argparse
from tqdm import tqdm
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset_name", help="Dataset name of huggingface repo", type=str, default="KorQuAD/squad_kor_v2", required=False)
    parser.add_argument("-m", "--output_path", help="Output path", type=str, default="/mnt/c/Users/thddm/Documents/dataset/korquad.jsonl")
    args = parser.parse_args()
    dataset = load_dataset(args.dataset_name)
    w = open(args.output_path, "w", encoding="utf-8")
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