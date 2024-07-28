import json
import os
from tqdm import tqdm
from transformers import AutoTokenizer
from bs4 import BeautifulSoup
import pandas as pd
import html_to_json
from io import StringIO
import warnings
import emoji
import unicodedata
import argparse
import re
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--input_dir", help="Input filename to clean", type=str, default="/mnt/c/Users/thddm/Documents/dataset/dataset.json", required=False)
    args = parser.parse_args()
    input_dir = args.input_dir
    dirname = os.path.dirname(input_dir)
    basename = os.path.basename(input_dir)
    output_dir = os.path.join(dirname, "cleaned_" + basename + "l")
    f = open(input_dir, "r")
    w = open(output_dir, "w", encoding="utf-8")
    f = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    for idx, i in tqdm(enumerate(f)):
        table, question, answer = i['table'], i['question'], i['answer']
        output = dict()
        for j in [table, question, answer]:
            input_key = namestr(j, globals())[0]
            j = unicodedata.normalize("NFC", emoji.replace_emoji(j, "").strip())
            soup = BeautifulSoup(j, "html.parser")
            br_tags = soup.find_all("br")
            for br in br_tags:
                br.replace_with("\n")
            if soup.find_all("br"):
                raise Exception(f"<br /> tag exists")
            html_tags = [tag.name for tag in soup.find_all()]
            if input_key != 'table' and html_tags:
                print(input_key)
                raise Exception(f"Invalid Format for {input_key} with value {html_tags}")
            elif input_key != 'table':
                continue
            else:
                tables = soup.find_all('table')
                for table in tables:
                    table_list = []
                    dfs = pd.read_html(StringIO(str(table)), header=0, flavor='bs4')
                    for df in dfs:
                        df = df.dropna(axis=1)
                        df = df.to_json(orient="records", force_ascii=False)
                        df = json.loads(df)
                        for row in df:
                            table_list.append("TABLE::" + json.dumps(row, ensure_ascii=False,separators=(', ', ': ')).replace("\"", "")[1:-1])
                    table.replace_with("\n".join(table_list))
            if [tag.name for tag in soup.find_all()]:
                raise Exception("html tags exist")

            output[input_key] = re.sub(r"[\t ]*\n[\t ]*", "\n", str(soup).replace("\t", " "))
        w.write(json.dumps(output, ensure_ascii=False) + "\n")
    print(f"Finish writing into {output_dir}")
    w.close()
