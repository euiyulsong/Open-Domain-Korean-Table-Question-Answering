import json
import os
from tqdm import tqdm
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO
import warnings
import emoji
import unicodedata
import argparse
import re

warnings.filterwarnings("ignore", category=UserWarning, module='bs4')
import logging
def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]
def replace_double_quotes(json_str):
    json_str = re.sub(r'\"(\w+)\":', r'\1:', json_str) 
    json_str = re.sub(r': \"(\w+)\"', r': \1', json_str)
    return json_str

def replace_single_quotes(tuple_str):
    tuple_str = re.sub(r"'(\w+)'(?=,|\))", r"\1", tuple_str)
    return tuple_str

def replace_json_quotes(json_str):
    json_str = re.sub(r"[{}]", "", json_str)
    return json_str
if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(encoding='utf-8', level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--input_dir", help="Input filename to clean", type=str, default="/mnt/c/Users/thddm/Documents/dataset/korquad.jsonl", required=False)
    args = parser.parse_args()
    input_dir = args.input_dir
    dirname = os.path.dirname(input_dir)
    basename = os.path.basename(input_dir)
    output_dir = os.path.join(dirname, "cleaned_" + basename)
    f = open(input_dir, "r")
    w = open(output_dir, "w", encoding="utf-8")
    if 'korquad' not in args.input_dir:
        f = json.load(f)
    he = -1

    for idx, i in tqdm(enumerate(f)):
        if 'korquad' in args.input_dir:
            i = json.loads(i)

        table, question, answer = i['table'], i['question'], i['answer']
        output = dict()
        for j in [table, question, answer]:
            input_key = namestr(j, globals())[0]
            j = unicodedata.normalize("NFC", emoji.replace_emoji(j, "").strip())
            soup = BeautifulSoup(j, "html.parser")


            for k in soup.find_all(['title']):
                i['title'] = k.string
                k.replaceWithChildren()
                
            for k in soup.find_all(['head']):
                k.extract()
            for k in soup.find_all(['html', 'dd', 'strong', 'annotation', 'i', 'abbr', 'munderover', 'input', 'sup', 'small', 'cite', 'hr', 'mi', 'tt', 'bdi', 'area', 'label', 'mpadded', 'ruby', 'dt', 'body', 'del', 'h5', 'link', 'big', 'none', 'track', 'sub', 'blockquote', 'p', 'noscript', 'rp', 'mspace', 'kbd', 'u', 'img', 'h3', 'div', 'rt', 'html', 'ins', 'math', 'mover', 'code', 'mrow', 'mstyle', 'h2', 'samp', 'munder', 'meta', 'msqrt', 'source', 'h4', 'caption', 'msubsup', 'mtext', 'a', 'map', 'video', 'wbr', 'mfrac', 'mo', 'rb', 'msup', 'mroot', 'pre', 's', 'mprescripts', 'var', 'strike', 'dl', 'font', 'semantics', 'style', 'mmultiscripts', 'span', 'menclose', 'b', 'center', 'form', 'h6', 'q', 'msub', 'mn', 'audio', 'h1']):
                k.replaceWithChildren()

            if "CON'C BOX" in str(soup):
                he = 0
            for br in soup.find_all("br"):
                br.replace_with("\n")
            html_tags = [tag.name for tag in soup.find_all()]

            list_tags = soup.find_all(["ol", "ul"])


            for list_tag in list_tags:
                li = list_tag.find_all("li")
                for l in li:
                    new_content = '- ' + l.get_text() + '\n'
                    l.replace_with(new_content)
                list_tag.replaceWithChildren()
            for l in soup.find_all("li"):
                l.replaceWithChildren()

            
            tables = soup.find_all('table')
            for table in tables:
                max_rowspan = 0
                max_firstrow = 0
                num_th = 0
                for idx2, row in enumerate(table.find_all("tr")):
                    slide_window_col = 0
                    if row.find_all("th"):
                        num_th += 1
                    for idx3, cell in enumerate(row.find_all(['th', 'td'])):
                        try:
                            rowspan = int(float(cell.get("rowspan", 1)))
                        except:
                            print(rowspan)
                            rowspan = 1
                        try:
                            colspan = int(float(cell.get("colspan", 1)))
                        except:
                            print(colspan)
                            colspan = 1
                            
                        if idx3 == 0:
                            max_firstrow = max(max_firstrow, rowspan)
                        colend = colspan
                        max_rowspan = max(max_rowspan, rowspan + idx2)
                        slide_window_col += 1
                    if max_rowspan == idx2 + 1:
                        break
                table_list = []

                # assert num_th == 0 or num_th  max_firstrow, "Error"
                # if num_th != 0:
                #     print(num_th)
                #     print(max_firstrow)
                #     max_firstrow = num_th
                try:
                    dfs = pd.read_html(StringIO(str(table)), header=[m for m in range(max_firstrow)], thousands=None, flavor='bs4')
                except:
                    table.extract()
                    continue
                columns = set()
                for df in dfs:
                    columns.update(set(dfs[0].columns.tolist()))
                dfs = pd.read_html(StringIO(str(table)), header=[m for m in range(max_firstrow)], thousands=None, converters={c: str for c in columns}, flavor='bs4')

                for df in dfs:
                    df = df.to_json(orient="records", force_ascii=False)
            
                    df = json.loads(df)

                        
                    for row in df:
                        invalid_list = []
                        for key, value in row.items():
                            if value is None:
                                invalid_list.append(key)
                        for invalid in invalid_list:
                            del row[invalid]
                        table_list.append(json.dumps(row, ensure_ascii=False,separators=(', ', ': ')))
                table.replace_with("\n".join(table_list))
            for k in soup.find_all():
                k.replaceWithChildren()
            
            assert not [tag.name for tag in soup.find_all()], str(soup)


            output[input_key] = re.sub(r'\s*\n\s*', "\n", " ".join(re.sub(r'\s*\n\s*', "\n", unicodedata.normalize('NFC',  str(soup).replace("<!DOCTYPE html>", "")).replace("&lt;", "<").replace("&gt;", ">").replace("\xa0", " ").replace("\t", " ").replace("\r", "\n")).split(" "))).strip()

        w.write(json.dumps(output, ensure_ascii=False) + "\n")
    logger.info(f"Finish writing into {output_dir}")
    w.close()
