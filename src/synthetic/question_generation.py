from openai import OpenAI
import os
from tqdm import tqdm
import json
import random
from pprint import pprint
import unicodedata
if __name__ == "__main__":
    instruction = [
        "주어진 정보를 바탕으로 다음 질문을 생성하세요: \n\n[정보]: ",
        "다음 정보를 기반으로 적절한 질문을 만들어 주세요: \n\n[정보]: ",
        "아래의 정보를 참고하여 관련된 질문을 작성해 주세요: \n\n[정보]: ",
        "주어진 데이터에서 추론할 수 있는 질문을 생성해 주세요: \n\n[정보]: ",
        "다음 설명을 보고 관련 질문을 만들어 주세요: \n\n[정보]: ",
        "아래 정보를 기반으로 새로운 질문을 형성해 주세요: \n\n[정보]: ",
        "제공된 정보를 활용하여 질문을 생성해 주세요: \n\n[정보]: ",
        "다음 정보를 바탕으로 적합한 질문을 제시해 주세요: \n\n[정보]: ",
        "아래의 정보에 대해 가능한 질문을 작성해 주세요: \n\n[정보]: ",
        "제공된 정보를 참고하여 관련된 질문을 만들어 주세요: \n\n[정보]: "
    ]
    client = OpenAI(api_key=os.getenv("OPENAI_TOKEN"))
    examples = []
    f = open("/mnt/c/Users/thddm/Documents/dataset/train.jsonl", "r")
    

    for i in tqdm(f):
        i = json.loads(i)
        examples.append(i['pos'][0] + " => " + f"[질문]: " + i['query'])
    
    f = open("/mnt/c/Users/thddm/Documents/dataset/cleaned_corpus.jsonl", "r")
    w = open("/mnt/c/Users/thddm/Documents/dataset/synthetic_questions_v2.jsonl", "w", encoding="utf-8")

    for i in tqdm(f):
        i = json.loads(i)
        i['pos'] = [i['content']]
        example = random.sample(examples, 5)
        inst = random.sample(instruction, 1)[0]
        example = "\n\n\n\n".join([inst + e for e in example])
        message = [{"role": "system", "content": unicodedata.normalize("NFC", f"안녕하세요. 저는 합성질문을 생성하는 Synthetic GPT입니다. 아래는 합성질문을 생성하는 예시입니다.\n{example}\n\n\n\n")}, {"role": "user", "content": unicodedata.normalize("NFC", f"위의 예시를 참고하여 {inst}{i['content']} => [질문]: ")}]
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=message
        )
    
        del i['content']
        i['query'] = completion.choices[0].message.content
        i['question'] = i['query']
        w.write(json.dumps(i, ensure_ascii=False) + "\n")
