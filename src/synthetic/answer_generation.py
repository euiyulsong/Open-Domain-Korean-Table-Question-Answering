from openai import OpenAI
import os
from tqdm import tqdm
import json
import random
from pprint import pprint
import unicodedata
if __name__ == "__main__":
    instruction = ["다음 질문에 주어진 문맥을 바탕으로 답변하세요.\n",
        "주어진 내용을 참고하여 질문에 답변을 작성하세요.\n",
        "문맥을 기반으로 질문에 대한 답변을 하세요.\n",
        "아래 질문에 대해 문맥을 참고하여 답변하십시오.\n",
        "주어진 문맥을 참고하여 다음 질문에 답변하세요.\n",
        "다음 문맥을 바탕으로 질문에 답변해 주세요.\n",
        "문맥을 읽고 질문에 적절한 답변을 작성하세요.\n",
        "주어진 문맥을 참고하여 질문에 대한 답변을 작성하십시오.\n",
        "다음 문맥을 바탕으로 질문에 대한 답을 작성하세요.\n",
        "문맥을 참고하여 질문에 대해 답변하십시오.\n"]
    
    client = OpenAI(api_key=os.getenv("OPENAI_TOKEN"))
    examples = []
    f = open("/mnt/c/Users/thddm/Documents/dataset/train.jsonl", "r")
    

    for i in tqdm(f):
        i = json.loads(i)
        examples.append(f"[문맥] {i['pos'][0]}\n[질문] {i['query']}\n=> [답변]: {i['chosen']}")
    
    f = open("/mnt/c/Users/thddm/Documents/dataset/synthetic_questions_v2.jsonl", "r")
    f2 = open("/mnt/c/Users/thddm/Documents/dataset/synthetic_qa_v2.jsonl", "r")
    cache = set()
    for i in f2:
        i = json.loads(i)
        cache.add(i['pos'][0])

    f2.close()
    w = open("/mnt/c/Users/thddm/Documents/dataset/synthetic_qa_v2.jsonl", "a", encoding="utf-8")

    for i in tqdm(f):
        i = json.loads(i)
        if i['pos'][0] in cache:
            continue
        example = random.sample(examples, 5)
        inst = random.sample(instruction, 1)[0]
        example = "\n\n\n\n".join([inst + e for e in example])
        message = [{"role": "system", "content": unicodedata.normalize("NFC", f"안녕하세요. 저는 문맥을 통하여 질문을 답변을 생성하는 질문답변 GPT입니다. 아래는 답변을 생성하는 예시입니다.\n{example}\n\n\n\n")}, {"role": "user", "content": unicodedata.normalize("NFC", f"위의 예시를 참고하여 {inst}[문맥] {i['pos'][0]}\n[질문] {i['query']}\n=> [답변]: ")}]

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=message
        )
    
        i['answer'] = completion.choices[0].message.content
        
        w.write(json.dumps(i, ensure_ascii=False) + "\n")