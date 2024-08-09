from datasets import Dataset, load_dataset
import argparse
from tqdm import tqdm
import json
import huggingface_hub
import os
import random
from transformers import AutoTokenizer
from pprint import pprint
if __name__ == "__main__":
    huggingface_hub.login(token=os.getenv("HF_ACCESS_TOKEN"))
    parser = argparse.ArgumentParser()
    
    
    """
    KKT SFT
    """
    #parser.add_argument("-d", "--input_dir", help="Input filename to load", type=str, default="/mnt/c/Users/thddm/Documents/dataset/retrieved_train.jsonl", required=False)
    """
    Korwiki, Korquad Instruct-FT
    """
    # parser.add_argument("-d", "--input_dir", help="Input filename to load", type=str, default="/mnt/c/Users/thddm/Documents/dataset/synthetic_qa_v2_rlaif_refine_step2.jsonl", required=False)
    """
    KKT Corpus
    """
    parser.add_argument("-d", "--input_dir", help="Input filename to load", type=str, default="/mnt/c/Users/thddm/Documents/dataset/synthetic_v2_rlaif_nns_1000.jsonl", required=False)
    parser.add_argument("-o", "--output_name", help="Output huggingface repo name to save", type=str, default="kkt_synth_od_1000", required=False)
    parser.add_argument("-v", "--view", help="View dataset", default=False, action="store_true")
    parser.add_argument("-s", "--split", help="Dataset split", type=str, default="train", required=False)
    parser.add_argument("-m", "--model_name", help="Model name for tokenization", type=str, default="google/gemma-2b", required=False)

    od_instructions = ["다음 질문에 주어진 문맥을 바탕으로 답변하세요.\n",
        "주어진 내용을 참고하여 질문에 답변을 작성하세요.\n",
        "문맥을 기반으로 질문에 대한 답변을 하세요.\n",
        "아래 질문에 대해 문맥을 참고하여 답변하십시오.\n",
        "주어진 문맥을 참고하여 다음 질문에 답변하세요.\n",
        "다음 문맥을 바탕으로 질문에 답변해 주세요.\n",
        "문맥을 읽고 질문에 적절한 답변을 작성하세요.\n",
        "주어진 문맥을 참고하여 질문에 대한 답변을 작성하십시오.\n",
        "다음 문맥을 바탕으로 질문에 대한 답을 작성하세요.\n",
        "문맥을 참고하여 질문에 대해 답변하십시오.\n"]
    
    cd_instructions = ["다음 질문에 답변하세요.\n",
        "질문에 답변을 작성하세요.\n",
        "질문에 대한 답변을 하세요.\n",
        "아래 질문에 대해 답변하십시오.\n",
        "질문에 답변해 주세요.\n",
        "질문에 적절한 답변을 작성하세요.\n",
        "질문에 대한 답변을 작성하십시오.\n",
        "질문에 대한 답을 작성하세요.\n",
        "질문에 대해 답변하십시오.\n"]
    
    args = parser.parse_args()
    input_dir = args.input_dir
    output_name = args.output_name
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    special_tokens_dict = {'additional_special_tokens': ['[질문]:', '[문맥]:', '[답변]:']}
    tokenizer.add_special_tokens(special_tokens_dict)
    
    if not args.view:
        count_big = 0
        f = open(input_dir, "r")
        output = []
        for i in tqdm(f):
            i = json.loads(i)
            if 'corpus' in args.output_name:
                text= i['content']
                
            if 'od' in args.output_name:

                text = f"<start_of_turn>user\n{random.sample(od_instructions, 1)[0]}\n\n[질문]: {i['question']}\n\n[문맥]: {i['table']}\n\n[답변]: <end_of_turn>\n<start_of_turn>model\n{i['answer']}<end_of_turn>"
                current_length = len(tokenizer.encode(text))
                if current_length > 1411:
                    count_big +=1
                    continue
                if "simpo" in args.output_name:
                    prompt = f"<start_of_turn>user\n{random.sample(od_instructions, 1)[0]}\n\n[질문]: {i['question']}\n\n[문맥]: {i['table']}\n\n[답변]: <end_of_turn>\n<start_of_turn>model\n"
                    rejected = f"{i['rejected']}<end_of_turn>"
                    chosen = f"{i['answer']}<end_of_turn>"

            else:
                if "simpo" in args.output_name:
                    rejected = f"{i['rejected']}<end_of_turn>"
                    chosen = f"{i['answer']}<end_of_turn>"
                    prompt = f"<start_of_turn>user\n{random.sample(od_instructions, 1)[0]}\n\n[질문]: {i['question']}\n\n[문맥]: {i['table']}\n\n[답변]: <end_of_turn>\n<start_of_turn>model\n"
                    text = f"{prompt}{chosen}"
                elif 'corpus' not in args.output_name:
                    text = f"<start_of_turn>user\n{random.sample(cd_instructions, 1)[0]}\n\n[질문]: {i['question']}\n\n[답변]: <end_of_turn>\n<start_of_turn>model\n{i['answer']}<end_of_turn>"
            if "simpo" not in args.output_name:
                current_length = len(tokenizer.encode(text))
                if "corpus" in args.output_name and current_length > 248:
                    continue
                output.append({"text": text})
            else:
                output.append({"text": text, "prompt": prompt, "rejected": rejected, "chosen": chosen})

        dataset = Dataset.from_list(output)
        split = 'train' if 'train' in args.split else 'test'
        print("Split: " + split)
        print(f"Count bigger than 1411: {count_big}")
        dataset.push_to_hub(f"euiyulsong/{output_name}", split=split, private=True)
    else:
        dataset = load_dataset(
            f"euiyulsong/{output_name}",
            split=args.split
        )
        max_length = 0
        bigger_than_1410 = 0
        count_invalid=0
        lengths = []
        for idx, i in enumerate(iter(dataset)):
            if 'corpus' not in args.output_name:
                answer = i['text'].split("<start_of_turn>model\n")[1].split("<end_of_turn>")[0]

                if idx == 0:
                    print(answer)
                    print("-" * 10)
                if answer not in i['text']:
                    count_invalid+=1
            current_length = len(tokenizer.encode(i['text']))
            if current_length > 1411:
                bigger_than_1410+=1
            max_length = max(current_length, max_length)
            lengths.append(current_length)
            if idx < 5:
                pprint({j: i[j] for j in ["prompt", "chosen", "rejected", "text"] if j in i})
        lengths.sort()
        print(f"Count longer than 1411: {bigger_than_1410}")
        print(f"Count invalid: {count_invalid}")
        print(f"Dataset: {args.output_name}")
        print(f"Split: {args.split}")
        print(f"Dataset Size: {len(dataset)}")
        print(f"Max Length: {max_length}")
        print(f"98%: {lengths[int(round(len(lengths)*.98+1, 0))]}")