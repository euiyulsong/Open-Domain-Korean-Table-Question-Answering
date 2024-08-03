from transformers import AutoTokenizer, AutoModelForCausalLM 
import argparse
import torch
import transformers
from datasets import load_dataset
import datetime
from tqdm import tqdm
from src.metrics.ko_em import *
from src.metrics.ko_rouge import *
from src.metrics.ko_f1 import *
import wandb

def compute_metrics(eval_preds, tokenizer):
    preds, labels, prompts = eval_preds
    stopping_token = set([tokenizer.eos_token_id, 107, -100])

    def stops(i, stopping_token, flag=0):
        for idx, j in enumerate(i):               
            new_p = []
            for e in j:
                if e not in stopping_token:
                    new_p.append(e)
                else:
                    if not flag:
                        new_p.append(tokenizer.pad_token_id)
                    else:
                        break
            preds[idx] = new_p
        return preds
    preds = stops(preds, stopping_token, 1)
    labels = stops(labels, stopping_token, 0)



    prediction = tokenizer.batch_decode(preds, skip_special_tokens=True)
    actual = tokenizer.batch_decode(labels, skip_special_tokens=True)
    prompt = tokenizer.batch_decode(prompts, skip_special_tokens=True)
    prediction_reconstruct = []
    for p, q in zip(prediction, prompt):
        prediction_reconstruct.append(p[len[q]:].strip())
    prediction= prediction_reconstruct
    print(actual[:2])
    print(prediction[:2])
    print(prompt[:2])

    eval_result = {"em": exact_match_score(prediction, actual), "f1": f1_score(prediction, actual), "rouge-l": rouge_l_score(prediction, actual)}
    #wandb.log(eval_result)
    return eval_result
class QADataset(torch.utils.data.Dataset):
    def __init__(self, instance):
        super().__init__()
        self.instance = instance

    def __len__(self):
        return len(self.instance)

    def __getitem__(self, idx):
        return self.instance[idx]

if __name__ in "__main__":
    config = {"max_seq_length": 1420, "batch_size": 16} # type: ignore
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset_name", help="Dataset name of huggingface repo", type=str, default="kkt_od_inst", required=False)
    parser.add_argument("-m", "--model_name", help="Output name of the trained model", type=str, default="kkt_instruction_tune_synth_sft_synth_simpo_f16")
    args = parser.parse_args()
    args.model_name = "/mnt/c/Users/thddm/Documents/model/" + args.model_name
    wandb.login(key=os.getenv("WANDB_TOKEN"), relogin=True)
    wandb.init(project=os.getenv("WANDB_PROJECT"), entity=os.getenv("WANDB_ID"), name=f"eval/{args.model_name}")
    model_name = args.model_name
    compute_dtype = "float16"
    dataset = load_dataset(
        f"euiyulsong/{args.dataset_name}",
    )
    print(next(iter(dataset)))
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=True)
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map='auto',
            torch_dtype=compute_dtype)
    
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer.padding_side = "right"

    if '<pad>' in tokenizer.get_vocab():
        print('<pad> token is in the tokenizer. Using <pad> for pad')
        # Set the pad token
        tokenizer.pad_token = '<pad>'
    elif '<unk>' in tokenizer.get_vocab():
        print('<unk> token is in the tokenizer. Using unk for pad')
        # Set the pad token
        tokenizer.pad_token = '<unk>'
    else:
        print(f'Using EOS token, {tokenizer.eos_token}, for padding')
        tokenizer.pad_token = tokenizer.eos_token

    model.pad_token_id = tokenizer.pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    datasets = []
    for i in iter(dataset['test']):
        splited = i['text'].split("<start_of_turn>model\n")
        prompt = splited[0].strip()
        answer = splited[1].split("<end_of_turn>")[0].strip()
        tokenized = tokenizer(prompt)
        tokenized['label'] = tokenizer.encode(answer)
        datasets.append(tokenized)

    end = 107
    datasets = QADataset(datasets)
    model.to("cuda")
    preds = []
    labels = []
    prompts = []
    with torch.no_grad():
        dataloader = torch.utils.data.DataLoader(datasets, batch_size=config['batch_size'], collate_fn=transformers.DataCollatorForSeq2Seq(tokenizer, model))
        for i in dataloader:
            label = i['labels'].cpu().numpy().tolist()
            prompt = i['input_ids'].cpu().numpy().tolist()
            prompts.extend(prompt)
            labels.extend(label)

            i.to("cuda")

            
            output = model.generate(**i, eos_token_id=end, max_length=config['max_seq_length'])
            output = output.cpu().detach().numpy().tolist()
            
            preds.extend(output)
            break
        print(args.model_name)
        print(preds)
        print(labels)
        print(prompts)
        print(compute_metrics([preds, labels, prompts], tokenizer))
        wandb.log(compute_metrics([preds, labels, prompts], tokenizer))