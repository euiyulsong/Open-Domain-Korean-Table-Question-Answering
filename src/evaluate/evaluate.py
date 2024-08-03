from transformers import AutoTokenizer, BatchEncoding, AutoModelForCausalLM
import argparse
import torch
from datasets import load_dataset
from tqdm import tqdm
from src.metrics.ko_em import *
from src.metrics.ko_rouge import *
from src.metrics.ko_f1 import *
import wandb
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import time

@dataclass
class CustomDataCollator:
    tokenizer: PreTrainedTokenizerBase
    return_tensors: str ="pt"
    def __call__(self, features):
        label_length = [len(feature["label"]) for feature in features]
        input_ids_length = [len(feature['input_ids']) for feature in features]
        max_label_length = max(label_length)
        max_input_length = max(input_ids_length)
        tobe = {}
        tobe['labels'] = []
        tobe['input_ids'] = []
        tobe['attention_mask'] = []
        for _, feature in enumerate(features):
            for key in ['input_ids', 'attention_mask']:
                if self.tokenizer.padding_side == "left":
                    tobe[key].append([self.tokenizer.pad_token_id] * (max_input_length - len(feature[key])) + feature[key])
                else:
                    tobe[key].append(feature[key] + [self.tokenizer.pad_token_id] * (max_input_length - len(feature[key])))
            if self.tokenizer.padding_side == "left":
                tobe['labels'].append([self.tokenizer.pad_token_id] * (max_label_length - len(feature['label'])) + feature['label'])
            else:
                tobe['labels'].append(feature['label'] + [self.tokenizer.pad_token_id] * (max_label_length - len(feature['label'])))
        features = BatchEncoding(tobe, tensor_type=self.return_tensors)
        return features


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    eval_result = {"em": exact_match_score(preds, labels), "f1": f1_score(preds, labels), "rouge-l": rouge_l_score(preds, labels)}
    for i in eval_result:
        eval_result[i] = round(eval_result[i] * 100, 3)
    return eval_result

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, instance):
        super().__init__()
        self.instance = instance

    def __len__(self):
        return len(self.instance)

    def __getitem__(self, idx):
        current = self.instance[idx]
        return self.instance[idx]
    

if __name__ in "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset_name", help="Dataset name of huggingface repo", type=str, default="kkt_od_inst", required=False)
    parser.add_argument("-m", "--model_name", help="Output name of the trained model", type=str, default="kkt_instruction_tune_synth_sft_synth_simpo_f16")
    args = parser.parse_args()
    args.model_name = "/mnt/c/Users/thddm/Documents/model/" + args.model_name
    config = {"kkt_od_inst": {"max_seq_length": 1420, "batch_size": 24}, "kkt_cd_inst": {"max_seq_length": 176, "batch_size": 128}} # type: ignore
    config = config[args.dataset_name]
    wandb.login(key=os.getenv("WANDB_TOKEN"), relogin=True)
    wandb.init(project=os.getenv("WANDB_PROJECT"), entity=os.getenv("WANDB_ID"), name=f"eval/{args.model_name}")
    model_name = args.model_name
    compute_dtype = "float16"
    dataset = load_dataset(
        f"euiyulsong/{args.dataset_name}",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=True)
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map='auto',
            torch_dtype=compute_dtype)
    
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer.padding_side = "left"

    if '<pad>' in tokenizer.get_vocab():
        print('<pad> token is in the tokenizer. Using <pad> for pad')
        tokenizer.pad_token = '<pad>'
    elif '<unk>' in tokenizer.get_vocab():
        print('<unk> token is in the tokenizer. Using unk for pad')
        tokenizer.pad_token = '<unk>'
    else:
        print(f'Using EOS token, {tokenizer.eos_token}, for padding')
        tokenizer.pad_token = tokenizer.eos_token

    model.pad_token_id = tokenizer.pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    datasets = []
    for idx, i in enumerate(iter(dataset['test'])):
        splited = i['text'].split("<start_of_turn>model\n")
        prompt = splited[0].strip() + "<start_of_turn>model\n"
        answer = splited[1].split("<end_of_turn>")[0].strip()
        tokenized = tokenizer(prompt)
        tokenized['input_ids'] = tokenized['input_ids'][:-1]
        tokenized['attention_mask'] = tokenized['attention_mask'][:-1]
        tokenized['label'] = tokenizer.encode(answer)[:-1]
        datasets.append(tokenized)

    sizes = len(datasets)
    end = 107
    datasets = CustomDataset(datasets)
    # model.to("cuda")
    preds = []
    labels = []

    with torch.no_grad():
        dataloader = torch.utils.data.DataLoader(datasets, batch_size=config['batch_size'], collate_fn=CustomDataCollator(tokenizer=tokenizer))
        start = time.time()
        for idx, i in tqdm(enumerate(dataloader)):
            label = i['labels']
            prompt = i['input_ids']
            i.to("cuda")

            output = model.generate(**i, eos_token_id=tokenizer.eos_token_id, max_length=config['max_seq_length'])
            output = output.cpu().detach().numpy().tolist()
            prompt = tokenizer.batch_decode(prompt, skip_special_tokens=True)
            output = tokenizer.batch_decode(output, skip_special_tokens=True)
            for idx2, (p, o) in enumerate(zip(prompt, output)):
                output[idx2] = o[len(p): ].strip()

            label = tokenizer.batch_decode(label, skip_special_tokens=True)
            for idx2, l in enumerate(label):
                label[idx2] = l.strip()

            preds.extend(output)
            labels.extend(label)

        end = time.time()
        out = compute_metrics([preds, labels])
        out['evaltime'] = end - start
        out['size_dataset'] = sizes
        wandb.log(out)