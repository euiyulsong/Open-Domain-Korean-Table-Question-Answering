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
    # labels = np.array(labels)
    # labels = np.where(labels == -100, tokenizer.pad_token_id, labels)



    # prediction = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # actual = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # prompts = tokenizer.batch_decode(prompts, skip_special_tokens=True)

    # print(prediction[0])
    # print(actual[0])
    eval_result = {"em": exact_match_score(preds, labels), "f1": f1_score(preds, labels), "rouge-l": rouge_l_score(preds, labels)}
    return eval_result
class QADataset(torch.utils.data.Dataset):
    def __init__(self, instance):
        super().__init__()
        self.instance = instance

    def __len__(self):
        return len(self.instance)

    def __getitem__(self, idx):
        current = self.instance[idx]
        return self.instance[idx]
import time
if __name__ in "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset_name", help="Dataset name of huggingface repo", type=str, default="kkt_od_inst", required=False)
    parser.add_argument("-m", "--model_name", help="Output name of the trained model", type=str, default="kkt_instruction_tune_synth_sft_synth_simpo_f16")
    args = parser.parse_args()
    args.model_name = "/mnt/c/Users/thddm/Documents/model/" + args.model_name
    config = {"kkt_od_inst": {"max_seq_length": 1420, "batch_size": 16}, "kkt_cd_inst": {"max_seq_length": 176, "batch_size": 128}} # type: ignore
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

    #tokenizer.padding_side = "right"

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
        prompt = splited[0].strip() + "<start_of_turn>model\n"
        answer = splited[1].split("<end_of_turn>")[0].strip()
        tokenized = tokenizer(prompt, return_tensors="pt")
        print(tokenizer.batch_decode(tokenized['input_ids']))
        tokenized['input_ids'] = tokenized['input_ids'][:, :-1]
        print(tokenizer.batch_decode(tokenized['input_ids']))
        tokenized['attention_mask'] = tokenized['attention_mask'][:, :-1]
        raise()

        tokenized['labels'] = tokenizer.encode(answer, return_tensors="pt")
        datasets.append(tokenized)
    sizes = len(datasets)
    end = 107
    #datasets = QADataset(datasets)
    model.to("cuda")
    preds = []
    labels = []
    prompts = []
    with torch.no_grad():
        #dataloader = torch.utils.data.DataLoader(datasets, batch_size=1, collate_fn=None)#transformers.DataCollatorWithPadding(tokenizer))
        start = time.time()
        for idx, i in tqdm(enumerate(datasets)):
            label = i['labels']
            prompt = i['input_ids']
            del i['labels']

            i.to("cuda")

            
            output = model.generate(**i, eos_token_id=tokenizer.eos_token_id, max_length=config['max_seq_length'])
            output = output.cpu().detach().numpy().tolist()
            prompt = tokenizer.decode(prompt[0], skip_special_tokens=True).strip()
            output = tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):].strip()
            label = tokenizer.decode(label[0], skip_special_tokens=True).strip()


        end = time.time()
        out = compute_metrics([preds, labels, prompts], tokenizer)
        out['time'] = end - start
        out['size'] = sizes
        wandb.log(out)
