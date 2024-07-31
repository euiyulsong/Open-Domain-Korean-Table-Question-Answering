import random
import os
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import LoraConfig, get_peft_model
from peft import prepare_model_for_kbit_training
from transformers import TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM 
import huggingface_hub
from src.metrics.ko_em import *
from src.metrics.ko_rouge import *
from src.metrics.ko_f1 import *
import random
import numpy as np
import gc
import wandb
import datetime

if __name__ == "__main__":
    huggingface_hub.login(token=os.getenv("HF_ACCESS_TOKEN"))
    wandb.login(key=os.getenv("WANDB_TOKEN"), relogin=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", help="model_name", type=str, default="/mnt/c/Users/thddm/Documents/model/kor_wiki_quad_od_instruct", required=False)
    parser.add_argument("-d", "--dataset_name", help="Dataset name of huggingface repo", type=str, default="kkt_synth_od_sft", required=False)
    parser.add_argument("-o", "--output_name", help="Output name of the trained model", type=str, default="/mnt/c/Users/thddm/Documents/model")
    parser.add_argument("-t", "--debug_mode", help="Determines if debug mode", default=False, action="store_true")
    parser.add_argument("-s", "--do_train", help="Determines if train mode", default=False, action="store_true")
    parser.add_argument("-e", "--do_eval", help="Determines if eval mode", default=False, action="store_true")
    args = parser.parse_args()

    model_hp = {"kkt_synth_od_sft": {"max_seq_length": 1411, "batch_size": 8, "lr": 2e-5}, 
                "kkt_od_inst": {"max_seq_length": 1411, "batch_size": 8, "lr": 2e-4},
                "kkt_cd_inst": {"max_seq_length": 153, "batch_size": 64,  "lr": 2e-4},
                "kor_wiki_quad_od_instruct": {"max_seq_length": 1411, "batch_size": 8,  "lr": 2e-4}}
    wandb.init(project=os.getenv("WANDB_PROJECT"), entity=os.getenv("WANDB_ID"), name=f"reader/{args.dataset_name}/{str(datetime.datetime.now()).replace(" ", "")}")

    args.output_name = os.path.join(f"{args.output_name}", args.dataset_name)
    dataset = load_dataset(
        f"euiyulsong/{args.dataset_name}",
    )

    model_name = args.model_name
    use_4bit = True
    use_nested_quant = False
    bnb_4bit_compute_dtype = "float16"
    bnb_4bit_quant_type = "nf4"
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    fp16 = True
    bf16 = False
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    def print_trainable_parameters(model):
        trainable_params = 0
        total_params = 0

        for _, param in model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        trainable_percent = 100 * trainable_params / total_params

        print(f"Trainable Parameters: {trainable_params}")
        print(f"Total Parameters: {total_params}")
        print(f"Trainable %: {trainable_percent:.2f}")
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
            print("=" * 80)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config, 
        device_map='auto',
        torch_dtype=compute_dtype)
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=True)
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

    assert model.pad_token_id == tokenizer.pad_token_id, "The model's pad token ID does not match the tokenizer's pad token ID!"

    print('Tokenizer pad token ID:', tokenizer.pad_token_id)
    print('Model pad token ID:', model.pad_token_id)
    print('Model config pad token ID:', model.config.pad_token_id)
    print('Number of tokens now in tokenizer:', tokenizer.vocab_size)
    special_tokens_dict = {'additional_special_tokens': ['[질문]:', '[문맥]:', '[답변]:']}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    if args.debug_mode:
        if args.do_train:
            dataset['train'] = dataset['train'].select(range(5))
        if args.do_eval:
            dataset['test'] = dataset['test'].select(range(5))
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules = ["q_proj", "v_proj"], 
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    ft_model = get_peft_model(
        model,
        config,
    )
    print_trainable_parameters(ft_model)
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        prediction = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        actual = tokenizer.batch_decode(labels, skip_special_tokens=True)
        if args.debug_mode:
            print(prediction, actual)
        eval_result = {"em": exact_match_score(prediction, actual), "f1": f1_score(prediction, actual), "rouge-l": rouge_l_score(prediction, actual)}
        wandb.log(eval_result)
        return eval_result
    training_arguments = TrainingArguments(
        output_dir=args.output_name if args.do_train else "/home/euiyul/tmp",
        num_train_epochs=1 if args.do_train else 0,
        per_device_train_batch_size=model_hp[args.dataset_name]['batch_size'],
        per_device_eval_batch_size=model_hp[args.dataset_name]['batch_size'], 
        gradient_accumulation_steps=1,
        optim="paged_adamw_8bit",
        save_steps=500,
        logging_steps=500,
        learning_rate=model_hp[args.dataset_name]['lr'],
        weight_decay=0.001,
        bf16=bf16,
        do_train=args.do_train,
        do_eval=args.do_eval,
        fp16=fp16,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.1,
        group_by_length=True,
        lr_scheduler_type="constant_with_warmup",
    )
    
    
    # Setting sft parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'] if args.do_eval else None,
        data_collator=DataCollatorForCompletionOnlyLM(response_template="<start_of_turn>model\n", tokenizer=tokenizer),
        peft_config=config,
        max_seq_length=model_hp[args.dataset_name]['max_seq_length'],
        dataset_text_field="text",
        tokenizer=tokenizer,
        compute_metrics=compute_metrics if args.do_eval else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        args=training_arguments,
        packing= False,
    )
    if args.do_train:
        trainer.train()
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(args.output_name, safe_serialization=True)
        tokenizer.save_pretrained(args.output_name)
        del merged_model
        gc.collect()
    if args.do_eval:
        metrics = trainer.evaluate()
        print(metrics)