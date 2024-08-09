import os
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import LoraConfig, get_peft_model
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
    parser.add_argument("-m", "--model_name", help="model_name", type=str, default="kor_wiki_quad_od_instruct_f16", required=False)
    parser.add_argument("-d", "--dataset_name", help="Dataset name of huggingface repo", type=str, default="kkt_synth_od_500_simpo", required=False)
    parser.add_argument("-o", "--output_name", help="Output name of the trained model", type=str, default="kkt_instruction_tune_synth_500_sft_f16")
    parser.add_argument("-t", "--debug_mode", help="Determines if debug mode", default=False, action="store_true")
    parser.add_argument("-s", "--do_train", help="Determines if train mode", default=False, action="store_true")
    parser.add_argument("-e", "--do_eval", help="Determines if eval mode", default=False, action="store_true")
    parser.add_argument("-f", "--is_float16", help="Determines if float16 mode", default=False, action="store_true")
    parser.add_argument("-i", "--is_int4", help="Determines if int4 mode", default=False, action="store_true")

    args = parser.parse_args()
    args.model_name = "/mnt/c/Users/thddm/Documents/model/" + args.model_name
    args.output_name = "/mnt/c/Users/thddm/Documents/model/" + args.output_name

    if args.do_eval:
        args.output_name = "/home/euiyul/tmp" 

    model_hp = {"kkt_synth_od_sft": {"max_seq_length": 1411, "batch_size": 6, "lr": 2e-5}, 
                "kkt_od_inst": {"max_seq_length": 1411, "batch_size": 6, "lr": 2e-4 if args.model_name == "google/gemma-2b" else 2e-6}, #divide by 2
                "kkt_cd_inst": {"max_seq_length": 176 if args.do_train else 1411, "batch_size": 64,  "lr": 2e-4 if "corpus" not in args.model_name else 2e-5}, 
                "kkt_corpus": {"max_seq_length": 248, "batch_size": 48,  "lr": 2e-4},
                "kor_wiki_quad_od_instruct": {"max_seq_length": 1411, "batch_size": 6,  "lr": 2e-4}, 
                "kkt_synth_od_500_simpo": {"max_seq_length": 1411, "batch_size": 6,  "lr": 2e-5},
                "kkt_synth_od_1000": {"max_seq_length": 1411, "batch_size": 6,  "lr": 2e-5},
                "kkt_synth_od_2000": {"max_seq_length": 1411, "batch_size": 6,  "lr": 2e-5},
                "kkt_synth_od_3000": {"max_seq_length": 1411, "batch_size": 6,  "lr": 2e-5}}

    dataset = load_dataset(
        f"euiyulsong/{args.dataset_name}",
    )

    model_name = args.model_name

    bnb_4bit_compute_dtype = "float16"

    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    fp16 = True
    bf16 = False


    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        torch_dtype=compute_dtype)
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=True)

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules = ["q_proj", "v_proj"], 
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model.gradient_checkpointing_enable()
    ft_model = get_peft_model(
        model,
        config,
    )

    training_arguments = TrainingArguments(
        output_dir=args.output_name if args.do_train else "/home/euiyul/tmp",
        num_train_epochs=1 if args.do_train else 0,
        per_device_train_batch_size=model_hp[args.dataset_name]['batch_size'],
        per_device_eval_batch_size=model_hp[args.dataset_name]['batch_size'], 
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
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
    
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'] if args.do_eval else None,
        data_collator=DataCollatorForCompletionOnlyLM(response_template="<start_of_turn>model\n", tokenizer=tokenizer) if "corpus" not in args.dataset_name else None,
        peft_config=config,
        max_seq_length=model_hp[args.dataset_name]['max_seq_length'],
        dataset_text_field="text",
        tokenizer=tokenizer,
        compute_metrics=None,
        args=training_arguments,
        packing=False
    )

    if args.do_train:
        trainer.train()
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(args.output_name, safe_serialization=True)
        tokenizer.save_pretrained(args.output_name)
        gc.collect()
