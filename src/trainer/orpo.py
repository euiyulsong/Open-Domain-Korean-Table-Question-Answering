import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import LoraConfig
import huggingface_hub
from src.metrics.ko_em import *
from src.metrics.ko_rouge import *
from src.metrics.ko_f1 import *
import wandb
from transformers import AutoModelForCausalLM
import argparse
from trl import ORPOConfig, ORPOTrainer

def main():
    huggingface_hub.login(token=os.getenv("HF_ACCESS_TOKEN"))
    wandb.login(key=os.getenv("WANDB_TOKEN"), relogin=True)
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--is_float16", help="Determines if float16 mode", default=False, action="store_true")
    parser.add_argument("-s", "--is_synthetic", help="Determines if synthetic dataset", default=False, action="store_true")

    args = parser.parse_args()
    synthetic_orpo = {"name": "/mnt/c/Users/thddm/Documents/model/kor_wiki_quad_od_instruct_f16",
                       "out": "/mnt/c/Users/thddm/Documents/model/kkt_instruction_tune_synth_500_orpo_f16",
                       "lr": 2e-5, "bs": 1, "dataname": "euiyulsong/kkt_synth_od_500_simpo"}
    
    
    orpo = {"name": "/mnt/c/Users/thddm/Documents/model/kkt_instruction_tune_synth_500_orpo_f16",
             "out": "/mnt/c/Users/thddm/Documents/model/kkt_instruction_tune_synth_500_orpo_real_orpo_f16",
             "lr": 2e-6, "bs": 1, "dataname": "euiyulsong/kkt_od_simpo"}

    current = synthetic_orpo if args.is_synthetic else orpo


    model_name = current['name']
    output_dir = current['out']
    bs = current['bs']
    dataname = current['dataname']
    raw_datasets = load_dataset(
        dataname
    )
    optim = "paged_adamw_32bit"


    bnb_4bit_compute_dtype = "float16"

    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)


    model = AutoModelForCausalLM.from_pretrained(
        model_name,
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


    orpo_args = ORPOConfig(
        learning_rate=1e-6,
        lr_scheduler_type="constant_with_warmup",
        max_length=1411,
        max_prompt_length=1411,
        beta=0.1,
        per_device_train_batch_size=bs,
        save_steps=500,
        gradient_accumulation_steps=1,
        optim=optim,
        num_train_epochs=1,
        logging_steps=500,
        warmup_ratio=0.1,
        output_dir=output_dir
    )

    trainer = ORPOTrainer(
        model=model,
        args=orpo_args,
        train_dataset=raw_datasets["train"],
        peft_config=config,
        tokenizer=tokenizer,
    )


    trainer.train()
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(orpo_args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(orpo_args.output_dir)

if __name__ == "__main__":
    main()