import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import LoraConfig, get_peft_model
from peft import prepare_model_for_kbit_training
import huggingface_hub
from src.metrics.ko_em import *
from src.metrics.ko_rouge import *
from src.metrics.ko_f1 import *
import wandb
from transformers import AutoModelForCausalLM
from trl import CPOTrainer, CPOConfig
import argparse


def main():
    huggingface_hub.login(token=os.getenv("HF_ACCESS_TOKEN"))
    wandb.login(key=os.getenv("WANDB_TOKEN"), relogin=True)
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--is_float16", help="Determines if float16 mode", default=False, action="store_true")
    parser.add_argument("-s", "--is_synthetic", help="Determines if synthetic dataset", default=False, action="store_true")

    args = parser.parse_args()

    synthetic_simpo = {"name": "/mnt/c/Users/thddm/Documents/model/f32_instruction_tuning",
                       "out": "/mnt/c/Users/thddm/Documents/model/f32_inst_tu_simpo_synthetic",
                       "lr": 2e-6, "bs": 1, "dataname": "euiyulsong/kkt_synth_od_simpo"}
    

    simpo = {"name": "/mnt/c/Users/thddm/Documents/model/f32_inst_tu_simpo_synthetic",
             "out": "/mnt/c/Users/thddm/Documents/model/f32_inst_tu_simpo_synthetic_simpo_real",
             "lr": 2e-7, "bs": 1, "dataname": "euiyulsong/kkt_od_simpo"}

    current = synthetic_simpo if args.is_synthetic else simpo



    model_name = current['name']
    output_dir = current['out']
    lr = current['lr']
    bs = current['bs']
    dataname = current['dataname']
    raw_datasets = load_dataset(
        dataname
    )
    optim = "paged_adamw_32bit"

    for i in iter(raw_datasets):
        raw_datasets[i] = raw_datasets[i].map(lambda example: {"prompt": example['prompt'], "chosen": example["chosen"] + "<eos>", "rejected": example["rejected"] + "<eos>"}, batched=False)

    print(next(iter(raw_datasets)))
    # With synthetic
    # model_name = "/mnt/c/Users/thddm/Documents/model/kor_wiki_quad_od_instruct"
    # output_dir = "/mnt/c/Users/thddm/Documents/model/kkt-simpo-synth"
    # lr = 2e-6


    bnb_4bit_compute_dtype = "float32" if not args.is_float16 else "float16"

    # bnb_4bit_quant_type = "nf4"
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    fp16 = False if not args.is_float16 else True
    # use_4bit = True
    # use_nested_quant = False

    bf16 = False
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=use_4bit,
    #     bnb_4bit_quant_type=bnb_4bit_quant_type,
    #     bnb_4bit_compute_dtype=compute_dtype,
    #     bnb_4bit_use_double_quant=use_nested_quant,
    # )

    def print_trainable_parameters(model):
        trainable_params = 0
        total_params = 0

        for _, param in model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        trainable_percent = 100 * trainable_params / total_params


    training_arguments = CPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs, 
        gradient_accumulation_steps=1,
        optim=optim,
        save_steps=500,
        logging_steps=500,
        learning_rate=lr,
        weight_decay=0.001,
        bf16=bf16,
        do_train=True,
        do_eval=False,
        fp16=fp16,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.1,
        max_length=1411,
        max_prompt_length=1411,
        beta=10.0,
        cpo_alpha=.05,
        simpo_gamma=5.4,
        loss_type="simpo",
        group_by_length=False,
        lr_scheduler_type="constant_with_warmup",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # quantization_config=bnb_config, 
        device_map='auto',
        torch_dtype=compute_dtype)
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=True)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    if '<pad>' in tokenizer.get_vocab():
        # Set the pad token
        tokenizer.pad_token = '<pad>'
    elif '<unk>' in tokenizer.get_vocab():
        # Set the pad token
        tokenizer.pad_token = '<unk>'
    else:
        tokenizer.pad_token = tokenizer.eos_token

    model.pad_token_id = tokenizer.pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    assert model.pad_token_id == tokenizer.pad_token_id, "The model's pad token ID does not match the tokenizer's pad token ID!"

    special_tokens_dict = {'additional_special_tokens': ['[질문]:', '[문맥]:', '[답변]:']}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
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

    trainer = CPOTrainer(
        model=model,
        args=training_arguments,
        peft_config=config,
        train_dataset=raw_datasets["train"],
        tokenizer=tokenizer,
        packing=False
    )

    trainer.train()
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(training_arguments.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(training_arguments.output_dir)

if __name__ == "__main__":
    main()