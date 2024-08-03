import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
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

    synthetic_simpo = {"name": "/mnt/c/Users/thddm/Documents/model/kkt_instruction_tune_synth_f16",
                       "out": "/mnt/c/Users/thddm/Documents/model/f32_inst_tu_simpo_synthetic", #"/mnt/c/Users/thddm/Documents/model/f32_inst_tu_simpo_synthetic", 
                       "lr": 2e-6, "bs": 3, "dataname": "euiyulsong/kkt_synth_od_simpo"}
    
    if args.is_float16:
        synthetic_simpo['name'] = "/mnt/c/Users/thddm/Documents/model/kor_wiki_quad_od_instruct_f16"
        synthetic_simpo['out'] = "/mnt/c/Users/thddm/Documents/model/inst_tu_simpo_synthetic_f16"

    simpo = {"name": "/mnt/c/Users/thddm/Documents/model/kkt_instruction_tune_synth_f16",
             "out": "/mnt/c/Users/thddm/Documents/model/kkt_instruction_tune_synth_sft_real_orpo_f16", # actually orpo
             "lr": 2e-6, "bs": 3, "dataname": "euiyulsong/kkt_od_simpo"}

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

    # for i in iter(raw_datasets):
    #     raw_datasets[i] = raw_datasets[i].map(lambda example: {"prompt": example['prompt'], "chosen": example["chosen"] + "<eos>", "rejected": example["rejected"] + "<eos>"}, batched=False)

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

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # quantization_config=bnb_config, 
        torch_dtype=compute_dtype)
    # if args.is_float16:
    #     model.push_to_hub("euiyulsong/f16_insttuned", private=True)

    # else:
    #     model.push_to_hub("euiyulsong/f32_insttuned", private=True)
    # raise()
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
    # model = prepare_model_for_kbit_training(model)
    ft_model = get_peft_model(
        model,
        config,
    )
    from trl import ORPOConfig, ORPOTrainer

    orpo_args = ORPOConfig(
        learning_rate=1e-6, #for two stage pre-training 1e-6, #for orpo semi with synthetic 1e-6 , #1e-5 orpo 1e-4 for the semi orpo with synthetic
        lr_scheduler_type="constant_with_warmup",
        max_length=1411,
        max_prompt_length=1411,
        beta=0.1,
        per_device_train_batch_size=bs, #20,
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