import logging
import random
import sys
from transformers import TrainingArguments
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

import wandb
import transformers
from transformers import AutoModelForCausalLM, set_seed

from alignment import (
    DataArguments,
    DPOConfig,
    H4ArgumentParser,
    ModelArguments,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    is_adapter_model,
)
from peft import PeftConfig, PeftModel
from simpo_trainer import SimPOTrainer
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

@dataclass
class SimPOConfig(DPOConfig):
    gamma: Optional[float] = field(
        default=0.5,
        metadata={"help": "The target reward margin term in SimPO loss."},
    )

def main():
    huggingface_hub.login(token=os.getenv("HF_ACCESS_TOKEN"))
    wandb.login(key=os.getenv("WANDB_TOKEN"), relogin=True)
    parser = H4ArgumentParser((ModelArguments, DataArguments, SimPOConfig))
    model_args, data_args, training_args = parser.parse()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    set_seed(training_args.seed)


    raw_datasets = get_datasets(
        "kkt_od_simpo",
        splits=data_args.dataset_splits,
        configs=data_args.dataset_configs,
        columns_to_keep=["chosen", "rejected", "prompt"],
    )

    data_args.truncation_side = "left"
    tokenizer = get_tokenizer(model_args, data_args)

    for index in random.sample(range(len(raw_datasets["train"])), 3):
        logger.info(f"Prompt sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt']}")
        logger.info(f"Chosen sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['chosen']}")
        logger.info(f"Rejected sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['rejected']}")


    model_name = ""
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

    #if model_args.use_peft is True:
    ref_model = None
    training_arguments = TrainingArguments(
        output_dir="/mnt/c/Users/thddm/Documents/model/kkt-simpo",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4, 
        gradient_accumulation_steps=1,
        optim="paged_adamw_8bit",
        save_steps=500,
        logging_steps=500,
        learning_rate=2e-6,
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
    trainer = SimPOTrainer(
        model=model,
        ref_model=ref_model, 
        model_init_kwargs=model_kwargs,
        args=training_arguments,
        beta=training_args.beta,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["test"],
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
        peft_config=get_peft_config(model_args),
        loss_type=training_args.loss_type,
    )


    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(raw_datasets["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": list(data_args.dataset_mixer.keys()),
        "dataset_tags": list(data_args.dataset_mixer.keys()),
        "tags": ["alignment-handbook"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(raw_datasets["test"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)

    logger.info("*** Training complete! ***")


if __name__ == "__main__":
    main()