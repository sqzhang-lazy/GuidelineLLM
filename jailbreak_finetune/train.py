from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
from utils_data import Llama2JailbreakDataset, VicunaJailbreakDataset
from dataclasses import dataclass, field
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, PreTrainedTokenizer
import json
import os
from datasets import load_dataset
from tqdm import tqdm
from rich.table import Column, Table
from rich import box
from rich.console import Console
console = Console(record=True)

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
import random

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_int8_training
import typing
from typing import Dict, Optional, Sequence
import random

IGNORE_INDEX = -100



@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: typing.List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    lora_weight_path: str = ""
    bias: str = "none"


def init_parser():
    parser = ArgumentParser(description="Finetuning Init")
    parser.add_argument(
        "--model",
        type=str,
        default="llama2-7B-chat",
        choices=["llama2-7B-chat", "vicuna-7B", "vicuna-13B", "llama2-7B"],
        help="Select model for inference(llama2-7B-chat, vicuna)"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="",
        help="Select one jailbreak prompt benchmark"
    )
    parser.add_argument("--input_path", type=str, default="", help="Input Data")
    parser.add_argument("--save_dir", type=str, default="", help="Checkpoints save dir.")
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--warmup_ratio', type=float, default=0.03)
    parser.add_argument('--input_len', type=int, default=1536)
    parser.add_argument('--output_len', type=int, default=2560)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument("--gpu_count",type=int,default=1,help="")
    parser.add_argument("--local-rank", default=-1, type=int)
    parser.add_argument("--task", default="", type=str)
    args = parser.parse_args()

    return args

def get_system_prompt():
    return "Task: You are a creative AI. Use different stylistic methods to construct an input example."

def load_model_tokenizer(args):
    from transformers import AutoTokenizer, LlamaForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir, padding_side="right", use_fast=False)
    model = LlamaForCausalLM.from_pretrained(args.checkpoint_dir, torch_dtype=torch.bfloat16, device_map='cpu')
    return model, tokenizer



def load_Jailbreak_data(input_path):
    with open(input_path, 'r') as rp:
        datas = json.load(rp)
    sources = [i["source"] for i in datas]
    targets = [i["target"] for i in datas]
    print("data num: ", len(sources), len(targets))
    return sources, targets


    
@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


if __name__ == '__main__':
    training_logger = Table(
        Column("Epoch", justify="center"),
        Column("Steps", justify="center"),
        Column("Loss", justify="center"),
        title="Training Status",
        pad_edge=False,
        box=box.ASCII,
    )

    args = init_parser()

    local_rank = args.local_rank

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    if local_rank == 0:
        print(json.dumps(vars(args), indent=2, sort_keys=False))
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
    random.seed(args.seed)
    torch.manual_seed(args.seed)  # pytorch random seed
    np.random.seed(args.seed)  # numpy random seed
    torch.backends.cudnn.deterministic = True

    lora_args = LoraArguments()
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.lora_target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.bias,
        task_type="CAUSAL_LM",
    )


    model, tokenizer = load_model_tokenizer(args)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.truncation_side = "left"

    model = get_peft_model(model, lora_config)


    system_prompt = get_system_prompt()

    training_datas = load_Jailbreak_data(args.input_path)

    if "llama" in args.model:
        train_set = Llama2JailbreakDataset(local_rank, training_datas, tokenizer, args.input_len, args.output_len-args.input_len, system_prompt)
    elif "vicuna" in args.model:
        train_set = VicunaJailbreakDataset(local_rank, training_datas, tokenizer, args.input_len, args.output_len-args.input_len, system_prompt)


    training_args = Seq2SeqTrainingArguments(
        args.save_dir,
        do_train=True,
        do_eval=False,
        warmup_ratio=args.warmup_ratio,
        evaluation_strategy="no",
        logging_strategy="steps",
        logging_dir="",
        logging_steps=50,
        save_strategy="steps",
        save_steps=200,
        # save_total_limit = 2,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        weight_decay=0.01,
        num_train_epochs=args.epoch,
        report_to="none",
        local_rank=args.local_rank
    )

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=None,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    if local_rank == 0:
        console.log(f"[Trainer]: Trainer is ok...\n")
    trainer.train()
    trainer.save_model(f"{args.save_dir}/checkpoint-last")