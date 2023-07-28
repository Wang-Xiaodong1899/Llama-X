#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import sys
sys.path.append('/workspace/Llama-X')

import os
import json
import copy
import logging
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Sequence, Tuple, Union

import torch
from torch import nn
import torch.distributed
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import TrainingArguments
from datasets import load_dataset
from torchvision.transforms.functional import InterpolationMode


from utils import *
from llama.attention import *

IGNORE_INDEX = -100
DEFAULT_BOS_TOKEN = '<s>'
DEFAULT_EOS_TOKEN = '</s>'
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_UNK_TOKEN = "<unk>"

DEFAULT_IMG_START_TOKEN = "[IMG]"
DEFAULT_IMG_END_TOKEN = "[/IMG]"
DEFAULT_IMG_TOKEN = "<image>"
USER_TOKEN = '[USER]'
ASSISTANT_TOKEN = '[ASSISTANT]'

image_placeholder = "[IMG]" + "<image>" * 32 + "[/IMG]"


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

class Emu_Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        image_features = inputs["image"] # image 32 features
        text_input = inputs["input_ids"] # text input_ids
        input_mask = inputs["attention_mask"] # text attention_mask
        image = inputs["image"] # image 32 tensor
        outputs = model(image=image, text_input=text_input, input_mask=input_mask)
        llm_loss = outputs.llm_loss
        regression_loss = outputs.regression_loss
        loss = llm_loss + regression_loss
        
        return loss


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    label_names: List[str] = field(default_factory=lambda: ['input_ids', 'label', 'image_names', 'image'])
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    image_names: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, label=labels, image_names=image_names)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ]
        )

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "label"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        print(f'input_ids: {input_ids.shape}')
        
        # Add image features
        image_names = [instance['image_names'] for instance in instances] # batchsize [name1, name2]
        image_tensors = []
        for name in image_names:
            path = os.path.join('/f_data/G/dataset/mscoco/train2017', name)
            image_tensor = self.transform(Image.open(path).convert('RGB'))
            image_tensors.append(image_tensor)
        image_tensors = torch.stack(image_tensors, dim=0)
        print(f'image: {image_tensors.shape}')
        
        return dict(
            input_ids=input_ids,
            label=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            image=image_tensors,
        )

def train_tokenize_function(examples, tokenizer):
    captions = [output for output in examples['caption']]
    image_names = [output for output in examples['image_name']]
    targets = [image_placeholder + tokenizer.eos_token] * len(captions)
    
    data_dict = preprocess(captions, image_names, targets, tokenizer)
    
    return data_dict

class llamaconfig():
    def __init__(self) -> None:
        self.ckpt_path = os.path.join('/f_data/G', "Emu/Emu/Emu-instruct.pt")
        self.instruct = True
        self.model_config_file = '/workspace/Llama-X/emu/Emu-14B.json'
        self.max_seq_length = 256

def quick_freeze(model):
    for name, param in model.named_parameters():
        param.requires_grad = False
    return model


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Debug by LLaMA
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if "llama" in model_args.model_name_or_path:
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )
    
    # TODO EMU
    # from emu.modeling_emu import Emu
    # args = llamaconfig()
    
    # emu_config = file2data(args.model_config_file)
    # model = Emu(**emu_config, cast_dtype=torch.float, args=args)
    
    # print('Patching LoRA...')
    # from peft import LoraConfig, get_peft_model
    # lora_config = LoraConfig(
    #     r=16,
    #     lora_alpha=16,
    #     target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    #     lora_dropout=0.05,
    #     bias="none",
    #     task_type="CAUSAL_LM",
    # )
    # model.decoder.lm = get_peft_model(model.decoder.lm, lora_config)
    # model = quick_freeze(model)
    # model.visual = model.visual.eval().half()
    # model.ln_visual = model.ln_visual.eval().half()
    # model.cformer = model.cformer.eval().half()
    
    # ckpt = torch.load(args.ckpt_path, map_location="cpu")
    # adaptively_load_state_dict(model, ckpt)
    
    print("INFO begin to init llama")
    # state_dict = torch.load(args.llama_7b_state_dict, map_location="cpu")
    # model_parameters = model.load_state_dict(state_dict)

    
    #TODO model.decoder.tokenizer
    special_token_list = [DEFAULT_IMG_START_TOKEN, DEFAULT_IMG_END_TOKEN, DEFAULT_IMG_TOKEN, USER_TOKEN,
                                  ASSISTANT_TOKEN]
    # lm.tokenizer will be set in initialzation of decoder, No Action Need!
    
    # from emu.modeling_llama import LLaMAForClsAndRegression
    # if isinstance(model.decoder, LLaMAForClsAndRegression):
    #     model.decoder.tokenizer.padding_side = "right"
    # tokenizer = model.decoder.tokenizer
        
    
    # model Ingore

    raw_train_datasets = load_dataset('json', data_files=data_args.data_path, split="train", cache_dir=training_args.cache_dir)
    if training_args.local_rank > 0: 
        torch.distributed.barrier()

    train_dataset = raw_train_datasets.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True, # not args.overwrite_cache
        desc="Running tokenizer on train dataset",
        fn_kwargs={"tokenizer": tokenizer}
    )

    if training_args.local_rank == 0:
        torch.distributed.barrier()
    
    if training_args.local_rank == 0:
        print(len(train_dataset))
        for index in random.sample(range(len(train_dataset)), 3):
            print(f"Sample {index} of the training set: {train_dataset[index]}.")
    print(f'pad id: {tokenizer.pad_token_id}')
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

    #Tell Trainer not to attempt DataParallel
    model.is_parallelizable = True
    model.model_parallel = True

    trainer = Emu_Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
