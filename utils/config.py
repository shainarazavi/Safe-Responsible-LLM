import logging
from dataclasses import dataclass, field

from typing import Tuple, Dict, Optional
import torch 

from utils.params import TRAIN_PARAM_DICT,INFER_PARAM_DICT, PROMPT_FORMAT_DICT

from transformers import (
    BitsAndBytesConfig,
    TrainingArguments
)

from peft import (
    LoraConfig
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class CustomTrainConfig:
    base_model: str = TRAIN_PARAM_DICT.get('base_model')[1]
    input_data_path: str = TRAIN_PARAM_DICT.get('input_data_path')[1]
    test_size: float = TRAIN_PARAM_DICT.get('test_size')[1]
    output_dir: str = TRAIN_PARAM_DICT.get('output_dir')[1]
    # TRAINER CONFIG
    num_train_epochs: int = TRAIN_PARAM_DICT.get('num_train_epochs')[1]
    per_device_train_batch_size: int = TRAIN_PARAM_DICT.get('per_device_train_batch_size')[1]
    gradient_accumulation_steps: float = TRAIN_PARAM_DICT.get('gradient_accumulation_steps')[1]
    optimizer: str = TRAIN_PARAM_DICT.get('optimizer')[1]
    save_steps: int = TRAIN_PARAM_DICT.get('save_steps')[1]
    logging_steps: int = TRAIN_PARAM_DICT.get('logging_steps')[1]
    logging_dir: str = TRAIN_PARAM_DICT.get('logging_dir')[1]
    learning_rate: float = TRAIN_PARAM_DICT.get('learning_rate')[1]
    weight_decay: int = TRAIN_PARAM_DICT.get('weight_decay')[1]
    fp16: bool = TRAIN_PARAM_DICT.get('fp16')[1]
    bf16: bool = TRAIN_PARAM_DICT.get('bf16')[1]
    max_grad_norm: float = TRAIN_PARAM_DICT.get('max_grad_norm')[1]
    max_steps: int = TRAIN_PARAM_DICT.get('max_steps')[1]
    warmup_ratio: float = TRAIN_PARAM_DICT.get('warmup_ratio')[1]
    group_by_length: bool = TRAIN_PARAM_DICT.get('group_by_length')[1]
    lr_scheduler_type: str = TRAIN_PARAM_DICT.get('lr_scheduler_type')[1]
    report_to: str = TRAIN_PARAM_DICT.get('report_to')[1]
    remove_unused_columns: bool = TRAIN_PARAM_DICT.get('remove_unused_columns')[1]
    dataset_text_field: str = TRAIN_PARAM_DICT.get('dataset_text_field')[1]
    max_seq_length: int = TRAIN_PARAM_DICT.get('max_seq_length')[1]
    packing: bool = TRAIN_PARAM_DICT.get('packing')[1]
    remove_unused_columns: bool = TRAIN_PARAM_DICT.get('remove_unused_columns')[1]

@dataclass
class CustomPEFTConfig:
    lora_alpha: int = TRAIN_PARAM_DICT.get('lora_alpha')[1]
    lora_dropout: int = TRAIN_PARAM_DICT.get('lora_dropout')[1]
    lora_r: int = TRAIN_PARAM_DICT.get('lora_r')[1]
    bias: str = TRAIN_PARAM_DICT.get('bias')[1]
    task_type: str = TRAIN_PARAM_DICT.get('task_type')[1]

@dataclass
class CustomBNBConfig:
    use_4bit: bool = TRAIN_PARAM_DICT.get('use_4bit')[1]
    bnb_4bit_quant_type: str = TRAIN_PARAM_DICT.get('bnb_4bit_quant_type')[1]
    bnb_4bit_compute_dtype: str = TRAIN_PARAM_DICT.get('bnb_4bit_compute_dtype')[1]
    use_nested_quant: bool = TRAIN_PARAM_DICT.get('use_nested_quant')[1]

@dataclass
class CustomInferenceConfig:
    base_model:str = INFER_PARAM_DICT.get('base_model')[1]
    adapter_model:str = INFER_PARAM_DICT.get('adapter_model')[1]
    single_inference:bool = INFER_PARAM_DICT.get('single_inference')[1]
    prompt_dictionary:dict = field(default_factory=lambda: (dict, PROMPT_FORMAT_DICT)) # because this is a muable datatype
    input_csv:str = INFER_PARAM_DICT.get('input_csv')[1]
    max_length:int = INFER_PARAM_DICT.get('max_length')[1]
    output_dir:str = INFER_PARAM_DICT.get('output_dir')[1]


def create_bnb_config(config:CustomBNBConfig) -> BitsAndBytesConfig:
    """

    """
    compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)
    logger.info(f"Defined Model config as 4bit:{config.use_4bit} - Qunatization Type:{config.bnb_4bit_quant_type}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.use_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config.use_nested_quant,
    )
    return bnb_config


def create_peft_config(config:CustomPEFTConfig) -> LoraConfig:

    logger.info(f"Defined PEFT config as ALPHA:{config.lora_alpha} - DROPOUT:{config.lora_dropout} - R:{config.lora_r}")
    peft_config = LoraConfig(
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        r=config.lora_r,
        bias=config.bias,
        task_type=config.task_type
    )
    return peft_config 


def create_training_config(config:CustomTrainConfig)-> Tuple[TrainingArguments, Dict]:
    """
    Creates all training parameters i.e parameters for SFTTrainer and trainer_arguments
    """
    training_arguments = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        optim=config.optimizer,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        logging_dir=config.logging_dir,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        fp16=config.fp16,
        bf16=config.bf16,
        max_grad_norm=config.max_grad_norm,
        max_steps=config.max_steps,
        warmup_ratio=config.warmup_ratio,
        group_by_length=config.group_by_length,
        lr_scheduler_type=config.lr_scheduler_type,
        report_to=config.report_to,
        remove_unused_columns=config.remove_unused_columns,
    )

    sft_trainer_args = {'dataset_text_field': config.dataset_text_field,
        'max_seq_length': config.max_seq_length,
        'packing': config.packing,
        'remove_unused_columns':config.remove_unused_columns,}
    
    return training_arguments, sft_trainer_args
