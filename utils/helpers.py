import logging
import os 
from typing import Union
import torch
from datasets import Dataset, load_dataset

from utils.params import (
    PROMPT_FORMAT_DICT)
from utils.config import CustomTrainConfig, CustomInferenceConfig


from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
    PreTrainedTokenizer,
    logging as transformers_logging,
)
from peft import PeftModel

transformers_logging.set_verbosity_info()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def create_hf_dataset_from_csv(input_data_path: str = None,dataset_column:str = "train",dataset_folder:str = "datasets") -> Dataset:
    """
    Loads CSV file and convert to Huggingface Dataset
    """
    current_script_dir = os.path.dirname(os.path.realpath(__file__))
    datasets_dir = os.path.dirname(current_script_dir)

    # Construct the path to data.csv
    input_data_path = os.path.join(datasets_dir, dataset_folder, input_data_path)

    dataset = load_dataset('csv', data_files=input_data_path)
    return dataset[dataset_column]


def format_hf_dataset_with_prompt(row,prompt_format_definition):
    """
    Creates a row of data formatted in line with predefined prompt sample format
    """

    biased_text_row = row['biased_text']
    debiased_text_row = row['debiased_text']
    
    start = prompt_format_definition['OPENING_BRACE']
    sys_message = prompt_format_definition['SYSTEM_MESSAGE']
    
    instruction_start = prompt_format_definition['OPENING_INSTRUCTION_BRACE']
    prompt_instruction = prompt_format_definition['INSTRUCTION']
    instruction_end = prompt_format_definition['CLOSING_INSTRUCTION_BRACE']
    
    instruction = f"{instruction_start} {prompt_instruction} {biased_text_row} {instruction_end}"
    
    response = debiased_text_row
    end = prompt_format_definition['CLOSING_BRACE']

    parts = [part for part in [start, sys_message, instruction, response, end] if part]

    formatted_prompt = "\n".join(parts)
    print("formatted_prompt",formatted_prompt)

    row["text"] = formatted_prompt
    return row

def load_hf_dataset(input_data_path: str = None,prompt_format_definition:dict = None) -> Dataset:
    """
    Loads the Dataset, Converts to Prompt Format and Returns it 
    """
    prompt_format_definition = prompt_format_definition if prompt_format_definition is not None else PROMPT_FORMAT_DICT
    dataset = create_hf_dataset_from_csv(input_data_path)
    dataset = dataset.map(lambda row: format_hf_dataset_with_prompt(row,prompt_format_definition))
    return dataset


def load_tokenizer(config:Union[CustomTrainConfig,CustomInferenceConfig]) -> PreTrainedTokenizer:
    """
    Load a HuggingFace Tokenizer for the Model
    """
    logger.info(f"Loading {config.base_model} tokenizer ")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "left"
    return tokenizer

def load_model(model_config:CustomTrainConfig,bnb_config:BitsAndBytesConfig,inference:bool = False) -> AutoModelForCausalLM:
    """
    Load a Huggingface model
    """
    
    logger.info(f"Loading {model_config.base_model} model")
    if inference:
        model = AutoModelForCausalLM.from_pretrained(model_config.base_model,
                                                    low_cpu_mem_usage=True,
                                                    return_dict=True,
                                                    torch_dtype=torch.float16,
                                                    device_map={"":0})
    else:
        model = AutoModelForCausalLM.from_pretrained(model_config.base_model,
                                                    quantization_config = bnb_config, 
                                                    use_cache=False)
    
    return model

def load_adapter(base_model: AutoModelForCausalLM,adapter_model: PeftModel):
    """
    Merges a Base model and the Finetuned Adapter Model
    """
    model = PeftModel.from_pretrained(base_model, adapter_model)
    merged_model = model.merge_and_unload()
    return merged_model

def push_to_huggingface(model,tokenizer,hub_name,HF_REPO):
    """
    Pushes a model to a huggingface REPO
    """
    HF_REPO = f"{HF_REPO}/{hub_name}"
    model.push_to_hub(HF_REPO)
    tokenizer.push_to_hub(HF_REPO)
