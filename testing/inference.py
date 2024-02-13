import os
import argparse

from utils.helpers import load_tokenizer, load_model, load_adapter, create_hf_dataset_from_csv, format_hf_dataset_with_prompt
from utils.config import CustomInferenceConfig, CustomBNBConfig, create_bnb_config
from utils.params import  SCRIPT_ARGUMENT_PARAMETERS_INFER

import pandas as pd
import torch

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from transformers.pipelines.pt_utils import KeyDataset


def inference_single(inference_config:CustomInferenceConfig,
                     input_text:str,
                     model:AutoModelForCausalLM = None,
                     tokenizer:AutoTokenizer = None)->str:
    """
    Generate Text for a single input prompt
    """

    system_message = inference_config.prompt_dictionary[1]['SYSTEM_MESSAGE']
    instruction = inference_config.prompt_dictionary[1]['INSTRUCTION']

    opening_brace = inference_config.prompt_dictionary[1]['OPENING_BRACE']
    opening_instruction_brace = inference_config.prompt_dictionary[1]['OPENING_INSTRUCTION_BRACE']
    closing_instruction_brace = inference_config.prompt_dictionary[1]['CLOSING_INSTRUCTION_BRACE']

    input_text = ''.join(input_text)
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=inference_config.max_length)
    result = pipe(f"{opening_brace} {system_message} {opening_instruction_brace} {instruction} {input_text} {closing_instruction_brace}")
    output = result[0]['generated_text']
    return output


def inference_csv(inference_config: CustomInferenceConfig,
                 model:AutoModelForCausalLM = None,
                 tokenizer:AutoTokenizer = None)->str:
    """
    Run inference on a batch of text from a csv
    """
    # load csv
    # convert to hf dataset
    # pass hf dataset to tokenizer
    # store results in output 
    output_filename = f"results/{inference_config.input_csv}_output.csv"
    if not os.path.exists(output_filename):
        df = pd.DataFrame(columns=['debiased_text'])
        df.to_csv(output_filename, index=False)
    
    prompt_format_definition = inference_config.prompt_dictionary[1]
    
    dataset = create_hf_dataset_from_csv(input_data_path=inference_config.input_csv,
                                         dataset_column="train") # TODO: fix this as it should be infer , but this works 
    
    dataset = dataset.map(lambda row: format_hf_dataset_with_prompt(row,prompt_format_definition))

    # pipe = pipeline("text-generation",
    #                 model=model,
    #                 tokenizer=tokenizer,
    #                 torch_dtype=torch.float16,
    #                 device_map="auto",
    #                 )
    
    # print("myds",dataset)
    
    # for out in pipe(KeyDataset(dataset, "text")): #text is the output column where the formatted prompt are stored
    #     print("out",out)
 
    #     # debiased_text = result[0]['generated_text']
    #     # df = pd.DataFrame({'debiased_text': [debiased_text]})
    #     # df.to_csv(output_filename, mode='a', header=False, index=False)

    debiased_text_llama = []
    count = 0
    for row in dataset:
        biased_text_formatted = row['text'] 
        encoded = tokenizer.encode(biased_text_formatted, return_tensors="pt").to("cuda")
        generate_ids = model.generate(encoded, max_length=len(biased_text_formatted) * 1.5) 
        generate_ids = generate_ids[:, encoded.shape[1]:]
        debiased_text = tokenizer.batch_decode(generate_ids)
        debiased_text_llama.append(debiased_text)
        count += 1

    dataset = dataset.add_column(name="Debiased_Text_LLaMA", column=debiased_text_llama)
    
    dataset.to_csv(output_filename, index=False)

    return dataset


def infer(inference_config:CustomInferenceConfig,bnb_config:CustomBNBConfig,input_text:str)->None:
    """
    Carry out inference on a model
    """
    adapter_model = inference_config.adapter_model

    bnb_config = create_bnb_config(config=bnb_config)

    tokenizer = load_tokenizer(config=inference_config)
    base_model = load_model(model_config=inference_config, bnb_config=bnb_config, inference=True)
    merged_model = load_adapter(base_model,adapter_model)

    if inference_config.single_inference:
        results = inference_single(inference_config=inference_config,
                         input_text=input_text,
                         model=merged_model,
                         tokenizer=tokenizer,
                         )
    elif inference_config.input_csv is not None:
        results = inference_csv(inference_config=inference_config,
                         model=merged_model,
                         tokenizer=tokenizer)
        
    return results


def main():
    def create_parser():
        parser = argparse.ArgumentParser(description='Inference Script')
        for param, param_type in SCRIPT_ARGUMENT_PARAMETERS_INFER:
            parser.add_argument(f"--{param}", type=param_type)

        args, _ = parser.parse_known_args()  # Parse only the known arguments

        # Filter out parameters that have not been updated
        updated_args = {param: getattr(args, param) for param in vars(args) if getattr(args, param) is not None}
        return args, updated_args
    
    def validate_arguments(config_class, args):
        # Get the attributes of the configuration class
        valid_attributes = set(vars(config_class))
        args_namespace = argparse.Namespace(**args)

        # Filter out parameters that are not relevant to the configuration class
        filtered_args = {k: v for k, v in vars(args_namespace).items() if k in valid_attributes}

        return argparse.Namespace(**filtered_args)
    
    args, updated_args = create_parser()
    print("ARGU",args)

    bnb_config_args = validate_arguments(CustomBNBConfig, updated_args)
    inference_config_args = validate_arguments(CustomInferenceConfig, updated_args)

    bnb_config = CustomBNBConfig(**vars(bnb_config_args))
    inference_config = CustomInferenceConfig(**vars(inference_config_args))

    results = infer(inference_config=inference_config,bnb_config=bnb_config,input_text=args.input_text)
    print("results",results)


if __name__ == "__main__":
    main()