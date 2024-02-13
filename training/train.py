import os
import logging
import argparse
from dotenv import load_dotenv

from utils.params import (
    SCRIPT_ARGUMENT_PARAMETERS_TRAIN)

from utils.helpers import (load_model,
                    load_tokenizer,
                    load_hf_dataset)

from utils.config import (CustomTrainConfig,
                    CustomPEFTConfig,
                    CustomBNBConfig,
                    create_peft_config, 
                    create_training_config,
                    create_bnb_config)

from datasets import DatasetDict

from peft import (
    prepare_model_for_kbit_training,
    get_peft_model
)

from trl import SFTTrainer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
load_dotenv()


def train(train_config:CustomTrainConfig,
          bnb_config:CustomBNBConfig,
          peft_config:CustomPEFTConfig)->None:
    """
    This function trains the model
    """
    logger.info(f"Loading {train_config.input_data_path} data with {train_config.test_size} split, storing into {train_config.output_dir}")
    

    dataset = load_hf_dataset(input_data_path=train_config.input_data_path)
    split_dataset = dataset.train_test_split(test_size=train_config.test_size)
    datasets = DatasetDict({
        'train': split_dataset['train'],
        'validation': split_dataset['test']
    })

    bnb_config = create_bnb_config(config=bnb_config)
    peft_config = create_peft_config(config=peft_config)
    training_configuration = create_training_config(config=train_config)


    model = load_model(model_config=train_config,bnb_config=bnb_config)
    tokenizer = load_tokenizer(config=train_config)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    
    training_arguments = training_configuration[0]
    sft_trainer_arguments = training_configuration[1]
    
    dataset_text_field = sft_trainer_arguments['dataset_text_field']
    max_seq_length = sft_trainer_arguments['max_seq_length']
    packing = sft_trainer_arguments['packing']

    
    trainer = SFTTrainer(
        model=model,
        train_dataset=datasets['train'],
        eval_dataset=datasets['validation'],
        peft_config=peft_config,
        dataset_text_field=dataset_text_field,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=packing,
    )
    
    logger.info(f"Training model {train_config.base_model}")
    trainer.train()
   
    trainer.model.save_pretrained(train_config.output_dir)
     

def main():
    """
    Runs all the code
    """
    def create_parser():
        parser = argparse.ArgumentParser(description='PEFT Finetuning Script')
        for param, param_type in SCRIPT_ARGUMENT_PARAMETERS_TRAIN:
            parser.add_argument(f"--{param}", type=param_type)

        args, _ = parser.parse_known_args()  # Parse only the known arguments
        
        # Filter out parameters that have not been updated
        updated_args = {param: getattr(args, param) for param in vars(args) if getattr(args, param) is not None}
    
        return argparse.Namespace(**updated_args)
    
    def validate_arguments(config_class, args):
        # Get the attributes of the configuration class
        valid_attributes = set(vars(config_class))

        # Filter out parameters that are not relevant to the configuration class
        filtered_args = {k: v for k, v in vars(args).items() if k in valid_attributes}

        return argparse.Namespace(**filtered_args)

    args = create_parser()
    
    # Get the relevant arguments for each configuration class
    train_config_args = validate_arguments(CustomTrainConfig, args)
    bnb_config_args = validate_arguments(CustomBNBConfig, args)
    peft_config_args = validate_arguments(CustomPEFTConfig, args)

    train_config = CustomTrainConfig(**vars(train_config_args))
    bnb_config = CustomBNBConfig(**vars(bnb_config_args))
    peft_config = CustomPEFTConfig(**vars(peft_config_args))

    train(train_config=train_config,
          bnb_config=bnb_config,
          peft_config=peft_config)

if __name__ == "__main__":
    main()
