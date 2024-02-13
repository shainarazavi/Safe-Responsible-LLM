import os
import logging
import argparse
import time

from utils.params import (
    default_input_file_path, 
    default_output_file_path, 
    default_backup_output_file_path,
    default_input_column,
    default_output_column,
    default_completions_model,
    context_prompt_with_bias_analysis as default_prompt,
    SCRIPT_ARGUMENT_PARAMETERS_ANNOTATE
    )

import pandas as pd
import openai
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

openai.api_key = os.environ['OPEN_AI_TOKEN']

def annotate(input_file_path:str = None,
             output_file_path:str = None,
             backup_output_file_path:str = None,
             input_column:str = None,
             output_column:str = None,
             prompt:str = None,
             completions_model:str = None,
             rate_limit_index:int = 5)->None:
    """
    Annotates Data using the defined format
    """
    # TODO - Create a Config class for each of these configs, with the default defined and put in the consts file 
    input_file_path = input_file_path if input_file_path is not None else default_input_file_path
    output_file_path = output_file_path if output_file_path is not None else default_output_file_path
    backup_output_file_path = backup_output_file_path if backup_output_file_path is not None else default_backup_output_file_path
    input_column = input_column if input_column is not None else default_input_column
    output_column = output_column if output_column is not None else default_output_column
    prompt = prompt if prompt is not None else default_prompt 
    completions_model = completions_model if completions_model is not None else default_completions_model
    rate_limit_index = rate_limit_index if rate_limit_index is not None else 5

    logger.info(f"Annotation Starting on {input_file_path}, to store in {output_file_path}, intermediate results in {backup_output_file_path}")
    rate_limit_index = rate_limit_index
    input_data = pd.read_csv(input_file_path)
    input_data[output_column] = ''

    # Check if the CSV file exists
    if os.path.exists(output_file_path):
        # Load the existing CSV file
        input_data = pd.read_csv(output_file_path)
        last_filled_index = input_data.index[input_data[output_column].notnull()].max()
        start_index = last_filled_index + 1 if not pd.isnull(last_filled_index) else 0
    else:
        start_index = 0

    for index, row in input_data.iterrows():
        if index < start_index:
            # Skip the rows that have already been processed and saved
            continue

        if row['label'] == "neutral":
            output = row['text']
        else:
            row_biased_text = row[input_column]
            sample_biased_text = """ '''{}''' """.format(row_biased_text)

            retry = True
            while retry:
                try:
                    completion = openai.ChatCompletion.create(
                        model=completions_model,
                        messages=[
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": sample_biased_text}
                        ]
                    )
                    output = completion['choices'][0]['message']['content']
                    retry = False  # Request succeeded, exit the retry loop
                except Exception as e:
                    logger.info(f"API request failed. Retrying in 30 seconds...")
                    logger.error(f"{str(e)} ")
                    time.sleep(30)  # Wait for 30 seconds before retrying

            logger.info(f"Index: {index} processed")


        # Store the result
        input_data.at[index, output_column] = output

        rate_limit_index += 1  # Increment rate limit index
        # Check if rate limit reached and introduce a delay
        if rate_limit_index == 30:
            time.sleep(15)  # Delay for 60 seconds (1 minute)
            logger.info(f"Waiting for Delay....")
            rate_limit_index = 0  # Reset rate limit index
            input_data.to_csv(output_file_path, index=False) # save results at every 30 rows

    # Save the final DataFrame to a new CSV file
    input_data.to_csv(backup_output_file_path, index=False)

def main(args):
    """
    Runs all the code
    """
    annotate(input_file_path=args.input_file_path,
             output_file_path=args.output_file_path,
             backup_output_file_path=args.backup_output_file_path,
             input_column=args.input_column,
             output_column=args.output_column,
             prompt=args.prompt,
             completions_model=args.completions_model)

if __name__ == "__main__":
    def create_parser():
        parser = argparse.ArgumentParser(description='Training Data Generation Script')

        for param, param_type, default in SCRIPT_ARGUMENT_PARAMETERS_ANNOTATE:
            parser.add_argument(f"--{param}", default=default, type=param_type)

        return parser

    args = create_parser().parse_args()
    main(args)