import pandas as pd
import os 

PROMPT_FORMAT_DICT = {
    'INSTRUCTION' : "Debias this text by rephrasing it to be free of bias :",
    'SYSTEM_MESSAGE' : "<<SYS>>You are a text debiasing bot, you take as input a text and you output its debiased version by rephrasing it to be free from any age, gender, political, social or socio-economic biases ,  without any extra outputs<</SYS>>",
    'OPENING_BRACE' : "<s>",
    'CLOSING_BRACE' : "<s/>",
    'OPENING_INSTRUCTION_BRACE' : "[INST]",
    'CLOSING_INSTRUCTION_BRACE' : "[/INST]",
    'RESPONSE_KEY' : " ",
    'END_KEY' : ""
}

TRAIN_PARAM_DICT = {
    # BNB CONFIG
    'use_4bit' : (bool,True),  # Activate 4-bit precision base model loading
    'bnb_4bit_compute_dtype' : (str,"float16"),  # Compute dtype for 4-bit base models : either float16 or bfloat16, bfloat16 is recommended as it produces less nans ** Note bnb_4bit_compute_dtype for merging.
    'bnb_4bit_quant_type': (str,"nf4"),  # Quantization type (fp4 or nf4)
    'use_nested_quant' : (bool,False),  # Activate nested quantization for 4-bit base models (double quantization) 
    # LORA CONFIG
    "lora_r" : (int,64), # LoRA attention dimension
    "lora_alpha" : (int,16), # Alpha parameter for LoRA scaling. This parameter controls the scaling of the low-rank approximation. Higher values might make the approximation more influential in the fine-tuning process, affecting both performance and computational cost.
    "lora_dropout" : (float,0.2), # Dropout probability for LoRA layers. This is the probability that each neuron’s output is set to zero during training, used to prevent overfitting.
    "task_type" : ("str","CAUSAL_LM"),
    "bias": (str,"none"),
    # TRAIN CONFIG
    "test_size": (float,0.3),
    "base_model": (str,'meta-llama/Llama-2-7b-chat-hf'),
    "output_dir":(str,"models/Llama-2-7b-hf-finetuned"),
    "input_data_path": (str,"train_500.csv"),
    "num_train_epochs": (int,1),
    "fp16": (bool,True),
    "bf16": (bool,False),
    "per_device_train_batch_size": (int,4),
    "per_device_eval_batch_size": (int,4),
    "gradient_accumulation_steps": (int,1),
    "gradient_checkpointing": (bool,True),
    "max_grad_norm": (float,0.3),
    "learning_rate": (float,2e-4),
    "weight_decay": (float,0.001),
    "optimizer": (str,"paged_adamw_32bit"),
    "lr_scheduler_type": (str,"constant"),
    "max_steps": (int,-1),
    "warmup_ratio": (float,0.03),
    "group_by_length": (bool,True),
    "save_steps": (int,25),
    "logging_steps": (int,25),
    "max_seq_length": (int,1024),
    "packing": (bool,False),
    "device_map": (dict,{"": 0}),
    "logging_dir": (str,"logs/"),
    # these below are for the SFT Trainer 
    "dataset_text_field": (str,"text"),
    "max_seq_length": (int,512) ,
    "packing" : (bool,False),
    "report_to": (str, "tensorboard"),
    "remove_unused_columns": (bool,False), #investigate why setting this to True causes all data to be deleted, ensure the inputs are actually being created
}


INFER_PARAM_DICT = {
    "base_model": (str, TRAIN_PARAM_DICT['base_model'][1]),
    "adapter_model": (str, TRAIN_PARAM_DICT['output_dir'][1]),
    'use_4bit' : (bool,TRAIN_PARAM_DICT['use_4bit']),  # Activate 4-bit precision base model loading
    'bnb_4bit_compute_dtype' : (str,TRAIN_PARAM_DICT['bnb_4bit_compute_dtype'][1]),  # Compute dtype for 4-bit base models : either float16 or bfloat16, bfloat16 is recommended as it produces less nans ** Note bnb_4bit_compute_dtype for merging.
    'bnb_4bit_quant_type': (str,TRAIN_PARAM_DICT['bnb_4bit_quant_type'][1]),  # Quantization type (fp4 or nf4)
    'use_nested_quant' : (bool,TRAIN_PARAM_DICT['use_nested_quant'][1]),  # Activate nested quantization for 4-bit base models (double quantization) 
    "single_inference": (bool, False),
    "prompt_dictionary": (dict, PROMPT_FORMAT_DICT),
    "input_text": (str, None),
    "input_csv": (str, "infer_50.csv"),
    "max_length": (int, 500),
    "output_dir": (str,TRAIN_PARAM_DICT['output_dir'][1])
}

# SCRIPT_ARGUMENT_PARAMETERS_TRAIN = [
#     (param_name, param_type, PARAM_DICT.get(param_name, (None, None))[1]) for param_name, (param_type, _) in PARAM_DICT.items()
# ]


SCRIPT_ARGUMENT_PARAMETERS_TRAIN = [
    ("base_model", str),
    ("input_data_path", str),
    ("output_dir", str),
    ("test_size", float),
    ("lora_alpha", int),
    ("lora_dropout", int),
    ("lora_r", int),
    ("bias", str),
    ("task_type", str),
    ("use_4bit", bool),
    ("bnb_4bit_quant_type", str),
    ("bnb_4bit_compute_dtype", str),
    ("use_nested_quant", bool),
    ("num_train_epochs", int),
    ("per_device_train_batch_size", int),
    ("gradient_accumulation_steps", float),
    ("optimizer", str),
    ("save_steps", int),
    ("logging_steps", int),
    ("logging_dir", str),
    ("learning_rate", float),
    ("weight_decay", int),
    ("fp16", bool),
    ("bf16", bool),
    ("max_grad_norm", float),
    ("max_steps", int),
    ("warmup_ratio", float),
    ("group_by_length", bool),
    ("lr_scheduler_type", str),
    ("report_to", str),
    ("dataset_text_field", str),
    ("max_seq_length", int),
    ("packing", bool),
    ("remove_unused_columns", bool),
]

SCRIPT_ARGUMENT_PARAMETERS_INFER = [
    ("base_model", str),
    ("adapter_model", str),
    ("use_4bit", bool),
    ("bnb_4bit_quant_type", str),
    ("bnb_4bit_compute_dtype", str),
    ("use_nested_quant", bool),
    ("single_inference", bool),
    ("prompt_dictionary", dict),
    ("input_text", list),
    ("input_csv", list),
    ("max_length", int),
    ("output_dir",int)
]

SCRIPT_ARGUMENT_PARAMETERS_ANNOTATE = [
    ("input_file_path", str, None),
    ("output_file_path", str, None),
    ("backup_output_file_path", str, None),
    ("input_column", str, None),
    ("output_column", str, None),
    ("completions_model", str, None),
    ("prompt", str, None),
]

# ---------------------------------------------------
# Annotation Script Parameters 

default_input_file_path = "path/to/file/.csv"
default_output_file_path = "path/to/file/.csv"
default_backup_output_file_path = "path/to/file/.csv"
default_completions_model = "gpt-3.5-turbo"
default_input_column = "biased_text"
default_output_column = 'debiased_text'

context_prompt = """You are a text de-biasing bot. You have this example of a biased text and its debiased output as context : “Biased Text: Billie EIlish issues apology for mouthing anti-Asian derogatory term in resurfaced video. Debiased Text: Billie EIlish issues apology for using anti-Asian remark in a resurfaced video”.
Use the following step-by-step instructions to respond to users' inputs. Step 1: Read and understand the biased text input provided by the user in triple quotes.
Step 2: Consider the context of the example provided.
Step 3: Identify and consider words with heavy biased(negative or positive sentiment) in the biased text input that can be replaced without losing significant meaning i.e Disgusted to Uncomfortable.
Step 4: Consider potential political, gender,religious,societal, racial and occupational biases present.
Step 5: Output ONLY a Re-phrased debiased text that is based on the example and considerations and is ethical, concise and allowable in a professional setting, but doesnt lose the general informmation of the sentence.
"""

context_prompt_with_bias_analysis = """You are a text de-biasing bot.
Take ex_bias_ip and ex_debias_op as context : ex_bias_ip: Billie Eilish issues apology for mouthing anti-Asian derogatory term in resurfaced video. ex_debias_op: Billie EIlish issues apology for using anti-Asian remark in a resurfaced video”.
Do the step-by-step instructions to respond to inputs.
Step 1: Take the biased_text_input as the text provided by the user in triple quotes.
Step 2: Consider ex_bias_ip and ex_debias_op for context.
Step 3: Identify and consider words with heavy bias(negative or positive sentiment) in biased_text_input that can be replaced without losing significant meaning i.e Disgusted to Uncomfortable, or remove words like Nigger
Step 4: Generate a re-phrased debiased text output based on the examples and considerations.
Step 5: Provide a reason for the bias in the original statement and Explain why the debiased statement isn't biased anymore, label this part of your output as 'A' :
Step 6: Provide the re-phrased debiased output, label this part of your output as 'B' :
Step 7 : ONLY output A and B
"""