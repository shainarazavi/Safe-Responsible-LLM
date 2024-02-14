This repository contains code for training, testing and evaluating Safe-LLM.


## Overview 
The content of the folders in this repo is as follows :

**Datasets**: The Content-Moderation-Dataset training and Counterfactual testset are provided.
We also provide guideline for LLM based annotation. - This contains subset(50,500 sample) datasets for testing iterations of desired prompts, and larger datasets(7000, 16000 sample) for training the models in line with defined annotation format, and prompt format defined in the default parameters file(params.py).

- **generate_data** - This contains code for machine-augmented data annotation using OpenAI API's and our defined prompts. Given the LLM's ability to annotate both accurately and quickly, we provide these guidelines for annotating our schema through LLMs. The script uses either default parameters in the params.py file , or parameters passed in the terminal as arguments. 

- **training** - contains code for finetuning the model based on the data in the datasets folder. Our default parametrs are defined in the params.py file, but a user can pass in any other parameters as an argument to the script. The train.py file should save the trained adapter model within the directory the code is stored in. 

- **testing** - contains code for running inference on the merged (base + adapter model), and storing the results in the datasets folder. 

- **utils** - contains utility code i.e configs.py - configuration dataclasses for moving parameters across methods; helpers.py - helper functions for loading files, merging models e.t.c, params.py - default parameters for training our model that can be overwritten by defining arguments for respective scripts. 

