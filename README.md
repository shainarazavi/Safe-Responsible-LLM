
# Safe-Responsible-LLM

This repository contains code for training, testing, and evaluating the Safe-Responsible-LLM.

## Overview

The content of the folders in this repository is as follows:

### Datasets
- **Content-Moderation-Dataset**: Includes the training dataset and the Counterfactual test set.
- **LLM-Based Annotation Guideline**: Provides datasets for testing iterations of desired prompts and a larger dataset for training models in line with the defined annotation and prompt format (as specified in the `params.py` file).

### Folders

- **generate_data**: Contains code for machine-augmented data annotation using OpenAI APIs and our defined prompts. Given the LLM's ability to annotate both accurately and quickly, these guidelines help annotate our schema through LLMs. The script uses either default parameters from the `params.py` file or parameters passed as terminal arguments.

- **training**: Contains code for fine-tuning the model based on the data in the `datasets` folder. The default parameters are defined in the `params.py` file, but users can pass other parameters as arguments to the script. The `train.py` file saves the trained adapter model within the directory where the code is stored.

- **testing**: Contains code for running inference on the merged (base + adapter) model and storing the results in the `datasets` folder.

- **utils**: Contains utility code, including:
  - `configs.py`: Configuration dataclasses for moving parameters across methods.
  - `helpers.py`: Helper functions for loading files, merging models, etc.
  - `params.py`: Default parameters for training our model, which can be overwritten by defining arguments for the respective scripts.

### Dataset
You can access the dataset here: will be shared upon release.

### Citation
If you use this work, please cite us:
```
@article{Safe-Responsible-LLM,
  title={Safe and Responsible Large Language Model : Can We Balance Bias Reduction and Language Understanding in Large Language Models?},
  author={Shaina Raza∗, Oluwanifemi Bamgbosea, Shardul Ghugea, Fatemeh Tavakolia, Deepak John
Rejic, Syed Raza Bashir},
  journal={arXiv preprint arXiv:2404.01399},
  year={2024}
}
```
Paper link: [https://arxiv.org/abs/2404.01399](https://arxiv.org/abs/2404.01399)

---

