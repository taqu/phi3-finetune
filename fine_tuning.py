# This code block is importing necessary modules and functions for fine-tuning a language model.

# 'randrange' is a function from the 'random' module that generates a random number within the specified range.
from random import randrange, shuffle

# 'torch' is the PyTorch library, a popular open-source machine learning library for Python.
import torch

# 'load_dataset' is a function from the 'datasets' library by Hugging Face which allows you to load a dataset.
from datasets import load_dataset
from datasets import Dataset
from datasets import DatasetDict

# 'LoraConfig' and 'prepare_model_for_kbit_training' are from the 'peft' library. 
# 'LoraConfig' is used to configure the LoRA (Learning from Random Architecture) model.
# 'prepare_model_for_kbit_training' is a function that prepares a model for k-bit training.
# 'TaskType' contains differenct types of tasks supported by PEFT
# 'PeftModel' base model class for specifying the base Transformer model and configuration to apply a PEFT method to.
from peft import LoraConfig, prepare_model_for_kbit_training, TaskType, PeftModel

# Several classes and functions are imported from the 'transformers' library by Hugging Face.
# 'AutoModelForCausalLM' is a class that provides a generic transformer model for causal language modeling.
# 'AutoTokenizer' is a class that provides a generic tokenizer class.
# 'BitsAndBytesConfig' is a class for configuring the Bits and Bytes optimizer.
# 'TrainingArguments' is a class that defines the arguments used for training a model.
# 'set_seed' is a function that sets the seed for generating random numbers.
# 'pipeline' is a function that creates a pipeline that can process data and make predictions.
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
    pipeline
)

# 'SFTTrainer' is a class from the 'trl' library that provides a trainer for soft fine-tuning.
from trl import SFTTrainer,SFTConfig

# This code block is setting up the configuration for fine-tuning a language model.

# 'model_id' and 'model_name' are the identifiers for the pre-trained model that you want to fine-tune. 
# In this case, it's the 'Phi-3-mini-4k-instruct' model from Microsoft.
# Model Names 
# microsoft/Phi-3-mini-4k-instruct
# microsoft/Phi-3-mini-128k-instruct
# microsoft/Phi-3-small-8k-instruct
# microsoft/Phi-3-small-128k-instruct
# microsoft/Phi-3-medium-4k-instruct
# microsoft/Phi-3-medium-128k-instruct
# microsoft/Phi-3-vision-128k-instruct
# microsoft/Phi-3-mini-4k-instruct-onnx
# microsoft/Phi-3-mini-4k-instruct-onnx-web
# microsoft/Phi-3-mini-128k-instruct-onnx
# microsoft/Phi-3-small-8k-instruct-onnx-cuda
# microsoft/Phi-3-small-128k-instruct-onnx-cuda
# microsoft/Phi-3-medium-4k-instruct-onnx-cpu
# microsoft/Phi-3-medium-4k-instruct-onnx-cuda
# microsoft/Phi-3-medium-4k-instruct-onnx-directml
# microsoft/Phi-3-medium-128k-instruct-onnx-cpu
# microsoft/Phi-3-medium-128k-instruct-onnx-cuda
# microsoft/Phi-3-medium-128k-instruct-onnx-directml
# microsoft/Phi-3-mini-4k-instruct-gguf

model_id = "microsoft/Phi-3.5-mini-instruct"
model_name = "microsoft/Phi-3.5-mini-instruct"

# 'dataset_name' is the identifier for the dataset that you want to use for fine-tuning. 
# In this case, it's the 'python_code_instructions_18k_alpaca' dataset from iamtarun (Ex: iamtarun/python_code_instructions_18k_alpaca).
# Update Dataset Name to your dataset name
dataset_name = "Insert your dataset name here"
dataset_path= "./DoxygenStrings-Short"
dataset_split = "train"

new_model = "Phi-3.5-mini-instruct-doxygen"
hf_model_repo=""

# 'device_map' is a dictionary that maps the model to the GPU device. 
# In this case, the entire model is loaded on GPU 0.
device_map = {"": 0}

# The following are parameters for the LoRA (Learning from Random Architecture) model.

# 'lora_r' is the dimension of the LoRA attention.
lora_r = 16

# 'lora_alpha' is the alpha parameter for LoRA scaling.
lora_alpha = 16

# 'lora_dropout' is the dropout probability for LoRA layers.
lora_dropout = 0.05

# 'target_modules' is a list of the modules in the model that will be replaced with LoRA layers.
target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]

# 'set_seed' is a function that sets the seed for generating random numbers, 
# which is used for reproducibility of the results.
seed = 6700417
set_seed(seed)

# 'login' is a function from the 'huggingface_hub' library that logs you in to the Hugging Face Model Hub using an API token.
from huggingface_hub import login

# 'load_dotenv' is a function from the 'python-dotenv' library that loads environment variables from a .env file.
from dotenv import load_dotenv

# 'os' is a standard Python library that provides functions for interacting with the operating system.
import os
import json
from typing import Union

# Call the 'load_dotenv' function to load the environment variables from the .env file.
load_dotenv()

# Call the 'login' function with the 'HF_HUB_TOKEN' environment variable to log in to the Hugging Face Model Hub.
# 'os.getenv' is a function that gets the value of an environment variable.
login(token=os.getenv("HF_HUB_TOKEN"))

# 'load_dataset' is a function from the 'datasets' library that loads a dataset from the Hugging Face Dataset Hub.
# 'dataset_name' is the name of the dataset to load, and 'dataset_split' is the split of the dataset to load (e.g., 'train', 'test').
dataset = DatasetDict.load_from_disk(dataset_dict_path=dataset_path)

# The 'len' function is used to get the size of the dataset, which is then printed.
# print(dataset)

# 'randrange' is a function from the 'random' module that generates a random number within the specified range.
# Here it's used to select a random example from the dataset, which is then printed.
#print(dataset[randrange(len(dataset['train']))])

# 'tokenizer_id' is set to the 'model_id', which is the identifier for the pre-trained model.
# This assumes that the tokenizer associated with the model has the same identifier as the model.
tokenizer_id = model_id

# 'AutoTokenizer.from_pretrained' is a method that loads a tokenizer from the Hugging Face Model Hub.
# 'tokenizer_id' is passed as an argument to specify which tokenizer to load.
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

# 'tokenizer.padding_side' is a property that specifies which side to pad when the input sequence is shorter than the maximum sequence length.
# Setting it to 'right' means that padding tokens will be added to the right (end) of the sequence.
# This is done to prevent warnings that can occur when the padding side is not explicitly set.
tokenizer.padding_side = 'left'

max_length = 4096

def tokenize(prompt, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < max_length
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()
    return result

value_ratio = 0.25
add_eos_token = False
train_on_inputs = True
templates = {}

def load_templates(template_name: str = "", verbose: bool = False):
    global templates
    if not template_name:
        # Enforce the default here, so the constructor can be called with '' and will not break.
        template_name = "doxygen"
    file_name = os.path.join("templates", f"{template_name}.json")
    if not os.path.exists(file_name):
        raise ValueError(f"Can't read {file_name}")
    with open(file_name) as fp:
        templates = json.load(fp)
    if verbose:
        print(
            f"Using prompt template {template_name}: {templates['description']}"
        )

load_templates(template_name='doxygen')

def generate_prompt(
    instruction: str,
    input: Union[None, str] = None,
    output: Union[None, str] = None,
    verbose: bool = False
) -> str:
    # returns the full prompt from instruction and optional input
    # if a label (=response, =output) is provided, it's also appended.
    if input:
        res = templates["prompt_input"].format(
            instruction=instruction, input=input, answer=output
        )
    else:
        res = templates["prompt_no_input"].format(
            instruction=instruction, answer=output
        )
    if verbose:
        print(res)
    return res

def formatting_prompts_func(data_point):
    full_prompt = generate_prompt(
        data_point["question"],
        data_point["context"],
        data_point["answer"],
    )
    return {"text": full_prompt}

def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(
        data_point["question"],
        data_point["context"],
        data_point["answer"],
    )
    tokenized_full_prompt = tokenize(full_prompt)
    if not train_on_inputs:
        user_prompt = generate_prompt(
            data_point["instruction"], data_point["input"]
        )
        tokenized_user_prompt = tokenize(
            user_prompt, add_eos_token=add_eos_token
        )
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        if add_eos_token:
            user_prompt_len -= 1

        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]  # could be speed up, probably
    return tokenized_full_prompt

train_dataset = dataset["train"].shuffle().map(formatting_prompts_func)
eval_dataset = dataset["test"].shuffle().map(formatting_prompts_func)
print(type(train_dataset))
print(train_dataset)
print(train_dataset['text'][8])
print(type(eval_dataset))
print(eval_dataset)
print(eval_dataset['text'][8])

# 'torch.cuda.is_bf16_supported()' is a function that checks if bfloat16 is supported on the current CUDA device.
# If bfloat16 is supported, 'compute_dtype' is set to 'torch.bfloat16' and 'attn_implementation' is set to 'flash_attention_2'.
if torch.cuda.is_bf16_supported():
    compute_dtype = torch.bfloat16
    attn_implementation = 'flash_attention_2'
# If bfloat16 is not supported, 'compute_dtype' is set to 'torch.float16' and 'attn_implementation' is set to 'sdpa'.
else:
    compute_dtype = torch.float16
    attn_implementation = 'sdpa'

# This line of code is used to print the value of 'attn_implementation', which indicates the chosen attention implementation.
print(attn_implementation)


# 'AutoTokenizer.from_pretrained' is a method that loads a tokenizer from the Hugging Face Model Hub.
# 'model_id' is passed as an argument to specify which tokenizer to load.
# 'trust_remote_code' is set to True to trust the remote code in the tokenizer files.
# 'add_eos_token' is set to True to add an end-of-sentence token to the tokenizer.
# 'use_fast' is set to True to use the fast version of the tokenizer.
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, add_eos_token=True, use_fast=True)

# The padding token is set to the unknown token.
tokenizer.pad_token = tokenizer.unk_token

# The ID of the padding token is set to the ID of the unknown token.
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

# The padding side is set to 'left', meaning that padding tokens will be added to the left (start) of the sequence.
tokenizer.padding_side = 'left'

# 'AutoModelForCausalLM.from_pretrained' is a method that loads a pre-trained model for causal language modeling from the Hugging Face Model Hub.
# 'model_id' is passed as an argument to specify which model to load.
# 'torch_dtype' is set to the compute data type determined earlier.
# 'trust_remote_code' is set to True to trust the remote code in the model files.
# 'device_map' is passed as an argument to specify the device mapping for distributed training.
# 'attn_implementation' is set to the attention implementation determined earlier.
model = AutoModelForCausalLM.from_pretrained(
          model_id, torch_dtype=compute_dtype, trust_remote_code=True, device_map=device_map,
          attn_implementation=attn_implementation
)

# 'TrainingArguments' is a class that holds the arguments for training a model.
# 'output_dir' is the directory where the model and its checkpoints will be saved.
# 'evaluation_strategy' is set to "steps", meaning that evaluation will be performed after a certain number of training steps.
# 'do_eval' is set to True, meaning that evaluation will be performed.
# 'optim' is set to "adamw_torch", meaning that the AdamW optimizer from PyTorch will be used.
# 'per_device_train_batch_size' and 'per_device_eval_batch_size' are set to 8, meaning that the batch size for training and evaluation will be 8 per device.
# 'gradient_accumulation_steps' is set to 4, meaning that gradients will be accumulated over 4 steps before performing a backward/update pass.
# 'log_level' is set to "debug", meaning that all log messages will be printed.
# 'save_strategy' is set to "epoch", meaning that the model will be saved after each epoch.
# 'logging_steps' is set to 100, meaning that log messages will be printed every 100 steps.
# 'learning_rate' is set to 1e-4, which is the learning rate for the optimizer.
# 'fp16' is set to the opposite of whether bfloat16 is supported on the current CUDA device.
# 'bf16' is set to whether bfloat16 is supported on the current CUDA device.
# 'eval_steps' is set to 100, meaning that evaluation will be performed every 100 steps.
# 'num_train_epochs' is set to 3, meaning that the model will be trained for 3 epochs.
# 'warmup_ratio' is set to 0.1, meaning that 10% of the total training steps will be used for the warmup phase.
# 'lr_scheduler_type' is set to "linear", meaning that a linear learning rate scheduler will be used.
# 'report_to' is set to "wandb", meaning that training and evaluation metrics will be reported to Weights & Biases.
# 'seed' is set to 42, which is the seed for the random number generator.

# LoraConfig object is created with the following parameters:
# 'r' (rank of the low-rank approximation) is set to 16,
# 'lora_alpha' (scaling factor) is set to 16,
# 'lora_dropout' dropout probability for Lora layers is set to 0.05,
# 'task_type' (set to TaskType.CAUSAL_LM indicating the task type),
# 'target_modules' (the modules to which LoRA is applied) choosing linear layers except the output layer..


# args = TrainingArguments(
#         output_dir="./phi-3.5-mini-instruct-LoRA",
#         evaluation_strategy="steps",
#         do_eval=True,
#         optim="adamw_torch",
#         per_device_train_batch_size=8,
#         gradient_accumulation_steps=4,
#         per_device_eval_batch_size=8,
#         log_level="debug",
#         save_strategy="epoch",
#         logging_steps=100,
#         learning_rate=1e-4,
#         fp16 = not torch.cuda.is_bf16_supported(),
#         bf16 = torch.cuda.is_bf16_supported(),
#         eval_steps=100,
#         num_train_epochs=1,
#         warmup_ratio=0.1,
#         lr_scheduler_type="linear",
#         report_to="wandb",
#         seed=seed,
# )

args = SFTConfig(
    output_dir="./phi-3.5-mini-instruct-LoRA",
    eval_strategy="steps",
    do_eval=True,
    optim="adamw_torch",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=8,
    log_level="debug",
    save_strategy="epoch",
    logging_steps=100,
    learning_rate=1e-4,
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    eval_steps=100,
    num_train_epochs=1,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    dataset_text_field='text',
    max_seq_length=max_length,
    report_to="wandb",
    seed=seed,
)

peft_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    task_type=TaskType.CAUSAL_LM,
    target_modules=target_modules,
)

# 'import wandb' is used to import the wandb library.
import wandb

# 'wandb.login()' is a method that logs you into your Weights & Biases account.
# If you're not already logged in, it will prompt you to log in.
# Once you're logged in, you can use Weights & Biases to track and visualize your experiments.
wandb.login()

# 'project_name' is set to the name of the project in Weights & Biases.
project_name = "phi-3.5-mini-instruct-LoRA"

# 'wandb.init' is a method that initializes a new Weights & Biases run.
# 'project' is set to 'project_name', meaning that the run will be associated with this project.
# 'name' is set to "phi-3-mini-ft-py-3e", which is the name of the run.
# Each run has a unique name which can be used to identify it in the Weights & Biases dashboard.
wandb.init(project=project_name, name = "phi-3-mini-ft-py-3e")

# 'model' is the model that will be trained.
# 'train_dataset' and 'eval_dataset' are the datasets that will be used for training and evaluation, respectively.
# 'peft_config' is the configuration for peft, which is used for instruction tuning.
# 'dataset_text_field' is set to "text", meaning that the 'text' field of the dataset will be used as the input for the model.
# 'max_seq_length' is set to 512, meaning that the maximum length of the sequences that will be fed to the model is 512 tokens.
# 'tokenizer' is the tokenizer that will be used to tokenize the input text.
# 'args' are the training arguments that were defined earlier.

trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=args,
)


# This code block is used to train the model and save it locally.

# 'trainer.train()' is a method that starts the training of the model.
# It uses the training dataset, evaluation dataset, and training arguments that were provided when the trainer was initialized.
trainer.train()

# 'trainer.save_model()' is a method that saves the trained model locally.
# The model will be saved in the directory specified by 'output_dir' in the training arguments.
trainer.save_model()

