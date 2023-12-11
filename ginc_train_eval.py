import sys
sys.path.append('/home/jupyter/project/beyond-scale-language-data-diversity/src/diversity')
sys.path.append('/home/jupyter/project/beyond-scale-language-data-diversity/src')

from copy import deepcopy
from pathlib import Path
import os
import argparse
import json
import numpy as np
import torch
import math

from task2vec1 import Task2Vec
import task_similarity

from datasets import load_dataset
from transformers import AutoConfig, GPT2LMHeadModel, PreTrainedTokenizerFast


def ginc_tokenizer(train_file, tokenizer_name,block_size=512,seed=42):

    data_files = {}
    data_files["train"] = train_file
    
    extension = (
        train_file.split(".")[-1]
    )
    if extension == "txt":
        extension = "text"
        
    ds = load_dataset(extension, data_files=data_files, split="train")

    eot = '[endoftext]'
    tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_name,
            bos_token=eot,
            eos_token=eot,
            unk_token=eot)

    column_names = ds.column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = ds.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        load_from_cache_file=False,
    )
    
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: torch.from_numpy(np.array([t[i : i + block_size] for i in range(0, total_length, block_size)]))
            for k, t in concatenated_examples.items()
        }
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        load_from_cache_file=False,
    ).with_format("torch")
    
    
    return lm_datasets, tokenizer

def model_training(model,lm_datasets,classifier_opts,batch_size=128,seed=42):

    num_tasks = math.ceil(len(lm_datasets) / batch_size)
    #num_tasks = min(num_tasks, 2)
    print("LEN_LM_DATASETS:", len(lm_datasets))
    print("NUM_TASKS: ", num_tasks)
    embeddings, losses = [], []
    for task_num in range(num_tasks):
        print(f'--> {task_num=}\n')
        seed = seed + task_num
        
        end_index = task_num * args.batch_size + args.batch_size
        if end_index > len(lm_datasets):
            end_index = len(lm_datasets)
        tokenized_task_dataset = lm_datasets.select(range(task_num * args.batch_size, end_index))

        #probe_network = model
        embedding, loss = Task2Vec(model, classifier_opts=classifier_opts).embed(tokenized_task_dataset)
        print(f'{embedding.hessian.shape=}')
        embeddings.append(embedding)
        if loss is not None:
            print("LOSS HERE: ", loss)
            losses.append(loss)

        num_tasks_processed = task_num + 1

    distance_matrix: np.ndarray = task_similarity.pdist(embeddings, distance='cosine')
    div_coeff, div_coeff_ci = task_similarity.stats_of_distance_matrix(distance_matrix)
    

    print(f'{distance_matrix=}')
    #'embeddings': [embed for embed in embeddings],
 
    results: dict = {
                     'distance_matrix': distance_matrix,
                     'losses': [loss for loss in losses],
                     "div_coeff_mu": div_coeff,
                     "div_coeff_var" : div_coeff_ci,
                     "num_tasks": num_tasks}
    #print(results)
    return results

def model_eval(model, lm_datasets,classifier_opts):
    print("########## model evaluation ##########")
    tokenized_task_dataset = lm_datasets
    loss = Task2Vec(model, classifier_opts=classifier_opts).eval_classifier(tokenized_task_dataset)
    return loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./output_dir_1", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--num_tasks", default=None, type=int,
                        help="The number of tasks to sample from data and compute diversity for.")
    parser.add_argument("--finetune", default=True, action='store_true',
                        help="Whether to run finetuning on probe network.")
    parser.add_argument("--pretrained", default=False, action='store_true',
                        help="Whether or not to use a pretrained probe network.")
    parser.add_argument("--break_early", default=False,
                        help="Break after 1 iteration.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--epochs", default=5, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--batch_size", default=128, type=int,
                        help="Batch size per GPU/CPU for training.")
    
    args = parser.parse_args()
    
    ginc_dataset_root = "../ginc-output-repro/data/"
    val_file = '../ginc-output-repro/val/GINC_trans0.1_start10.0_nsymbols13_nvalues12_nslots11_nsamples32_nhmms20000_seed1111/train.json'

    for path in os.listdir(ginc_dataset_root):
        print(path)
        if path == '.ipynb_checkpoints':
            continue
        torch.cuda.empty_cache()
        ginc_dataset_dir_path = os.path.join(ginc_dataset_root, path)
        tokenizer_name = os.path.join(ginc_dataset_dir_path, 'tokenizer.json')
        train_file = os.path.join(ginc_dataset_dir_path, 'train.json')
        print(tokenizer_name)
        print(train_file)
        
        lm_datasets_train, tokenizer = ginc_tokenizer(train_file,tokenizer_name)
       
        
        classifier_opts = {'break_early': args.break_early, "finetune": True, "seed": args.seed, "epochs": args.epochs, "task_batch_size": args.batch_size}

        model = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir=None)
        model.config.vocab_size = tokenizer.vocab_size
        device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.resize_token_embeddings(len(tokenizer))

        results = model_training(model,lm_datasets_train,classifier_opts)
        
        torch.cuda.empty_cache()
        lm_datasets_eval, _ = ginc_tokenizer(val_file,tokenizer_name)
        
        eval_loss = model_eval(model,lm_datasets_eval,classifier_opts)
        

        results['eval_loss'] = eval_loss
        print("EVAL LOSS: ", eval_loss)
        
        if not os.path.exists(args.output_dir): 
            os.makedirs(args.output_dir)
        np.save(os.path.join(args.output_dir, 'results_{}.npy'.format(path)), results)
        
        
        results["distance_matrix"] = results["distance_matrix"].tolist()
        results['losses'] = list(results['losses'])


        with open(os.path.join(args.output_dir, 'results_{}.json'.format(path)), "w") as fp:
            results_str = json.dumps(results) 
            fp.write(results_str)
        
