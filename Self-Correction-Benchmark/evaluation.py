import sys
sys.path.append('/mnt/yuanzenghui/Self-Correction-Benchmark')
from utils.process_config import open_config
from model import create_model
from task import create_task
from method import create_method
from tqdm import tqdm
import argparse
import os
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, default='/mnt/yuanzenghui/Self-Correction-Benchmark/config/model_config/api_llama_config.json')
    parser.add_argument('--task_config', type=str, default='/mnt/yuanzenghui/Self-Correction-Benchmark/config/task_config/gsm.json')
    parser.add_argument('--method', type=str, default='rci')
    parser.add_argument('--prompting_style', type=str, default='zero-shot-cot')
    args = parser.parse_args()
    return args


def load_model(args):
    model_config = open_config(config_path=args.model_config)
    model = create_model(model_config)
    return model

def load_data(args):
    task_config = open_config(config_path=args.task_config)
    task = create_task(task_config)
    data = task.get_data()
    return task, data

def load_method(args):
    method = create_method(args, model, task)
    return method


if "__main__" == __name__:
    
    args = parse_args()
    model = load_model(args)
    task, data = load_data(args)
    method = load_method(args)

    '''Create a directory to store the results'''
    results_path = f'./results/{args.method}/{task.task_name}/'
    results_file = f'{results_path}/{model.name}_results.json'
    dic = os.path.dirname(results_file)
    if not os.path.exists(dic):
        os.makedirs(dic)
    with open(results_file, 'w') as f:
        json.dump({}, f)
    print(f"Make a new file {results_file} to save the result.")

    final_results = []
    correct_number = 0
    total_number = 0
    for i in tqdm(range(len(data['question']))):
        total_number = total_number + 1
        record = method(data['question'][i], data['final_answer'][i])
        final_results.append(record)
        if record['correct'] == True:
            correct_number = correct_number + 1
    
    ACC = correct_number/total_number
    print(f"Method: {args.method}\nTask: {task.task_name}\nModel: {model.name}\n Accuracy: {ACC:.2f}")        
            
        
    '''Save the results to a json file'''
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=4)
    print(f"Results saved to {results_file}")
    
    
    