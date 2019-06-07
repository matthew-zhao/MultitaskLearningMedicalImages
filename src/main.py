import torch
import pretrainedmodels
import os
import pandas as pd

import time, copy
from torchnet import meter
import types
import matplotlib.pyplot as plt
import argparse

from dataloader import MURALoader
from agent import MultiTaskSeparateAgent

def get_study_level_data(study_type, base_dir, data_cat):
    """
    Returns a dict, with keys 'train' and 'valid' and respective values as study level dataframes, 
    these dataframes contain three columns 'Path', 'Count', 'Label'
    Args:
        study_type (string): one of the seven study type folder names in 'train/valid/test' dataset 
    """
    study_data = {}
    study_label = {'positive': 1, 'negative': 0}
    for phase in data_cat:
        BASE_DIR = os.path.join(base_dir, '%s/%s/' % (phase, study_type))
        patients = list(os.walk(BASE_DIR))[0][1] # list of patient folder names
        study_data[phase] = pd.DataFrame(columns=['Path', 'Count', 'Label'])
        i = 0
        for patient in patients: # for each patient folder
            for study in os.listdir(BASE_DIR + patient): # for each study in that patient folder
                label = study_label[study.split('_')[1]] # get label 0 or 1
                path = BASE_DIR + patient + '/' + study + '/' # path to this study
                study_data[phase].loc[i] = [path, len(os.listdir(path)), label] # add new row
                i+=1
    return study_data

def train_and_evaluate_model(pretrained_model, num_phases, batch_size, num_classes, input_size, base_dir, 
        num_minibatches, sample_with_replacement, study_type):
    model_name = pretrained_model
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')

    data_cat = ['train', 'valid'] # data categories

    study_types = [folder for folder in os.listdir(os.path.join(base_dir, 'train'))]

    if study_type:
        data_task_list = [get_study_level_data(study_type, base_dir, data_cat)]
    else:
        data_task_list = [get_study_level_data(study_type, base_dir, data_cat) for study_type in study_types]

    train_data = MURALoader(data_task_list, batch_size=batch_size, num_minibatches=num_minibatches, train=True, drop_last=True, 
        rescale_size=input_size, sample_with_replacement=sample_with_replacement)
    test_data = MURALoader(data_task_list, batch_size=batch_size, num_minibatches=num_minibatches, train=False, drop_last=False, 
        rescale_size=input_size, sample_with_replacement=sample_with_replacement)

    num_classes_multi = train_data.num_classes_multi(num_tasks=1 if study_type else len(study_types))
    num_channels = train_data.num_channels

    agent = MultiTaskSeparateAgent(num_classes=num_classes_multi, model=model)
    agent.train(train_data=train_data,
                    test_data=test_data,
                    num_phases=num_phases,
                    save_history=True,
                    save_path=os.path.join(base_dir, '..'),
                    verbose=True
                )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model', '-w', default='densenet121', help='type of torchvision model used for pretrained weights')
    parser.add_argument('--num_phases', '-p', default=5, help='number of phases to train on')
    parser.add_argument('--batch_size', '-b', default=16, help='set the batch size')
    parser.add_argument('--num_classes', '-c',default=2, help='the number of classes each task has')
    parser.add_argument('--input_size', '-i', default=224, help='the size of the images to rescale to')
    parser.add_argument('--base_dir', '-d', required=True, help='directory in which train and valid folders are')
    parser.add_argument('--num_minibatches', '-m', default=5, help='number of minibatches per phase')
    parser.add_argument('--sample_with_replacement', '-r', action='store_true')
    parser.add_argument('--study_type', '-t', default=None, help='If specified, we will only train a single task model on that study')

    args = parser.parse_args()

    func_arguments = {}
    for (key, value) in vars(args).items():
        if key == 'sample_with_replacement':
            if args.sample_with_replacement:
                func_arguments['sample_with_replacement'] = True
            else:
                func_arguments['sample_with_replacement'] = False
        else:
            func_arguments[key] = value

    train_and_evaluate_model(func_arguments['pretrained_model'], func_arguments['num_phases'], func_arguments['batch_size'], 
        func_arguments['num_classes'], func_arguments['input_size'], func_arguments['base_dir'], func_arguments['num_minibatches'],
        func_arguments['sample_with_replacement'], func_arguments['study_type'])

if __name__ == "__main__":
    main()




