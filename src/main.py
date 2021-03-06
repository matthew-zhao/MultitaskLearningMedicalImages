import torch
import torch.nn as nn
from torch.autograd import Variable
import pretrainedmodels
import os
import pandas as pd
import numpy as np

import time, copy
from torchnet import meter
import types
import matplotlib.pyplot as plt
import argparse

from dataloader import RadiographLoader
from agent import MultiTaskSeparateAgent
from loss import MaskedBCEWithLogitsLoss

STRATEGIES_USING_BCE = set([
    "u_ones",
    "u_zeros"
])

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
        if study_type == "XR":
            BASE_DIR = os.path.join(base_dir, '%s/' % (phase))
            patients = [os.path.join(study_type, patient) for study_type in os.listdir(BASE_DIR) for patient in os.listdir(BASE_DIR + study_type)]
        else:
            BASE_DIR = os.path.join(base_dir, '%s/%s/' % (phase, study_type))
            patients = os.listdir(BASE_DIR) # list of patient folder names
        study_data[phase] = pd.DataFrame(columns=['Path', 'Count', 'Label'])
        i = 0
        for patient in patients: # for each patient folder
            for study in os.listdir(BASE_DIR + patient): # for each study in that patient folder
                label = study_label[study.split('_')[1]] # get label 0 or 1
                path = BASE_DIR + patient + '/' + study + '/' # path to this study
                dir_wo_hidden = [file for file in os.listdir(path) if not file.startswith(".")]
                study_data[phase].loc[i] = [path, len(dir_wo_hidden), label] # add new row
                i+=1
    return study_type, study_data

def get_chexpert_data(base_dir, data_cat):
    """
    Returns a dict, with keys 'train' and 'valid' and respective values as study level dataframes,
    these dataframes contain three columns 'Path', 'Count', 'Label'
    Args:
        study_type (string): one of the seven study type folder names in 'train/valid/test' dataset
    """
    study_data = {}
    parent_dir_of_base_dir = os.path.dirname(os.path.dirname(base_dir)) if base_dir.endswith('/') else os.path.dirname(base_dir)
    for phase in data_cat:
        csv_file = os.path.join(base_dir, phase + '.csv')
        labeled_studies = pd.read_csv(csv_file)
        parent_dirs = []
        new_paths = []
        for path in labeled_studies['Path']:
            path_parent_dir = os.path.dirname(path)
            parent_dirs.append(path_parent_dir)
            new_paths.append(os.path.join(parent_dir_of_base_dir,path))
        levels_dict = {_: i for i, _ in enumerate(np.unique(parent_dirs))}
        levels = [levels_dict[_] for _ in parent_dirs]
        labeled_studies['Level'] = levels
        labeled_studies['Path'] = new_paths
        study_data[phase] = labeled_studies
    return "chexpert", study_data

def n_p(x):
    '''convert numpy float to Variable tensor float'''
    return Variable(torch.cuda.FloatTensor([x]), requires_grad=False)

def get_count(df, cat):
    '''
    Returns number of images in a study type dataframe which are of abnormal or normal
    Args:
    df -- dataframe
    cat -- category, "positive" for abnormal and "negative" for normal
    '''
    return df[df['Path'].str.contains(cat)]['Count'].sum()

def train_and_evaluate_model(pretrained_model, num_phases, num_head_phases, batch_size, num_classes, input_size, base_dir, 
        second_base_dir, num_minibatches, sample_with_replacement, study_type, uncertainty_strategy):
    model_name = pretrained_model
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet', drop_rate=0.2)

    data_cat = ['train', 'valid'] # data categories

    data_task_list = []
    if base_dir:
        study_types = [folder for folder in os.listdir(os.path.join(base_dir, 'train'))]
        if study_type:
            data_task_list = [get_study_level_data(study_type, base_dir, data_cat)]
        else:
            data_task_list = [get_study_level_data(study_type, base_dir, data_cat) for study_type in study_types]

    if second_base_dir:
        data_task_list.append(get_chexpert_data(second_base_dir, data_cat))

    train_data = RadiographLoader(data_task_list, batch_size=batch_size, num_minibatches=num_minibatches, train=True, drop_last=True,
        rescale_size=input_size, sample_with_replacement=sample_with_replacement)
    test_data = RadiographLoader(data_task_list, batch_size=batch_size, num_minibatches=num_minibatches, train=False, drop_last=False,
        rescale_size=input_size, sample_with_replacement=sample_with_replacement)

    num_classes_multi = train_data.num_classes_multi(uncertainty_strategy)
    num_channels = train_data.num_channels

    tais = {study_type: {x: get_count(study_data[x], 'positive') for x in data_cat} for study_type, study_data in data_task_list if study_type != 'chexpert'}
    tnis = {study_type: {x: get_count(study_data[x], 'negative') for x in data_cat} for study_type, study_data in data_task_list if study_type != 'chexpert'}
    Wt0_list = {study_type: {x: (np.log(float(tnis[study_type][x] + tais[study_type][x]) / tnis[study_type][x]) + 1) for x in data_cat} for study_type, study_data in data_task_list if study_type != 'chexpert'}
    Wt1_list = {study_type: {x: (np.log(float(tnis[study_type][x] + tais[study_type][x]) / tais[study_type][x]) + 1) for x in data_cat} for study_type, study_data in data_task_list if study_type != 'chexpert'}
    Wt0_weight = {study_type: {x: n_p(Wt0_list[study_type]['train'] / (Wt0_list[study_type]['train'] + Wt1_list[study_type]['train'])) for x in data_cat} for study_type in Wt0_list}
    Wt1_weight = {study_type: {x: n_p(Wt1_list[study_type]['train'] / (Wt0_list[study_type]['train'] + Wt1_list[study_type]['train'])) for x in data_cat} for study_type in Wt1_list}

    criterions = {study_type: nn.CrossEntropyLoss(weight=torch.cat((Wt0_weight[study_type]['train'], Wt1_weight[study_type]['train']), 0)) for study_type in Wt0_weight}
    if uncertainty_strategy == "u_ignore":
        criterions['chexpert'] = nn.MaskedBCEWithLogitsLoss()
    elif uncertainty_strategy == "u_multiclass" :
        # currently unsupported, don't know how to represent 3 labels in a multi-label multi-hot encoded vector
        criterions['chexpert'] = nn.CrossEntropyLoss()
    elif uncertainty_strategy in STRATEGIES_USING_BCE:
        criterions['chexpert'] = nn.BCEWithLogitsLoss()

    agent = MultiTaskSeparateAgent(num_classes=num_classes_multi, model=model, input_size=input_size, uncertainty_strategy=uncertainty_strategy)
    agent.train_head(criterions=criterions,
                     train_data=train_data,
                     num_head_phases=num_head_phases)
    agent.train(criterions=criterions,
                train_data=train_data,
                test_data=test_data,
                num_phases=num_phases,
                save_history=True,
                verbose=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model', '-w', default='densenet121', help='type of torchvision model used for pretrained weights')
    parser.add_argument('--num_phases', '-p', default=10, help='number of phases to train on')
    parser.add_argument('--num_head_phases', '-f', default=5, help='number of phases to train last classifier layer on')
    parser.add_argument('--batch_size', '-b', default=16, help='set the batch size')
    parser.add_argument('--num_classes', '-c',default=2, help='the number of classes each task has')
    parser.add_argument('--input_size', '-i', default=224, help='the size of the images to rescale to')
    parser.add_argument('--base_dir', '-d', default=None, help='directory in which train and valid folders are')
    parser.add_argument('--second_base_dir', '-d2', default=None, help='optional second directory for which to also run multitask training on')
    parser.add_argument('--num_minibatches', '-m', default=5, help='number of minibatches per phase')
    parser.add_argument('--sample_with_replacement', '-r', action='store_true')
    parser.add_argument('--study_type', '-t', default=None, help='If specified, we will only train a single task model on that study')
    parser.add_argument('--uncertainty_strategy', '-u', default="u_ones", help='for chexpert unlabelled data, how to treat uncertain labels. Default is convert them to positive labels')

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

    train_and_evaluate_model(func_arguments['pretrained_model'], int(func_arguments['num_phases']), int(func_arguments['num_head_phases']), int(func_arguments['batch_size']),
        int(func_arguments['num_classes']), int(func_arguments['input_size']), func_arguments['base_dir'], func_arguments['second_base_dir'], int(func_arguments['num_minibatches']),
        func_arguments['sample_with_replacement'], func_arguments['study_type'], func_arguments['uncertainty_strategy'])

if __name__ == "__main__":
    main()




