import os
import csv
import logging
from tqdm import tqdm
import torch
import deepcancer.fpgnn.models.model as md
model_path_main = './deepcancer/files/models_pkl'

def mkdir(path,isdir = True):
    if isdir == False:
        path = os.path.dirname(path)
    if path != '':
        os.makedirs(path, exist_ok = True)
    
def load_model(category,model_name):
    if model_name == 'all':
        models_dict = load_models(category)
        return models_dict
    else:
        if type(model_name) != list:
            model_name = [model_name]
        models_dict = load_single_model(category,model_name)
        return models_dict

def load_models(category):
    models_dict = {}
    model_path = model_path_main+'/'+category
    # 进度条显示
    with tqdm(total=len(os.listdir(model_path))) as pbar:
        pbar.set_description('Loading ' + model_path + " models")
        for x in os.listdir(model_path):
            pbar.update(1)
            state = torch.load(os.path.join(model_path, x), map_location=lambda storage, loc: storage)
            args, loaded_state_dict = state['args'], state['state_dict']
            model = md.build_model(args)
            # load more model args
            model_state_dict = model.state_dict()
            pretrained_state_dict = {}
            for param_name in loaded_state_dict.keys():
                pretrained_state_dict[param_name] = loaded_state_dict[param_name]
            model_state_dict.update(pretrained_state_dict)
            model.load_state_dict(model_state_dict)
            models_dict.update({x[:-3]:model})
    return models_dict

def load_single_model(category,models_name):
    models_dict = {}
    for model_name in models_name:
        model_path = model_path_main + '/' + category + "/" + model_name + ".pt"
        state = torch.load(model_path, map_location=lambda storage, loc: storage)
        args, loaded_state_dict = state['args'], state['state_dict']
        model = md.build_model(args)
        # load more model args
        model_state_dict = model.state_dict()
        pretrained_state_dict = {}
        for param_name in loaded_state_dict.keys():
            pretrained_state_dict[param_name] = loaded_state_dict[param_name]
        model_state_dict.update(pretrained_state_dict)
        model.load_state_dict(model_state_dict)
        # return md.build_model(args)
        models_dict.update({model_name:model})
    return models_dict


def set_log(name,save_path):
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    
    log_stream = logging.StreamHandler()
    log_stream.setLevel(logging.DEBUG)
    log.addHandler(log_stream)
    
    mkdir(save_path)
    
    log_file_d = logging.FileHandler(os.path.join(save_path, 'debug.log'))
    log_file_d.setLevel(logging.DEBUG)
    log.addHandler(log_file_d)
    
    return log