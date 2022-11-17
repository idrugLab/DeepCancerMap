import torch
import multitask.model.fpgnn as md
from multitask.tool.tool import load_data_from_smiles
from deepcancer.tool import set_prediction_argument,set_log
from multitask.train import predict
from rdkit import Chem
import pandas as pd
import os

def predicting(args,log):
    savepath = args.save_path
    info = log.info
    smile_list = []
    #1.get SMILES
    info(f'Loading molecule(s)')
    if str(args.molecule)[-4:] == '.csv':
        df_mol = pd.read_csv(args.molecule,encoding='utf-8',low_memory=False)
        smile_list = df_mol[df_mol.columns[0]].to_list()
        WithFile = True
    elif str(args.molecule)[-4] == '.':
        info(f'Please check the name and format of the file containing the molecules expected for predicting')
        return None
    else:
        smile_list = [args.molecule]
    mol_list = []
    for smile in smile_list:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            info(f'Smile "{smile}" is invalid')
            return None
        else:
            mol_list.append(mol)
    # check whether it's empty list
    if len(smile_list) == 0:
        info(f'Please enter smile(s) or upload smile file')
        return None

    info(f'Loading model(s)')
    # load model
    path = './multitask/model.pt'  #.pt
    state = torch.load(path, map_location=lambda storage, loc: storage)
    args, loaded_state_dict = state['args'], state['state_dict']
    args.cuda = False
    model = md.FPGNN(args)
    model_state_dict = model.state_dict()
    pretrained_state_dict = {}
    for param in loaded_state_dict.keys():
        #pretrained_state_dict[param_name] = loaded_state_dict[param_name]
        if param not in model_state_dict:
            print(f'Parameter is not found: {param}.')
        elif model_state_dict[param].shape != loaded_state_dict[param].shape:
            print(f'Shape of parameter is error: {param}.')
        else:
            pretrained_state_dict[param] = loaded_state_dict[param]
            # print(f'Load parameter: {param}.')
    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(model_state_dict)


    #3.do predict
    smiles_data = load_data_from_smiles(smile_list,args=args)
    test_preds = predict(model=model, data=smiles_data, batch_size=128, scaler=None)

    #4.save result
    info(f'Done! and saving result file to the path you determin')
    df_res = pd.read_csv('./multitask/multitask_info.csv')
    for i in range(len(smile_list)):
        df_res[smile_list[i]]=test_preds[i]
        df_res[smile_list[i]] = df_res[smile_list[i]].apply(lambda x: round(float(x),4))
    df_res.to_csv(os.path.join(savepath,'PredictResult.csv'),index=False)


if __name__=='__main__':
    args = set_prediction_argument()
    log = set_log('Multitask_Predict',args.log_path)
    predicting(args,log)
    