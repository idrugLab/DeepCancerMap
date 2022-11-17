import os
import pandas as pd
from rdkit import Chem
from deepcancer.tool import set_prediction_argument,set_log,load_model
from deepcancer.fpgnn.data.utils import get_data_from_smiles
from deepcancer.fpgnn.train import predict

def predicting(args,log):
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
    smiles_data = get_data_from_smiles(smile_list, skip_invalid_smiles=False)

    info(f'Loading model(s)')
    #2.load models
    model_name = args.model
    category = args.category
    if model_name[-4:] == '.csv':
        df_models = pd.read_csv(model_name)
        model_name = df_models[df_models.columns[0]].to_list()
    models_dict = load_model(category,model_name)
    info(f'Performing prediction')
    #3.do predict
    predict_result = {'SMILES':smile_list}
    for model_name_each in models_dict:
        pre_y = predict(model=models_dict[model_name_each], data=smiles_data, batch_size=1, scaler=None)
        pre_y = [round(y[0], ndigits=3) for y in pre_y]
        predict_result.update({model_name_each:pre_y})
    #4.save file
    info(f'Done! and saving result file to the path you determin')
    df_res = pd.DataFrame(predict_result,index=None)
    df_res.to_csv(os.path.join(args.save_path,'PredictResult.csv'),index=False)







if __name__=='__main__':
    args = set_prediction_argument()
    log = set_log('Activity_Predict',args.log_path)
    predicting(args,log)
    