import csv
import os
import pandas as pd
from deepcancer.tool import set_vs_argument,set_log,load_model
from rdkit import Chem
from deepcancer.fpgnn.data.utils import get_data_from_smiles
from deepcancer.fpgnn.train import predict

def VS(args,log):
    info = log.info
    #1.get SMILES
    if str(args.molecule)[-4:] == '.csv':
        df_mol = pd.read_csv(args.molecule,encoding='utf-8',low_memory=False)
        df_mol.columns = ['smiles']
    else:
        info(f'Please check the name and format of the file containing the molecules expected for predicting')
        return None
    # check whether it's empty list
    if df_mol.shape[0] == 0:
        info(f'Please enter smile(s) or upload smile file')
        return None
    def smi2mol(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            info(f'Smile "{smi}" is invalid')
            raise ValueError('Find invalid smiles in CSV file')
        else:
            return mol
    df_mol['mol'] = df_mol['smiles'].apply(smi2mol)
    smiles_data = get_data_from_smiles(df_mol['smiles'].tolist(), skip_invalid_smiles=False)


    #2.load models
    model_name = args.model
    # if model_name == 'all':
    #     info(f'Please enter the name of a cell line or a target.')
    #     return None
    category = args.category
    models_dict = load_model(category,model_name)
    #3.predict
    predict_result = {'SMILES':df_mol['smiles'].tolist()}
    for model_name_each in models_dict:
        pre_y = predict(model=models_dict[model_name_each], data=smiles_data, batch_size=1, scaler=None)
        pre_y = [round(y[0], ndigits=3) for y in pre_y]
        predict_result.update({model_name_each:pre_y})
    #4.save file
    df_res = pd.DataFrame(predict_result,index=None)
    df_res.to_csv(os.path.join(args.save_path,'VSresult.csv'),index=False)







if __name__=='__main__':
    args = set_vs_argument()
    log = set_log('Virtual_Screen',args.log_path)
    VS(args,log)
    