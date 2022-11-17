import csv
import os
import pandas as pd
import pickle
from deepcancer.tool import set_similar_argument,set_log,load_model
from rdkit import DataStructs, Chem
from rdkit.Chem import AllChem
def searching(args,log):
    info=log.info
    #1.get molecules(submitting and storing)
    
    info(f'Loading molecule(s)')
    if str(args.molecule)[-4:] == '.csv':
        df_mol = pd.read_csv(args.molecule,encoding='utf-8',low_memory=False)
        smile_list = [df_mol[df_mol.columns[0]].to_list()[0]]
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

    #2.ensure the fingerprint and do featurize
    fingerprint = args.fingerprint
    if fingerprint == 'MACCS':
        submit_fp = AllChem.GetMACCSKeysFingerprint(Chem.MolFromSmiles(smile))
        fp_col = 'fingerprint_MACCS'
    elif fingerprint == 'Morgan':
        submit_fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smile), 2, nBits=1024)
        fp_col = 'fingerprint_Morgan'
    elif fingerprint == 'AtomPairs':
        submit_fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(Chem.MolFromSmiles(smile))
        fp_col = 'fingerprint_AtomPairs'
    else:
        return None


    df_similar = pickle.load(open('./deepcancer/files/dataset/df_similar.pkl', 'rb'))
    if args.dataset_category == 'CancerCell':
        df_similar = df_similar.dropna(subset=['cell'])
        df_similar = df_similar[df_similar['cell'] != 'nan']
        target_col = 'cell'
    elif args.dataset_category == 'AnticancerTarget':
        df_similar = df_similar.dropna(subset=['target'])
        df_similar = df_similar[df_similar['target'] != 'nan']
        target_col = 'target'
    elif args.dataset_category == 'NCI-60':
        df_similar = df_similar.dropna(subset=['nci60'])
        df_similar = df_similar[df_similar['nci60'] != 'nan']
        target_col = 'nci60'
    else:
        return None
    #3.calculating similarity

    df_similar['compute_similar'] = df_similar[fp_col].apply(
        lambda x: DataStructs.FingerprintSimilarity(submit_fp, x, metric=DataStructs.DiceSimilarity))
    df_similar[target_col+"_info"] = df_similar[target_col]
    def join_target(target_dict_list):
        target=''
        if type(target_dict_list) == list:
            for target_dict in target_dict_list:
                target += target_dict['name']
            return target
        else:
            target = ','.join(target_dict_list.split('@'))
            return target
    df_similar[target_col] = df_similar[target_col].apply(join_target)

    #4.save result to file
    df_similar = df_similar[['smiles',target_col,'compute_similar']]
    df_similar.to_csv(os.path.join(args.save_path,'SimilaritySearchResult.csv'),index=False)


if __name__=='__main__':
    args = set_similar_argument()
    log = set_log('Similarity_Search',args.log_path)
    searching(args,log)
    