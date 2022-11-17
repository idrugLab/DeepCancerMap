from argparse import ArgumentParser, Namespace
from .tool import mkdir

def add_prediction_argument(p):
    p.add_argument('--category',type=str,choices=['CancerCell','AnticancerTarget','NCI-60'],
                    help="Choose a category in ['CancerCell','AnticancerTarget','NCI-60']")
    p.add_argument('--model',type=str,
                    help="Choose cell line(s) or target(s) by entering its name or submitting a CSV file containing the list")
    p.add_argument('--log_path',type=str,default='log',
                   help="The dir of output log file.")
    p.add_argument('--save_path',type=str,default='./result/Activitiy_Precit',
                   help="The dir of storing calculating result file.")
    p.add_argument('--molecule',type=str,
                   help="The molecule(s) expected for predicting activity, both pasting SMILES and using CSV file containing molecules are available")


def add_vs_argument(p):
    p.add_argument('--category',type=str,choices=['CancerCell','AnticancerTarget','NCI-60'],
                    help="Choose a category in ['CancerCell','AnticancerTarget','NCI-60']")
    p.add_argument('--model',type=str,
                    help="Choose a cell line or a target by entering its name")
    p.add_argument('--molecule',type=str,
                   help="The molecule(s) expected for Virtual Screening, determining a CSV file containing molecules")
                   
    p.add_argument('--log_path',type=str,default='log',
                   help="The dir of output log file.")
    p.add_argument('--save_path',type=str,default='./result/Virtual_Screen',
                   help="The dir of storing calculating result file.")
    

def add_similar_argument(p):
    p.add_argument('--dataset_category',type=str,choices=['CancerCell','AnticancerTarget','NCI-60'],
                    help="Choose a category in ['CancerCell','AnticancerTarget','NCI-60']")
    p.add_argument('--molecule',type=str,
                   help="The molecule expected for Similarity Search, pasting the specific SMILES or determining a CSV file containing molecule")
    p.add_argument('--fingerprint',type=str,choices=['Morgan','AtomPairs','MACCS'],
                   help="Select a FingerPrint format to featurize the molecules for calculating similarity (among 'Morgan','AtomPairs','MACCS')")

    p.add_argument('--log_path',type=str,default='log',
                   help="The dir of output log file.")
    p.add_argument('--save_path',type=str,default='./result/Similarity_Search',
                   help="The dir of storing calculating result file.")


def set_prediction_argument():
    p = ArgumentParser()
    add_prediction_argument(p)
    args = p.parse_args()

    # assert args.data_path

    mkdir(args.save_path)

    return args

def set_vs_argument():
    p = ArgumentParser()
    add_vs_argument(p)
    args = p.parse_args()

    mkdir(args.save_path)
    return args

def set_similar_argument():
    p = ArgumentParser()
    add_similar_argument(p)
    args = p.parse_args()

    mkdir(args.save_path)
    return args