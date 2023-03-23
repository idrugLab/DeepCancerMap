# **Environment**

The most important python packages are:
- python == 3.6.7
- pytorch == 1.2.0
- torch == 0.4.1
- tensorboard == 1.13.1
- rdkit == 2019.09.3
- scikit-learn == 0.22.2.post1
- hyperopt == 0.2.5
- numpy == 1.18.2
- pandas == 1.3.0

DeepCancer was packed into such a local version for performing high-throughput calculation without limitation based on users' device. This package realizes the main funcions on the online platform. For using our software package more conveniently, we provide the environment file *<environment.txt>* to install environment directly.

Limited to the uploading data size provided by Github, the files in directory "/deepcancer/files" should be downloaded from our webserver https://deepcancer.idruglab.cn. Due to the naming scheme, the symbol "/" was replaced with "%", which should be noticed in entering the name of cancer cell lines.

---
# **Command**

### **1. Activity Predict**
Use activity_predict.py

Args:
  - category : The category of interested cell lines or targets (choosing among CancerCell, AnticancerTarget and NCI-60). *E.g. CancerCell*
  - model : The name of interest cell line(s) or target(s). You could set as its name or set as "all", also assigning a file containing the name list are all available. *E.g. MDA-MB-361 or all or targets.csv*
  - molecule : The molecule(s) expected for predicting activity, both pasting SMILES and using CSV file containing molecules are available. *E.g. smiles.csv*
  - save_path : The path to store the predicting result file. *E.g. result*
  - log_path : The path to record and save the result of prediction. *E.g. log*

E.g.

`python activity_predict.py  --category CancerCell --model "MDA-MB-361" --molecule "Cc1ccc(cc1)C(=O)Nc3ncc(Cc2cccc(c2)Cl)s3" --save_path "./result/Activitiy_Predict" --log_path log`

### **2. Virtual Screen**
Use virtual_screen.py

Args:
  - category : The category of interested cell line or target (choosing among CancerCell, AnticancerTarget and NCI-60). *E.g. CancerCell*
  - model : The name of the interest cell line or target, assigning by entering its name. *E.g. MDA-MB-361*  
  - molecule : The molecules expected for virtual screen, only using CSV file containing molecules are available. *E.g. smiles.csv*
  - save_path : The path to store the predicting result file. *E.g. result*
  - log_path : The path to record and save the result of prediction. *E.g. log*

E.g.

`python virtual_screen.py  --category CancerCell --model "MDA-MB-361" --molecule smiles.csv --save_path "./result/Virtual_Screen" --log_path log`

### **3. Similarity Search**
Use similarity_search.py

Args:
  - dataset_category : The categoryname of the interested dataset for similarity searching. (choosing among CancerCell, AnticancerTarget and NCI-60). *E.g. CancerCell*
  - molecule : The SMILES of molecule expected for virtual screen, both pasting SMILES and assigning a CSV file are available, but only the first molecule will be token in file. *E.g. smiles.csv*
  - save_path : The path to store the predicting result file. *E.g. result*
  - log_path : The path to record and save the result of prediction. *E.g. log*

E.g.

`python similarity_search.py --dataset_category CancerCell --molecule smiles.csv --fingerprint Morgan --save_path "./result/Similarity" --log_path log`


### **4. Multitask Predict**
Use multitask_predict.py, only multitask on NCI-60 tumor cell lines was supported.

Args:
  - molecule : The molecule(s) expected for predicting activity, both pasting SMILES and using CSV file containing molecules are available. *E.g. smiles.csv*
  - save_path : The path to store the predicting result file. *E.g. result*
  - log_path : The path to record and save the result of prediction. *E.g. log*

E.g.

`python multitask_predict.py --molecule "Cc1ccc(cc1)C(=O)Nc3ncc(Cc2cccc(c2)Cl)s3" --save_path "./result/Multitask_Predict" --log_path log`


---
# **File**
The file used in this programme should be **CSV** format and accord with the following content format(i.e. The header is a must):

```
SMILES
O(C(=O)C(=O)NCC(OC)=O)C
FC1=CNC(=O)NC1=O
...

```

