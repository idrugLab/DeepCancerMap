import os
from argparse import Namespace
from logging import Logger
from typing import Tuple

import numpy as np

from deepcancer.fpgnn.data.utils import get_task_names
from deepcancer.fpgnn.utils import makedirs
from .run_training import run_training

"""
def function(s:str) -> int:
python3的新特性。s:str意思是，s是形参，str是形参的注解 ->int是返回值的注解
"""


def cross_validate(args: Namespace, logger: Logger = None) -> Tuple[float, float]:
    """k-fold cross validation"""
    info = logger.info if logger is not None else print
    
    #输出到txt
    refile=open('result.txt','a+')
    
    # Initialize relevant variables 初始化相关变量
    init_seed = args.seed
    save_dir = args.save_dir
    task_names = get_task_names(args.data_path)
    data_path=args.data_path
    

    # Run training on different random seeds for each fold 每折对不同的随机种子进行训练
    all_scores = []
    for fold_num in range(args.num_folds):
    #numfolds：进行交叉验证时的折叠数，缺省为1
        info(f'Fold {fold_num}')
        #info函数，logging库里的，创建一条严重级别为info的日志记录。
        #f'Fold {fold_num}' 是python3的新特性，也可用在print()里，比如print(f'Fold {fold_num}')。意思是：打印Fold+{}的内容，{}里的内容是变量fold_num，运行时改变
        args.seed = init_seed + fold_num
        #随机数种子根据折叠数改变
        args.save_dir = os.path.join(save_dir, f'fold_{fold_num}')
        #存储目录也根据折叠数改变
        makedirs(args.save_dir) 
        #运行
        model_scores = run_training(args, logger)
        #model_scores 包括了每个tasks的score，是[a,b,c,...] 
        all_scores.append(model_scores)
    all_scores = np.array(all_scores)
    #all_scores 包括了每折的每个task的 是[[a,b,c,...],[a,b,c,...],...]

    # Report results 报告结果
    info(f'{args.num_folds}-fold cross validation')

    # Report scores for each fold 报告每一折的分数
    for fold_num, scores in enumerate(all_scores):
    #enumerate(a) 将一个可遍历的数据对象（如：列表、元组、字符串等）组合为一个索引序列，同时列出：数据和数据下标。a是list，返回出两个值，一个索引，一个值
    #for a,b in c: 是循环，每一遍都得到a,b的值
        info(f'Seed {init_seed + fold_num} ==> test {args.metric} = {np.nanmean(scores):.6f}')
        #args.metric是检测指标，auc or其他
        if args.show_individual_scores:
            for task_name, score in zip(task_names, scores):
            #zip(a,b)  结果是(a[0],b[0]),(a[1],b[1]),... 返回的是 tuple 可以强制list(zip(a,b))转换成list格式
                info(f'Seed {init_seed + fold_num} ==> test {task_name} {args.metric} = {score:.6f}')

    # Report scores across models 报告各个模型的分数
    avg_scores = np.nanmean(all_scores, axis=1)  # average score for each model across tasks
    #np.nanmean() 沿指定轴计算算术平均值，忽略NaN。返回数组元素的平均值。 axis=0，那么输出矩阵是1行，求每一列的平均；axis=1，输出矩阵是1列，求每一行的平均
    #all_scores 是[[a,b,c,...],[a,b,c,...],...] 计算出avg_scores是[[avg1],[avg2],[avg3],...] avg1是每折的task平均
    mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)
    #np.nanstd()是标准差
    info(f'Overall test {args.metric} = {mean_score:.6f} +/- {std_score:.6f}')
    print(f'Overall test {args.metric} = {mean_score:.6f} +/- {std_score:.6f}',file=refile)
    print('Data_path:',data_path,file=refile)
    print('',file=refile)
    print('',file=refile)
    print('',file=refile)
    refile.close()

    if args.show_individual_scores:
        for task_num, task_name in enumerate(task_names):
            info(f'Overall test {task_name} {args.metric} = '
                 f'{np.nanmean(all_scores[:, task_num]):.6f} +/- {np.nanstd(all_scores[:, task_num]):.6f}')
                 #all_scores[:, task_num]意思是第task_num列   A[ : 2]:表示索引 0至1行（=A[0:2]）；  A[ :, 2]:表示所有行的第【3】列

    return mean_score, std_score
