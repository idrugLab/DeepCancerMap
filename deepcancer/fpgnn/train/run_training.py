import csv
import os
# Tqdm 是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息，用户只需要封装任意的迭代器 tqdm(iterator)
import pickle
from argparse import Namespace
from logging import Logger
from pprint import pformat
from typing import List

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR

from deepcancer.fpgnn.data import StandardScaler
from deepcancer.fpgnn.data.utils import get_class_sizes, get_data, get_task_names, split_data
from deepcancer.fpgnn.models import build_model
from deepcancer.fpgnn.nn_utils import param_count
from deepcancer.fpgnn.utils import build_optimizer, build_lr_scheduler, get_loss_func, get_metric_func, load_checkpoint, \
    makedirs, save_checkpoint
from .evaluate import evaluate, evaluate_predictions
from .predict import predict
from .train import train


def run_training(args: Namespace, logger: Logger = None) -> List[float]:
    """
    Trains a model and returns test scores on the model checkpoint with the highest validation score.

    :param args: Arguments.
    :param logger: Logger.
    :return: A list of ensemble scores for each task.
    """
    """
    训练模型，并在模型检查点上以最高验证分数返回测试分数。

     ：param args：参数。
     ：param logger：记录器。
     ：return：每个任务的合奏乐谱列表。
    """
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    refile=open('result.txt','a+')
    
    # Set GPU
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        #print('????????gpu:',args.gpu) 0
    #print('!!!!!!!!!!!!!!!gpu:',args.gpu) 0

    # Print args
    debug(pformat(vars(args)))

    # Get data
    #此处get_data抛出无效smiles序号，目前暂时关闭
    debug('Loading data')
    args.task_names = get_task_names(args.data_path)
    data = get_data(path=args.data_path, args=args, logger=logger)
    args.num_tasks = data.num_tasks()
    args.features_size = data.features_size()
    debug(f'Number of tasks = {args.num_tasks}')

    # Split data
    debug(f'Splitting data with seed {args.seed}')
    if args.separate_test_path:
        test_data = get_data(path=args.separate_test_path, args=args, features_path=args.separate_test_features_path, logger=logger)
    if args.separate_val_path:
        val_data = get_data(path=args.separate_val_path, args=args, features_path=args.separate_val_features_path, logger=logger)

    if args.separate_val_path and args.separate_test_path:
        train_data = data
    elif args.separate_val_path:
        train_data, _, test_data = split_data(data=data, split_type=args.split_type, sizes=(0.8, 0.2, 0.0), seed=args.seed, args=args, logger=logger)
    elif args.separate_test_path:
        train_data, val_data, _ = split_data(data=data, split_type=args.split_type, sizes=(0.8, 0.2, 0.0), seed=args.seed, args=args, logger=logger)
    else:
        train_data, val_data, test_data = split_data(data=data, split_type=args.split_type, sizes=args.split_sizes, seed=args.seed, args=args, logger=logger)

    if args.dataset_type == 'classification':
        class_sizes = get_class_sizes(data)
        debug('Class sizes')
        print('Class sizes',file=refile)
        for i, task_class_sizes in enumerate(class_sizes):
        #enumerate(a) 将一个可遍历的数据对象（如：列表、元组、字符串等）组合为一个索引序列，同时列出：数据和数据下标。a是list，返回出两个值，一个索引，一个值
            info(f'{args.task_names[i]} '
                  f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')
            print(f'{args.task_names[i]} '
                  f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}',file=refile)

    if args.save_smiles_splits:
    #为每个train/ VAL /test段保存smile，以便以后进行预测。缺省为0
        with open(args.data_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)

            lines_by_smiles = {}
            indices_by_smiles = {}
            for i, line in enumerate(reader):
                smiles = line[0]
                lines_by_smiles[smiles] = line
                indices_by_smiles[smiles] = i

        all_split_indices = []
        for dataset, name in [(train_data, 'train'), (val_data, 'val'), (test_data, 'test')]:
            with open(os.path.join(args.save_dir, name + '_smiles.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['smiles'])
                for smiles in dataset.smiles():
                    writer.writerow([smiles])
            with open(os.path.join(args.save_dir, name + '_full.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for smiles in dataset.smiles():
                    writer.writerow(lines_by_smiles[smiles])
            split_indices = []
            for smiles in dataset.smiles():
                split_indices.append(indices_by_smiles[smiles])
                split_indices = sorted(split_indices)
            all_split_indices.append(split_indices)
        with open(os.path.join(args.save_dir, 'split_indices.pckl'), 'wb') as f:
            pickle.dump(all_split_indices, f)

    if args.features_scaling:
    #关闭功能缩放 缺省为1
        features_scaler = train_data.normalize_features(replace_nan_token=0)
        val_data.normalize_features(features_scaler)
        test_data.normalize_features(features_scaler)
    else:
        features_scaler = None

    args.train_data_size = len(train_data)
    
    debug(f'Total size = {len(data):,} | '
          f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')
    #至此数据准备完毕
    print(f'Total size = {len(data):,} | '
          f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}',file=refile)

    # Initialize scaler and scale training targets by subtracting mean and dividing standard deviation (regression only) 通过减去均值并除以标准偏差来初始化定标器和定标训练目标（仅回归）
    if args.dataset_type == 'regression':
        debug('Fitting scaler')
        train_smiles, train_targets = train_data.smiles(), train_data.targets()
        scaler = StandardScaler().fit(train_targets)
        scaled_targets = scaler.transform(train_targets).tolist()
        train_data.set_targets(scaled_targets)
    else:
        scaler = None

    # Get loss and metric functions 获取损失和指标功能
    loss_func = get_loss_func(args)
    metric_func = get_metric_func(metric=args.metric)

    # Set up test set evaluation 设置测试集评估
    test_smiles, test_targets = test_data.smiles(), test_data.targets()
    if args.dataset_type == 'multiclass':
        sum_test_preds = np.zeros((len(test_smiles), args.num_tasks, args.multiclass_num_classes))
    else:
        sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))

    # Train ensemble of models
    for model_idx in range(args.ensemble_size):
    #ensemble_size 缺省为1
        # Tensorboard writer
        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        makedirs(save_dir)
        try:
            writer = SummaryWriter(log_dir=save_dir)
        except:
            writer = SummaryWriter(logdir=save_dir)
        # Load/build model 加载/构建模型  若checkpoint_paths有，则加载指定模型，若无，则构建模型
        if args.checkpoint_paths is not None:
            debug(f'Loading model {model_idx} from {args.checkpoint_paths[model_idx]}')
            model = load_checkpoint(args.checkpoint_paths[model_idx], current_args=args, logger=logger)
        else:
            debug(f'Building model {model_idx}')
            #由此开始构建模型
            model = build_model(args)

        debug(model)
        debug(f'Number of parameters = {param_count(model):,}')
        if args.cuda:
            debug('Moving model to cuda')
            model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        #print('!!!!!!!!!!!!!!!!!!!!!!!')
        #print('准备print model')
        #print('!!!!!!!!!!!!!!!!!!!!!!!')
        #print(model)
        
        # Ensure that model is saved in correct location for evaluation if 0 epochs 如果0个时期，请确保将模型保存在正确的位置以进行评估
        save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)

        # Optimizers 优化器
        optimizer = build_optimizer(model, args)

        # Learning rate schedulers 学习率调度器
        scheduler = build_lr_scheduler(optimizer, args)

        # Run training
        best_score = float('inf') if args.minimize_score else -float('inf')
        best_epoch, n_iter = 0, 0
        #for epoch in trange(args.epochs):
        for epoch in range(args.epochs):
            info(f'Epoch {epoch}')
            print(f'Epoch {epoch}',file=refile)

            n_iter = train(
                model=model,
                data=train_data,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                logger=logger,
                writer=writer
            )
            #print('val tp:')
            if isinstance(scheduler, ExponentialLR):
            #isinstance() 函数来判断一个对象是否是一个已知的类型，类似 type()。
                print('此处是否执行')
                scheduler.step()
            val_scores = evaluate(
                model=model,
                data=val_data,
                num_tasks=args.num_tasks,
                metric_func=metric_func,
                batch_size=args.batch_size,
                dataset_type=args.dataset_type,
                scaler=scaler,
                logger=logger
            )
            #val_scores 在evaluate里面算
            
            #print('train tp:')
            #算train的数据
            train_scores = evaluate(
                model=model,
                data=train_data,
                num_tasks=args.num_tasks,
                metric_func=metric_func,
                batch_size=args.batch_size,
                dataset_type=args.dataset_type,
                scaler=scaler,
                logger=logger
            )
            #
            
            print('train auc:',train_scores)
            print('train auc:',train_scores,file=refile)

            # Average validation score  平均验证分数
            avg_val_score = np.nanmean(val_scores)
            debug(f'Validation1111 {args.metric} = {avg_val_score:.6f}')
            print(f'Validation1111 {args.metric} = {avg_val_score:.6f}',file=refile)
            writer.add_scalar(f'validation2222_{args.metric}', avg_val_score, n_iter)

            if args.show_individual_scores:
                # Individual validation scores  个人验证分数
                for task_name, val_score in zip(args.task_names, val_scores):
                    info(f'Validation {task_name} {args.metric} = {val_score:.6f}')
                    writer.add_scalar(f'validation_{task_name}_{args.metric}', val_score, n_iter)

            # Save model checkpoint if improved validation score  如果提高了验证分数，则保存模型检查点
            if args.minimize_score and avg_val_score < best_score or \
                    not args.minimize_score and avg_val_score > best_score:
                best_score, best_epoch = avg_val_score, epoch
                save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)        

        # Evaluate on test set using model with best validation score  使用具有最佳验证分数的模型评估测试集
        info(f'Model {model_idx} best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')
        print(f'Model {model_idx} best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}',file=refile)
        refile.close()
        model = load_checkpoint(os.path.join(save_dir, 'model.pt'), cuda=args.cuda, logger=logger)
        #加载最佳数值保存点的模型
        
        test_preds = predict(
            model=model,
            data=test_data,
            batch_size=args.batch_size,
            scaler=scaler
        )
        test_scores = evaluate_predictions(
            preds=test_preds,
            targets=test_targets,
            num_tasks=args.num_tasks,
            metric_func=metric_func,
            dataset_type=args.dataset_type,
            logger=logger
        )
        '''
        print('test tp:')
        test_pred2=[x for [x] in test_preds]
        test_pre2 = [int(item > 0.5) for item in test_pred2]
        #print('准备打印TP等')
        tn_valid, fp_valid, fn_valid, tp_valid = confusion_matrix(test_targets, test_pre2).ravel()
        print('TN:',tn_valid, 'FP:',fp_valid, 'FN:',fn_valid, 'TP:',tp_valid)
       '''

        if len(test_preds) != 0:
            sum_test_preds += np.array(test_preds)

        # Average test score
        avg_test_score = np.nanmean(test_scores)
        info(f'Model {model_idx} test {args.metric} = {avg_test_score:.6f}')
        writer.add_scalar(f'test_{args.metric}', avg_test_score, 0)

        if args.show_individual_scores:
            # Individual test scores
            for task_name, test_score in zip(args.task_names, test_scores):
                info(f'Model {model_idx} test {task_name} {args.metric} = {test_score:.6f}')
                writer.add_scalar(f'test_{task_name}_{args.metric}', test_score, n_iter)

    # Evaluate ensemble on test set 评估测试集的合奏
    avg_test_preds = (sum_test_preds / args.ensemble_size).tolist()

    ensemble_scores = evaluate_predictions(
        preds=avg_test_preds,
        targets=test_targets,
        num_tasks=args.num_tasks,
        metric_func=metric_func,
        dataset_type=args.dataset_type,
        logger=logger
    )

    # Average ensemble score 
    avg_ensemble_test_score = np.nanmean(ensemble_scores)
    info(f'Ensemble test {args.metric} = {avg_ensemble_test_score:.6f}')
    writer.add_scalar(f'ensemble_test_{args.metric}', avg_ensemble_test_score, 0)

    # Individual ensemble scores
    if args.show_individual_scores:
        for task_name, ensemble_score in zip(args.task_names, ensemble_scores):
            info(f'Ensemble test {task_name} {args.metric} = {ensemble_score:.6f}')

    return ensemble_scores
