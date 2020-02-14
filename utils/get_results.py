"""
Script for finding the best hyperparameters from a list of files.
"""

import argparse
import os
import csv
from gen_run_cmds import gen_command

def get_number(line):
    """Get the value from a sentence like 'Something is 0.9854'"""
    return float(line[line.index('is ') + 3:])

def get_para_value(fileName, para_name):
    """Get a parameter value from the fileName like 'Eta0_0.1_Alpha_0.2"""
    if fileName.find(para_name) != -1:
        res_str = fileName[fileName.find(para_name) + len(para_name) + 1:]
        return res_str[:res_str.find('_')]
    else:
        return ''

def get_results(folder, output_name):
    best_results = {}

    result_folder = './tuning_results'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # Merge all results into one CSV file for convenient look-up.
    with open(result_folder + '/' + output_name + '.csv', 'w', newline='') as results_csv:
        writer = csv.writer(results_csv)
        writer.writerow(['Method', 'Eta0', 'Weight Decay', 'Alpha', 'Momentum',
                         'Milestones', 'Patience', 'Threshold', 'Train Loss',
                         'Train Accuracy', 'Test Loss', 'Test Accuracy'])
        for fileName in os.listdir(folder):
            # Parse hyperparameter values.
            method = fileName[fileName.find('_') + 1 : fileName.find('_Eta0')]
            eta0 = get_para_value(fileName, 'Eta0')
            weight_decay = get_para_value(fileName, 'WD')
            alpha = get_para_value(fileName, 'alpha')
            momentum = get_para_value(fileName, 'Mom')
            patience = get_para_value(fileName, 'Patience')
            thres = get_para_value(fileName, 'Thres')
            if method == 'SGD_Stage_Decay':
                milestones = fileName[fileName.find('Milestones_') + 11 : fileName.find('_Epoch')]
            else:
                milestones = ''

            with open(folder + '/' + fileName, 'r') as file_fp:
                lines = file_fp.readlines()
                tr_loss = get_number(lines[4])
                tr_accu = get_number(lines[5])
                ts_loss = get_number(lines[10])
                ts_accu = get_number(lines[11])           
                writer.writerow([method, eta0, weight_decay, alpha, momentum,
                                 milestones, patience, thres, tr_loss,
                                 tr_accu, ts_loss, ts_accu])

            if (method not in best_results) or (best_results[method]['metric'] < ts_accu):
                best_results[method] = {'metric':ts_accu, 'fileName':fileName}

    # Generate the command for running the test on the best hyperparameters.
    with open(result_folder + '/' + output_name + '_test_cmds.txt', 'w') as f:
        for method, result in best_results.items():
            # Parse hyperparameter values.
            fileName = result['fileName']
            folder = './src'
            eta0 = get_para_value(fileName, 'Eta0')
            weight_decay = get_para_value(fileName, 'WD')
            alpha = get_para_value(fileName, 'alpha')
            nesterov = True
            momentum = get_para_value(fileName, 'Mom')
            patience = get_para_value(fileName, 'Patience')
            thres = get_para_value(fileName, 'Thres')
            if method == 'SGD_Stage_Decay':
                milestones = fileName[fileName.find('Milestones_') + 11 : fileName.find('_Epoch')].split()
            else:
                milestones = ''
            train_epochs = get_para_value(fileName, 'Epoch')
            batchsize = get_para_value(fileName, 'Batch')
            validation = False
            val_ratio = 0
            eval_interval = 1
            dataset = fileName[:fileName.find('_')]
            dataroot = './data'
            use_cuda = True
            reproducible = False
            seed = 0
            log_folder = output_name

            cmd = gen_command(folder, method, eta0, alpha, nesterov, momentum,
                              weight_decay, milestones, patience, thres,
                              train_epochs, batchsize, validation, val_ratio,
                              eval_interval, dataset, dataroot, use_cuda,
                              reproducible, seed, log_folder)
            print('Best hyperparameters for %s: %s' % (method,
                      cmd[cmd.find('--eta0') : cmd.find('--train-epochs')]))
            f.write(cmd + '\n\n')


def load_args():
    parser = argparse.ArgumentParser(description='Find best hyperparameters.')

    parser.add_argument('--folder', type=str, default='./logs/tests',
                        help='log folder path')
    parser.add_argument('--output-name', type=str, default='cifar10-decay',
                        help='output file name')

    return parser.parse_args()


args = load_args()
get_results(args.folder, args.output_name)
