"""
Generating commands for running the main function given a list of arguments.
"""

import argparse
from itertools import combinations 
import math
import os

def load_args():
    parser = argparse.ArgumentParser(description='Generating commands')

    parser.add_argument('--folder', type=str, default='./src',
                        help='Location of the source files')
    parser.add_argument('--optim-methods', nargs='*', default=['SGD'],
                        help=("A list of possible optimizers to use (currently "
                              "only supports "
                              "{'Adam', 'SGD', 'SGD_Exp_Decay', 'SGD_1t_Decay', "
                              "'SGD_1sqrt_Decay', 'SGD_Stage_Decay', "
                              "'SGD_ReduceLROnPlateau'}) (default: SGD)."))
    parser.add_argument('--eta0s', nargs='*', default=[0.1],
                        help='A list of initial step sizes to be tried (default: 0.1).')
    parser.add_argument('--alphas', nargs='*', default=[0.1],
                        help='A list of decay factors to be tried (default: 0.1).') 
    parser.add_argument('--alpha-etaT-eta0', action='store_true',
                        help=("Use this option if you want the alpha values to "
                              "be converted to etaT/eta0, note this requires "
                              "the train-size argument (do not consider validation "
                              "as its size will be automatically deducted according "
                              "to the val-ratio argument) (default: False)."))
    parser.add_argument('--nesterov', action='store_true',
                        help='Use nesterov momentum (default: False).') 
    parser.add_argument('--momentums', nargs='*', default=[0.9],
                        help='A list of momentums to be tried (default: 0.9).')    
    parser.add_argument('--weight-decays', nargs='*', default=[0.0005],
                        help='A list of weight decays used to be tried (default: 0.0005).')
    parser.add_argument('--milestones', nargs='*', default=[],
                        help=("Used for SGD stagewise decay denoting when to "
                              "decrease the step size, unit in iteration "
                              "(default: [])."))
    parser.add_argument('--milestone-comb-order', type=int, default=0,
                        help=("If 0, just take milestones as is; otherwise, "
                              "try all combinations of at most this number of "
                              "elements from milestones (default: 0)."))
    parser.add_argument('--patiences', nargs='*', default=[10],
                        help=("A list of patience value to be tried in "
                              "ReduceLROnPlateau denoting number of epochs "
                              "with no improvement after which learning rate "
                              "will be reduced (default: 10)."))
    parser.add_argument('--thresholds', nargs='*', default=[1e-4],
                        help=("A list of patience value to be tried in "
                              "ReduceLROnPlateau for measuring the new, "
                              "optimum to only focus on significant changes "
                              "(default: 1e-4)."))

    parser.add_argument('--train-epochs', nargs='*', default=[100],
                        help='A list of number of train epochs to be tried (default: 100).')
    parser.add_argument('--batchsizes', nargs='*', default=[128],
                        help='A list of batchsizes to be tried (default: 128).')
    parser.add_argument('--train-size', type=int, default=50000,
                        help=("The number of samples the whole training set contains"
                              "(default: 50000)."))
    parser.add_argument('--validation', action='store_true',
                        help='Do validation (True) or test (False) (default: False).')        
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Percentage of training samples used as validation (default: 0.1).')
    parser.add_argument('--eval-interval', type=int, default=10,
                        help=("How often should the model be evaluated during"
                              "training, unit in epochs (default: 10)."))

    parser.add_argument('--datasets', nargs='*', default=['CIFAR10'],
                        help=("A list of datasets to be tried (currently only "
                              "supports CIFAR10 and CIFAR100) "
                              "(default: CIFAR10)."))
    parser.add_argument('--dataroot', type=str, default='../data',
                        help='Where to retrieve data (default: ../data).')        
    parser.add_argument('--use-cuda', action='store_true',
                        help='Use CUDA (default: False).')
    parser.add_argument('--reproducible', action='store_true',
                        help='Ensure reproducibility (default: False).')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='Random seed (default: 0).')

    parser.add_argument('--log-folder', type=str, default='../logs',
                        help='Where to store results.')
    parser.add_argument('--output-name', type=str, default='cmds',
                        help="Output file name (will be saved as .txt) (defaults: cmds).")

    return parser.parse_args()

def gen_command(folder, optim_method, eta0, alpha, nesterov, momentum,
                weight_decay, milestones, patience, threshold, train_epochs,
                batchsize, validation, val_ratio, eval_interval, dataset,
                dataroot, use_cuda, reproducible, seed, log_folder):
    cmd = ('python '
           + ('%s/main.py ' % folder)
           + ('--optim-method %s ' % optim_method)
           + ('--eta0 %g ' % float(eta0))
           + (('--alpha %g ' % float(alpha))
               if optim_method not in ['Adam', 'SGD'] else '')
           + ('--nesterov '
               if (optim_method.startswith('SGD') and nesterov) else '')
           + (('--momentum %g ' % float(momentum))
               if optim_method.startswith('SGD') else '')
           + ('--weight-decay %g ' % float(weight_decay))
           + (('--milestones ' + (' '.join(milestones)) + ' ')
               if optim_method == 'SGD_Stage_Decay' else '')
           + (('--patience %d --threshold %g ' % (int(patience), float(threshold)))
               if optim_method == 'SGD_ReduceLROnPlateau' else '')
           + ('--train-epochs %d ' % int(train_epochs))
           + ('--batchsize %d ' % int(batchsize))
           + ('--validation ' if validation else '')
           + (('--val-ratio %g ' % float(val_ratio)) if validation else '')
           + ('--eval-interval %d ' % int(eval_interval))
           + ('--dataset %s ' % dataset)
           + ('--dataroot %s ' % dataroot)
           + ('--use-cuda ' if use_cuda else '')
           + ('--reproducible ' if reproducible else '')
           + (('--seed %d ' % int(seed)) if reproducible else '')
           + ('--log-folder %s ' % log_folder))
    return cmd

def main():
    args = load_args()

    train_size = float(args.train_size)
    if args.validation:
        train_size *= 1 - float(args.val_ratio)

    cmd_folder = './cmds'
    if not os.path.exists(cmd_folder):
        os.makedirs(cmd_folder)
    with open(cmd_folder + '/' + args.output_name + '.txt', 'w') as fCMD:
        for num_epoch in args.train_epochs:
        
            for batchsize in args.batchsizes:        

                for optim_method in args.optim_methods:

                    for eta0 in args.eta0s:

                        alphas = [1]
                        if optim_method not in ['Adam', 'SGD']:
                            alphas = [float(x) for x in args.alphas]
                            if args.alpha_etaT_eta0:
                                num_rounds = math.ceil(train_size / float(batchsize)) * float(num_epoch)
                                if optim_method == 'SGD_Exp_Decay':
                                    alphas = [x**(1.0 / num_rounds) for x in alphas]
                                elif optim_method == 'SGD_1t_Decay':
                                    alphas = [(1.0/x - 1)/num_rounds for x in alphas]
                                elif optim_method == 'SGD_1sqrt_Decay':
                                    alphas = [(1.0/x - 1)/math.sqrt(num_rounds) for x in alphas]
                        for alpha in alphas:

                            momentums = [0.9]
                            if optim_method.startswith('SGD'):
                                momentums = args.momentums
                            for momentum in momentums:

                                for weight_decay in args.weight_decays:

                                    milestones = [[]]
                                    if optim_method == 'SGD_Stage_Decay':                                
                                        if args.milestone_comb_order == 0:
                                            milestones = [args.milestones]
                                        else:
                                            milestones = []
                                            for i in range(1, args.milestone_comb_order + 1):
                                                milestones.extend(list(combinations(args.milestones, i)))
                                    for milestone in milestones:

                                        patiences = [1]
                                        thresholds = [1]
                                        if optim_method == 'SGD_ReduceLROnPlateau':
                                            patiences = args.patiences
                                            thresholds = args.thresholds
                                        for patience in patiences:
                                            for threshold in thresholds:

                                                for dataset in args.datasets:

                                                    cur_cmd = gen_command(args.folder,
                                                                          optim_method,
                                                                          eta0,
                                                                          alpha,
                                                                          args.nesterov,
                                                                          momentum,
                                                                          weight_decay,
                                                                          milestone,
                                                                          patience,
                                                                          threshold,
                                                                          num_epoch,
                                                                          batchsize, 
                                                                          args.validation,
                                                                          args.val_ratio,
                                                                          args.eval_interval,
                                                                          dataset,
                                                                          args.dataroot,
                                                                          args.use_cuda,
                                                                          args.reproducible,
                                                                          args.seed,
                                                                          args.log_folder)
                                                    fCMD.write(cur_cmd + '\n\n')

main()
