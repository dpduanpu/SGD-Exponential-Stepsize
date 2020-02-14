"""
Convert between alpha and etaT/eta0.
"""

import argparse
import math

def load_args():
    parser = argparse.ArgumentParser(description='Convert between alpha and etaT/eta0.')

    parser.add_argument('--optim-method', type=str, default='SGD_Exp_Decay',
                        choices=['SGD_Exp_Decay', 'SGD_1t_Decay', 
                                 'SGD_1sqrt_Decay'],
                        help="Decay schemes (default: SGD_Exp_Decay).")
    parser.add_argument('--alpha', type=float, default=-1,
                        help=("Decay factor, you can only specify either alpha"
                              "or etaT-eta0 (default: -1)."))
    parser.add_argument('--etaT-eta0', type=float, default=-1,
                        help='etaT/eta0 (default: -1).')
    parser.add_argument('--train-epochs', type=int, default=100,
                        help='Number of train epochs (default: 100).')
    parser.add_argument('--batchsize', type=int, default=128,
                        help='Batchsize in train (default: 128).')
    parser.add_argument('--train-size', type=int, default=50000,
                        help=("The number of samples the whole training set "
                              "contains (default: 50000)."))
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Percentage of training samples used as validation (default: 0.1).')

    return parser.parse_args()

def main():
    args = load_args()

    train_size = args.train_size * (1 - args.val_ratio)
    num_rounds = math.ceil(train_size / args.batchsize) * args.train_epochs
    if args.alpha == -1:
        if args.optim_method == 'SGD_Exp_Decay':
            result = args.etaT_eta0**(1.0 / num_rounds)
        elif args.optim_method == 'SGD_1t_Decay':
            result = (1.0/args.etaT_eta0 - 1)/num_rounds
        elif args.optim_method == 'SGD_1sqrt_Decay':
            result = (1.0/args.etaT_eta0 - 1)/math.sqrt(num_rounds)
        print('Given etaT / eta0 = %g, alpha is %g.' % (args.etaT_eta0, result))
    else:
        if args.optim_method == 'SGD_Exp_Decay':
            result = args.alpha**num_rounds
        elif args.optim_method == 'SGD_1t_Decay':
            result = 1.0 / (1 + args.alpha*num_rounds)
        elif args.optim_method == 'SGD_1sqrt_Decay':
            result = 1.0 / (1 + args.alpha*(num_rounds**0.5))
        print('Given alpha = %g, etaT / eta0 is %g.' % (args.alpha, result))

main()
