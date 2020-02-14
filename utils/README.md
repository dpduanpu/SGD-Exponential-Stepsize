This folder contains some helper functions for generating commands for grid searching, processing hyperparameter tuning results, and drawing figures.

### `gen_fun_cmds.py`
Mainly used for generating batch of commands for hyperparameter tuning as it can combine searching grids of a bunch of hyperparameters. Commands will be saved in a sub-directory named `cmds` as a txt file with the name from `--output-name`. Options are specified as flags (see --help for a detailed list and explanation). Nonetheless, we would like to point out two options specifically:
* `--alpha-etaT-eta0`: if specified, the values you input for `--alphas` will be considered as <img src="https://render.githubusercontent.com/render/math?math=\eta_T/\eta_0"> and corresponding <img src="https://render.githubusercontent.com/render/math?math=\alpha"> will be automatically computed. To use this option, you need to also give the `--train-size` in order to calculate T. However, you don't need to deduct the validation set size manually, as the actual training data size will be automatically inferred from input arguments.
* `--milestone-comb-order`: this option specifies how many elements can be picked from the `--milestone` argument at most to form a combination of milestones. An example might make it more clear. Suppose `--milestones` is [32000, 40000, 48000], then setting this option to 0 will take `--milestones` as is, namely [32000, 40000, 48000]. Setting this option to 1 will take [[32000], [40000], [48000]] as milestones. And setting this option to 2 will take [[32000], [40000], [48000], [32000 40000], [32000 48000], [40000 48000]] as milestone.

#### Hyperparameter tuning for comparing Adam, exponential decay, O(1/t) decay, and O(1/sqrt(t)) decay for training a 20-layer Residual Network to do image classification on CIFAR-10:
```shell
python gen_run_cmds.py --folder ./src --optim-methods Adam SGD_Exp_Decay SGD_1t_Decay SGD_1sqrt_Decay --eta0s 0.00001 0.0001 0.001 0.01 0.1 1 --alphas 0.00001 0.0001 0.001 0.01 0.1 1 --alpha-etaT-eta0 --nesterov --momentums 0.9 --weight-decays 0.0005 --train-epochs 100 --batchsizes 128  --train-size 50000 --validation --val-ratio 0.1 --eval-interval 10 --datasets CIFAR10 --dataroot ./data --use-cuda --reproducible --log-folder ./logs/cifar10_decay_val --output-name cifar10_decay_val_cmds
```

#### Hyperparameter tuning for comparing Adam, exponential decay, O(1/t) decay, and O(1/sqrt(t)) decay for training a 100-layer DenseNet to do image classification on CIFAR-100:
```shell
python gen_run_cmds.py --folder ./src --optim-methods Adam SGD_Exp_Decay SGD_1t_Decay SGD_1sqrt_Decay --eta0s 0.00001 0.0001 0.001 0.01 0.1 1 --alphas 0.00001 0.0001 0.001 0.01 0.1 1 --alpha-etaT-eta0 --nesterov --momentums 0.9 --weight-decays 0.0005 --train-epochs 50 --batchsizes 128  --train-size 50000 --validation --val-ratio 0.1 --eval-interval 10 --datasets CIFAR100 --dataroot ./data --use-cuda --reproducible --log-folder ./logs/cifar100_decay_val --output-name cifar100_decay_val_cmds
```

#### Hyperparameter tuning for comparing exponential decay with stagewise SGD and PyTorch's ReduceLROnPlateau for training a 20-layer Residual Network to do imageclassification on CIFAR-10:

```shell
# SGD searching for a good eta0 range:
python gen_run_cmds.py --folder ./src --optim-methods SGD --eta0s 0.00001 0.0001 0.001 0.01 0.1 1 --nesterov --momentums 0.9 --weight-decays 0.0001 --train-epochs 164 --batchsizes 128  --train-size 50000 --validation --val-ratio 0.1 --eval-interval 10 --datasets CIFAR10 --dataroot ./data --use-cuda --reproducible --log-folder ./logs/cifar10_stage_sgd --output-name cifar10_stage_sgd_cmds


# Hyperparameter tuning:
python gen_run_cmds.py --folder ./src --optim-methods SGD SGD_Stage_Decay SGD_ReduceLROnPlateau --eta0s 0.007 0.01 0.04 0.07 0.1 0.4 --alphas 0.1 --nesterov --momentums 0.9 --weight-decays 0.0001 --milestones 16000 24000 32000 40000 48000 56000 --milestone-comb-order 2 --patiences 5 10 --thresholds 0.0001 0.001 0.01 0.1 --train-epochs 164 --batchsizes 128  --train-size 50000 --validation --val-ratio 0.1 --eval-interval 10 --datasets CIFAR10 --dataroot ./data --use-cuda --reproducible --log-folder ./logs/cifar10_stage_val --output-name cifar10_stage_val_cmds

python gen_run_cmds.py --folder ./src --optim-methods SGD_Exp_Decay --eta0s 0.007 0.01 0.04 0.07 0.1 0.4 --alphas 0.007 0.01 0.04 0.07 0.1 0.4 --alpha-etaT-eta0 --nesterov --momentums 0.9 --weight-decays 0.0005 --train-epochs 164 --batchsizes 128  --train-size 50000 --validation --val-ratio 0.1 --eval-interval 10 --datasets CIFAR10 --dataroot ./data --use-cuda --reproducible --log-folder ./logs/cifar10_stage_val --output-name cifar10_stage_exp_val_cmds
```

### `get_results.py`:
Given the folder where all validation results for one experiment are stored, this script will merge all results into a CSV file for later fast look-up. It also computes best hyperparameters for each method based on validation accuracy, and generating corresponding commands for running test using best hyperparameters. Generated files will be saved in a sub-directory named `tuning_results` as one CSV file for summary and one txt file for commands with names from `--output-name`. Note that you should only put results obtained from the same dataset on a folder. For example:
```
python get_results.py --folder ./logs/cifar10_decay_val --output-name cifar10_decay
```

### `draw_comps.py`
This script draws figures comparing different methods by simply taking the `folder` option which denotes the place where results to be drawn are stored. Like the above script, you should only put results obtained from the same dataset on one folder. For example:
```
python draw_comps.py --folder ./logs/cifar10_decay
```

### `alpha_converter.py`
This script is used to convert between <img src="https://render.githubusercontent.com/render/math?math=\alpha"> and <img src="https://render.githubusercontent.com/render/math?math=\eta_T/\eta_0">, with T computed from `--train-epochs`, `--batchsize`, `--train-size`, and `--val-ratio` as
```shell
T = train_epochs * math.ceil(train_size * (1 - val_ratio) / batchsize))
```
Example usages are:
```shell
python .\alpha_converter.py --optim-method 'SGD_Exp_Decay' --alpha 0.99 --train-epochs 100 --batchsize 128 --train-size 50000 --val-ratio 0.1

python .\alpha_converter.py --optim-method 'SGD_1t_Decay' --etaT-eta0 0.01 --train-epochs 100 --batchsize 128 --train-size 50000 --val-ratio 0.1
```
