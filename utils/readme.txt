Compare exponential decay with polynomial decays and Adam:

CIFAR10 Hyperparameter tuning:
python gen_run_cmds.py --folder ./src --optim-methods Adam SGD_Exp_Decay SGD_1t_Decay SGD_1sqrt_Decay --eta0s 0.00001 0.0001 0.001 0.01 0.1 1 --alphas 0.00001 0.0001 0.001 0.01 0.1 1 --alpha-etaT-eta0 --nesterov --momentums 0.9 --weight-decays 0.0005 --train-epochs 100 --batchsizes 128  --train-size 50000 --validation --val-ratio 0.1 --eval-interval 10 --datasets CIFAR10 --dataroot ./data --use-cuda --reproducible --log-folder ./logs/cifar10_decay_val --output-name cifar10_decay_val_cmds


CIFAR100 Hyperparameter tuning:
python gen_run_cmds.py --folder ./src --optim-methods Adam SGD_Exp_Decay SGD_1t_Decay SGD_1sqrt_Decay --eta0s 0.00001 0.0001 0.001 0.01 0.1 1 --alphas 0.00001 0.0001 0.001 0.01 0.1 1 --alpha-etaT-eta0 --nesterov --momentums 0.9 --weight-decays 0.0005 --train-epochs 50 --batchsizes 128  --train-size 50000 --validation --val-ratio 0.1 --eval-interval 10 --datasets CIFAR100 --dataroot ./data --use-cuda --reproducible --log-folder ./logs/cifar100_decay_val --output-name cifar100_decay_val_cmds



Compare exponential decay with stagewise SGD and ReduceLROnPlateau:

SGD searching for a good eta0 range:
python gen_run_cmds.py --folder ./src --optim-methods SGD --eta0s 0.00001 0.0001 0.001 0.01 0.1 1 --nesterov --momentums 0.9 --weight-decays 0.0001 --train-epochs 164 --batchsizes 128  --train-size 50000 --validation --val-ratio 0.1 --eval-interval 10 --datasets CIFAR10 --dataroot ./data --use-cuda --reproducible --log-folder ./logs/cifar10_stage_sgd --output-name cifar10_stage_sgd_cmds


Hyperparameter tuning:
python gen_run_cmds.py --folder ./src --optim-methods SGD SGD_Stage_Decay SGD_ReduceLROnPlateau --eta0s 0.007 0.01 0.04 0.07 0.1 0.4 --alphas 0.1 --nesterov --momentums 0.9 --weight-decays 0.0001 --milestones 16000 24000 32000 40000 48000 56000 --milestone-comb-order 2 --patiences 5 10 --thresholds 0.0001 0.001 0.01 0.1 --train-epochs 164 --batchsizes 128  --train-size 50000 --validation --val-ratio 0.1 --eval-interval 10 --datasets CIFAR10 --dataroot ./data --use-cuda --reproducible --log-folder ./logs/cifar10_stage_val --output-name cifar10_stage_val_cmds

python gen_run_cmds.py --folder ./src --optim-methods SGD_Exp_Decay --eta0s 0.007 0.01 0.04 0.07 0.1 0.4 --alphas 0.007 0.01 0.04 0.07 0.1 0.4 --alpha-etaT-eta0 --nesterov --momentums 0.9 --weight-decays 0.0005 --train-epochs 164 --batchsizes 128  --train-size 50000 --validation --val-ratio 0.1 --eval-interval 10 --datasets CIFAR10 --dataroot ./data --use-cuda --reproducible --log-folder ./logs/cifar10_stage_val --output-name cifar10_stage_exp_val_cmds