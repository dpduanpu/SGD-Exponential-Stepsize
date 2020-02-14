# Exponential Step Sizes for Non-Convex Optimization
This repository contains PyTorch codes for the experiments on deep learning in the paper:

**[Exponential Step Sizes for Non-Convex Optimization](https://arxiv.org/abs/2002.05273)**  
Xiaoyu Li*, Zhenxun Zhuang*, Francesco Orabona

### Description
Stochastic Gradient Descent (SGD) is a popular tool in large scale optimization of machine learning objective functions. However, the performance is greatly variable, depending on the choice of the step sizes. In this paper, we introduce the exponential step sizes for stochastic optimization of smooth non-convex functions which satisfy the Polyak-Lojasiewicz (PL) condition. We show that, without any information on the level of noise over the stochastic gradients, these step sizes guarantee a convergence rate for the last iterate that automatically interpolates between a linear rate (in the noisy-free case) and an O(1/T) rate (in the noisy case), up to poly-logarithmic factors. Moreover, if without the PL condition, the exponential step sizes still guarantee optimal convergence to a critical point, up to logarithmic factors. We also validate our theoretical results with empirical experiments on real-world datasets with deep learning architectures.

### Code & Usage
`src` folder contains codes for training a deep neural network to do image classification on CIFAR10/100. You can train models with the `main.py` script, with hyper-parameters being specified as flags (see --help for a detailed list and explanation).

`utils` folder contains codes for generating running commands given a set of hyperparameter searching grids, finding the best hyperparameters from all tuning results, and drawing figures. Look inside the folder for details.

### Results
Below, exponential decay means <img src="https://render.githubusercontent.com/render/math?math=\eta_t=\eta_0\cdot\alpha^t">, O(1/t) decay means &eta;<sub>t</sub> = &eta;<sub>0</sub>/(1+&alpha;t), and O(1/sqrt(t)) means &eta;<sub>t</sub> = &eta;<sub>0</sub>/(1+&alpha;&radic;t). They are variants of the vanilla SGD by using a decaying step size instead of a constant one.

#### Compare [Adam](https://arxiv.org/abs/1412.6980), exponential decay, O(1/t) decay, and O(1/sqrt(t)) decay for training a 20-layer Residual Network to do image classification on CIFAR-10:
<p float="left">
  <img src="/figs/cifar10/Train_Loss.png" width="49%" />
  <img src="/figs/cifar10/Test_Accuracy.png" width="49%" />
</p>

```shell
python ./src/main.py --optim-method Adam --eta0 0.001 --weight-decay 0.0005 --train-epochs 100 --batchsize 128 --eval-interval 1 --dataset CIFAR10 --dataroot ./data --use-cuda --log-folder ./logs/cifar10_decay 

python ./src/main.py --optim-method SGD_1sqrt_Decay --eta0 0.2 --alpha 0.1 --nesterov --momentum 0.9 --weight-decay 0.0005 --train-epochs 100 --batchsize 128 --eval-interval 1 --dataset CIFAR10 --dataroot ./data --use-cuda --log-folder ./logs/cifar10_decay 

python ./src/main.py --optim-method SGD_1t_Decay --eta0 0.08 --alpha 0.0006 --nesterov --momentum 0.9 --weight-decay 0.0005 --train-epochs 100 --batchsize 128 --eval-interval 1 --dataset CIFAR10 --dataroot ./data --use-cuda --log-folder ./logs/cifar10_decay 

python ./src/main.py --optim-method SGD_Exp_Decay --eta0 0.08 --alpha 0.9999 --nesterov --momentum 0.9 --weight-decay 0.0005 --train-epochs 100 --batchsize 128 --eval-interval 1 --dataset CIFAR10 --dataroot ./data --use-cuda --log-folder ./logs/cifar10_decay 
```

After obtaining the results, to see the figures simply run:
```shell
python ./utils/draw_comps.py --folder ./logs/cifar10_decay
```

#### Compare Adam, exponential decay, O(1/t) decay, and O(1/sqrt(t)) decay for training a 100-layer DenseNet to do image classification on CIFAR-100:
<p float="left">
  <img src="/figs/cifar100/Train_Loss.png" width="49%" />
  <img src="/figs/cifar100/Test_Accuracy.png" width="49%" />
</p>

```shell
python ./src/main.py --optim-method Adam --eta0 0.001 --weight-decay 0.0005 --train-epochs 50 --batchsize 128 --eval-interval 1 --dataset CIFAR100 --dataroot ./data --use-cuda --log-folder ./logs/cifar100_decay 

python ./src/main.py --optim-method SGD_1sqrt_Decay --eta0 0.1 --alpha 0.015 --nesterov --momentum 0.9 --weight-decay 0.0005 --train-epochs 50 --batchsize 128 --eval-interval 1 --dataset CIFAR100 --dataroot ./data --use-cuda --log-folder ./logs/cifar100_decay 

python ./src/main.py --optim-method SGD_1t_Decay --eta0 0.8 --alpha 0.004 --nesterov --momentum 0.9 --weight-decay 0.0005 --train-epochs 50 --batchsize 128 --eval-interval 1 --dataset CIFAR100 --dataroot ./data --use-cuda --log-folder ./logs/cifar100_decay 

python ./src/main.py --optim-method SGD_Exp_Decay --eta0 0.08 --alpha 0.99985 --nesterov --momentum 0.9 --weight-decay 0.0005 --train-epochs 50 --batchsize 128 --eval-interval 1 --dataset CIFAR100 --dataroot ./data --use-cuda --log-folder ./logs/cifar100_decay 
```

After obtaining the results, to see the figures simply run:
```shell
python ./utils/draw_comps.py --folder ./logs/cifar100_decay
```

#### Compare exponential decay with stagewise SGD and PyTorch's [ReduceLROnPlateau](https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau) for training a 20-layer Residual Network to do imageclassification on CIFAR-10:
<p float="left">
  <img src="/figs/stage_sgd/Train_Loss.png" width="49%" />
  <img src="/figs/stage_sgd/Test_Accuracy.png" width="49%" />
</p>

```shell
python ./src/main.py --optim-method SGD --eta0 0.07 --nesterov --momentum 0.9 --weight-decay 0.0001 --train-epochs 164 --batchsize 128 --eval-interval 1 --dataset CIFAR10 --dataroot ./data --use-cuda --log-folder ./logs/cifar10_stage 

python ./src/main.py --optim-method SGD_Exp_Decay --eta0 0.1 --alpha 0.99991 --nesterov --momentum 0.9 --weight-decay 0.0005 --train-epochs 164 --batchsize 128 --eval-interval 1 --dataset CIFAR10 --dataroot ./data --use-cuda --log-folder ./logs/cifar10_stage 

python ./src/main.py --optim-method SGD_ReduceLROnPlateau --eta0 0.07 --alpha 0.1 --nesterov --momentum 0.9 --weight-decay 0.0001 --patience 10 --threshold 0.001 --train-epochs 164 --batchsize 128 --eval-interval 1 --dataset CIFAR10 --dataroot ./data --use-cuda --log-folder ./logs/cifar10_stage 

python ./src/main.py --optim-method SGD_Stage_Decay --eta0 0.2 --alpha 0.1 --nesterov --momentum 0.9 --weight-decay 0.0001 --milestones 32000 40000 --train-epochs 164 --batchsize 128 --eval-interval 1 --dataset CIFAR10 --dataroot ./data --use-cuda --log-folder ./logs/cifar10_stage

python ./src/main.py --optim-method SGD_Stage_Decay --eta0 0.1 --alpha 0.1 --nesterov --momentum 0.9 --weight-decay 0.0001 --milestones 32000 --train-epochs 164 --batchsize 128 --eval-interval 1 --dataset CIFAR10 --dataroot ./data --use-cuda --log-folder ./logs/cifar10_stage 
```

After obtaining the results, to see the figures simply run:
```shell
python ./utils/draw_comps.py --folder ./logs/cifar10_stage
```
