# High-Dimensional Bayesian Optimization via Semi-supervised Learning with Optimized Unlabelled Data Sampling

## Initialization 
Dataset initialization and environment installation can be found in T-LBO repository (https://github.com/huawei-noah/HEBO/tree/master/T-LBO).

## Run TSBO
First enter the root folder.

### Topology task
```shell script
bash ./weighted_retraining/scripts/robust_opt/robust_opt_topology.sh
```

### Expression task
```shell script
bash ./weighted_retraining/scripts/robust_opt/robust_opt_expr.sh
```
### Chemical design task
```shell script
bash ./weighted_retraining/scripts/robust_opt/robust_opt_chem.sh
```

## Acknowledgements
We build our code upon on [High-Dimensional Bayesian Optimisation withVariational Autoencoders and Deep Metric Learning](https://arxiv.org/abs/2106.03609).

