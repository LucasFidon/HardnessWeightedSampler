# Distributionally Robust Deep Learning using Hardness Weighted Sampling

ERM is the optimization problem used to define the training process of most state-of-the-art deep learning pipelines.
Distributionally Robust Optimization (DRO) is a robust generalization of Empirical Risk Minimization (EMR).

In contrast to ERM, a deep learning network trained with DRO seeks to perform more consistently 
over the entire training dataset.
As a result, DRO can lead to models that perform better on underrepresented subsets of the training set.

To train deep neural networks with DRO, we propose Hardness Weighted Sampling, a novel training data sampling method
that can be easily plugged in any state-of-the-art deep learning pipeline.


## Installation
```bash
pip install git+https://github.com/LucasFidon/HardnessWeightedSampler.git
```
After installation you can run the tests using
```bash
sh run_tests.sh
```

## Example
TO DO

## How to cite
TO DO