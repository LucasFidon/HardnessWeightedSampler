# Distributionally Robust Deep Learning using Hardness Weighted Sampling

Empirical Risk Minimization (ERM) is the optimization problem used for the training of most state-of-the-art deep learning pipelines.
Distributionally Robust Optimization (DRO) is a robust generalization of ERM.

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
For an example of how to use the hardness weighted sampler please see the folder ```nnUNet_plugin```.


## Computational Time and Memory Overhead
You can compute the additional computational time and memory that will be required to use the hardness weighted sampler 
for your application using
```bash
python compute_time_and_memory.py --num_samples <number of samples in your training dataset>
```
batch size and beta can also be adjusted. For more details, see
```bash
python compute_time_and_memory.py -h
```

## How to cite
If you use the hardness weighted sampler in your work please cite
* L. Fidon, M. Aertsen, T. Deprest, D. Emam, F. Guffens, N. Mufti, E. Van Elslander,
  E. Schwarz, M. Ebner, D. Prayer, G. Kasprian, A. L. David, A. Melbourne, S. Ourselin,
  J. Deprest, G. Lang, T. Vercauteren.
[Distributionally Robust Deep Learning using Hardness Weighted Sampling][main_paper].
arXiv preprint arXiv:2001.02658.

BibTeX:
```
@article{fidon2020distributionally,
  title={Distributionally robust deep learning using hardness weighted sampling},
  author={Fidon, Lucas and Aertsen, Michael and Deprest, Thomas and Emam, Doaa and
    Guffens, Frédéric and Mufti, Nada and Van Elslander, Esther and 
    Schwartz, Ernst and Ebner, Michael and Prayer, Daniela and Kasprian, Gregor and 
    David, Anna L. and Melbourne, Andrew and Ourselin, Sébastien and Deprest, Jan and 
    Langs, Georg and Vercauteren, Tom},
  journal={arXiv preprint arXiv:2001.02658},
  year={2020}
}
```

[main_paper]: https://arxiv.org/abs/2001.02658