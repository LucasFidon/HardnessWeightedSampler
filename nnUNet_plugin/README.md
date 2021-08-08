This folder provides the code to use the proposed hardness weighted sampler
to train a deep neural network
with Distributionally Robust Optimization (DRO) with [nnU-Net][nnunet].

## How to use with nnU-Net
First, you need to install the hardness_weighted_sampler library with
```bash
pip install git+https://github.com/LucasFidon/HardnessWeightedSampler.git
```
Second, you have to put the files of this folder in the [nnUNet repository][nnunet]
* put the file ```nnUNetTrainerV2_DRO.py``` in ```nnUNet/nnunet/training/network_training/```
* put the file ```dro_dataset_loading.py``` in ```nnUNet/nnunet/training/dataloading/```
and install nnU-Net from source.

[nnunet]: https://github.com/MIC-DKFZ/nnUNet