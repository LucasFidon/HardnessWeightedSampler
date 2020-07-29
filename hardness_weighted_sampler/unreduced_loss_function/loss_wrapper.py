"""
@brief  PyTorch code for Unreduced Loss Function.

        Usual loss functions in PyTorch returned one scalar loss value per batch
        (e.g. average or sum of the losses for the samples of the batch).
        However, we may want instead to get a batch of scalar loss values
        with one scalar loss value per sample.

        Unreduced Loss Functions still return the reduced loss (mean or sum) when called,
        but in addition they store internally an unreduced loss value for the last batch
        that was processed with one scalar value per sample in the batch.

@author Lucas Fidon (lucas.fidon@kcl.ac.uk)
@date   June 2020.
"""

import torch
import torch.nn as nn


class UnreducedLossFunction(nn.Module):
    def __init__(self, reduced_loss_func, reduction='mean'):
        """
        Wrapper for PyTorch loss functions that keeps the batch of loss values in memory
        instead of returning only a reduced version of the batch loss (e.g. mean or sum).
        :param reduced_loss_func: instance of nn.Module; a PyTorch loss function
        """
        super(UnreducedLossFunction, self).__init__()
        assert isinstance(reduced_loss_func, nn.Module), \
            "reduced_loss_func must be an instance of nn.Module. " \
            "Found an instance of %s instead." % type(reduced_loss_func)
        # wrapped PyTorch loss function
        self.func = reduced_loss_func
        # internal unreduced loss for the last batch that has been evaluated.
        # this is the attribute that is specific to unreduced loss functions.
        self.loss_batch = None
        assert reduction in ['mean', 'sum'], "Only 'mean' and 'sum' are supported " \
                                             "for the reduction parameter."
        self.reduction = reduction

    def cuda(self, device=None):
        """
        Moves all the internal loss function parameters and buffers to the GPU.
        :param device: int; (optional) if specified, all parameters will be
                copied to that device
        """
        self.func.cuda(device)

    def cpu(self):
        """
        Moves all the internal loss function parameters and buffers to the CPU.
        """
        self.func.cpu()

    def __post_init__(self):
        """
        Called once after the __init__ function.
        """
        # set the reduction parameter of the internal
        # PyTorch loss function to 'none'
        self.func.reduction = 'none'

    def __call__(self, input, target):
        """
        Return the reduced batch loss (mean or sum)
        and saved the batch loss in self.loss
        :param input: pytorch tensor
        :param target: pytorch tensor
        :return: pytorch scalar tensor
        """
        self.func.reduction = 'none'

        # store the unreduced batch loss
        self.loss_batch = self.func.forward(input, target)

        # make sure the loss is averaged over the pixel/voxel positions
        # self.loss_batch should be a 1D tensor of size = batch size
        while len(self.loss_batch.shape) > 1:
            # average over the last dimension
            self.loss_batch = torch.mean(self.loss_batch, -1)

        # then return the reduced loss
        if self.reduction == 'sum':
            return self.loss_batch.sum()
        else:  # self.reduction == 'mean'
            return self.loss_batch.mean()
