from hardness_weighted_sampler.unreduced_loss_function.loss_wrapper import UnreducedLossFunction
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import unittest


# Segmentations for test in 2D
# Prediction (logits)
NUM_CLASSES = 5
S1 = F.one_hot(torch.tensor([[0, 0, 0, 0],
                             [1, 0, 3, 1],
                             [1, 0, 0, 1],
                             [1, 1, 1, 1]]),
               num_classes=NUM_CLASSES).permute(2, 0, 1).float()
S1 = F.softmax(S1, dim=1).unsqueeze(0)

list_pred = [0.1 * S1, S1, 10. * S1]
batch_pred = torch.cat(list_pred, 0).cpu()

# Ground truth
S2 = torch.tensor([[0, 0, 3, 0],
                   [1, 0, 3, 1],
                   [1, 0, 0, 1],
                   [1, 0, 0, 1]]).unsqueeze(0)
list_gt = [S2] * 3
batch_gt = torch.cat(list_gt, 0).cpu()


class TestUnreducedLossFunctionWrapper(unittest.TestCase):
    """
    Test class for the unreduced loss function wrapper
    using the cross entropy loss.
    """

    def test_example_2d_cpu(self):
        """
        Test the consistency between the original CrossEntropyLoss
        and the UnreducedCrossEntropyLoss.
        """
        ce = nn.CrossEntropyLoss(reduction='mean')
        unreduced_ce = UnreducedLossFunction(
            nn.CrossEntropyLoss(), reduction='mean')

        # compute the batch loss with PyTorch CrossEntropyLoss
        true_list_loss = []
        for i in range(len(list_pred)):
            true_list_loss.append(
                ce.forward(
                    list_pred[i].cpu(),
                    list_gt[i].cpu()))
        ref_batch_loss = torch.stack(true_list_loss).numpy()
        ref_mean_loss = ce.forward(batch_pred.cpu(), batch_gt.cpu())

        # compute it with the UnreducedCrossEntropyLoss
        mean_loss = unreduced_ce(batch_pred.cpu(), batch_gt.cpu())
        batch_loss = unreduced_ce.loss_batch.numpy()

        # check that the returned mean loss is correct
        self.assertAlmostEqual(
            ref_mean_loss.numpy(),
            mean_loss.numpy(),
            places=3
        )

        # check that the batch of loss values are close
        self.assertAlmostEqual(
            np.sum(np.abs(batch_loss - ref_batch_loss)),
            0.,
            places=3
        )

    def test_example_2d_gpu(self):
        """
        Test the consistency between the original CrossEntropyLoss
        and the UnreducedCrossEntropyLoss.
        """
        if not torch.cuda.is_available():
            print('No GPU available. Skip test of the unreduced loss wrapper on GPU')
        else:
            ce = nn.CrossEntropyLoss(reduction='mean')
            unreduced_ce = UnreducedLossFunction(
                nn.CrossEntropyLoss(), reduction='mean')

            # compute the batch loss with PyTorch CrossEntropyLoss
            true_list_loss = []
            for i in range(len(list_pred)):
                true_list_loss.append(
                    ce.forward(
                        list_pred[i].cuda(),
                        list_gt[i].cuda()))
            ref_batch_loss = torch.stack(true_list_loss).cpu().numpy()
            ref_mean_loss = ce.forward(batch_pred.cuda(), batch_gt.cuda())

            # compute it with the UnreducedCrossEntropyLoss
            mean_loss = unreduced_ce(batch_pred.cuda(), batch_gt.cuda())
            batch_loss = unreduced_ce.loss_batch.cpu().numpy()

            # check that the returned mean loss is correct
            self.assertAlmostEqual(
                ref_mean_loss.cpu().numpy(),
                mean_loss.cpu().numpy(),
                places=3
            )

            # check that the batch of loss values are close
            self.assertAlmostEqual(
                np.sum(np.abs(batch_loss - ref_batch_loss)),
                0.,
                places=3
            )
