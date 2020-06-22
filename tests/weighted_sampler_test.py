from hardness_weighted_sampler.sampler.weighted_sampler import WeightedSampler
from hardness_weighted_sampler.sampler.batch_weighted_sampler import BatchWeightedSampler
from torch.utils.data import Sampler
from scipy.special import softmax
import numpy as np
import torch
import unittest


class TestWeightedSampler(unittest.TestCase):
    """
    Test class for the weighted sampler.
    """

    def test_init(self):
        """
        Test that the weighted sampler is initialized correctly.
        """
        num_samples = 10
        sampler = WeightedSampler(num_samples=num_samples)
        # test initialization with uniform weights + noise
        self.assertTrue(
            np.allclose(sampler.weights.numpy(),
                        torch.ones(num_samples).numpy(),
                        rtol=0.01)
        )

        init_weights = torch.tensor([0, 1, 2, 3, 4, 5])
        sampler = WeightedSampler(weights_init=init_weights)
        # test initialization of the weights from previous weights
        self.assertTrue(
            np.allclose(sampler.weights.numpy(), init_weights.numpy(), rtol=0.0001)
        )
        # num_sample must have been initialized
        self.assertTrue(sampler.num_samples == init_weights.shape[0])

        # an error should be raised if num_samples and weights_init are specified
        # but the size of weights_init is not equal to num_samples
        with self.assertRaises(AssertionError):
            WeightedSampler(num_samples=12, weights_init=init_weights)

    def test_weighted_sampler(self):
        """
        Test that the weighted sampling is correct,
        i.e. it draws indices with respect to the distribution:
            softmax(beta * weights)
        """
        # weights to test
        weights_list = [0., 2., 4., 6.]
        weights_torch = torch.tensor(weights_list)

        # initialised the sampler
        sampler = WeightedSampler(weights_init=weights_torch, beta=0.1)

        # compute proba with numpy and scipy for comparison
        weights_np = np.array(weights_list)
        proba_np = softmax(sampler.beta * weights_np)

        # estimate empirical proba by sampling
        counts = np.zeros(len(weights_list))
        n_sampling = 20000
        for _ in range(n_sampling):
            s = sampler.draw_samples(1)[0]
            counts[s] += 1.
        proba_empirical = counts / n_sampling
        self.assertAlmostEqual(
            np.mean(
                np.abs(
                    proba_np -
                    proba_empirical)),
            0.,
            places=2)

    def test_hardest_samples_indices(self):
        # weights to test
        weights_list = [10., 2., 14., 6.]
        weights_torch = torch.tensor(weights_list)

        # initialised the sampler
        sampler = WeightedSampler(weights_init=weights_torch, beta=1.)

        hardest_samples = sampler.hardest_samples_indices(num=2)
        # the two hardest samples must be indices 2 and 0 here
        self.assertEqual(hardest_samples[0], 2)
        self.assertEqual(hardest_samples[1], 0)


class TestBatchWeightedSampler(unittest.TestCase):
    """
    Test class for the batch weighted sampler.
    """

    def test_init(self):
        """
        Test the initialization of the batch weighted sampler.
        """
        # initialize the batch sampler
        batch_size = 4
        init_weights = torch.tensor([0, 1, 2, 3, 4, 5])
        sampler = WeightedSampler(weights_init=init_weights, beta=27)
        batch_sampler = BatchWeightedSampler(
            sampler=sampler, batch_size=batch_size)

        # test initialization of the robustness parameter (beta).
        self.assertTrue(batch_sampler.sampler.beta == 27)

        # test initialization of the weights from previous weights
        self.assertTrue(
            torch.equal(
                batch_sampler.sampler.weights,
                init_weights.float()))

        # an error should be raised if the sampler is not a WeightedSampler
        with self.assertRaises(AssertionError):
            classic_sampler = Sampler([0, 1, 2, 3, 4, 5])
            batch_sampler = BatchWeightedSampler(
                sampler=classic_sampler, batch_size=batch_size)

    def test_weighted_sampler(self):
        """
        Test that the weighted sampling is correct,
        i.e. it draws indices with respect to the distribution:
            softmax(beta * weights)
        """
        # weights to test
        batch_size = 1
        weights_list = [0., 1., 2., 3.]
        weights_torch = torch.tensor(weights_list)

        # initialised the batch sampler
        sampler = WeightedSampler(weights_init=weights_torch, beta=0.1)
        batch_sampler = BatchWeightedSampler(
            sampler=sampler, batch_size=batch_size)

        # compute proba with numpy and scipy for comparison
        weights_np = np.array(weights_list)
        proba_np = softmax(sampler.beta * weights_np)

        # estimate empirical proba by sampling
        idx_counts = np.zeros(len(weights_list))
        n_epoch_sampling = 200000
        for epoch in range(n_epoch_sampling):
            for batch in batch_sampler:
                for idx in batch:
                    idx_counts[idx] += 1.

        # check convergence to the true distribution
        proba_empirical = idx_counts / np.sum(idx_counts)
        self.assertAlmostEqual(
            np.mean(
                np.abs(
                    proba_np -
                    proba_empirical)),
            0.,
            places=2)

    def test_update_weights(self):
        """
        Test that the weights can be updated dynamically properly.
        """
        def dummy_loss(batch):
            """
            Dummy loss where the loss is equal to the index.
            :param batch: batch of indices
            """
            return torch.tensor(np.array(batch, dtype=float)).float()
        # start with uniform distribution
        batch_size = 1
        num_samples = 4
        sampler = WeightedSampler(num_samples=num_samples, beta=0.1)
        batch_sampler = BatchWeightedSampler(
            sampler=sampler, batch_size=batch_size)

        # compute the true proba for the dummy loss with numpy and scipy for
        # comparison
        weights_np = np.array([i for i in range(num_samples)])
        proba_np = softmax(batch_sampler.beta * weights_np)

        # learn the loss history and estimate the empirical proba by sampling
        idx_counts = np.zeros_like(weights_np)
        n_epoch_sampling = 10000
        for epoch in range(n_epoch_sampling):
            for batch in batch_sampler:
                # let time for the loss history to be learnt
                # count the cumulative occurence of the indices
                for idx in batch:
                    idx_counts[idx] += 1.
                # update the loss history dynamically
                batch_loss = dummy_loss(batch)
                batch_sampler.update_weights(
                    batch_new_weights=batch_loss, indices=batch)

        # check convergence to the true distribution
        proba_empirical = idx_counts / np.sum(idx_counts)
        self.assertAlmostEqual(
            np.mean(
                np.abs(
                    proba_np -
                    proba_empirical)),
            0.,
            places=2)

    def test_consistency_uniform_distribution(self):
        """
        Test that when beta=0 the hardness weighted sampler
        behaves like a uniform sampler.
        """
        def dummy_loss(batch):
            """
            Dummy loss where the loss is equal to the index.
            Here it should not matter what is returned,
            since beta is equal to 0.
            Therefore, the hardness weighted sampler should be uniform
            (independently of the stale per-example loss vector).
            :param batch: batch of indices
            """
            return torch.tensor(np.array(batch, dtype=float)).float()
        # start with uniform distribution (warm-up)
        batch_size = 1
        num_samples = 4
        sampler = WeightedSampler(num_samples=num_samples, beta=0.)
        batch_sampler = BatchWeightedSampler(
            sampler=sampler, batch_size=batch_size)

        # compute the true proba (uniform distribution).
        true_proba_np = np.array([1./num_samples]*num_samples)

        # learn the loss history and estimate the empirical proba by sampling
        idx_counts = np.zeros_like(true_proba_np)
        n_epoch_sampling = 10000
        for epoch in range(n_epoch_sampling):
            for batch in batch_sampler:
                # count the cumulative occurence of the indices
                for idx in batch:
                    idx_counts[idx] += 1.
                # update the loss history dynamically
                batch_loss = dummy_loss(batch)
                batch_sampler.update_weights(
                    batch_new_weights=batch_loss, indices=batch)

        # check convergence to the true distribution
        proba_empirical = idx_counts / np.sum(idx_counts)
        self.assertAlmostEqual(
            np.mean(
                np.abs(
                    true_proba_np -
                    proba_empirical)),
            0.,
            places=2)

    def test_importance_sampling(self):
        """
        Test that the computation of the importance weights
        for the importance sampling is correct.
        """
        def dummy_loss(batch):
            """
            Dummy loss where the loss is equal to the index.
            :param batch: batch of indices
            """
            return torch.tensor(np.array(batch, dtype=float)).float()
        # Start with a uniform distribution
        batch_size = 1
        num_samples = 4
        beta =1.
        weights_init = torch.tensor([1] * num_samples)
        sampler = WeightedSampler(
            num_samples=num_samples, beta=beta, weights_init=weights_init)
        batch_sampler = BatchWeightedSampler(
            sampler=sampler, batch_size=batch_size)
        for i in range(num_samples):
            true_is_weight = float(np.exp(beta * (i - 1.)))
            loss_val = dummy_loss([i])
            ind = torch.tensor([i])
            is_weight = batch_sampler.get_importance_sampling_weights(
                loss_val, ind
            )
            self.assertAlmostEqual(
                true_is_weight, float(is_weight.numpy()), places=3)


if __name__ == '__main__':
    unittest.main()
