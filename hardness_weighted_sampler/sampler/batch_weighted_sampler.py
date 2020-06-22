from torch.utils.data import BatchSampler
from hardness_weighted_sampler.sampler.weighted_sampler import WeightedSampler


class BatchWeightedSampler(BatchSampler):
    def __init__(self, sampler, batch_size, drop_last=False):
        """
        Custom Batch Sampler that calls the sampler once per iteration
        instead of once per epoch.
        An epoch consists in n iterations, where n is equal
        to the number of examples in dataset.
        When the sampler is an instance of WeightedSampler,
        this allows to implement dynamic sampling methods
        that change after each iteration.
        :param sampler: WeightedSampler; a PyTorch sampler
        :param batch_size: int; number of samples per batch.
        :param drop_last: bool; if True, incomplete batch at the end
        of an epoch are dropped.
        """
        assert isinstance(sampler, WeightedSampler), \
            "The sampler used in the BatchWeightedSampler must be a WeightedSampler"
        super(BatchWeightedSampler, self).__init__(
            sampler,
            batch_size,
            drop_last
        )

    @property
    def num_samples(self):
        return self.sampler.num_samples

    @property
    def beta(self):
        return self.sampler.beta

    def update_weights(self, batch_new_weights, indices):
        """
        Update the weights for the last batch.
        The indices corresponding the the weights in batch_new_weights
        should be the indices that have been copied into self.batch
        :param batch_new_weights: float or double array; new weights value for the last batch.
        :param indices: int list; indices of the samples to update.
        """
        assert len(indices) == batch_new_weights.size()[0], "number of weights in " \
                                                               "input batch does not " \
                                                               "correspond to the number " \
                                                               "of indices."
        # Update the weights for all the indices in self.batch
        for idx, new_weight in zip(indices, batch_new_weights):
            self.sampler.update_weight(idx, new_weight)

    def get_importance_sampling_weights(self, batch_new_weights, indices):
        importance_weights = self.sampler.get_importance_sampling_weights(
            batch_new_weights, indices)
        return importance_weights

    def __iter__(self):
        for _ in range(len(self)):
            batch = self.sampler.draw_samples(self.batch_size)
            self.batch = [x for x in batch]
            yield batch

    def __len__(self):
        """
        :return: int; number of batches per epoch.
        """
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
