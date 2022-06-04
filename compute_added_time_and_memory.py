from hardness_weighted_sampler.sampler.weighted_sampler import WeightedSampler
from hardness_weighted_sampler.sampler.batch_weighted_sampler import BatchWeightedSampler
from argparse import ArgumentParser
from time import time
import numpy as np

parser = ArgumentParser()
parser.add_argument('--num_samples', required=True, type=int,
                    help='Number of elements in the training dataset on which the sampling has to be performed')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--beta', type=float, default=100)

N_ITER = 10000  # Number of times the sampling operation will be performed


def main(args):
    print('Hardness weighted sampling for %d samples / batch size=%d / beta=%.2f' %
          (args.num_samples, args.batch_size, args.beta))
    # Prepare the initial random weights / loss values
    vec = np.ones(args.num_samples, np.float64)
    weight = np.random.normal(loc=vec, scale=0.1*vec, size=args.num_samples)
    memory_size = weight.size * weight.itemsize
    print('Additional memory on CPU: %f MB' % (memory_size / (1024 * 1024)))

    # Initialize the hardness weighted sampler
    sampler = WeightedSampler(weights_init=weight, beta=args.beta)
    batch_sampler = BatchWeightedSampler(
        sampler=sampler, batch_size=args.batch_size)

    # Measure sampling time
    n_epoch = 1 + N_ITER // args.num_samples
    iter = 0
    t_start = time()
    for e in range(n_epoch):
        for batch in batch_sampler:  # Sample batches with the hardness weighted sampler
            iter +=1
            if iter >= N_ITER:
                break
    t_end = time()
    t_diff = t_end - t_start
    print('Additional time on CPU: %f seconds per iteration' % (t_diff / iter))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
