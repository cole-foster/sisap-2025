import argparse
import h5py
import numpy as np
import os
from pathlib import Path
import time
from utils.datasets import DATASETS, prepare, get_fn

def store_results(dst, algo, dataset, task, D, I, buildtime, querytime, params):
    os.makedirs(Path(dst).parent, exist_ok=True)
    f = h5py.File(dst, 'w')
    f.attrs['algo'] = algo
    f.attrs['dataset'] = dataset
    f.attrs['task'] = task
    f.attrs['buildtime'] = buildtime
    f.attrs['querytime'] = querytime
    f.attrs['params'] = params
    f.create_dataset('knns', I.shape, dtype=I.dtype)[:] = I
    f.create_dataset('dists', D.shape, dtype=D.dtype)[:] = D
    f.close()

def run(dataset, k=15):
    task = 'task2'
    print(f'Running {task} on {dataset}, k={k}')
    prepare(dataset, task)

    # get the dataset
    fn, _ = get_fn(dataset, task)
    f = h5py.File(fn)
    data = np.array(DATASETS[dataset][task]['data'](f))
    queries = np.array(DATASETS[dataset][task]['queries'](f))
    f.close()
    n, d = data.shape

    # done!
    print(f"done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        choices=DATASETS.keys(),
        default='ccnews-small'
    )
    args = parser.parse_args()
    run(args.dataset)