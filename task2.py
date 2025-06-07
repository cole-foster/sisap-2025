import argparse
import h5py
import numpy as np
import os
from pathlib import Path
import time
from utils.datasets import DATASETS, prepare, get_fn
import Submission

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
    index_identifier = "BrownCICESE"

    # get the dataset
    fn, _ = get_fn(dataset, task)
    f = h5py.File(fn)
    data = np.array(DATASETS[dataset][task]['data'](f)).astype(np.float32)
    N,D = data.shape

    # initialize index
    time_start = time.time()
    index = Submission.Graph(N, D)
    index.add_items(data)
    elapsed_build = time.time() - time_start
    f.close()

    # create knn graph
    time_start = time.time()
    index.create_knn()
    neighbors,distances = index.return_knn(k)
    elapsed_search = time.time() - time_start
    print(f"kNN Graph construction in {elapsed_search}s.")
    neighbors = neighbors + 1 # FAISS is 0-indexed, groundtruth is 1-indexed
    identifier = f"index=({index_identifier})"
    store_results(os.path.join("results/", dataset, task, f"{identifier}.h5"), index_identifier, dataset, task, distances, neighbors, elapsed_build, elapsed_search, identifier)






    # now create the graph





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        choices=DATASETS.keys(),
        default='ccnews-small'
    )
    args = parser.parse_args()
    run(args.dataset)