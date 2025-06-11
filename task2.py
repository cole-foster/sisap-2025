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
    data_disk = DATASETS[dataset][task]['data'](f)
    N,D = data_disk.shape

    # parameters for different datasets
    if (dataset == 'ccnews-small'):
        arr_num_neighbors = [64]
        arr_num_hops = [32]
        arr_omap_size = [N//2] 
        arr_num_iterations = [1]
    elif (dataset == 'gooaq'):
        arr_num_neighbors = [32, 36, 48, 64]
        arr_num_hops = [32, 32, 36, 42]
        arr_omap_size = [1000000] * 4
        arr_num_iterations = [1] * 4
    
    # do this many times
    path_list = [] 
    for i in range(len(arr_num_neighbors)):
        num_neighbors = arr_num_neighbors[i]
        num_hops = arr_num_hops[i]
        omap_size = arr_omap_size[i]
        num_iterations = arr_num_iterations[i]
        print(f"Running with num_neighbors={num_neighbors}, num_hops={num_hops}, omap_size={omap_size}, num_iterations={num_iterations}")
        identifier = f"index=({index_identifier}),query=(M={num_neighbors},H={num_hops},O={omap_size},I={num_iterations})"

        # initialize index and add dataset items in batches
        time_start = time.time()
        index = Submission.Task2(N, D)
        start = 0
        while start < N:
            end = min(start + 200000, N)
            subset = np.array(data_disk[start:end]).astype(np.float32)
            index.add_items(subset)
            start = end
        elapsed_build = time.time() - time_start

        # create knn graph
        time_start = time.time()
        neighbors,distances = index.create_knn(k, num_neighbors, num_hops, omap_size, num_iterations)
        elapsed_search = time.time() - time_start
        print(f"kNN Graph construction in {elapsed_search}s.")
        neighbors = neighbors + 1 # FAISS is 0-indexed, groundtruth is 1-indexed
        store_results(os.path.join("results/", dataset, task, f"{identifier}.h5"), index_identifier, dataset, task, distances, neighbors, elapsed_build, elapsed_search, identifier)
        path_list.append(os.path.join("results/", dataset, task, f"{identifier}.h5"))

        # explicitly delete for next iteration just in case
        del index



 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        choices=DATASETS.keys(),
        default='ccnews-small'
    )
    args = parser.parse_args()
    run(args.dataset)