import argparse
import h5py
import numpy as np
import os
from pathlib import Path
import time
from utils.datasets import DATASETS, prepare, get_fn
from Submission import Task1

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

def run(dataset, k=30):
    task = 'task1'
    print(f'Running {task} on {dataset}, k={k}')
    prepare(dataset, task)
    index_identifier = "BrownCICESE"

    # get the dataset
    fn, _ = get_fn(dataset, task)
    f = h5py.File(fn)
    data_disk = DATASETS[dataset][task]['data'](f)
    N,D = data_disk.shape
    
    # initialize index and add dataset items in batches
    num_bits = 4
    num_neighbors = 64
    time_start = time.time()
    index = Task1(N, D, num_neighbors, num_bits)

    ''' train quantizer '''
    print("Training quantizer...")
    start = 0
    while start < N/4:
        end = min(start + 100000, N)
        subset = np.array(data_disk[start:end]).astype(np.float32)
        index.train(subset)
        start = end

    ''' load the data into the index '''
    print("Loading data into index...")
    start = 0
    while start < N:
        end = min(start + 100000, N)
        subset = np.array(data_disk[start:end]).astype(np.float32)
        index.add_items(subset)
        start = end

    # ''' perform the graph construction '''
    num_candidates = 128
    num_hops = 64
    num_iterations = 1
    index.build(num_candidates, num_hops, num_iterations)
    elapsed_build = time.time() - time_start
    print(f"Time taken to build index: {elapsed_build:.3f} seconds")

    queries = np.array(DATASETS[dataset][task]['queries'](f)).astype(np.float32)
    num_queries, _ = queries.shape
    for beam_size in [30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 200]:
        identifier = f"index=({index_identifier}),query=(b={beam_size})"

        print(f"Querying with beam size: {beam_size}")
        time_start = time.time()
        neighbors,distances = index.search(queries, k=k, beam_size=beam_size)
        time_end = time.time()
        elapsed_search = time_end - time_start
        throughput = num_queries / elapsed_search
        print(f"Time taken to query: {elapsed_search:.3f} seconds, throughput: {throughput:.2f} queries/sec")

        # convert to 0-indexing
        neighbors += 1
        store_results(os.path.join("results/", dataset, task, f"{identifier}.h5"), index_identifier, dataset, task, distances, neighbors, elapsed_build, elapsed_search, identifier)

    print("done!")
    f.close()

    

 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        choices=DATASETS.keys(),
        default='ccnews-small'
    )
    args = parser.parse_args()
    run(args.dataset)