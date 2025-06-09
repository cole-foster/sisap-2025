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
    queries = np.array(DATASETS[dataset][task]['queries'](f)).astype(np.float32)
    gt_knn = np.array(DATASETS[dataset][task]['gt_I'](f)).astype(np.uint32)
    N,D = data_disk.shape
    num_queries, _ = queries.shape
    

    # initialize index and add dataset items in batches
    num_bits = 0
    num_neighbors = 64
    time_start = time.time()
    index = Task1(N, D, num_neighbors, num_bits)

    ''' train quantizer '''
    start = 0
    while start < N:
        end = min(start + 200000, N)
        subset = np.array(data_disk[start:end]).astype(np.float32)
        index.train(subset)
        start = end

    ''' load the data into the index '''
    start = 0
    while start < N:
        end = min(start + 200000, N)
        subset = np.array(data_disk[start:end]).astype(np.float32)
        index.add_items(subset)
        start = end

    ''' perform the graph construction '''
    num_candidates = 200
    index.build(num_candidates, 100, 2)
    elapsed_build = time.time() - time_start
    print(f"Time taken to build index: {elapsed_build:.3f} seconds")

    for beam_size in [30, 35, 40, 45, 50, 60, 70, 80, 90]:

        print(f"Querying with beam size: {beam_size}")
        time_start = time.time()
        neighbors,_ = index.search(queries, k=k, beam_size=beam_size)
        time_end = time.time()
        elapsed_query = time_end - time_start
        throughput = num_queries / elapsed_query

        # convert to 0-indexing
        neighbors += 1

        # measure recall
        recall = 0
        for i in range(num_queries):
            est_neighbors = neighbors[i, :k]
            gt_neighbors = gt_knn[i, :k]
            recall += len(set(est_neighbors) & set(gt_neighbors))
        recall /= (num_queries * k)
        print(f" * beam_size={beam_size}, recall={recall:.4f}, throughput={throughput:.2f} queries/sec")


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