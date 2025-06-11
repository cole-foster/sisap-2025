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

def run(dataset, k=30, flag_test=False):
    task = 'task1'
    print(f'Running {task} on {dataset}, k={k}')
    prepare(dataset, task)

    # get the dataset
    fn, _ = get_fn(dataset, task)
    f = h5py.File(fn)
    data_disk = DATASETS[dataset][task]['data'](f)
    N,D = data_disk.shape

    if (dataset == 'ccnews-small'):
        num_neighbors = 36
        num_candidates = 48
        num_hops = 64
        num_nodes_top = 100000
        num_iterations = 1
    elif (dataset == 'pubmed'):
        num_neighbors = 32
        num_candidates = 48
        num_hops = 64
        num_nodes_top = 400000
        num_iterations = 1
    index_identifier = "BrownCICESE"

    
    # initialize index and add dataset items in batches
    num_bits = 8
    time_start = time.time()
    index = Task1(N, D, num_neighbors, num_bits)

    ''' train quantizer '''
    print("Training quantizer...")
    start = 0
    while start < N:
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
    index.build(num_candidates, num_hops, num_nodes_top, num_iterations)
    elapsed_build = time.time() - time_start
    print(f"Time taken to build index: {elapsed_build:.3f} seconds")

    queries = np.array(DATASETS[dataset][task]['queries'](f)).astype(np.float32)
    num_queries, _ = queries.shape
    for beam_size in [30, 30, 35, 40, 45, 50, 60, 70, 80, 100, 150, 200, 300, 400, 500, 1000]:
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

        # measure recall
        if (flag_test):
            gt_I = DATASETS[dataset][task]['gt_I'](f)

            # compute recall
            recall = 0
            for i in range(num_queries):
                gt_neighbors = gt_I[i, :k]
                est_neighbors = neighbors[i, :k]
                recall += len(set(gt_neighbors) & set(est_neighbors))
            recall /= (num_queries * k)
            print(f"Recall for beam size {beam_size}: {recall:.4f}")


    print("done!")
    f.close()

    

 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        choices=DATASETS.keys(),
        default='ccnews-small'
    )
    parser.add_argument(
        '--flag_test',
        action='store_true',
        help='Set this flag to enable test mode'
    )
    args = parser.parse_args()
    run(args.dataset, flag_test=args.flag_test)