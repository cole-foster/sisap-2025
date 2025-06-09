import argparse
import h5py
import numpy as np
import os
from pathlib import Path
import time
from utils.datasets import DATASETS, prepare, get_fn
# import Submission
import hnswlib

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

    # get the dataset
    fn, _ = get_fn(dataset, task)
    f = h5py.File(fn)
    data = np.array(DATASETS[dataset][task]['data'](f)).astype(np.float32)
    N,D = data.shape
    queries = np.array(DATASETS[dataset][task]['queries'](f)).astype(np.float32)
    num_queries,_ = queries.shape
    knns = np.array(DATASETS[dataset][task]['gt_I'](f)).astype(np.uint32)
    f.close()

    # building the hnsw index
    print("Begin constructing HNSW...")
    time_start = time.time()
    p = hnswlib.Index(space='ip', dim=D)  # 'ip' is for inner product
    p.init_index(max_elements=N, ef_construction=200, M=64)
    p.add_items(data)
    time_end = time.time()
    time_construction = time_end - time_start
    print(f"Time taken to build HNSW index: {time_construction:.3f} seconds")

    # querying the entire dataset
    for ef in [30, 35, 40, 45, 50, 80, 100, 120, 150]:
        p.set_ef(ef)
        time_start = time.time()
        neighbors,_ = p.knn_query(queries, k=k)
        time_end = time.time()
        time_query = time_end - time_start
        throughput = num_queries / time_query

        # measure recall
        recall = 0
        for i in range(num_queries):
            gt_neighbors = knns[i, 0:k] - 1
            est_neighbors= neighbors[i, 0:k]
            recall += len(set(gt_neighbors) & set(est_neighbors))
        recall /= (num_queries * k)
        print(f" * ef={ef}, recall={recall:.4f}, throughput={throughput:.2f} queries/sec")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        choices=DATASETS.keys(),
        default='ccnews-small'
    )
    args = parser.parse_args()
    run(args.dataset)