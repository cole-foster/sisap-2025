import os
import h5py
import numpy as np
import time
import math
import hnswlib

def measure_accuracy(graph, graph_gt, k):
    N = graph.shape[0]
    if graph_gt.shape[0] != N:
        raise ValueError("Ground truth neighbors must have the same number of samples as the neighbors array.")\
    
    # measure recall
    accuracy = 0
    for i in range(N):
        neighbors = graph[i, 1:k+1] # get rid of the first
        neighbors_gt = graph_gt[i, 1:k+1]  # get rid of first
        accuracy += len(set(neighbors) & set(neighbors_gt))
    accuracy = accuracy / (N * k)
    return accuracy


def main(dataset, subset_size):
    k = 15
    M = 32
    ef_construction = 32
    print(f"Begin kNN graph construciton by HNSW:, M={M}, ef_construction={ef_construction}")
    print(f" * dataset: {dataset}")
    print(f" * subset size: {subset_size}")
    print(f" * M: {M}")
    print(f" * ef_construction: {ef_construction}")

    # load the dataset
    fn = "/users/cfoste18/data/cfoste18/knn-construction/data/gooaq/gooaq.h5"
    f = h5py.File(fn)
    print("Dataset Details:")
    print(f.keys())
    print(f" * dataset: {dataset}")
    data_disk = f['train']
    N,D = data_disk.shape
    print(" * size:", data_disk.shape)
    print(" * data_type:", data_disk.dtype)
    if subset_size > 0 and subset_size < N:
        N = subset_size
    print(f" * using subset of size: {N}")
    data = data_disk[:N]

    # building the hnsw index
    time_start = time.time()
    p = hnswlib.Index(space='ip', dim=D)  # 'ip' is for inner product
    p.init_index(max_elements=N, ef_construction=ef_construction, M=M)
    p.add_items(data)
    time_end = time.time()
    time_construction = time_end - time_start
    print(f"Time taken to build HNSW index: {time_construction:.3f} seconds")

    # querying the entire dataset
    p.set_ef(24)
    neighbors, _ = p.knn_query(data, k=k+1)
    time_end = time.time()
    time_query = time_end - time_start
    print(f"Total knn graph construction time: {time_query:.3f} seconds")
    print(neighbors.shape)

    # evaluating ground truth
    knn_gt = np.fromfile("/users/cfoste18/data/cfoste18/knn-construction/data/knn-gooaq-N-3001496-k-32-int32.bin", dtype=np.int32)
    knn_gt = knn_gt.reshape((3001496, 32))
    knn_gt -= 1  # convert to zero-based 
    print(knn_gt.shape)

    # measure accuracy
    accuracy = measure_accuracy(neighbors, knn_gt, k)
    print(f"Accuracy of graph: {accuracy * 100:.3f}")

    


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Approximate k-NN graph construction")
    parser.add_argument('--dataset', default='gooaq')
    parser.add_argument('-N', type=int, default=0, help='dataset size to use')
    args = parser.parse_args()
    main(args.dataset, args.N)
    print(f"Done! Have a good day! :)")