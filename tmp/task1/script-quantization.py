import numpy as np
import h5py
# import torch
# import tensorflow as tf
import os
import math
# import pandas as pd
# from datasets import load_dataset
from sklearn.preprocessing import normalize
from sklearn.metrics import precision_score
from typing import Tuple

DATASETS = {
    'gooaq': '/users/cfoste18/data/cfoste18/knn-construction/data/gooaq/gooaq.h5',
    'pubmed23': '/users/cfoste18/data/cfoste18/knn-construction/data/pubmed23/pubmed23.h5',
    'ccnews': '/users/cfoste18/data/cfoste18/knn-construction/data/ccnews/ccnews.h5',
}
def load_dataset(dataset: str) -> Tuple[np.ndarray]:
    """
    Loads the benchmark file of vectors (train, itest/otest) and returns base and queries as PyTorch tensors.
    """


def load_testset(dataset: str, testset: str='otest') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the benchmark file of vectors (train, itest/otest) and returns base and queries as PyTorch tensors.
    """
    file_path = DATASETS[dataset]
    with h5py.File(file_path, 'r') as f:
        queries = np.array(f[testset]['queries']).astype(np.float32)  # Query vectors from the 'otest' group
        dists = np.array(f[testset]['dists']).astype(np.float32)  # Distances to nearest neighbors
        knns = np.array(f[testset]['knns']).astype(np.uint32)  # Indices of nearest neighbors

    return queries, dists, knns

import sys
def get_size_in_bytes(obj):
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return sum(get_size_in_bytes(item) for item in obj)
    elif isinstance(obj, dict):
        return sum(get_size_in_bytes(k) + get_size_in_bytes(v) for k, v in obj.items())
    else:
        return sys.getsizeof(obj)

# Generates an n x n rotation matrix Q. The columns of Q are orthonormal vectors, and the matrix has a determinant of 1.
def generate_rotation_matrix_Q(d: int, seed: int = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)

    # Generate a random matrix
    random_matrix = np.random.randn(d, d).astype(np.float32)

    # Apply QR decomposition to obtain an orthonormal matrix Q
    Q,_ = np.linalg.qr(random_matrix)

    # Ensure that Q is a rotation matrix (det(Q) = 1)
    det_Q = np.linalg.det(Q)
    if det_Q < 0:
        # Adjust the sign of the last column to ensure det(Q) = 1
        Q[:, -1] *= -1  # Multiply the last column by -1
    return Q



# convert the float vec to a packed bitstring
def quantize_vector(x, num_bits, l_arr, r_arr):
    assert num_bits <= 8, "num_bits must be less than or equal to 8"
    d = x.shape[0]  # dimension of the vector

    L = 2**num_bits # number of quantization levels
    delta = (r_arr - l_arr) / (L - 1) # the step size between quantization levels
    q = np.floor((x - l_arr) / delta + 0.5) # the level it belongs to
    q = np.clip(q, 0, L - 1).astype(np.uint8)
    
    # Convert each integer to a list of its B bits
    bit_array = np.zeros(d*num_bits, dtype=np.uint8) 
    bit_array = (q[:,np.newaxis] >> np.arange(num_bits - 1, -1, -1)) & 1  # Extract bits from the integer
    packed_bitstring = np.packbits(bit_array, axis=None, bitorder='big')
    return packed_bitstring

# reconstruct float from bitstring
def dequantize_vector(q_packed, num_bits, l_arr, r_arr):
    d = l_arr.shape[0]  # dimension of the vector

    # convert from bitstring to integer vec
    total_bits = d * num_bits
    bits = np.unpackbits(q_packed, bitorder='big')[:total_bits]
    bits = bits.reshape(d, num_bits)  # Reshape to (d, num_bits)
    powers = 2 ** np.arange(num_bits - 1, -1, -1)
    q = bits.dot(powers)

    # convert to flaot vec
    L = 2**num_bits
    delta = (r_arr - l_arr) / (L - 1)
    x_hat = q * delta + l_arr
    return x_hat

def train_quantizer(data_sample):
    l_arr = np.min(data_sample, axis=0)  # Lower bound
    u_arr = np.max(data_sample, axis=0)  # Upper bound
    return l_arr,u_arr

def compress_dataset(data, num_bits):
    N,D = data.shape

    # get the min/max bounds for each dimension
    l_arr,r_arr = train_quantizer(data)
    l_arr = np.where((l_arr <= 0) & (l_arr > -0.0001), -0.0001, l_arr)  # avoid zero
    r_arr = np.where((r_arr >= 0) & (r_arr < 0.0001), 0.0001, r_arr)  # avoid zero

    # quantize the dataset, pack to bits
    size_per_el = np.ceil(D*num_bits / 8).astype(int)  # size in bytes per element
    compressed_dataset = np.zeros((N,size_per_el), dtype=np.uint8)
    for i in range(N):
        compressed_dataset[i,:] = quantize_vector(data[i,:], num_bits, l_arr, r_arr)

    # return 
    result = {
        'compressed_dataset': compressed_dataset,
        'N': N,
        'D': D,
        'l_arr': l_arr,
        'r_arr': r_arr,
        'num_bits': num_bits
    }
    return result

def uncompress_dataset(compressed_dataset_object):
    N,D = compressed_dataset_object['N'], compressed_dataset_object['D']
    data = np.zeros((N,D), dtype=np.float32)

    # decompress each row
    compressed_dataset = compressed_dataset_object['compressed_dataset']
    num_bits = compressed_dataset_object['num_bits']
    l_arr = compressed_dataset_object['l_arr']
    r_arr = compressed_dataset_object['r_arr']
    for i in range(N):
        data[i,:] = dequantize_vector(compressed_dataset[i,:], num_bits, l_arr, r_arr)
    return data

# compute similarities
def knn(queries, dataset, k):
    distances = 1 - np.dot(queries, dataset.T)  # shape: (num_queries, num_base)
    knn_indices = np.argsort(distances, axis=1)[:, :k]  # shape: (num_queries, k)
    knn_distances = np.take_along_axis(distances, knn_indices, axis=1)
    return knn_indices, knn_distances



def main(dataset):

    # load the dataset
    file_path = DATASETS[dataset]
    f = h5py.File(file_path, 'r')
    dataset_disk = f['train']
    print("dataset:", dataset_disk.shape)
    N,D = dataset_disk.shape
    print("No rotation applied to the dataset")

    # rotate the dataset
    # print(f"Rotating the dataset")
    # Q = generate_rotation_matrix_Q(D, seed=12)
    # rotated_base = np.matmul(dataset_disk, Q.T).astype(np.float32)

    # load the queries
    num_queries = 1000 #queries.shape[0]
    queries = np.array(f['otest']['queries']).astype(np.float32)  # Query vectors from the 'otest' group
    knn_gt = np.array(f['otest']['knns']).astype(np.uint32)  # Indices of nearest neighbors
    # queries = np.matmul(queries, Q.T).astype(np.float32)  # Rotate the queries as well

    # quantize with different numbers of bits
    for num_bits in [3, 4, 5, 6, 7, 8]:
        print(f"Compressing with num_bits={num_bits}")
        result = compress_dataset(dataset_disk, num_bits=num_bits)
        compressed_size = get_size_in_bytes(result)
        print("compressed_size: ", compressed_size/1024/1024)

        # uncompress the dataset
        uncompressed_dataset = uncompress_dataset(result)
        uncompressed_size = get_size_in_bytes(uncompressed_dataset)
        print("uncompressed_size: ", uncompressed_size/1024/1024)
        
        # find the knn of the query over this, compute recall
        knn_indices, _ = knn(queries[:num_queries,:], uncompressed_dataset, k=30)
        recall_arr = [0, 0, 0, 0, 0]
        for i in range(num_queries):
            for j, k in enumerate([1, 5, 10, 20, 30]):
                recall_arr[j] += len(set(knn_indices[i,:k]) & set(knn_gt[i,:k] - 1))
        for j, k in enumerate([1, 5, 10, 20, 30]):
            recall_arr[j] /= (num_queries * k)
            print(f"{recall_arr[j]:.4f}", end=',')
        print("")

        del result, uncompressed_dataset


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantization")
    parser.add_argument('--dataset', choices=DATASETS.keys(), default='ccnews')
    args = parser.parse_args()
    main(args.dataset)
    print(f"Done! Have a good day! :)")