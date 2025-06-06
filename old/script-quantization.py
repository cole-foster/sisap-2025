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
# from tqdm import tqdm

DATASETS = {
    # 'gooaq': 'https://huggingface.co/datasets/sadit/SISAP2025/resolve/main/benchmark-dev-gooaq.h5?download=true',
    # 'pubmed23': 'https://huggingface.co/datasets/sadit/SISAP2025/resolve/main/benchmark-dev-pubmed23.h5?download=true',
    # 'ccnews': 'https://huggingface.co/datasets/sadit/SISAP2025/resolve/main/benchmark-dev-ccnews-fp16.h5?download=true'
    'gooaq': '/users/cfoste18/data/cfoste18/knn-construction/data/gooaq/gooaq.h5',
    'pubmed23': '/users/cfoste18/data/cfoste18/knn-construction/data/pubmed23/pubmed23.h5',
    'ccnews': '/users/cfoste18/data/cfoste18/knn-construction/data/ccnews/ccnews.h5',
}
def load_dataset(dataset: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the benchmark file of vectors (train, itest/otest) and returns base and queries as PyTorch tensors.
    """
    file_path = DATASETS[dataset]
    with h5py.File(file_path, 'r') as f:
        # print("Keys available in the file:", list(f.keys()))
        # Ensure the expected keys are present
        if 'train' not in f or 'otest' not in f:
            raise KeyError("Keys 'train' or 'otest' are not found in the file.")
        # print(f"original_dtype: {f['train'].dtype}")

        base = np.array(f['train']).astype(np.float32)  # Training vectors (base)
        queries = np.array(f['otest']['queries']).astype(np.float32)  # Query vectors from the 'otest' group
        dists = np.array(f['otest']['dists']).astype(np.float32)  # Distances to nearest neighbors
        knns = np.array(f['otest']['knns']).astype(np.uint32)  # Indices of nearest neighbors

    return base, queries, dists, knns

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


# Finds the points where the slope is equal to ±45 degrees, indicating concavity changes.
def find_concavity_changes(array, slope_threshold=1.0, atol=1e-2):
    # if array.dim() != 1:
    #     raise ValueError("The tensor must be one-dimensional (flattened).")
    if array.shape[0] > 2048:
        raise ValueError("Max size of 2048")

    # Calculate dx based on the maximum value of the array
    dx = 10**(math.floor(math.log10(np.max(np.abs(array)).item())) - 2)

    # Compute the first derivative using vectorized finite differences
    first_derivative = (array[1:] - array[:-1]) / dx

    def find_indices(atol):
        condition = (
            (np.abs(first_derivative - slope_threshold) <= atol) |
            (np.abs(first_derivative + slope_threshold) <= atol)
        )
        return np.where(condition)[0].tolist()

    def ensure_two_elements(atol_initial, atol_max, step_size):
        atol = atol_initial
        while atol <= atol_max:
            slope_indices = find_indices(atol)
            if len(slope_indices) >= 2:
                return slope_indices
            atol += step_size  # Increase tolerance
        return slope_indices  # Return best option

    # Search for indices with the initial tolerance
    slope_indices = ensure_two_elements(atol_initial=1e-6, atol_max=1, step_size=1e-1)

    # If not enough indices are found, apply an alternative strategy
    if len(slope_indices) < 2:
        print("Warning: Less than 2 indices found. You may need to adjust your parameters.")

    first_curve = slope_indices[0]
    second_curve = slope_indices[-1]
    return first_curve, second_curve

# etimate upper/lower bounds for inliers based on concavity changes
def estimate_global_concavity_thresholds(vectors: np.ndarray, sample_size: int = 5000, seed: int = None) -> Tuple[int, int]:
    n = vectors.shape[0]
    sample_size = min(sample_size, n)

    if seed is not None:
        np.random.seed(seed)
    indices = np.random.permutation(n)[:sample_size]

    # find the upper/lower bounds for each vector sampled
    lower_limits = []
    upper_limits = []
    for idx in indices:
        sorted_vec = np.sort(vectors[idx])
        lower, upper = find_concavity_changes(sorted_vec)
        lower_limits.append(lower)
        upper_limits.append(upper)

    avg_lower = np.mean(lower_limits)
    avg_upper = np.mean(upper_limits)
    return int(avg_lower), int(avg_upper)


# Encoding function
def orthoQuant_encode(data, lower_limit, upper_limit, num_bits):
    n,d = data.shape
    num_segments = 2**num_bits - 1          # total quantization values
    less_segments = num_segments // 2       # negative quantization values
    more_segments = num_segments - less_segments    # positive quantization values

    # fixed number of outliers on both sides
    mask_og = np.zeros((n, d), dtype=bool)
    mask_og[:,:lower_limit] = 1
    mask_og[:,upper_limit:] = 1

    # getting the outliers for saving in full precision
    non_zero_elements = data[mask_og].reshape(n, -1)
    print(f"shape outliers: {non_zero_elements.shape}")

    # getting the inliers for compression
    mask_in_range = ~mask_og # inliers mask
    mask_neg = (data < 0) & mask_in_range
    mask_pos = (0 <= data) & mask_in_range

    # Fixed number of segments
    # count_neg = np.sum(mask_neg[0]) # total number of lower inliers for first element
    # count_pos = np.sum(mask_pos[0]) # total number of upper inliers for first element

    # uneven number of quantization values, favor the one with more inliers
    num_segments_l = less_segments #more_segments if count_neg > count_pos else less_segments
    num_segments_r = more_segments #num_segments - num_segments_l

    # generate evenly spaced quantization values between [-1,0] and [0,1]
    neg_edges = np.linspace(-1, 0, num_segments_l + 1).reshape(1, -1)
    pos_edges = np.linspace(0, 1, num_segments_r + 1).reshape(1, -1)

    # Scale edges based on thresholds... each row has a custom quantization?    
    sorted_data = np.sort(data, axis=1)
    l = sorted_data[:, lower_limit]  # (n,)
    r = sorted_data[:, upper_limit]  # (n,)
    l = np.where(l > 0, -np.abs(r), l)
    r = np.where(r < 0, np.abs(l), r)
    neg_edges_scaled = l.reshape(-1, 1) * (-1 * neg_edges)
    pos_edges_scaled = r.reshape(-1, 1) * pos_edges

    # iterate through each segment, computing the average value to get a quantization value
    avg_values = np.zeros((n, num_segments+1), dtype=np.float32)
    for seg_idx in range(num_segments_l):

        # get the values within this segment
        lower = neg_edges_scaled[:, seg_idx][:, np.newaxis]
        upper = neg_edges_scaled[:, seg_idx + 1][:, np.newaxis]
        seg_mask = (data >= lower) & (data < upper) & mask_neg
        seg_vals = np.where(seg_mask, data, np.zeros_like(data))

        # compute the average value for this segment, for each row individually
        counts = np.sum(seg_mask,axis=1)
        valid = counts > 0
        sums = np.sum(seg_vals,axis=1)
        avg_values[valid, seg_idx+1] = sums[valid] / counts[valid]

    # iterate through each segment, computing the average value to get a quantization value
    for seg_idx in range(num_segments_r):

        # get the values within this segment
        lower = pos_edges_scaled[:, seg_idx][:, np.newaxis]
        upper = pos_edges_scaled[:, seg_idx + 1][:, np.newaxis]
        seg_mask = (data >= lower) & (data <= upper) & mask_pos
        seg_vals = np.where(seg_mask, data, np.zeros_like(data))

        # compute the average value for this segment, for each row individually
        counts = np.sum(seg_mask,axis=1)
        valid = counts > 0
        sums = np.sum(seg_vals,axis=1)
        avg_values[valid, num_segments_l + seg_idx + 1] = sums[valid] / counts[valid]
    
    mask_out_of_range = (avg_values == 0)
    print(f" num outliers avg_values: {np.sum(mask_out_of_range, axis=None)}")

    # encoding the values for the quantization for each index
    index_matrix = np.zeros_like(data, dtype=np.uint32)
    for seg_idx in range(num_segments_l):
        lower = neg_edges_scaled[:, seg_idx][:, np.newaxis]
        upper = neg_edges_scaled[:, seg_idx + 1][:, np.newaxis]
        seg_mask = (data >= lower) & (data < upper) & mask_neg
        index_matrix[seg_mask] = seg_idx + 1 # corresponding to this segment
    for seg_idx in range(num_segments_r):
        lower = pos_edges_scaled[:, seg_idx][:, np.newaxis]
        upper = pos_edges_scaled[:, seg_idx + 1][:, np.newaxis]
        seg_mask = (data >= lower) & (data <= upper) & mask_pos
        index_matrix[seg_mask] = num_segments_l + seg_idx + 1 # corresponding to this segment

    mask_out_of_range = (index_matrix == 0)
    print(f" num outliers index_matrix: {np.sum(mask_out_of_range, axis=None)}")

    # Convert segment indices to binary
    binary_matrix = (index_matrix[:, :, np.newaxis] >> np.arange(num_bits - 1, -1, -1) & 1).astype(np.uint8)
    binary_matrix = binary_matrix.reshape(n, num_bits * d)

    # Compress binary representation
    packed_binary_matrix = np.packbits(binary_matrix, axis=1)
    og_shape_bin = binary_matrix.shape
    print(f"shape packed binary matrix: {packed_binary_matrix.shape}")

    compressed_batch = {
        'outliers': non_zero_elements,
        'avg_values': avg_values,
        'packed_binary_matrix': packed_binary_matrix,
        'og_shape_bin': og_shape_bin
    }
    return compressed_batch

# compress the entire dataset in batches
def orthoQuant_encode_in_batches(dataset, lower_limit, upper_limit, num_bits, batch_size = 10000):
    total_vectors = dataset.shape[0]

    # Initialize lists to accumulate results
    all_outliers = []
    all_avg_values = []
    all_packed_binaries = []
    for i in range(0, total_vectors, batch_size):
        batch = dataset[i:i + batch_size]
        compressed = orthoQuant_encode(batch, lower_limit, upper_limit, num_bits)
        all_outliers.append(compressed['outliers'])
        all_avg_values.append(compressed['avg_values'])
        all_packed_binaries.append(compressed['packed_binary_matrix'])
        if i == 0:
            og_shape_bin = compressed['og_shape_bin']  # assumed to be the same for all batches

    result = {
        'outliers': np.concatenate(all_outliers, axis=0),
        'avg_values': np.concatenate(all_avg_values, axis=0),
        'packed_binary_matrix': np.concatenate(all_packed_binaries, axis=0),
        'og_shape_bin': og_shape_bin
    }

    # Compute global mean per column and replace avg_values by this mean
    mean_per_column = np.mean(result['avg_values'], axis=0).astype(np.float32)
    result['avg_values'] = mean_per_column  # Shape changes from (total_vectors, d) to (d,)
    print(mean_per_column.shape)
    return result

# reconstructs an approximation of the original array
def orthoQuant_decode_database(outliers, avg_values, packed_binary_matrix, og_shape_bin, num_bits):
    N = packed_binary_matrix.shape[0]
    d = og_shape_bin[1] // num_bits

    # Unpack the encoded indices
    binary_matrix = np.unpackbits(packed_binary_matrix, axis=1)
    index_matrix = binary_matrix.reshape(N, d, num_bits)
    print(f"shape index matrix: {index_matrix.shape}")

    indices = np.zeros((N, d), dtype=np.uint32)
    for bit in range(num_bits):
        indices = (indices << 1) | index_matrix[:, :, bit].astype(np.uint32)

    # Initialize output tensor
    D_reconstructed = np.zeros((N, d), dtype=np.float32)
    print(f"shape D_reconstructed matrix: {D_reconstructed.shape}")

    # Insert approximated in-range values (segmented)
    D_reconstructed = avg_values[indices]

    # Insert original out-of-range values (index 0)
    mask_out_of_range = (indices == 0)
    print(f"shape D_reconstructed matrix: {np.sum(mask_out_of_range, axis=None)}")
    
    # assert np.all(mask_out_of_range,axis=1) == outliers.shape[1], "Number of out-of-range elements does not match"
    D_reconstructed[mask_out_of_range] = outliers.flatten()
    return D_reconstructed


def main(dataset):

    # load the dataset
    base, queries, dists, knns = load_dataset(dataset)
    print("Base:", base.shape)
    print("Queries:", queries.shape)
    print("Ground truth dists:", dists.shape)
    print("Ground truth knns:", knns.shape)
    N,D = base.shape
    subset = 1000
    # base_subset = base[:subset, :]

    Q = generate_rotation_matrix_Q(D, seed=12)
    rotated_base = np.matmul(base, Q.T).astype(np.float32)

    indices = np.random.permutation(N)[:subset]
    print("d=0")
    for i in rotated_base[indices,0]:
        print(f"{i:.3f}", end=',')
    print("")
    print("d=100")
    for i in rotated_base[indices,100]:
        print(f"{i:.3f}", end=',')
    print("")
    print("d=200")
    for i in rotated_base[indices,200]:
        print(f"{i:.3f}", end=',')
    print("")
    print("d=300")
    for i in rotated_base[indices,300]:
        print(f"{i:.3f}", end=',')
    print("")
    print("d=383")
    for i in rotated_base[indices,383]:
        print(f"{i:.3f}", end=',')
    print("")



    # generate rotation matrix
    # Q = generate_rotation_matrix_Q(D, seed=12)

    # rotate base subset
    # rotated_subset = np.matmul(base_subset, Q.T).astype(np.float32)
    # original_size = get_size_in_bytes(rotated_subset)
    # print("Base subset size:", get_size_in_bytes(base_subset) / (1024 * 1024), "MB")

    # # compute the outlier bounds
    # lower, upper = estimate_global_concavity_thresholds(rotated_subset, sample_size=5000)
    # print(lower, upper)

    # # compress the vectors
    # num_bits = 6
    # res = orthoQuant_encode_in_batches(rotated_subset, lower, upper, num_bits)
    # new_size = get_size_in_bytes(res)
    # print("Compressed size:", new_size / (1024 * 1024), "MB")
    # print(f" * avg_values: {get_size_in_bytes(res['avg_values'])}")
    # print(f" * outliers: {get_size_in_bytes(res['outliers'])}")
    # print(f" * packed_binary_matrix: {get_size_in_bytes(res['packed_binary_matrix'])}")
    # print(f" * og_shape_bin: {get_size_in_bytes(res['og_shape_bin'])}")

    # print(f"Compression ratio: {original_size / new_size:.2f}x")

    # # uncompress
    # uncomp = orthoQuant_decode_database(res['outliers'], res['avg_values'], res['packed_binary_matrix'], res['og_shape_bin'], num_bits)
    # print("Uncompressed size:", get_size_in_bytes(uncomp) / (1024 * 1024), "MB")
    # print(rotated_subset[0,:])
    # print(uncomp[0,:])












import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantization")
    parser.add_argument('--dataset', choices=DATASETS.keys(), default='ccnews')
    args = parser.parse_args()
    main(args.dataset)
    print(f"Done! Have a good day! :)")