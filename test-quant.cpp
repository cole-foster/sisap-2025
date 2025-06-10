// interact -q batch -n 32 -m 64g -t 12:00:00
// g++ main-quantization.cpp -o m_quant -fopenmp -O3
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <fstream>
#include <mutex>
#include <numeric>
#include <queue>
#include <vector>
#include <immintrin.h>
#include <cstring>
#include "src/distances.h"

/* dataset */
size_t dataset_size, testset_size, dimension;
float* data_pointer = nullptr;
float* test_pointer = nullptr;
distances::SpaceInterface<float>* space;
DISTFUNC<float> distFunc_;
void* distFuncParam_{nullptr};

std::vector<std::vector<uint>> gt_knn;
void load_dataset_ccnews(const std::string& filename, float*& data_pointer, size_t& dataset_size, size_t& dimension);
void load_queries_ccnews(const std::string& filename, float*& data_pointer, size_t& dataset_size, size_t& dimension);
void load_gt_ccnews(const std::string& filename, std::vector<std::vector<uint>>& knn_gt);

float compute_distance(float* index1_ptr, float* index2_ptr) {
    return distFunc_(index1_ptr, index2_ptr, distFuncParam_);
}

void pack_4bit_aligned(const std::vector<uint8_t>& values, uint8_t* out_bytes) {
    size_t num_bytes = (values.size() + 1) / 2;
    std::fill(out_bytes, out_bytes + num_bytes, 0);  // zero out

    for (size_t i = 0; i < values.size(); ++i) {
        uint8_t val = values[i] & 0x0F;
        if (i % 2 == 0)
            out_bytes[i / 2] |= val;
        else
            out_bytes[i / 2] |= val << 4;
    }
}

// Packs a vector of integers (each with num_bits bits) into a byte array
// void pack_bits(const std::vector<uint8_t>& values, uint num_bits, uint8_t* out_bytes) {
//     std::fill(out_bytes, out_bytes + ((values.size() * num_bits + 7) / 8), 0); // zero-out

//     size_t bit_offset = 0;
//     for (size_t i = 0; i < values.size(); ++i) {
//         uint8_t value = values[i] & ((1 << num_bits) - 1); // mask to num_bits
//         size_t byte_idx = bit_offset / 8;
//         size_t bit_idx  = bit_offset % 8;

//         out_bytes[byte_idx] |= value << bit_idx;

//         if (bit_idx + num_bits > 8) {
//             out_bytes[byte_idx + 1] |= value >> (8 - bit_idx);
//         }

//         bit_offset += num_bits;
//     }
// }


// // Unpacks a byte array into a vector of integers (each with num_bits bits)
// void unpack_bits(const uint8_t* in_bytes, size_t num_values, uint num_bits, std::vector<uint8_t>& out_values) {
//     out_values.resize(num_values);
//     size_t bit_offset = 0;

//     for (size_t i = 0; i < num_values; ++i) {
//         size_t byte_idx = bit_offset / 8;
//         size_t bit_idx  = bit_offset % 8;

//         uint16_t word = in_bytes[byte_idx];
//         if (bit_idx + num_bits > 8) {
//             word |= static_cast<uint16_t>(in_bytes[byte_idx + 1]) << 8;
//         }

//         out_values[i] = (word >> bit_idx) & ((1 << num_bits) - 1);
//         bit_offset += num_bits;
//     }
// }

void pack_bits(const std::vector<uint8_t>& values, uint num_bits, uint8_t* out_bytes) {
    std::fill(out_bytes, out_bytes + ((values.size() * num_bits + 7) / 8), 0); // zero-out

    size_t bit_offset = 0;
    for (size_t i = 0; i < values.size(); ++i) {
        uint8_t value = values[i] & ((1 << num_bits) - 1); // mask to num_bits
        size_t byte_idx = bit_offset / 8;
        size_t bit_idx  = bit_offset % 8;

        if (num_bits == 8 && bit_idx == 0) {
            // Fast path: write whole byte
            out_bytes[byte_idx] = value;
        } else {
            out_bytes[byte_idx] |= value << bit_idx;

            if (bit_idx + num_bits > 8) {
                out_bytes[byte_idx + 1] |= value >> (8 - bit_idx);
            }
        }

        bit_offset += num_bits;
    }
}
void unpack_bits(const uint8_t* in_bytes, size_t num_values, uint num_bits, std::vector<uint8_t>& out_values) {
    out_values.resize(num_values);
    size_t bit_offset = 0;

    for (size_t i = 0; i < num_values; ++i) {
        size_t byte_idx = bit_offset / 8;
        size_t bit_idx  = bit_offset % 8;

        if (num_bits == 8 && bit_idx == 0) {
            // Fast path: read whole byte
            out_values[i] = in_bytes[byte_idx];
        } else {
            uint16_t word = in_bytes[byte_idx];
            if (bit_idx + num_bits > 8) {
                word |= static_cast<uint16_t>(in_bytes[byte_idx + 1]) << 8;
            }
            out_values[i] = (word >> bit_idx) & ((1 << num_bits) - 1);
        }

        bit_offset += num_bits;
    }
}


void unpack_bits_4bit_scalar(std::vector<uint8_t>& out_values, uint num_bits, const uint8_t* in_bytes, size_t num_values) {
    for (size_t i = 0; i < num_values / 2; ++i) {
        uint8_t byte = in_bytes[i];
        out_values[2 * i]     = byte & 0x0F;
        out_values[2 * i + 1] = byte >> 4;
    }
}


// inline void unpack_bits_4bit_simd_avx2(std::vector<uint8_t>& out_values, uint num_bits, const uint8_t* in_bytes, size_t num_values) {
//     out_values.resize(num_values);
//     uint8_t* out_ptr = out_values.data();

//     size_t num_bytes = num_values / 2;
//     size_t i = 0;

//     const __m256i mask_0F = _mm256_set1_epi8(0x0F);

//     for (; i + 32 <= num_bytes; i += 32) {
//         __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in_bytes + i));

//         // Extract lower nibbles
//         __m256i lo = _mm256_and_si256(v, mask_0F);

//         // To extract upper nibbles:
//         // Promote to 16-bit lanes, shift right by 4, then narrow back to 8-bit
//         __m256i v_16 = _mm256_and_si256(_mm256_srli_epi16(v, 4), mask_0F);
//         __m256i hi = v_16;

//         // Interleave lo and hi nibbles into final 64-byte output
//         __m256i interleaved_lo = _mm256_unpacklo_epi8(lo, hi); // 32 unpacked
//         __m256i interleaved_hi = _mm256_unpackhi_epi8(lo, hi); // 32 unpacked

//         size_t out_idx = 2 * i;
//         _mm256_storeu_si256(reinterpret_cast<__m256i*>(out_ptr + out_idx), interleaved_lo);
//         _mm256_storeu_si256(reinterpret_cast<__m256i*>(out_ptr + out_idx + 32), interleaved_hi);
//     }

//     // Fallback scalar for remainder
//     for (; i < num_bytes; ++i) {
//         uint8_t byte = in_bytes[i];
//         out_ptr[2 * i]     = byte & 0x0F;
//         out_ptr[2 * i + 1] = byte >> 4;
//     }
// }

inline void unpack_4bit_aligned_avx2(const uint8_t* in_bytes, size_t num_values, std::vector<uint8_t>& out_values) {
    out_values.resize(num_values);
    uint8_t* out_ptr = out_values.data();

    size_t num_pairs = num_values / 2;
    size_t i = 0;

    const __m256i mask_0F = _mm256_set1_epi8(0x0F);

    // Process 32 input bytes → 64 output nibbles
    for (; i + 32 <= num_pairs; i += 32) {
        __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in_bytes + i));
        __m256i lo = _mm256_and_si256(packed, mask_0F);
        __m256i hi = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask_0F);

        __m256i interleaved_lo = _mm256_unpacklo_epi8(lo, hi);
        __m256i interleaved_hi = _mm256_unpackhi_epi8(lo, hi);

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out_ptr + 2 * i), interleaved_lo);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out_ptr + 2 * i + 32), interleaved_hi);
    }

    // Scalar fallback
    for (; i < num_pairs; ++i) {
        uint8_t byte = in_bytes[i];
        out_ptr[2 * i]     = byte & 0x0F;
        out_ptr[2 * i + 1] = byte >> 4;
    }

    // Odd value fallback
    if (num_values % 2 != 0) {
        out_ptr[num_values - 1] = in_bytes[num_pairs] & 0x0F;
    }
}


// Computes the Hamming distance between two uint8_t arrays of length D
int hamming_distance(const uint8_t* a, const uint8_t* b, size_t D) {
    int dist = 0;
    for (size_t i = 0; i < D; ++i) {
        dist += __builtin_popcount(a[i] ^ b[i]);
    }
    return dist;
}

// MARK: MAIN
int main(int argc, char** argv) {
    uint num_bits = 4;
    uint bytes_per_vector_ = (uint) (num_bits*384 + 7) / 8; // number of bytes per vector

    std::vector<uint8_t> input = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    size_t packed_size = (input.size() + 1) / 2;
    std::vector<uint8_t> packed(packed_size);
    std::vector<uint8_t> unpacked;

    pack_4bit_aligned(input, packed.data());
    unpack_4bit_aligned_avx2(packed.data(), input.size(), unpacked);

    for (size_t i = 0; i < input.size(); ++i) {
        printf("%d ", unpacked[i]);
    }
    printf("\n");

    /* Loading dataset */
    printf("Loading dataset...\n");
    std::string data_filename = "data/ccnews-data-N-603664-D-384.bin";
    load_dataset_ccnews(data_filename, data_pointer, dataset_size, dimension);
    std::string test_filename = "data/ccnews-queries-T-11000-D-384.bin";
    load_queries_ccnews(test_filename, test_pointer, testset_size, dimension);
    std::string gt_filename = "data/ccnews-gt-T-11000-K-30.bin";
    load_gt_ccnews(gt_filename, gt_knn);
    testset_size = 1000;

    /* Initialize distances */
    printf("Initializing the distances \n");
    space = new distances::InnerProductSpace(dimension);
    distFunc_ = space->get_dist_func();
    distFuncParam_ = space->get_dist_func_param();

    /* find bounds */
    std::vector<float> lower_bounds(dimension, 10000);
    std::vector<float> upper_bounds(dimension, -10000);
    for (size_t i = 0; i < dataset_size; i++) {
        for (size_t d = 0; d < dimension; d++) {
            lower_bounds[d] = std::min(lower_bounds[d], data_pointer[i * dimension + d]);
            upper_bounds[d] = std::max(upper_bounds[d], data_pointer[i * dimension + d]);
        }
    }
    for (size_t d = 0; d < dimension; d++) {
        lower_bounds[d] /= 1;
        upper_bounds[d] /= 1;
        if (upper_bounds[d] - lower_bounds[d] < 0.001) {
            float ave = (upper_bounds[d] + lower_bounds[d]) / 2.0f;
            lower_bounds[d] = ave - 0.001f;
            upper_bounds[d] = ave + 0.001f;
        }
    }

    /* quantize the dataset */
    float L = (float) std::pow(2, num_bits); // number of quantization levels
    uint bits_per_vector = num_bits*dimension;
    uint bytes_per_vector = (bits_per_vector + 7) / 8; // number of bytes per vector
    printf("Quantizing the dataset with %d bits per dimension, %d levels, %d bytes per vector\n", num_bits, (int)L, bytes_per_vector);

    
    uint8_t* quantized_dataset = new uint8_t[dataset_size*bytes_per_vector];
    for (size_t i = 0; i < dataset_size; i++) {
        std::vector<uint8_t> quantized_vector(dimension, 0);
        for (size_t d = 0; d < dimension; d++) {
            float value = data_pointer[i * dimension + d];
            float lval = lower_bounds[d];
            float uval = upper_bounds[d];
            float delta = (uval - lval) / (L - 1);

            // quantize the data
            uint8_t quantized_value = static_cast<uint8_t>(std::floor((value - lval) / delta + 0.5));
            if (quantized_value < 0) quantized_value = 0;
            if (quantized_value >= L) quantized_value = L - 1;
            quantized_vector[d] = quantized_value;
        }
        // pack_bits(quantized_vector, num_bits, quantized_dataset+i*bytes_per_vector);
        // pack_4bit_aligned(quantized_vector, num_bits, quantized_dataset+i*bytes_per_vector);
        // pack_bits(quantized_vector, num_bits, quantized_dataset+i*bytes_per_vector);
        pack_4bit_aligned(quantized_vector, quantized_dataset+i*bytes_per_vector);
    }


    /* dequantize the dataset */
    auto tStart = std::chrono::high_resolution_clock::now();
    printf("Quantizing the dataset...\n");
    float* dequantized_dataset = new float[dataset_size*dimension];
    for (size_t i = 0; i < dataset_size; i++) {
        std::vector<uint8_t> quantized_vector(dimension, 0);
        // unpack_bits(quantized_vector, num_bits, quantized_dataset+i*bytes_per_vector, dimension);
        // unpack_bits_4bit_simd_avx2(quantized_vector, num_bits, quantized_dataset+i*bytes_per_vector, dimension);
        // unpack_bits_4bit_simd_avx2(quantized_vector, num_bits, , dimension);
        // unpack_bits(quantized_dataset+i*bytes_per_vector, dimension, num_bits, quantized_vector);
        unpack_4bit_aligned_avx2(quantized_dataset+i*bytes_per_vector, dimension, quantized_vector);
        for (size_t d = 0; d < dimension; d++) {
            float quantized_value = quantized_vector[d];
            float lval = lower_bounds[d];
            float uval = upper_bounds[d];
            float delta = (uval - lval) / (L - 1);

            // dequantize the dataset
            float dequantized_value = quantized_value * delta + lval;
            dequantized_dataset[i * dimension + d] = dequantized_value;
        }
    }
    auto tEnd = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd - tStart).count();
    printf("Dequantization done in %.3f seconds, %.3f per vec\n", time, 1000*time / (double) dataset_size);

    /* evaluate */
    uint k = 30;
    double recall = 0;
    for (uint q = 0; q < testset_size; q++) {
        float* query_ptr = test_pointer + q * dimension;

        // quantize the query
        std::vector<uint8_t> quantized_vector(dimension, 0);
        for (size_t d = 0; d < dimension; d++) {
            float value = query_ptr[d];
            float lval = lower_bounds[d];
            float uval = upper_bounds[d];
            float delta = (uval - lval) / (L - 1);

            // quantize the data
            uint8_t quantized_value = static_cast<uint8_t>(std::floor((value - lval) / delta + 0.5));
            if (quantized_value < 0) quantized_value = 0;
            if (quantized_value >= L) quantized_value = L - 1;
            quantized_vector[d] = quantized_value;
        }
        std::vector<uint8_t> packed_vector(bytes_per_vector_, 0);
        pack_bits(quantized_vector, num_bits, packed_vector.data());


        // dequantize the query
        std::vector<float> dequantized_vector(dimension, 0);
        for (size_t d = 0; d < dimension; d++) {
            float quantized_value = quantized_vector[d];
            float lval = lower_bounds[d];
            float uval = upper_bounds[d];
            float delta = (uval - lval) / (L - 1);

            // dequantize the dataset
            float dequantized_value = quantized_value * delta + lval;
            dequantized_vector[d] = dequantized_value;
        }



        // collect the knn
        std::priority_queue<std::pair<float,uint>> pq;
        for (uint x = 0; x < dataset_size; x++) {
            // float dist = hamming_distance(packed_vector.data(), )
            float dist = compute_distance(dequantized_vector.data(), dequantized_dataset + x*dimension);
            pq.emplace(dist, x);
            if (pq.size() > k) {
                pq.pop();
            }
        }

        /* compare to gt */
        std::vector<uint> neighbors = gt_knn[q];
        neighbors.resize(k);
        while (pq.size() > 0) {
            uint est_neighbor = pq.top().second;
            if (std::find(neighbors.begin(), neighbors.end(), est_neighbor) != neighbors.end()) {
                recall += 1.0;
            }
            pq.pop();
        }
    }
    recall /= (double) (testset_size * k);
    printf("Recall@%d: %.4f\n", k, recall);


    delete[] dequantized_dataset;
    delete[] quantized_dataset;

    delete space;
    if (data_pointer != nullptr) delete[] data_pointer;
    printf("Done! Have a good day!\n");
    return 0;
}


void load_dataset_ccnews(const std::string& filename, float*& data_pointer, size_t& dataset_size, size_t& dimension) {
    dataset_size = 603664;
    dimension = 384;
    printf("CCNews dataset\n");
    printf(" * N=%u\n", (uint)dataset_size);
    printf(" * D=%u\n", (uint)dimension);

    // open the file
    std::ifstream input_file(filename, std::ios::binary);
    if (!input_file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    // Create a container for the vectors
    data_pointer = new float[dataset_size * dimension];

    // Read all the vectors
    input_file.read(reinterpret_cast<char*>(data_pointer), dataset_size * dimension * sizeof(float));
    input_file.close();
    return;
}

void load_queries_ccnews(const std::string& filename, float*& data_pointer, size_t& dataset_size, size_t& dimension) {
    dataset_size = 11000;
    dimension = 384;
    printf("CCNews queries\n");
    printf(" * N=%u\n", (uint)dataset_size);
    printf(" * D=%u\n", (uint)dimension);

    // open the file
    std::ifstream input_file(filename, std::ios::binary);
    if (!input_file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    // Create a container for the vectors
    data_pointer = new float[dataset_size * dimension];

    // Read all the vectors
    input_file.read(reinterpret_cast<char*>(data_pointer), dataset_size * dimension * sizeof(float));
    input_file.close();
    return;
}

void load_gt_ccnews(const std::string& filename, std::vector<std::vector<uint>>& knn_gt) {
    uint dataset_size = 11000;
    uint k = 30;

    // open the file
    std::ifstream input_file(filename, std::ios::binary);
    if (!input_file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    // init
    knn_gt.clear();
    knn_gt.resize(dataset_size);

    // load for each vec
    for (uint i = 0; i < dataset_size; i++) {
        knn_gt[i].resize(k);
        input_file.read(reinterpret_cast<char*>(knn_gt[i].data()), k * sizeof(uint));
    }
    input_file.close();

    // make it zero indexed
    for (uint i = 0; i < dataset_size; i++) {
        for (uint j = 0; j < k; j++) {
            knn_gt[i][j]--;
        }
    }
    return;
}
