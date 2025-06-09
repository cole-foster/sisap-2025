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

// Packs a vector of integers (each with num_bits bits) into a byte array
void pack_bits(const std::vector<uint8_t>& values, uint num_bits, uint8_t* out_bytes) {

    size_t bit_offset = 0;
    for (size_t i = 0; i < values.size(); ++i) {
        uint8_t value = values[i] & ((1 << num_bits) - 1); // mask to num_bits
        size_t byte_idx = bit_offset / 8;
        size_t bit_idx = bit_offset % 8;

        // Write the lower bits of value into the current byte
        out_bytes[byte_idx] |= value << bit_idx;

        // If value spans two bytes, write the upper bits into the next byte
        if (bit_idx + num_bits > 8) {
            out_bytes[byte_idx + 1] |= value >> (8 - bit_idx);
        }

        bit_offset += num_bits;
    }
}

// Unpacks a byte array into a vector of integers (each with num_bits bits)
void unpack_bits(std::vector<uint8_t>& out_values, uint num_bits, const uint8_t* in_bytes, size_t num_values) {
    out_values.resize(num_values);
    size_t bit_offset = 0;
    for (size_t i = 0; i < num_values; ++i) {
        size_t byte_idx = bit_offset / 8;
        size_t bit_idx = bit_offset % 8;

        // Read the bits for this value
        uint16_t val = in_bytes[byte_idx] >> bit_idx;
        if (bit_idx + num_bits > 8) {
            // Value spans two bytes
            val |= (uint16_t(in_bytes[byte_idx + 1]) << (8 - bit_idx));
        }
        out_values[i] = uint8_t(val & ((1 << num_bits) - 1));
        bit_offset += num_bits;
    }
}



// MARK: MAIN
int main(int argc, char** argv) {
    uint num_bits = 5;

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
        pack_bits(quantized_vector, num_bits, quantized_dataset+i*bytes_per_vector);
    }


    /* dequantize the dataset */
    auto tStart = std::chrono::high_resolution_clock::now();
    printf("Quantizing the dataset...\n");
    float* dequantized_dataset = new float[dataset_size*dimension];
    for (size_t i = 0; i < dataset_size; i++) {
        std::vector<uint8_t> quantized_vector(dimension, 0);
        unpack_bits(quantized_vector, num_bits, quantized_dataset+i*bytes_per_vector, dimension);
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

        // collect the knn
        std::priority_queue<std::pair<float,uint>> pq;
        for (uint x = 0; x < dataset_size; x++) {
            float dist = compute_distance(query_ptr, dequantized_dataset + x*dimension);
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
