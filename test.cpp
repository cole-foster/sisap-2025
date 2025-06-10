// interact -q batch -n 32 -m 64g -t 12:00:00
// g++ main-task1.cpp -o m_t1 -fopenmp -O3
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <vector>
#include "src/graph.hpp"

/* dataset */
size_t dataset_size, testset_size, dimension;
float* data_pointer = nullptr;
float* test_pointer = nullptr;
distances::SpaceInterface<float>* space;
std::vector<std::vector<uint>> gt_knn;
void load_dataset_ccnews(const std::string& filename, float*& data_pointer, size_t& dataset_size, size_t& dimension);
void load_queries_ccnews(const std::string& filename, float*& data_pointer, size_t& dataset_size, size_t& dimension);
void load_gt_ccnews(const std::string& filename, std::vector<std::vector<uint>>& knn_gt);


// MARK: MAIN
int main(int argc, char** argv) {
    uint num_neighbors = 48;
    printf("Running with %d cores\n", omp_get_num_procs());

    /* Loading dataset */
    printf("Loading dataset...\n");
    std::string data_filename = "data/ccnews-data-N-603664-D-384.bin";
    load_dataset_ccnews(data_filename, data_pointer, dataset_size, dimension);
    std::string test_filename = "data/ccnews-queries-T-11000-D-384.bin";
    load_queries_ccnews(test_filename, test_pointer, testset_size, dimension);
    std::string gt_filename = "data/ccnews-gt-T-11000-K-30.bin";
    load_gt_ccnews(gt_filename, gt_knn);
    // dataset_size = 100000;
    testset_size = 1000;

    // MARK: QUANTIZE 
    uint num_bits = 4;
    float L = (float) std::pow(2, num_bits); // number of quantization levels
    uint bits_per_vector = num_bits*dimension;
    uint bytes_per_vector = (bits_per_vector + 7) / 8; // number of bytes per vector
    printf("Quantizing the dataset with %d bits per dimension, %d levels, %d bytes per vector\n", num_bits, (int)L, bytes_per_vector);

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

    // quantize and dequantize
    float* dequantized_dataset = new float[dataset_size*dimension];   
    for (size_t i = 0; i < dataset_size; i++) {
        std::vector<uint8_t> quantized_vector(dimension, 0);
        for (size_t d = 0; d < dimension; d++) {

            // quantize the value 
            float value = data_pointer[i * dimension + d];
            float lval = lower_bounds[d];
            float uval = upper_bounds[d];
            float delta = (uval - lval) / (L - 1);
            uint8_t quantized_value = static_cast<uint8_t>(std::floor((value - lval) / delta + 0.5));
            if (quantized_value < 0) quantized_value = 0;
            if (quantized_value >= L) quantized_value = L - 1;

            // dequantize the value
            float dequantized_value = quantized_value * delta + lval;
            dequantized_dataset[i * dimension + d] = dequantized_value;
        }
    }


    /* Initialize Index */
    printf("Initializing the index\n");
    space = new distances::InnerProductSpace(dimension);
    Graph* alg = new Graph(dataset_size, dimension, num_neighbors, space, num_bits);
    alg->verbose_ = true;  // enable verbose output

    printf("training quantizer\n");
    alg->train_quantizer(data_pointer, dataset_size);

    printf("Adding dataset elements\n");
    // #pragma omp parallel for
    for (uint i = 0; i < dataset_size; i++) {
        alg->add_point(data_pointer + i * dimension, i);
    }
    alg->init_node_label_lookup();

    // printf("check a few elements\n");
    // {
    //     uint i = 334534;
    //     printf("Element %d\n: ", i);

    //     for (uint d = 0; d < 5; d++) {
    //         printf("%.4f ", data_pointer[i * dimension + d]);
    //     }
    //     printf("\n");

    //     for (uint d = 0; d < 5; d++) {
    //         printf("%.4f ", dequantized_dataset[i * dimension + d]);
    //     }
    //     printf("\n");

    //     std::vector<float> tmp;
    //     const float* element_ptr = alg->getRepresentation(alg->node_labels_map_[i], tmp);
    //     for (uint d = 0; d < 5; d++) {
    //         printf("%.4f ", element_ptr[d]);
    //     }
    //     printf("\n");
    // }

    printf("check a neighbors\n");
    {

        uint index = 0;
        uint id = alg->node_labels_map_[index];
        printf("%u --> %u\n", index, id);
        
        uint num_neighbors;
        uint* neighbors = alg->get_neighbors(id, num_neighbors);
        printf("num neighbors %u:", num_neighbors);
        for (uint d = 0; d < num_neighbors; d++) {
            printf("%u ", neighbors[d]);
        }
        printf("\n");
    }
    
    {
        alg->init_random_graph();
        printf("here\n");

        uint index = 0;
        uint id = alg->node_labels_map_[index];
        printf("%u --> %u\n", index, id);
        
        {
            uint num_neighbors;
            uint* neighbors = alg->get_neighbors(id, num_neighbors);
            printf("num neighbors %u:", num_neighbors);
            for (uint d = 0; d < num_neighbors; d++) {
                printf("%u,", neighbors[d]);
            }
            printf("\n");
        }

        std::vector<uint> new_neighbors = {34, 23, 43, 42};
        alg->set_neighbors(id, new_neighbors);
        
        {
            uint num_neighbors;
            uint* neighbors = alg->get_neighbors(id, num_neighbors);
            printf("num neighbors %u:", num_neighbors);
            for (uint d = 0; d < num_neighbors; d++) {
                printf("%u,", neighbors[d]);
            }
            printf("\n");
        }

    }
    




    
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
