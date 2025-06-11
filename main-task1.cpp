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
    uint num_neighbors = 32;
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
    // uint num_bits = 4;
    // float L = (float) std::pow(2, num_bits); // number of quantization levels
    // uint bits_per_vector = num_bits*dimension;
    // uint bytes_per_vector = (bits_per_vector + 7) / 8; // number of bytes per vector
    // printf("Quantizing the dataset with %d bits per dimension, %d levels, %d bytes per vector\n", num_bits, (int)L, bytes_per_vector);

    /* find bounds */
    // std::vector<float> lower_bounds(dimension, 10000);
    // std::vector<float> upper_bounds(dimension, -10000);
    // for (size_t i = 0; i < dataset_size; i++) {
    //     for (size_t d = 0; d < dimension; d++) {
    //         lower_bounds[d] = std::min(lower_bounds[d], data_pointer[i * dimension + d]);
    //         upper_bounds[d] = std::max(upper_bounds[d], data_pointer[i * dimension + d]);
    //     }
    // }
    // for (size_t d = 0; d < dimension; d++) {
    //     lower_bounds[d] /= 1;
    //     upper_bounds[d] /= 1;
    //     if (upper_bounds[d] - lower_bounds[d] < 0.001) {
    //         float ave = (upper_bounds[d] + lower_bounds[d]) / 2.0f;
    //         lower_bounds[d] = ave - 0.001f;
    //         upper_bounds[d] = ave + 0.001f;
    //     }
    // }

    // quantize and dequantize
    // float* dequantized_dataset = new float[dataset_size*dimension];   
    // for (size_t i = 0; i < dataset_size; i++) {
    //     std::vector<uint8_t> quantized_vector(dimension, 0);
    //     for (size_t d = 0; d < dimension; d++) {

    //         // quantize the value 
    //         float value = data_pointer[i * dimension + d];
    //         float lval = lower_bounds[d];
    //         float uval = upper_bounds[d];
    //         float delta = (uval - lval) / (L - 1);
    //         uint8_t quantized_value = static_cast<uint8_t>(std::floor((value - lval) / delta + 0.5));
    //         if (quantized_value < 0) quantized_value = 0;
    //         if (quantized_value >= L) quantized_value = L - 1;

    //         // dequantize the value
    //         float dequantized_value = quantized_value * delta + lval;
    //         dequantized_dataset[i * dimension + d] = dequantized_value;
    //     }
    // }



    // {
    //     /* Initialize Index */
    //     printf("Initializing the index\n");
    //     uint num_bits = 4;
    //     space = new distances::InnerProductSpace(dimension);
    //     Graph* alg = new Graph(dataset_size, dimension, num_neighbors, space, num_bits);
    //     alg->verbose_ = true;  // enable verbose output

    //     printf("training quantizer\n");
    //     alg->train_quantizer(dequantized_dataset, dataset_size);

    //     printf("Adding dataset elements\n");

    //     // #pragma omp parallel for
    //     for (uint i = 0; i < dataset_size; i++) {
    //         alg->add_point(data_pointer + i * dimension, i);
    //     }

    //     printf("intialize random graph\n");
    //     alg->init_random_graph();

    //     printf("begin graph refinement\n");
    //     uint num_candidates = 128;
    //     uint num_hops = 32;
    //     uint num_iterations = 20;
    //     for (uint i = 0; i < num_iterations; i++) {
    //         printf(" * iteration %u/%u\n", i + 1, num_iterations);
    //         auto tStart = std::chrono::high_resolution_clock::now();
    //         alg->init_top_layer_graph((uint) 4*sqrt(dataset_size), 32, 3);
    //         std::vector<uint> labels(dataset_size);
    //         for (uint i = 0; i < dataset_size; i++) {
    //             labels[i] = i;
    //         }

    //         printf(" * updating graph\n");
    //         alg->update_graph(data_pointer, labels.data(), dataset_size, num_candidates, num_hops);
    //         auto tEnd = std::chrono::high_resolution_clock::now();
    //         double time = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd - tStart).count();
    //         printf(" * refinement time: %.4f\n", time);

    //         alg->trim_graph_hsp();  // trim the graph to keep only the top 10% of neighbors
    //         alg->print_graph_stats();

    //         /* search */
    //         uint k = 30; 
    //         std::vector<uint> vec_beam_size = {30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 200, 400, 800, 1600};
    //         for (uint beam_size : vec_beam_size) {

    //             // begin search
    //             std::vector<std::vector<std::pair<float, uint>>> est_knn(testset_size);
    //             auto tStart = std::chrono::high_resolution_clock::now();
    //             {
    //                 #pragma omp parallel for
    //                 for (uint q = 0; q < testset_size; q++) {
    //                     float* queryPtr = test_pointer + q * dimension;
    //                     // est_knn[q] = alg->search_brute_force(queryPtr, k); // beam_size, k);
    //                     est_knn[q] = alg->search(queryPtr, beam_size, k);
    //                 }
    //             }
    //             auto tEnd = std::chrono::high_resolution_clock::now();
    //             double time = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd - tStart).count();
    //             double throughput = (double)testset_size / time;

    //             // compute recall@k
    //             double recall = 0.0;
    //             for (uint q = 0; q < testset_size; q++) {
    //                 auto est_neighbors = est_knn[q];
    //                 if (est_neighbors.size() > k) est_neighbors.resize(k);
    //                 std::vector<uint> gt_neighbors = gt_knn[q];
    //                 if (gt_neighbors.size() > k) gt_neighbors.resize(k);

    //                 for (uint j = 0; j < k; j++) {
    //                     uint est_neighbor = est_neighbors[j].second;
    //                     if (std::find(gt_neighbors.begin(), gt_neighbors.end(), est_neighbor) != gt_neighbors.end()) {
    //                         recall += 1.0;
    //                     }   
    //                 }
    //             }
    //             recall /= (double) (testset_size * k);

    //             printf("iteration, beam_size %u, throughput (qps): %.4f, recall@%d: %.4f\n", beam_size, throughput, k, recall);
    //         }
    //     }

    //     delete alg;
    // }

    // std::string gname = "tmp.bin";
    {
        /* Initialize Index */
        printf("Initializing the index\n");
        uint num_bits = 4;
        space = new distances::InnerProductSpace(dimension);
        Graph* alg = new Graph(dataset_size, dimension, num_neighbors, space, num_bits);
        alg->verbose_ = true; 

        printf("training quantizer\n");
        alg->train_quantizer(data_pointer, dataset_size);

        printf("Adding dataset elements\n");
        for (uint i = 0; i < dataset_size; i++) {
            alg->add_point(data_pointer + i * dimension, i);
        }

        printf("intialize random graph\n");
        // alg->init_random_graph();
        alg->init_empty_but_top_layer(4000);

        printf("begin graph refinement\n");
        uint num_candidates = 128;
        uint num_hops = 64;
        uint num_iterations = 5;
        for (uint i = 0; i < num_iterations; i++) {
            printf(" * iteration %u/%u\n", i + 1, num_iterations);

            // {
            //     auto tStart = std::chrono::high_resolution_clock::now();
            //     alg->init_top_layer_graph(4000, 32, 1);
            //     auto tEnd = std::chrono::high_resolution_clock::now();
            //     double time = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd - tStart).count();
            //     printf(" * omap time: %.4f\n", time);
            // }

            {
                auto tStart = std::chrono::high_resolution_clock::now();
                alg->graph_refinement_iteration(num_candidates, num_hops);
                auto tEnd = std::chrono::high_resolution_clock::now();
                double time = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd - tStart).count();
                printf(" * refinement time: %.4f\n", time);
            }

            // alg->trim_graph_hsp();  // trim the graph to keep only the top 10% of neighbors
            alg->print_graph_stats();

            /* search */
            uint k = 30; 
            std::vector<uint> vec_beam_size = {30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 200, 400, 800, 1600};
            for (uint beam_size : vec_beam_size) {

                // begin search
                std::vector<std::vector<std::pair<float, uint>>> est_knn(testset_size);
                auto tStart = std::chrono::high_resolution_clock::now();
                {
                    #pragma omp parallel for
                    for (uint q = 0; q < testset_size; q++) {
                        float* queryPtr = test_pointer + q * dimension;
                        est_knn[q] = alg->search(queryPtr, k, beam_size);
                    }
                }
                auto tEnd = std::chrono::high_resolution_clock::now();
                double time = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd - tStart).count();
                double throughput = (double)testset_size / time;

                // compute recall@k
                double recall = 0.0;
                for (uint q = 0; q < testset_size; q++) {
                    auto est_neighbors = est_knn[q];
                    if (est_neighbors.size() > k) est_neighbors.resize(k);
                    std::vector<uint> gt_neighbors = gt_knn[q];
                    if (gt_neighbors.size() > k) gt_neighbors.resize(k);

                    for (uint j = 0; j < k; j++) {
                        uint est_neighbor = est_neighbors[j].second;
                        if (std::find(gt_neighbors.begin(), gt_neighbors.end(), est_neighbor) != gt_neighbors.end()) {
                            recall += 1.0;
                        }   
                    }
                }
                recall /= (double) (testset_size * k);

                printf("iteration, beam_size %u, throughput (qps): %.4f, recall@%d: %.4f\n", beam_size, throughput, k, recall);
            }
        }

        // begin search
        // printf("begin exact search\n");
        // {
        //     uint k = 30;
        //     std::vector<std::vector<std::pair<float, uint>>> est_knn(testset_size);
        //     auto tStart = std::chrono::high_resolution_clock::now();
        //     {
        //         #pragma omp parallel for
        //         for (uint q = 0; q < testset_size; q++) {
        //             float* queryPtr = test_pointer + q * dimension;
        //             est_knn[q] = alg->search_brute_force(queryPtr, k);
        //         }
        //     }
        //     auto tEnd = std::chrono::high_resolution_clock::now();
        //     double time = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd - tStart).count();
        //     double throughput = (double)testset_size / time;

        //     // compute recall@k
        //     double recall = 0.0;
        //     for (uint q = 0; q < testset_size; q++) {
        //         auto est_neighbors = est_knn[q];
        //         if (est_neighbors.size() > k) est_neighbors.resize(k);
        //         std::vector<uint> gt_neighbors = gt_knn[q];
        //         if (gt_neighbors.size() > k) gt_neighbors.resize(k);

        //         for (uint j = 0; j < k; j++) {
        //             uint est_neighbor = est_neighbors[j].second;
        //             if (std::find(gt_neighbors.begin(), gt_neighbors.end(), est_neighbor) != gt_neighbors.end()) {
        //                 recall += 1.0;
        //             }   
        //         }
        //     }
        //     recall /= (double) (testset_size * k);
        //     printf("iteration, beam_size %u, throughput (qps): %.4f, recall@%d: %.4f\n", 0, throughput, k, recall);
        // }

        delete alg;
    }

    // {
    //     /* Initialize Index */
    //     printf("Initializing the index\n");
    //     uint num_bits = 5;
    //     space = new distances::InnerProductSpace(dimension);
    //     Graph* alg = new Graph(dataset_size, dimension, num_neighbors, space, num_bits);
    //     alg->verbose_ = true;  // enable verbose output
    //     // alg->set_num_cores(num_cores);

    //     printf("training quantizer\n");
    //     alg->train_quantizer(data_pointer, dataset_size);

    //     printf("Adding dataset elements\n");
    //     // #pragma omp parallel for
    //     for (uint i = 0; i < dataset_size; i++) {
    //         alg->add_point(data_pointer + i * dimension, i);
    //     }
    //     uint k = 30;  // number of neighbors to search for

    //     printf("loading...\n");
    //     alg->load_graph(gname);
        
    //     printf("search\n");
    //     /* search */
    //     std::vector<uint> vec_beam_size = {30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 200, 400, 800, 1600};
    //     for (uint beam_size : vec_beam_size) {

    //         // begin search
    //         std::vector<std::vector<std::pair<float, uint>>> est_knn(testset_size);
    //         auto tStart = std::chrono::high_resolution_clock::now();
    //         {
    //             #pragma omp parallel for
    //             for (uint q = 0; q < testset_size; q++) {
    //                 float* queryPtr = test_pointer + q * dimension;
    //                 est_knn[q] = alg->search(queryPtr, beam_size, k);
    //             }
    //         }
    //         auto tEnd = std::chrono::high_resolution_clock::now();
    //         double time = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd - tStart).count();
    //         double throughput = (double)testset_size / time;

    //         // compute recall@k
    //         double recall = 0.0;
    //         for (uint q = 0; q < testset_size; q++) {
    //             auto est_neighbors = est_knn[q];
    //             if (est_neighbors.size() > k) est_neighbors.resize(k);
    //             std::vector<uint> gt_neighbors = gt_knn[q];
    //             if (gt_neighbors.size() > k) gt_neighbors.resize(k);

    //             for (uint j = 0; j < k; j++) {
    //                 uint est_neighbor = est_neighbors[j].second;
    //                 if (std::find(gt_neighbors.begin(), gt_neighbors.end(), est_neighbor) != gt_neighbors.end()) {
    //                     recall += 1.0;
    //                 }   
    //             }
    //         }
    //         recall /= (double) (testset_size * k);

    //         printf("iteration, beam_size %u, throughput (qps): %.4f, recall@%d: %.4f\n", beam_size, throughput, k, recall);
    //     }
    //     delete alg;
    // }


    // begin search
    // printf("begin search\n");
    // std::vector<std::vector<std::pair<float, uint>>> est_knn(testset_size);
    // auto tStart = std::chrono::high_resolution_clock::now();
    // {
    //     #pragma omp parallel for
    //     for (uint q = 0; q < testset_size; q++) {
    //         float* queryPtr = test_pointer + q * dimension;
    //         est_knn[q] = alg->search_brute_force(queryPtr, k);
    //     }
    // }
    // auto tEnd = std::chrono::high_resolution_clock::now();
    // double time = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd - tStart).count();
    // double throughput = (double)testset_size / time;

    // compute recall@k
    // printf("Computing recall@%d\n", k);
    // double recall = 0.0;
    // for (uint q = 0; q < testset_size; q++) {
    //     auto est_neighbors = est_knn[q];
    //     if (est_neighbors.size() > k) est_neighbors.resize(k);
    //     std::vector<uint> gt_neighbors = gt_knn[q];
    //     if (gt_neighbors.size() > k) gt_neighbors.resize(k);

    //     for (uint j = 0; j < k; j++) {
    //         uint est_neighbor = est_neighbors[j].second;
    //         if (std::find(gt_neighbors.begin(), gt_neighbors.end(), est_neighbor) != gt_neighbors.end()) {
    //             recall += 1.0;
    //         }   
    //     }
    // }
    // recall /= (double) (testset_size * k);
    // printf("throughput (qps): %.4f, recall@%d: %.4f\n", throughput, k, recall);

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
