// interact -q batch -n 32 -m 64g -t 12:00:00
// g++ main1.cpp -o main1 -fopenmp -O3
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <queue>
#include <vector>
#include <memory>
#include <mutex>
#include <numeric>

#include "utils.hpp"
#include "src/distances.h"
#include "src/graph.hpp"

/* dataset */
size_t dataset_size, dimension, gt_k;
float* data_pointer = nullptr;
distances::SpaceInterface<float>* space;
std::vector<std::vector<uint>> gt_graph;
void load_dataset_google_qa(const std::string& filename, float*& data_pointer, size_t& dataset_size, size_t& dimension);
void load_gt_google_qa(const std::string& filename, std::vector<std::vector<uint>>& graph, size_t& dataset_size, size_t& k);

/* evaluation */
double measure_neighbors_accuracy(const std::vector<uint>& gt_neighbors, const std::vector<std::pair<float,uint>>& neighbors, int k);
double measure_graph_quality(const std::vector<std::vector<uint>>& gt_graph, const std::vector<std::vector<std::pair<float, uint>>>& knn_graph, uint k);
double measure_graph_quality(const std::vector<std::vector<uint>>& gt_graph, const std::vector<std::vector<uint>>& knn_graph, uint k);

// MARK: MAIN
int main(int argc, char** argv) {
    uint k = 15;

    uint num_neighbors = 48;
    uint num_hops = 15;
    uint omap_size = 1000;
    uint omap_neighbors = 32;
    if (argc > 3) {
        num_neighbors = (uint) atoi(argv[1]);
        num_hops = (uint) atoi(argv[2]);
        omap_size = (uint) atoi(argv[3]);
    }
    uint num_iterations = 10;
    uint num_cores = omp_get_num_procs();
    printf("Running with %d cores\n", num_cores);
    printf("Parameters: num_neighbors=%u, num_hops=%u, omap_size=%u, omap_neighbors=%u\n", 
           num_neighbors, num_hops, omap_size, omap_neighbors);

    /* Loading dataset */
    printf("Loading dataset...\n");
    std::string data_filename = "/users/cfoste18/data/cfoste18/knn-construction/data/gooaq-N-3001496-D-384-fp32.bin";
    load_dataset_google_qa(data_filename, data_pointer, dataset_size, dimension);

    /* Loading ground truth */
    std::string gt_filename = "/users/cfoste18/data/cfoste18/knn-construction/data/knn-gooaq-N-3001496-k-32-int32.bin";
    load_gt_google_qa(gt_filename, gt_graph, dataset_size, gt_k);
    printf(" * gt_k=%u\n", (uint) gt_k);

    /* Initialize Index */
    space = new distances::InnerProductSpace(dimension);
    Graph* alg = new Graph(dataset_size, dimension, space);
    alg->load_dataset_float(data_pointer);
    alg->set_num_cores(num_cores);

    /* build the knn graph */
    printf("Begin refinement-based knn graph construction\n");
    auto tStart = std::chrono::high_resolution_clock::now();
    for (uint i = 0; i < num_iterations; i++) {
        printf("Iteration %d/%d\n", i+1, num_iterations);
        int random_seed = i * 71; // use a different seed for each iteration
        alg->set_omap_params(omap_size, omap_neighbors);
        alg->iterate_knn_refinement(num_neighbors, num_hops, random_seed);

        // measure time 
        auto tEnd = std::chrono::high_resolution_clock::now();
        double time = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd - tStart).count();

        // measure memory
        double peak_memory = (double) getPeakRSS() / 1024 / 1024; // in MB
        double current_memory = (double) getCurrentRSS() / 1024 / 1024; // in MB

        // measure accuracy
        double accuracy = measure_graph_quality(gt_graph, *(alg->graph_with_distances_), k);

        // print
        printf("Iteration, time, accuracy, current mem, peak mem\n");
        printf("%d, %.3f, %.5f, %.3f, %.3f\n", i,  time, 100 * accuracy, current_memory, peak_memory);
        fflush(stdout);
    }

    /* save the knn to disk so we can delete memory */
    printf("save the graph to disk\n");
    std::string graph_filename = "tmp-knn-new.bin";
    alg->save_graph(graph_filename);
    alg->clear_graphs();
    {
        size_t peak_memory = getPeakRSS();
        size_t current_memory = getCurrentRSS();
        printf("memory (MB) after knn graph construction: %.3f, %.3f\n",  (double) current_memory / 1024 / 1024, (double) peak_memory / 1024 / 1024);
    }

    /* load the graph from disk */
    alg->load_graph(graph_filename);
    {
        size_t peak_memory = getPeakRSS();
        size_t current_memory = getCurrentRSS();
        printf("memory (MB) after graph load: %.3f, %.3f\n",  (double) current_memory / 1024 / 1024, (double) peak_memory / 1024 / 1024);
    }

    /* refine the graph */
    alg->refine_graph();
    {
        size_t peak_memory = getPeakRSS();
        size_t current_memory = getCurrentRSS();
        printf("memory (MB) after graph load: %.3f, %.3f\n",  (double) current_memory / 1024 / 1024, (double) peak_memory / 1024 / 1024);
    }
    std::string graph_filename_hsp = "tmp-hsp-new.bin";
    alg->save_graph(graph_filename_hsp);

    delete alg;
    delete space;
    if (data_pointer != nullptr) delete[] data_pointer;
    printf("Done! Have a good day!\n");
    return 0;
}

double measure_neighbors_accuracy(const std::vector<uint>& gt_neighbors, const std::vector<std::pair<float,uint>>& neighbors, int k) {
    double recall = 0;
    for (auto val : neighbors) {
        uint node = val.second;
        if (std::find(gt_neighbors.begin(), gt_neighbors.end(), node) != gt_neighbors.end()) {
            recall += 1.0f;
        }
    }
    return (recall / (double) k);
}

double measure_graph_quality(const std::vector<std::vector<uint>>& gt_graph,
                             const std::vector<std::vector<std::pair<float, uint>>>& knn_graph, uint k) {
    size_t total_num_correct = 0;
    for (uint x = 0; x < dataset_size; x++) {
        auto gt_neighbors = gt_graph[x];
        auto est_neighbors = knn_graph[x];
        for (uint i = 0; i < k; i++) {
            uint est_node = est_neighbors[i].second;
            if (std::find(gt_neighbors.begin(), gt_neighbors.end(), est_node) != gt_neighbors.end()) {
                total_num_correct++;
            }
        }
    }
    double accuracy = (double)total_num_correct / (dataset_size * (size_t)k);

    return accuracy;
}
double measure_graph_quality(const std::vector<std::vector<uint>>& gt_graph,
                             const std::vector<std::vector<uint>>& knn_graph, uint k) {
    size_t total_num_correct = 0;
    for (uint x = 0; x < dataset_size; x++) {
        auto gt_neighbors = gt_graph[x];
        auto est_neighbors = knn_graph[x];
        for (uint i = 0; i < k; i++) {
            uint est_node = est_neighbors[i];
            if (std::find(gt_neighbors.begin()+1, gt_neighbors.end()+k+1, est_node) != gt_neighbors.end()+k+1) {
                total_num_correct++;
            }
        }
    }
    double accuracy = (double)total_num_correct / (dataset_size * (size_t)k);

    return accuracy;
}


void load_dataset_google_qa(const std::string& filename, float*& data_pointer, size_t& dataset_size, size_t& dimension) {
    dataset_size = 3001496;
    dimension = 384;
    printf("Google QA dataset\n");
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


void load_gt_google_qa(const std::string& filename, std::vector<std::vector<uint>>& graph, size_t& dataset_size, size_t& k) {
    dataset_size = 3001496;
    k = 32;

    // open the file
    std::ifstream input_file(filename, std::ios::binary);
    if (!input_file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    // Create a container for the vectors
    graph.resize(dataset_size, std::vector<uint>(k));

    // Read all the vectors
    std::vector<int32_t> temp_vec(k);
    for (size_t i = 0; i < dataset_size; ++i) {
        input_file.read(reinterpret_cast<char*>(temp_vec.data()), k * sizeof(int32_t));
        for (size_t j = 0; j < k; ++j) {
            graph[i][j] = static_cast<uint> (temp_vec[j]) - 1; // BECAUSE GT WAS 1-INDEXED
        }
    }
    input_file.close();
    return; 
}