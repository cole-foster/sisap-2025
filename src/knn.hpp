#pragma once
// #define _RECORD_METRICS
// #ifndef NUM_CORES
//     #define NUM_CORES 1
// #endif

#include <omp.h>

#include <algorithm>
#include <atomic>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <unordered_map>
#include <vector>

#include "distances.h"
#include "graph.hpp"
#include "unique-priority-queue.hpp"
#include "visited-list-pool.h"

typedef uint32_t uint;
class KNN {
   public:
    float MAX_FLOAT = std::numeric_limits<float>::max();
    uint MAX_UINT = std::numeric_limits<uint>::max();

    // data
    bool data_owner_ = false;
    char* data_pointer_{nullptr};  // pointer to the data in the graph
    uint dataset_size_{0};         // max size of dataset
    uint dimension_{0};            // dimensionality of the dataset
    uint max_neighbors_{64};       // number of bottom layer neighbors
    std::unique_ptr<std::vector<std::vector<std::pair<float, uint>>>> graph_{nullptr};
    // std::unique_ptr<std::vector<std::vector<std::pair<float, uint>>>> graph_with_distances_{nullptr};
    // std::unique_ptr<std::unordered_map<uint, std::vector<uint>>> graph_top_{nullptr};
    std::vector<uint> start_nodes_{}; 

    // top layer
    std::vector<uint> top_layer_nodes_{}; 
    std::unique_ptr<Graph> top_layer_graph_{nullptr}; 

    // parallelism
    uint num_cores_ = 1;
    std::vector<std::mutex> node_neighbors_locks_;  // locks for parallel neighbor updates

    // distances
    distances::SpaceInterface<float>* space_;
    DISTFUNC<float> distFunc_;
    void* distFuncParam_{nullptr};
    size_t data_size_{0};

    // search params
    std::unique_ptr<VisitedListPool> visitedListPool_{nullptr};

    // metrics
    std::atomic<size_t> metrics_hops_{0};
    std::atomic<size_t> metrics_distance_computations_{0};

    // MARK: CONSTRUCTOR
    KNN(uint dataset_size, uint dimension, distances::SpaceInterface<float>* s)
        : node_neighbors_locks_(dataset_size) {
        // printf("Constructor...\n");
        dataset_size_ = dataset_size;
        dimension_ = dimension;
        space_ = s;
        distFunc_ = space_->get_dist_func();
        distFuncParam_ = space_->get_dist_func_param();
        data_size_ = dimension_ * sizeof(float);  // EDIT FOR QUANTIZATION
        visitedListPool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, dataset_size_));
    }
    ~KNN() {
        if (data_owner_ && data_pointer_ != nullptr) {
            free(data_pointer_);
            data_pointer_ = nullptr;
        }
        data_owner_ = false;
    };

    /* loading dataset, which is managed outside of here */
    // void load_dataset_float(float*& data_pointer) {
    //     printf("Loading dataset...\n");
    //     data_pointer_ = reinterpret_cast<char*>(data_pointer);
    //     data_owner_ = false;
    // }

    void init_dataset() {
        data_owner_ = true;
        size_t total_bytes = (size_t)dataset_size_ * (size_t)data_size_;
        data_pointer_ = (char*)malloc(total_bytes);
        if (data_pointer_ == nullptr) throw std::runtime_error("Not enough memory");
    }
    void add_point(const void* element_ptr, uint element_id) {
        memcpy(data_pointer_ + element_id * data_size_, element_ptr, data_size_);
    }
    void set_num_cores(uint num_cores) { num_cores_ = num_cores; }

    //--------------- MARK: I/O

    /* save the graph to disk (no distances) */
    // void save_graph(std::string filename) {
    //     printf("Saving Graph to: %s\n", filename.c_str());
    //     std::ofstream outputFileStream(filename, std::ios::binary);
    //     if (!outputFileStream.is_open()) {
    //         printf("Open File Error\n");
    //         exit(0);
    //     }
    //     printf(" * saving graph without distances...\n");

    //     // output the datasetSize, graphNeighborsTop, graphNeighborsBottom, sizeElementLinks_
    //     outputFileStream.write((char*)&dataset_size_, sizeof(uint));
    //     outputFileStream.write((char*)&dimension_, sizeof(uint));
    //     outputFileStream.write((char*)&max_neighbors_, sizeof(uint));

    //     if (graph_with_distances_ != nullptr && graph_with_distances_->size() == dataset_size_) {
    //         for (uint index = 0; index < dataset_size_; index++) {
    //             if ((*graph_with_distances_)[index].size() != max_neighbors_)
    //                 throw std::runtime_error("Graph size mismatch");
    //             for (uint m = 0; m < max_neighbors_; m++) {
    //                 uint neighbor = (*graph_with_distances_)[index][m].second;
    //                 outputFileStream.write((char*)&neighbor, sizeof(uint));
    //             }
    //         }
    //     } else if (graph_ != nullptr && graph_->size() == dataset_size_) {
    //         for (uint index = 0; index < dataset_size_; index++) {
    //             if ((*graph_)[index].size() != max_neighbors_) throw std::runtime_error("Graph size mismatch");
    //             for (uint m = 0; m < max_neighbors_; m++) {
    //                 uint neighbor = (*graph_)[index][m];
    //                 outputFileStream.write((char*)&neighbor, sizeof(uint));
    //             }
    //         }
    //     }

    //     outputFileStream.close();
    //     return;
    // }

    /* to save memory */
    // void clear_graphs() {
    //     printf("Clearing graph...\n");
    //     graph_with_distances_.reset();
    //     graph_with_distances_ = nullptr;
    //     graph_.reset();
    //     graph_ = nullptr;
    //     return;
    // }

    /* load the graph to disk (no distances) */
    // void load_graph(std::string filename) {
    //     printf("Loading graph from: %s\n", filename.c_str());
    //     std::ifstream inputFileStream(filename, std::ios::binary);
    //     if (!inputFileStream.is_open()) {
    //         printf("Open File Error\n");
    //         exit(0);
    //     }
    //     printf(" * loading graph without distances...\n");

    //     // check the parameters for consistency
    //     {
    //         uint dataset_size, dimension;
    //         inputFileStream.read((char*)&dataset_size, sizeof(uint));
    //         if (dataset_size != dataset_size_) throw std::runtime_error("dataset_size mismatch");
    //         inputFileStream.read((char*)&dimension, sizeof(uint));
    //         if (dimension != dimension_) throw std::runtime_error("dimension mismatch");
    //         inputFileStream.read((char*)&max_neighbors_, sizeof(uint));
    //     }
    //     clear_graphs();
    //     graph_ = std::make_unique<std::vector<std::vector<uint>>>(dataset_size_);

    //     /* write the top/bottom linked lists for each element */
    //     for (uint index = 0; index < dataset_size_; index++) {
    //         (*graph_)[index].resize(max_neighbors_);
    //         for (uint m = 0; m < max_neighbors_; m++) {
    //             inputFileStream.read((char*)&(*graph_)[index][m], sizeof(uint));
    //         }
    //     }
    //     inputFileStream.close();
    //     return;
    // }

    // void trim_graph(uint k) {
    //     for (uint index = 0; index < dataset_size_; index++) {
    //         if ((*graph_with_distances_)[index].size() > k) {
    //             (*graph_with_distances_)[index].resize(k);
    //         }
    //     }
    // }

    //--------------- MARK: DISTANCES
    inline char* getDataByInternalId(uint index) const { return (data_pointer_ + index * data_size_); }
    float compute_distance(void* index1_ptr, void* index2_ptr) const {
        return distFunc_(index1_ptr, index2_ptr, distFuncParam_);
    }
    float compute_distance(void* index1_ptr, uint index2) const {
        return distFunc_(index1_ptr, getDataByInternalId(index2), distFuncParam_);
    }
    float compute_distance(uint index1, void* index2_ptr) const {
        return distFunc_(getDataByInternalId(index1), index2_ptr, distFuncParam_);
    }
    float compute_distance(uint index1, uint index2) const {
        return distFunc_(getDataByInternalId(index1), getDataByInternalId(index2), distFuncParam_);
    }

    //--------------- MARK: NEIGHBORS

    const std::vector<std::pair<float, uint>>& get_neighbors_(uint index) const {
        return (*graph_)[index];
    }
    void set_neighbors(uint index, const std::vector<std::pair<float, uint>>& neighbors) {
        (*graph_)[index] = neighbors;
        return;
    }
    void queue_to_reverse_vector(std::priority_queue<std::pair<float, uint>>& pq,
                                 std::vector<std::pair<float, uint>>& vec) {
        vec.clear();
        while (!pq.empty()) {
            vec.push_back(pq.top());
            pq.pop();
        }
        std::reverse(vec.begin(), vec.end());
    }

    /* update the graph with these new neighbors. thread safe. */
    void update_graph_knn(uint node, const std::vector<std::pair<float, uint>>& new_neighbors) {
        /* Update the graph with outgoing links */
        {
            /* Add the new neighbors*/
            UniquePriorityQueue pq;
            for (auto val : new_neighbors) {
                if (pq.size() < max_neighbors_ || val.first < pq.top().first) {
                    pq.push(val);
                    if (pq.size() > max_neighbors_) pq.pop_fast();
                }
            }
            while (pq.size() > max_neighbors_) pq.pop_fast();

            /* Add the existing neighbors */
            std::lock_guard<std::mutex> lock(node_neighbors_locks_[node]);
            const std::vector<std::pair<float, uint>>& current_neighbors = get_neighbors_(node);
            for (auto val : current_neighbors) {
                if (pq.size() < max_neighbors_ || val.first < pq.top().first) {
                    pq.push(val);
                    if (pq.size() > max_neighbors_) pq.pop_fast();
                }
            }
            while (pq.size() > max_neighbors_) pq.pop_fast();

            /* Update the graph with these outgoing neighbors */
            std::vector<std::pair<float, uint>> neighbors(max_neighbors_);
            for (int m = max_neighbors_ - 1; m >= 0; m--) {
                neighbors[m] = pq.top();
                pq.pop_fast();
            }
            set_neighbors(node, neighbors);
        }

        /* Update the graph with ingoing links */
        for (auto val : new_neighbors) {
            uint neighbor_node = val.second;
            float neighbor_dist = val.first;
            if (neighbor_dist >= (*graph_)[neighbor_node][max_neighbors_ - 1].first) continue;
            std::lock_guard<std::mutex> lock(node_neighbors_locks_[neighbor_node]);

            /* find place to insert into neighbors' neighbors */
            std::vector<std::pair<float, uint>>& neighbors_neighbors = (*graph_)[neighbor_node];
            int insert_place = 10000;
            for (int m = max_neighbors_; m > 0; m--) {
                if (node == neighbors_neighbors[m - 1].second) {
                    insert_place = 10000;
                    break;
                }
                if (neighbor_dist < neighbors_neighbors[m - 1].first) {
                    insert_place = m - 1;
                }
            }

            /* insert into neighbors' neighbors */
            if (insert_place < max_neighbors_) {
                neighbors_neighbors.pop_back();  // remove last element first to avoid resizing vec;
                neighbors_neighbors.insert(neighbors_neighbors.begin() + insert_place, {neighbor_dist, node});
            }
        }
    }

    //====================================================================================================
    //
    //
    //
    //
    //                                      MARK: GRAPH
    //
    //
    //
    //
    //================================================================================================

    /* initialize with a random graph */
    int random_seed_ = 101;
    void init_random_graph(uint num_neighbors) {
        max_neighbors_ = num_neighbors;
        printf(" * begin random graph initialization...\n");
        graph_.reset();
        graph_ = std::make_unique<std::vector<std::vector<std::pair<float, uint>>>>(dataset_size_);


        random_seed_ = 0;
        srand(random_seed_);

        /* assign random neighbors for all nodes */
        printf("   - random neighbor assignment\n");
        srand(random_seed_);
        for (uint x = 0; x < dataset_size_; x++) {
            std::vector<std::pair<float, uint>> neighbors(max_neighbors_);
            for (uint m = 0; m < max_neighbors_; m++) {
                uint node = rand() % dataset_size_;
                float dist = HUGE_VALF;
                neighbors[m] = {dist, node};
            }
            set_neighbors(x, neighbors);
        }

        printf("   - random neighbor distance computations\n");
        #pragma omp parallel for 
        for (uint x = 0; x < dataset_size_; x++) {
            // compute distances
            for (uint m = 0; m < max_neighbors_; m++) {
                uint node = (*graph_)[x][m].second;
                (*graph_)[x][m].first = compute_distance(x, node);
            }

            // sort
            std::sort((*graph_)[x].begin(), (*graph_)[x].end());
        }
    }

    bool verbose_ = false;


    // construction of graph on subset, recursive construction
    void init_top_layer_graph(uint num_nodes, uint num_neighbors, uint num_iterations = 0) {
        if (verbose_) printf(" * begin omap initialization...\n");

        /* initialize the omap */
        top_layer_graph_.reset();
        top_layer_graph_ = std::make_unique<Graph>(num_nodes, dimension_, num_neighbors, space_);

        /* select omap nodes */
        top_layer_nodes_.clear();
        for (uint i = 0; i < num_nodes; i++) {
            uint node = rand() % dataset_size_;
            top_layer_nodes_.push_back(node);
        }

        /* add to the index */
        for (uint node : top_layer_nodes_) {
            float* node_ptr = reinterpret_cast<float*> (getDataByInternalId(node));
            top_layer_graph_->add_point(node_ptr, node);
        }

        /* brute force construction of top layer graph */
        if (num_nodes < 1000 || num_iterations == 0) {
            top_layer_graph_->brute_force_construction();
        } 
        /* refinement based construction of top layer graph */
        else {

            /* initialize the random graph*/
            // top_layer_graph_->init_random_graph();
            top_layer_graph_->init_empty_but_top_layer((uint) 10*sqrt(num_nodes));

            /* refinement based graph construction */
            for (uint i = 0; i < num_iterations; i++) {
                top_layer_graph_->graph_refinement_iteration(32, 32);
            }
        }

        return;
    }

    uint omap_size_ = 1000; // size of the observation map
    uint omap_num_neighbors_ = 32; // number of neighbors in the observation map
    void set_omap_params(uint num_nodes, uint num_neighbors = 32) {
        omap_size_ = num_nodes;
        omap_num_neighbors_ = num_neighbors;
    }



    /* Build the kNN Graph */
    void iterate_knn_refinement(uint num_neighbors, uint num_hops = 100) {
        /* initialize the random seed */

        if (graph_ == nullptr) {
            init_random_graph(num_neighbors);
        }
        uint beam_size = max_neighbors_;

        /* initialize start nodes for search on the graph */
        start_nodes_.clear();
        for (uint m = 0; m < max_neighbors_; m++) {
            start_nodes_.push_back(rand() % dataset_size_);
        }

        /* initialize the observation map */
        {
            auto tStart = std::chrono::high_resolution_clock::now();
            uint num_nodes_top = omap_size_; // (uint) sqrt(dataset_size_); // ((double) dataset_size_ / (double) max_neighbors_);
            if (num_nodes_top > 100) {
                uint num_neighbors_top = omap_num_neighbors_;
                uint num_iterations_top = 1;
                init_top_layer_graph(num_nodes_top, num_neighbors_top, num_iterations_top);
            }
            auto tEnd = std::chrono::high_resolution_clock::now();
            double time = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd - tStart).count();
            printf(" * top layer graph initialized in %.3f seconds\n", time);
        }

        /* create a random ordering for the nodes, to the dataset */
        std::vector<uint> random_ordering(dataset_size_);
        std::iota(random_ordering.begin(), random_ordering.end(), 0);
        std::random_shuffle(random_ordering.begin(), random_ordering.end());

        /* do the random batches */
        printf(" * creating graph in batches...\n");
        uint batch_size = 10000;
        std::vector<std::vector<std::pair<float, uint>>> batch_neighbors(batch_size);
        uint batch_count = 0;
        uint batch_begin = 0;
        while (batch_begin < dataset_size_) {
            uint batch_end = batch_begin + batch_size;
            if (batch_end > dataset_size_) batch_end = dataset_size_;
            batch_count++;

            /* perform the searches in parallel */
            #pragma omp parallel for
            for (uint q = batch_begin; q < batch_end; q++) {
                uint qid = q - batch_begin;
                uint query_node = random_ordering[q];
                char* query_ptr = getDataByInternalId(query_node);

                /* perform the beam search to collect candidates */
                std::priority_queue<std::pair<float, uint>> res;
                if (top_layer_graph_ == nullptr) {
                    res = internal_beam_search(query_ptr, start_nodes_, beam_size, num_hops);
                } else {
                    // if top layer graph exists, use it to find the start node
                    // uint start_node = top_layer_graph_->search_start_node(reinterpret_cast<float*>(query_ptr));
                    auto res1 = top_layer_graph_->search((float*)(query_ptr), num_neighbors, beam_size, num_hops);
                    res = internal_beam_search(query_ptr, {res1[0].second}, beam_size, num_hops);
                    for (auto val : res1) {
                        res.emplace(val.first, val.second);
                    }
                }
                // while (res.size() > num_neighbors) res.pop();

                /* save the neighbors */
                queue_to_reverse_vector(res, batch_neighbors[qid]);
            }

            /* batch update of the graph */
            #pragma omp parallel for
            for (uint q = batch_begin; q < batch_end; q++) {
                uint qid = q - batch_begin;
                uint query_node = random_ordering[q];
                update_graph_knn(query_node, batch_neighbors[qid]);
            }

            /* setup next batch */
            if (batch_count % 10 == 0) {
                printf(" %u/%u\n", batch_end, dataset_size_);
            }
            batch_begin = batch_end;
        }
    }

    //====================================================================================================
    //
    //
    //
    //
    //                             MARK: SEARCH
    //
    //
    //
    //
    //====================================================================================================

    /* Normal beam search: start from given start nodes */
    std::priority_queue<std::pair<float, uint>> internal_beam_search(char* query_ptr, const std::vector<uint>& start_nodes, uint beam_size,
                                                                     uint max_iterations = 1000) {
        // get the visited list
        VisitedList* vl = visitedListPool_->getFreeVisitedList();
        vl_type* visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        // initializing the beam with the given start nodes
        std::priority_queue<std::pair<float, uint>> topCandidates;  // top k points found (largest first)
        std::priority_queue<std::pair<float, uint>> candidateSet;   // points to explore (smallest first)
        for (uint start_node : start_nodes) {
            float dist = compute_distance(query_ptr, start_node);
            topCandidates.emplace(dist, start_node);
            candidateSet.emplace(-dist, start_node);
            visited_array[start_node] = visited_array_tag;
        }
        while (topCandidates.size() > beam_size) topCandidates.pop();
        float lower_bound = topCandidates.top().first;

        //> Perform the beam search loop
        uint num_iterations = 0;
        while (!candidateSet.empty()) {
            uint candidate_node = candidateSet.top().second;
            float candidate_distance = -candidateSet.top().first;

            // if we have explored all points in our beam, stop
            if ((candidate_distance > lower_bound) && (topCandidates.size() >= beam_size)) break;
            candidateSet.pop();

            // setting max iterations
            if (num_iterations++ > max_iterations) break;

            // iterate through node neighbors
            const std::vector<std::pair<float, uint>>& neighbors = get_neighbors_(candidate_node);
            for (int i = 0; i < (int)neighbors.size(); i++) {
                uint neighbor_node = neighbors[i].second;

                // skip if already visisted
                if (visited_array[neighbor_node] == visited_array_tag) continue;
                visited_array[neighbor_node] = visited_array_tag;

                // compute distance
                float const dist = compute_distance(query_ptr, neighbor_node);

                // add to beam if closer than some point in the beam, or beam not yet full
                if (topCandidates.size() < beam_size || dist <= lower_bound) {
                    candidateSet.emplace(-dist, neighbor_node);

                    // update beam
                    topCandidates.emplace(dist, neighbor_node);
                    if (topCandidates.size() > beam_size) topCandidates.pop();
                    lower_bound = topCandidates.top().first;
                }
            }
        }

        visitedListPool_->releaseVisitedList(vl);
        return topCandidates;
    }
};