#pragma once
// #define _RECORD_METRICS
// #ifndef NUM_CORES
//     #define NUM_CORES 1
// #endif

#include <vector>
#include <fstream>
#include <limits>
#include <queue>
#include <random>
#include <atomic>
#include <memory>
#include <omp.h>
#include <unordered_map>
#include <mutex>
#include <algorithm>

#include "distances.h"
#include "visited-list-pool.h"
#include "unique-priority-queue.hpp"

typedef uint32_t uint;
class Graph {
public:
    float MAX_FLOAT = std::numeric_limits<float>::max();
    uint MAX_UINT = std::numeric_limits<uint>::max();

    // data
    char *data_pointer_{nullptr};           // pointer to the data in the graph
    uint dataset_size_{0};                  // max size of dataset
    uint dimension_{0};                     // dimensionality of the dataset
    uint max_neighbors_{64};         // number of bottom layer neighbors 
    std::unique_ptr<std::vector<std::vector<uint>>> graph_{nullptr};
    std::unique_ptr<std::vector<std::vector<std::pair<float, uint>>>> graph_with_distances_{nullptr};
    std::unique_ptr<std::unordered_map<uint,std::vector<uint>>> graph_top_{nullptr};

    // parallelism
    uint num_cores_ = 1;
    std::vector<std::mutex> node_neighbors_locks_; // locks for parallel neighbor updates

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
    Graph(uint dataset_size, uint dimension, distances::SpaceInterface<float>* s):
        node_neighbors_locks_(dataset_size)
    {
        printf("Constructor...\n");
        dataset_size_ = dataset_size;
        dimension_ = dimension;
        space_ = s;
        distFunc_ = space_->get_dist_func();
        distFuncParam_ = space_->get_dist_func_param();
        data_size_ = dimension_ * sizeof(float);        // EDIT FOR QUANTIZATION
        visitedListPool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, dataset_size_));
    }
    ~Graph() {};

    void load_dataset_float(float*& data_pointer) {
        printf("Loading dataset...\n");
        data_pointer_ = reinterpret_cast<char*>(data_pointer);
    }

    void set_num_cores(uint num_cores) {
        num_cores_ = num_cores;
    }


    //--------------- MARK: I/O

    /* save the graph to disk (no distances) */
    void save_graph(std::string filename) {
        printf("Saving Graph to: %s\n", filename.c_str());
        std::ofstream outputFileStream(filename, std::ios::binary);
        if (!outputFileStream.is_open()) {
            printf("Open File Error\n");
            exit(0);
        }
        printf(" * saving graph without distances...\n");

        // output the datasetSize, graphNeighborsTop, graphNeighborsBottom, sizeElementLinks_
        outputFileStream.write((char*)&dataset_size_,    sizeof(uint));
        outputFileStream.write((char*)&dimension_, sizeof(uint));
        outputFileStream.write((char*)&max_neighbors_, sizeof(uint));

        if (graph_with_distances_ != nullptr && graph_with_distances_->size() == dataset_size_) {
            for (uint index = 0; index < dataset_size_; index++) {
                if ((*graph_with_distances_)[index].size() != max_neighbors_) throw std::runtime_error("Graph size mismatch");
                for (uint m = 0; m < max_neighbors_; m++) {
                    uint neighbor = (*graph_with_distances_)[index][m].second;
                    outputFileStream.write((char*)&neighbor, sizeof(uint));
                }
            }
        } else if (graph_ != nullptr && graph_->size() == dataset_size_) {
            for (uint index = 0; index < dataset_size_; index++) {
                if ((*graph_)[index].size() != max_neighbors_) throw std::runtime_error("Graph size mismatch");
                for (uint m = 0; m < max_neighbors_; m++) {
                    uint neighbor = (*graph_)[index][m];
                    outputFileStream.write((char*)&neighbor, sizeof(uint));
                }
            }
        }

        outputFileStream.close();
        return;
    }

    /* to save memory */
    void clear_graphs() {
        printf("Clearing graph...\n");
        graph_with_distances_.reset();
        graph_with_distances_ = nullptr;
        graph_.reset();
        graph_ = nullptr;
        return;
    }

    /* load the graph to disk (no distances) */
    void load_graph(std::string filename) {
        printf("Loading graph from: %s\n", filename.c_str());
        std::ifstream inputFileStream(filename, std::ios::binary);
        if (!inputFileStream.is_open()) {
            printf("Open File Error\n");
            exit(0);
        }
        printf(" * loading graph without distances...\n");

        // check the parameters for consistency
        {
            uint dataset_size, dimension;
            inputFileStream.read((char*)&dataset_size, sizeof(uint));
            if (dataset_size != dataset_size_) throw std::runtime_error("dataset_size mismatch");
            inputFileStream.read((char*)&dimension, sizeof(uint));
            if (dimension != dimension_) throw std::runtime_error("dimension mismatch");
            inputFileStream.read((char*)&max_neighbors_, sizeof(uint));
        }
        clear_graphs();
        graph_ = std::make_unique<std::vector<std::vector<uint>>>(dataset_size_);

        /* write the top/bottom linked lists for each element */
        for (uint index = 0; index < dataset_size_; index++) {
            (*graph_)[index].resize(max_neighbors_);
            for (uint m = 0; m < max_neighbors_; m++) {
                inputFileStream.read((char*)&(*graph_)[index][m], sizeof(uint));
            }
        }
        inputFileStream.close();
        return;
    }


    //--------------- MARK: DISTANCES
    inline char* getDataByInternalId(uint index) const {
        return (data_pointer_ + index * data_size_);
    }
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
        return (*graph_with_distances_)[index];
    }
    void set_neighbors(uint index, const std::vector<std::pair<float, uint>>& neighbors) {
        (*graph_with_distances_)[index] = neighbors;
        return;
    }
    void queue_to_reverse_vector(std::priority_queue<std::pair<float, uint>>& pq, std::vector<std::pair<float, uint>>& vec) {
        vec.clear();
        while (!pq.empty()) {
            vec.push_back(pq.top());
            pq.pop();
        }
        std::reverse(vec.begin(), vec.end());
    }

    /* update the graph with these new neighbors. thread safe. */
    void update_graph_knn(uint node, const std::vector<std::pair<float,uint>>& new_neighbors) {
        
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
            const std::vector<std::pair<float,uint>>& current_neighbors = get_neighbors_(node);
            for (auto val : current_neighbors) {
                if (pq.size() < max_neighbors_ || val.first < pq.top().first) {
                    pq.push(val);
                    if (pq.size() > max_neighbors_) pq.pop_fast();
                }
            }
            while (pq.size() > max_neighbors_) pq.pop_fast();

            /* Update the graph with these outgoing neighbors */
            std::vector<std::pair<float,uint>> neighbors(max_neighbors_);
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
            std::lock_guard<std::mutex> lock(node_neighbors_locks_[neighbor_node]); 
            
            /* find place to insert into neighbors' neighbors */
            std::vector<std::pair<float,uint>>& neighbors_neighbors = (*graph_with_distances_)[neighbor_node];
            int insert_place = 10000;
            for (int m = max_neighbors_; m > 0; m--) {
                if (node == neighbors_neighbors[m-1].second) {
                    insert_place = 10000;
                    break;
                }
                if (neighbor_dist < neighbors_neighbors[m-1].first) {
                    insert_place = m - 1;
                }
            }

            /* insert into neighbors' neighbors */
            if (insert_place < max_neighbors_) {     
                neighbors_neighbors.pop_back(); // remove last element first to avoid resizing vec; 
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
    int random_seed_ = 0;
    void init_random_graph(uint num_neighbors) {
        max_neighbors_ = num_neighbors;
        printf(" * begin random graph initialization...\n");
        graph_with_distances_.reset();
        graph_with_distances_ = std::make_unique<std::vector<std::vector<std::pair<float, uint>>>>(dataset_size_);

        /* assign random neighbors for all nodes */
        printf("   - random neighbor assignment\n");
        srand(random_seed_);
        for (uint x = 0; x < dataset_size_; x++) {
            std::vector<std::pair<float,uint>> neighbors(max_neighbors_);
            for (uint m = 0; m < max_neighbors_; m++) {
                uint node = rand() % dataset_size_;
                float dist = HUGE_VALF;
                neighbors[m] = {dist, node};
            }
            set_neighbors(x, neighbors);
        }

        printf("   - random neighbor distance computations\n");
        #pragma omp parallel for num_threads(num_cores_)
        for (uint x = 0; x < dataset_size_; x++) {

            // compute distances
            for (uint m = 0; m < max_neighbors_; m++) {
                uint node = (*graph_with_distances_)[x][m].second;
                (*graph_with_distances_)[x][m].first = compute_distance(x, node);
            }

            // sort
            std::sort((*graph_with_distances_)[x].begin(), (*graph_with_distances_)[x].end());
        }
    }

    uint omap_size_ = 1000;
    uint num_neighbors_top_ = 32;
    uint max_iterations_top_ = 100;
    void set_omap_params(uint omap_size, uint num_neighbors_top, uint max_iterations_top = 100) {
        omap_size_ = omap_size;
        num_neighbors_top_ = num_neighbors_top;
        max_iterations_top_ = max_iterations_top;
    }

    /* Build the kNN Graph */
    void iterate_knn_refinement(uint num_neighbors, uint num_hops=100, int random_seed = 0) {
        if (graph_with_distances_ == nullptr) {
            init_random_graph(num_neighbors);
        }
        uint beam_size = max_neighbors_;

        /* initialize the random seed */
        if (random_seed > 0 && random_seed != random_seed_) {
            random_seed_ = random_seed;
        } else {
            random_seed_ = time(nullptr);
            srand(random_seed_);
        }

        /* create the observation map */
        printf(" * creating observation map...\n");
        graph_top_.reset();
        graph_top_ = std::make_unique<std::unordered_map<uint,std::vector<uint>>>();
        std::vector<uint> node_samples{};
        for (uint i = 0; i < omap_size_; i++) {
            uint node = rand() % dataset_size_;
            node_samples.push_back(node);
            graph_top_->emplace(node, std::vector<uint>());
        }
        #pragma omp parallel for num_threads(num_cores_)
        for (uint node : node_samples) {
            char* node_ptr = getDataByInternalId(node);
            std::vector<uint> neighbors = node_samples;
            fixed_hsp_test(node, neighbors, num_neighbors_top_);
            if (neighbors.size() > num_neighbors_top_) neighbors.resize(num_neighbors_top_);
            graph_top_->at(node) = neighbors;
        }
        uint start_node_top = node_samples[rand() % omap_size_];

        /* create a random ordering for the nodes, to add to the dataset */
        srand(3*random_seed_+1);
        std::vector<uint> random_ordering(dataset_size_);
        std::iota(random_ordering.begin(), random_ordering.end(), 0);
        std::random_shuffle(random_ordering.begin(), random_ordering.end());

        /* do the random batches */
        printf(" * creating graph in batches...\n");
        uint batch_size = 10000;
        std::vector<std::vector<std::pair<float,uint>>> batch_neighbors(batch_size, std::vector<std::pair<float,uint>>(num_neighbors, {HUGE_VALF, 0}));
        uint batch_count = 0;
        uint batch_begin = 0;
        while (batch_begin < dataset_size_) {
            uint batch_end = batch_begin + batch_size;
            if (batch_end > dataset_size_) batch_end = dataset_size_;
            batch_count++;

            /* perform the searches in parallel */
            #pragma omp parallel for num_threads(num_cores_)
            for (uint q = batch_begin; q < batch_end; q++) {
                uint qid = q - batch_begin;
                uint query_node = random_ordering[q];
                char* query_ptr = getDataByInternalId(query_node);
                
                /* perform a beam search */
                uint start_node = internal_greedy_search(query_ptr, start_node_top);
                auto res = internal_beam_search(query_ptr, start_node, beam_size, num_hops);
                while (res.size() > num_neighbors) res.pop();

                /* save the neighbors */
                queue_to_reverse_vector(res, batch_neighbors[qid]);
            }

            /* batch update of the graph */
            #pragma omp parallel for num_threads(num_cores_)
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
    std::priority_queue<std::pair<float, uint>> internal_beam_search(char* query_ptr, uint start_node, uint beam_size, uint max_iterations = 1000) {

        // get the visited list
        VisitedList* vl = visitedListPool_->getFreeVisitedList();
        vl_type* visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        // initializing the beam with the given start nodes
        std::priority_queue<std::pair<float, uint>> topCandidates;  // top k points found (largest first)
        std::priority_queue<std::pair<float, uint>> candidateSet;   // points to explore (smallest first)
        {
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
            const std::vector<std::pair<float,uint>> &neighbors = get_neighbors_(candidate_node);
            for (int i = 0; i < (int) neighbors.size(); i++) {
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


    /* Greedy search from given start nodes, fast */
    uint internal_greedy_search(char* query_ptr, uint candidate_node) {
        float candidate_distance = compute_distance(query_ptr, candidate_node);

        /* greedy search loop */
        uint num_iterations = 0;
        bool flag_continue = true;
        while (flag_continue) {
            flag_continue = false;
            if (num_iterations++ >= max_iterations_top_) break;

            // iterate through node neighbors
            const std::vector<uint>& neighbors = (*graph_top_)[candidate_node];
            for (uint neighbor_node : neighbors) {
                float dist = compute_distance(query_ptr, neighbor_node);
                if (dist < candidate_distance) {
                    candidate_distance = dist;
                    candidate_node = neighbor_node;
                    flag_continue = true;
                    // break;
                }
            }
        }

        return candidate_node;
    }

    //====================================================================================================
    //
    //
    //
    //
    //                            MARK: REFINEMENT        
    //
    //
    //
    //
    //====================================================================================================

    void refine_graph(uint max_reverse_neighbors = 0) {
        printf("Starting graph refinement to hsp graph...\n");
        if (max_reverse_neighbors == 0) max_reverse_neighbors = max_neighbors_;

        /* construct the reverse graph */
        if (graph_ == nullptr) std::runtime_error("Graph is not initialized");
        std::unique_ptr<std::vector<std::vector<uint>>> reverse_graph = std::make_unique<std::vector<std::vector<uint>>>(dataset_size_);
        for (uint x = 0; x < dataset_size_; x++) {
            (*reverse_graph)[x].reserve(max_reverse_neighbors);
        }

        printf(" * creating reverse graph...\n");
        // adding the reverse links, but in a crazy order... it makes sense I swear
        for (uint m = 0; m < max_neighbors_; m++) {

            #pragma omp parallel for num_threads(num_cores_)
            for (uint x = 0; x < dataset_size_; x++) {
                uint neighbor = (*graph_)[x][m];
                std::lock_guard<std::mutex> lock(node_neighbors_locks_[neighbor]); // lock the neighbor node

                // skip if the reverse neighbors are full
                if ((*reverse_graph)[neighbor].size() >= max_reverse_neighbors) continue;

                // check that x is not already a neighbor of n
                if (std::find((*graph_)[neighbor].begin(), (*graph_)[neighbor].end(), x) != (*graph_)[neighbor].end()) continue;

                // add it to the reverse graph
                (*reverse_graph)[neighbor].push_back(x);
            }
        }

        /* refine with the hsp test */
        printf(" * refining with hsp condition...\n");
        #pragma omp parallel for num_threads(num_cores_)
        for (uint x = 0; x < dataset_size_; x++) {
            std::vector<uint> neighbors = (*graph_)[x];
            neighbors.insert(neighbors.end(), (*reverse_graph)[x].begin(), (*reverse_graph)[x].end());
            // knn_test(x, neighbors, max_neighbors_);
            fixed_hsp_test(x, neighbors, max_neighbors_);
            (*graph_)[x] = neighbors;
        }
    }

    void knn_test(uint x, std::vector<uint>& neighbors, uint m) {

        std::vector<std::pair<float, uint>> candidates;
        for (uint index : neighbors) {
            if (x == index) continue; // skip self
            float dist = compute_distance(x, index);
            candidates.push_back({dist, index});
        }
        std::sort(candidates.begin(), candidates.end());
        neighbors.clear();
        for (uint i = 0; i < candidates.size(); i++) {
            neighbors.push_back(candidates[i].second);
            if (i >= m) break;
        }
    }

    void fixed_hsp_test(uint x, std::vector<uint>& neighbors, uint m) {

        // - initialize the list with all points and distances
        std::vector<std::tuple<float, uint, int>> active_list;
        active_list.reserve(neighbors.size());
        for (uint index : neighbors) {
            if (index == x) continue;
            float distance = compute_distance(x, index);
            active_list.emplace_back(distance, index, 0);
        }
        neighbors.clear();

        // sort by increasing distance to the node
        std::sort(active_list.begin(), active_list.end());

        // Find each [modified] hsp neighbors, fixed at d
        for (uint i = 0; i < m; i++) {

            // find the next neighbor
            uint it1 = 0;  // iterator to next neighbor
            int interference1 = 1000000;
            float distance_Q1 = HUGE_VALF;
            for (uint it2 = 0; it2 < (uint)active_list.size(); it2++) {
                float interference2 = std::get<2>(active_list[it2]);
                float distance_Q2 = std::get<0>(active_list[it2]);
                if (interference2 < interference1) {
                    interference1 = interference2;
                    distance_Q1 = distance_Q2;
                    it1 = it2;
                }
                // else is not necessary: the distance will always be further, sorted
            }

            // add the hsp neighbor
            if (it1 >= active_list.size()) break;
            uint index1 = std::get<1>(active_list[it1]);
            neighbors.push_back(index1);
            std::get<2>(active_list[it1]) = 2000000;  // invalidate

            // iterate through all nodes (further away) to tally invalidation
            for (uint it2 = it1 + 1; it2 < (uint)active_list.size(); it2++) {
                uint index2 = std::get<1>(active_list[it2]);
                float distance_Q2 = std::get<0>(active_list[it2]);
                float distance_12 = compute_distance(index1, index2);
                if (distance_12 < distance_Q2) {
                    std::get<2>(active_list[it2])++;
                }
            }
        }

        return;
    }




};