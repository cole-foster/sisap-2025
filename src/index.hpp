#pragma once
// #define _RECORD_METRICS

#include <vector>
#include <fstream>
#include <queue>
#include <random>
#include <atomic>
#include <memory>
#include <omp.h>

#include "distances.h"
#include "visited-list-pool.h"

typedef uint32_t uint;
class Index {
  public:
    float MAX_FLOAT = std::numeric_limits<float>::max();
    uint MAX_UINT = std::numeric_limits<uint>::max();

    // MARK: VARIABLES
    char* data_{nullptr};               // Storing data and graph together
    uint dataset_size_{0};               // max size of dataset
    uint graphNeighborsTop_{64};        // number of top layer neighbors
    uint graphNeighborsBottom_{64};     // number of bottom layer neighbors 
    uint maxNeighbors_{128};            // total number of neighbors

    size_t sizeTopLinks_{0};            // memory for the top links in graph: (uint + graphNeighborsTop_*uint)
    size_t sizeBottomLinks_{0};         // memory for the top links in graph: (uint + graphNeighborsBottom_*uint)
    size_t sizeElementLinks_{0};        // memory for each node in graph: sizeTopLinks_ + sizeBottomLinks_
    size_t sizeElementData_{0};         // memory for each data vector: dimension * sizeof(DATA_T)
    size_t sizePerElement_{0};          // total memory for each node: sizeElementLinks_ + sizePerElement_

    // distances
    distances::SpaceInterface<float>* space_;
    DISTFUNC<float> distFunc_;
    void* distFuncParam_{nullptr};

    // selection
    int randomSeed_{0};
    std::default_random_engine level_generator_;

    // top layer graph shards
    int num_shards_{0};                                      // number of shards in top layer graph
    std::vector<uint> shard_assignments_{};                         // start nodes, some in each shard

    // add the construction parameters, search parameters (visited list, etc.)
    std::vector<std::vector<uint>> starts_{};   // start nodes in bottom layer graph
    int beamSize_{1};
    uint maxIterations_{100};
    std::unique_ptr<VisitedListPool> visitedListPool_{nullptr};
    std::unique_ptr<VisitedList> visitedList_{nullptr}; // quickly accessible

    // metrics
    std::atomic<size_t> metrics_hops_{0};
    std::atomic<size_t> metrics_distance_computations_{0};

    // MARK: CONSTRUCTOR
    Index(uint datasetSize, distances::SpaceInterface<float>* s, uint graphNeighborsTop=32, uint graphNeighborsBottom=32) {
        dataset_size_ = datasetSize;

        // distance function (from hnswlib)
        space_ = s;
        distFunc_ = space_->get_dist_func();
        distFuncParam_ = space_->get_dist_func_param();
        sizeElementData_ = space_->get_data_size();

        // graph
        graphNeighborsTop_ = graphNeighborsTop;
        graphNeighborsBottom_ = graphNeighborsBottom;

        // initialize data, storing graph + dataset together (from hnswlib)
        sizeTopLinks_  = sizeof(uint) + graphNeighborsTop_ * sizeof(uint);
        sizeBottomLinks_ = sizeof(uint) + graphNeighborsBottom_ * sizeof(uint);
        sizeElementLinks_ = sizeTopLinks_ + sizeBottomLinks_;
        sizePerElement_ = sizeElementLinks_ + sizeElementData_;
        data_ = (char*) malloc(dataset_size_ * sizePerElement_);
        if (data_ == nullptr) throw std::runtime_error("Not enough memory");

        // visited list initialization for beam search
        visitedListPool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(32, dataset_size_));
        visitedList_ = std::unique_ptr<VisitedList>(new VisitedList(dataset_size_));

        // define random start nodes
        set_start_nodes(1024);
        srand(randomSeed_);
    }
    ~Index() {
        if (data_ != nullptr) free(data_);
        data_ = nullptr;
    }

    //--------------- MARK: DATA

    // pointer to the vector of the element
    char* getDataByIndex(uint index) const { 
        return (data_ + index * sizePerElement_ + sizeElementLinks_);   // data is offset by links
    }

    // add each data point and initialize the bottom layer graph
    void addPoint(const void* element_ptr, uint element) {
        if (element >= dataset_size_) throw std::runtime_error("Element ID exceeded dataset_size_");

        // - initializing and setting data memory for bottom level
        memset(data_ + element * sizePerElement_, 0, sizePerElement_);
        memcpy(data_ + element * sizePerElement_ + sizeElementLinks_, element_ptr, sizeElementData_); // set the data
        return;
    }

    //--------------- MARK: DISTANCES

    float compute_distance(char* index1_ptr, char* index2_ptr) const {
        return distFunc_(index1_ptr, index2_ptr, distFuncParam_);
    }
    float compute_distance(char* index1_ptr, uint index2) const {
        return distFunc_(index1_ptr, getDataByIndex(index2), distFuncParam_);
    }
    float compute_distance(uint index1, char* index2_ptr) const {
        return distFunc_(getDataByIndex(index1), index2_ptr, distFuncParam_);
    }
    float compute_distance(uint index1, uint index2) const {
        return distFunc_(getDataByIndex(index1), getDataByIndex(index2), distFuncParam_);
    }

    //====================================================================================================
    //
    //
    //
    //
    //                                      MARK: NEIGHBORS
    //
    //
    //
    //
    //====================================================================================================

    // get the linked list of index for the top/bottom graph
    uint* get_linkedList(uint index, bool top = false) const {
        if (top) {
            return (uint*) (data_ + index * sizePerElement_);
        } else {
            return (uint*) (data_ + index * sizePerElement_ + sizeTopLinks_);
        }
    }

    // get number of neighbors in the linked list
    uint get_linkedListCount(uint* ptr) const {
        return *(ptr);
    }
    void set_linkedListCount(uint* ptr, uint count) const {
        *(ptr) = count;
    }

    // set the neighbors, given a vector
    void set_neighbors(uint index, std::vector<uint> const& neighbors, bool top = false) const {
        uint numNeighbors = (uint) neighbors.size();

        if (top) {
            if (numNeighbors > graphNeighborsTop_) numNeighbors = graphNeighborsTop_;
        } else {
            if (numNeighbors > graphNeighborsBottom_) numNeighbors = graphNeighborsBottom_;
        }

        uint* indexData = get_linkedList(index, top);
        set_linkedListCount(indexData, numNeighbors);
        
        uint* indexNeighbors = (uint*)(indexData + 1);
        for (uint i = 0; i < numNeighbors; i++) {
            indexNeighbors[i] = neighbors[i];
        }
    }
    uint* get_neighbors(uint index, uint& numNeighbors, bool top = false) const {
        uint* indexData = get_linkedList(index, top);
        numNeighbors = get_linkedListCount(indexData);
        return (uint*)(indexData + 1);
    }
    uint get_numNeighbors(uint index, bool top) const {
        uint* indexData = get_linkedList(index, top);
        return get_linkedListCount(indexData);
    }

    //====================================================================================================
    //
    //
    //
    //
    //                                      MARK: GRAPH I/O
    //
    //
    //
    //
    //====================================================================================================

    void saveGraph(std::string filename) {
        printf("Saving Graph to: %s\n", filename.c_str());
        std::ofstream outputFileStream(filename, std::ios::binary);
        if (!outputFileStream.is_open()) {
            printf("Open File Error\n");
            exit(0);
        }

        // output the datasetSize, graphNeighborsTop, graphNeighborsBottom, sizeElementLinks_
        outputFileStream.write((char*)&dataset_size_,    sizeof(uint));
        outputFileStream.write((char*)&graphNeighborsTop_, sizeof(uint));
        outputFileStream.write((char*)&graphNeighborsBottom_, sizeof(uint));
        outputFileStream.write((char*)&sizeElementLinks_, sizeof(size_t));

        // write the top/bottom linked lists for each element
        for (uint index = 0; index < dataset_size_; index++) {
            outputFileStream.write((char*) data_ + index * sizePerElement_, sizeElementLinks_);
        }

        outputFileStream.close();
        return;
    }

    void loadGraph(std::string filename) {
        printf("Loading graph from: %s\n", filename.c_str());
        printf("Please make sure all elements were added before loading graph...\n");
        std::ifstream inputFileStream(filename, std::ios::binary);
        if (!inputFileStream.is_open()) {
            printf("Open File Error\n");
            exit(0);
        }

        // ensure match with the datasetSize, sizeElementLinks
        {
            uint datasetSize;
            inputFileStream.read((char*)&datasetSize, sizeof(uint));
            if (datasetSize != dataset_size_) throw std::runtime_error("datasetSize mismatch");

            uint graphNeighborsTop;
            inputFileStream.read((char*)&graphNeighborsTop, sizeof(uint));
            if (graphNeighborsTop != graphNeighborsTop_) throw std::runtime_error("graphNeighborsTop mismatch");

            uint graphNeighborsBottom;
            inputFileStream.read((char*)&graphNeighborsBottom, sizeof(uint));
            if (graphNeighborsBottom != graphNeighborsBottom_) throw std::runtime_error("graphNeighborsBottom mismatch");

            size_t sizeElementLinks;
            inputFileStream.read((char*)&sizeElementLinks, sizeof(size_t));
            if (sizeElementLinks != sizeElementLinks_) throw std::runtime_error("sizeElementLinks mismatch");
        }

        // read the top/bottom linked lists for each element
        for (uint index = 0; index < dataset_size_; index++) {
            inputFileStream.read((char*) data_ + index * sizePerElement_, sizeElementLinks_);
        }

        inputFileStream.close();
        return;
    }

    void loadGraphBottom(std::vector<std::vector<uint>> const& graph) {
        if (graph.size() != dataset_size_) throw std::runtime_error("graph size mismatch");
        for (uint index = 0; index < dataset_size_; index++) {
            std::vector<uint> neighbors = graph[index];
            if (neighbors.size() > graphNeighborsBottom_) neighbors.resize(graphNeighborsBottom_);
            set_neighbors(index, graph[index], false);
        }
    }

    void loadGraphTop(std::vector<std::vector<uint>> const& graph) {
        if (graph.size() != dataset_size_) throw std::runtime_error("graph size mismatch");
        for (uint index = 0; index < dataset_size_; index++) {
            std::vector<uint> neighbors = graph[index];
            if (neighbors.size() > graphNeighborsTop_) neighbors.resize(graphNeighborsTop_);
            set_neighbors(index, graph[index], true);
        }
    }

    void printGraphStats() {
        printf("Graph Statistics:\n");
        printf(" * number of nodes: %u\n", dataset_size_);
        
        // top layer graph
        {
            size_t min_degree = 1000000;
            size_t max_degree = 0;
            size_t num_edges = 0.0f;
            for (uint node = 0; node < dataset_size_; node++) {
                size_t node_edges = get_numNeighbors(node, true);
                num_edges += node_edges;
                if (node_edges < min_degree) min_degree = node_edges;
                if (node_edges > max_degree) max_degree = node_edges;
            }
            double ave_degree = (double)num_edges / (double) dataset_size_;
            printf("Top Layer Graph [min/max/ave]: %u, %u, %.2f\n", min_degree, max_degree, ave_degree);
        }

        // bottom layer graph
        {
            size_t min_degree = 1000000;
            size_t max_degree = 0;
            size_t num_edges = 0.0f;
            for (uint node = 0; node < dataset_size_; node++) {
                size_t node_edges = get_numNeighbors(node, false);
                num_edges += node_edges;
                if (node_edges < min_degree) min_degree = node_edges;
                if (node_edges > max_degree) max_degree = node_edges;
            }
            double ave_degree = (double)num_edges / (double)dataset_size_;
            printf("Bottom Layer Graph [min/max/ave]: %u, %u, %.2f\n", min_degree, max_degree, ave_degree);
        }

        fflush(stdout);
        return;
    }

    //====================================================================================================
    //
    //
    //
    //
    //                                      MARK: STARTS
    //
    //
    //
    //
    //====================================================================================================

    // set random start nodes
    void set_start_nodes(uint max_searches = 1024, uint num_starts = 1) {
        srand(3);

        starts_.clear();
        starts_.resize(max_searches);
        for (uint p = 0; p < max_searches; p++) {
            starts_[p].resize(num_starts);
            for (uint n = 0; n < num_starts; n++) {
                starts_[p][n] = (uint) (rand() % dataset_size_);
            }
        }
    }

    // set random start nodes in each shard
    void set_shard_starts(const std::vector<uint>& shard_assignments, uint max_searches = 1024, uint num_starts = 1) {
        if (shard_assignments.size() != dataset_size_) throw std::runtime_error("shard_assignment size not matching dataset size");
        shard_assignments_ = shard_assignments;

        // find the number of shards
        num_shards_ = 0;
        for (uint i = 0; i < dataset_size_; i++) {
            if (shard_assignments[i] > num_shards_) num_shards_ = shard_assignments[i];
        }
        num_shards_++;
        printf(" * num_shards=%u\n", num_shards_);

        // organize into the shards
        std::vector<std::vector<uint>> shards(num_shards_);
        for (uint index = 0; index < dataset_size_; index++) {
            uint shard_id = shard_assignments_[index];
            shards[shard_id].push_back(index);
        }
        printf(" * size_shards=%u\n", shards[0].size());

        // get the start nodes in each shard, cycling through the shards
        srand(5);
        starts_.clear();
        starts_.resize(max_searches, std::vector<uint>(num_starts));
        uint shard_id = 0;
        for (uint m = 0; m < max_searches; m++) {
            uint shard_size = (uint) shards[shard_id].size();

            // fill with random starts in the shard
            for (uint i = 0; i < num_starts; i++) {
                uint rand_index = rand() % shard_size;
                starts_[m][i] = shards[shard_id][rand_index];
            }
            shard_id++;
            if (shard_id >= num_shards_) shard_id = 0;
        }
    }

    //====================================================================================================
    //
    //
    //
    //
    //                             GRAPH        
    //
    //
    //
    //
    //====================================================================================================

    /* initialize with a random graph */
    void init_random_graph(size_t random_seed = 3) {
        srand(random_seed);
        for (uint x = 0; x < dataset_size_; x++) {
            std::vector<uint> neighbors(graphNeighborsBottom_);
            for (uint m = 0; m < graphNeighborsBottom_; m++) {
                neighbors[m] = rand() % dataset_size_;
            }
            set_neighbors(x, neighbors, false);
        }
    }

    void update_graph(uint index, const std::vector<uint>& neighbors) {
        
    }

    //====================================================================================================
    //
    //
    //
    //
    //                             SEARCH        
    //
    //
    //
    //
    //====================================================================================================

    uint k_ = 1;
    uint beam_size_ = 1;
    uint max_iterations_ = 10000;
    uint params_max_iterations_top_ = 10000;
    uint max_starts_ = 10000;
    void set_search_params(uint k, uint beam_size, uint max_iterations = 10000, uint max_starts = 10000) {
        k_ = k;
        beam_size_ = beam_size;
        max_iterations_ = max_iterations;
        max_starts_ = max_starts;
    }

    // MARK: >> SOLO BEAM

    /* Normal beam search: start from given start nodes */
    std::priority_queue<std::pair<float, uint>> internal_beam_search(const void* query_ptr_v, const std::vector<uint>& startNodes, bool flag_top = false) {
        char* query_ptr = (char*) (query_ptr_v);

        #ifdef _RECORD_METRICS
            size_t temp_hops = 1; // start nodes
            size_t temp_distances = 0;
        #endif

        // get the visited list
        VisitedList* vl = visitedListPool_->getFreeVisitedList();
        vl_type* visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        // initializing the beam with the given start nodes
        std::priority_queue<std::pair<float, uint>> topCandidates;  // top k points found (largest first)
        std::priority_queue<std::pair<float, uint>> candidateSet;   // points to explore (smallest first)
        for (uint i = 0; i < max_starts_; i++) {
            if (i >= startNodes.size()) break;
            uint startNode = startNodes[i];

            float dist = compute_distance(query_ptr, startNode);
            topCandidates.emplace(dist, startNode);
            candidateSet.emplace(-dist, startNode);
            visited_array[startNode] = visited_array_tag;

            #ifdef _RECORD_METRICS
                temp_distances++;
            #endif
        }
        while (topCandidates.size() > beam_size_) topCandidates.pop();
        float lower_bound = topCandidates.top().first;

        //> Perform the beam search loop
        uint num_iterations = 0;
        while (!candidateSet.empty()) {
            uint candidate_node = candidateSet.top().second;
            float candidate_distance = -candidateSet.top().first;

            // if we have explored all points in our beam, stop
            // if ((candidate_distance > lower_bound) && (topCandidates.size() >= beam_size_)) break;
            if (candidate_distance > lower_bound) break;
            candidateSet.pop();

            // setting max iterations
            if (num_iterations++ > max_iterations_) break;

            // max number of iterations
            #ifdef _RECORD_METRICS
                temp_hops++;
            #endif

            // iterate through node neighbors
            uint num_neighbors;
            uint* neighbors = get_neighbors(candidate_node, num_neighbors, flag_top);
            for (int i = 0; i < num_neighbors; i++) {
                uint neighbor_node = neighbors[i];

                // skip if already visisted
                if (visited_array[neighbor_node] == visited_array_tag) continue;
                visited_array[neighbor_node] = visited_array_tag;

                // compute distance
                float const dist = compute_distance(query_ptr, neighbor_node); // FIX: turn to after continue
                #ifdef _RECORD_METRICS
                    temp_distances++;
                #endif

                // add to beam if closer than some point in the beam, or beam not yet full
                if (topCandidates.size() < beam_size_ || dist <= lower_bound) {
                    candidateSet.emplace(-dist, neighbor_node);

                    // update beam
                    topCandidates.emplace(dist, neighbor_node);
                    if (topCandidates.size() > beam_size_) topCandidates.pop();
                    lower_bound = topCandidates.top().first;
                }
            }
        }

        #ifdef _RECORD_METRICS
            metrics_hops_ += temp_hops;
            metrics_distance_computations_ += temp_distances;
        #endif

        visitedListPool_->releaseVisitedList(vl);
        return topCandidates;
    }

    /* Greedy search from given start nodes, fast */
    uint internal_greedy_search(const void* query_ptr_v, const std::vector<uint>& startNodes, bool flag_top = true) {
        char* query_ptr = (char*) (query_ptr_v);

        #ifdef _RECORD_METRICS
            size_t temp_metrics_hops_ = 1; // start nodes hop
            size_t temp_metrics_distance_computations_ = 0;
        #endif

        /* find the best start node */
        uint candidate_node;
        float candidate_distance = MAX_FLOAT;
        for (uint i = 0; i < max_starts_; i++) {
            if (i >= startNodes.size()) break;
            uint startNode = startNodes[i];
            float dist = compute_distance(query_ptr, startNode);
            if (dist < candidate_distance) {
                candidate_distance = dist;
                candidate_node = startNode;
            }
            #ifdef _RECORD_METRICS
                temp_metrics_distance_computations_++;
            #endif
        }

        /* greedy search loop */
        uint num_iterations = 0;
        bool flag_continue = true;
        while (flag_continue) {
            flag_continue = false;
            if (num_iterations++ >= params_max_iterations_top_) break;
            #ifdef _RECORD_METRICS
                temp_metrics_hops_++;
            #endif

            // iterate through node neighbors
            uint num_neighbors;
            uint* neighbors = get_neighbors(candidate_node, num_neighbors, flag_top);
            for (int i = 0; i < num_neighbors; i++) {
                uint neighbor_node = neighbors[i];

                // compute distance
                float dist = compute_distance(query_ptr, neighbor_node);
                #ifdef _RECORD_METRICS
                    temp_metrics_distance_computations_++;
                #endif

                if (dist < candidate_distance) {
                    candidate_distance = dist;
                    candidate_node = neighbor_node;
                    flag_continue = true;
                    // break;
                }
            }
        }

        #ifdef _RECORD_METRICS
            metrics_hops_ += temp_metrics_hops_;
            metrics_distance_computations_ += temp_metrics_distance_computations_;
        #endif

        return candidate_node;
    }




    //====================================================================================================
    //
    //
    //
    //
    //                                      MARK: SEARCH EXTERN
    //
    //
    //
    //
    //====================================================================================================

    /* normal beam search */
    void search_beam(const void* query_ptr_v, std::priority_queue<std::pair<float, uint>>& top_candidates, uint start_node, bool flag_top = false) {
        char* query_ptr = (char*) query_ptr_v;
        top_candidates = internal_beam_search(query_ptr, {start_node}, false);
        while (top_candidates.size() > k_) top_candidates.pop();
        return;
    }
    void search_beam(const void* query_ptr_v, std::priority_queue<std::pair<float, uint>>& top_candidates) {
        const char* query_ptr = (char*) query_ptr_v;
        top_candidates = internal_beam_search(query_ptr, starts_[0], false);
        while (top_candidates.size() > k_) top_candidates.pop();
        return;
    }
    void search_beam_top_down(const void* query_ptr_v, std::priority_queue<std::pair<float, uint>>& top_candidates) {
        const char* query_ptr = (char*) query_ptr_v;
        uint start_node = internal_greedy_search(query_ptr, starts_[0], true);
        top_candidates = internal_beam_search(query_ptr, {start_node}, false);
        while (top_candidates.size() > k_) top_candidates.pop();
        return;
    }
};