// interact -q batch -n 32 -m 64g -t 12:00:00
// g++ main.cpp -o main -fopenmp -O3
#define NUM_CORES 8

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <queue>
#include <vector>
#include <memory>
#include <mutex>
#include <numeric>
#include <memory>


#include "utils.hpp"
#include "src/distances.h"
#include "src/visited-list-pool.h"
#include "src/unique-priority-queue.hpp"

/* dataset */
size_t dataset_size, dimension, gt_k;
float* data_pointer = nullptr;
std::vector<std::vector<uint>> gt_graph;

/* distances */
distances::SpaceInterface<float>* space;
DISTFUNC<float> distFunc_;
void* distFuncParam_{nullptr};
float compute_distance(float* index1_ptr, float* index2_ptr) {
    return distFunc_(index1_ptr, index2_ptr, distFuncParam_);
}

/* graph + search */
std::vector<std::vector<std::pair<float,uint>>> graph;
std::vector<std::mutex> node_mutexes(3001496); // size should be dataset_size, but this is a quick fix for the example
uint num_neighbors = 64;
void init_random_graph();
std::unique_ptr<VisitedListPool> visitedListPool_{nullptr};
std::priority_queue<std::pair<float, uint>> internal_beam_search(float* query_ptr, uint start_node, uint beam_size=1, uint max_iterations=1000);
void update_graph(uint node, std::priority_queue<std::pair<float,uint>>& new_neighbors);
void update_graph1(uint node, std::priority_queue<std::pair<float,uint>>& new_neighbors);
void update_graph_parallel(uint node, const std::vector<std::pair<float,uint>>& new_neighbors);
double measure_neighbors_accuracy(const std::vector<uint>& gt_neighbors, const std::vector<std::pair<float,uint>>& neighbors, int k);
double measure_graph_quality(const std::vector<std::vector<uint>>& gt_graph, const std::vector<std::vector<std::pair<float, uint>>>& knn_graph, uint k);
void queue_to_reverse_vector(std::priority_queue<std::pair<float, uint>>& pq, std::vector<std::pair<float, uint>>& vec);


// MARK: MAIN
int main() {
    uint k = 15;
    num_neighbors = 30;
    uint beam_size = 30;

    /* Loading dataset */
    printf("Loading dataset...\n");
    std::string data_filename = "/users/cfoste18/data/cfoste18/knn-construction/data/gooaq-N-3001496-D-384-fp32.bin";
    load_dataset_google_qa(data_filename, data_pointer, dataset_size, dimension);
    printf("Google QA dataset\n");
    printf(" * N=%u\n", (uint)dataset_size);
    printf(" * D=%u\n", (uint)dimension);
    space = new distances::InnerProductSpace(dimension);
    distFunc_ = space->get_dist_func();
    distFuncParam_ = space->get_dist_func_param();
    visitedListPool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(NUM_CORES, dataset_size));
    // node_mutexes.resize(dataset_size);

    /* Loading ground truth */
    std::string gt_filename = "/users/cfoste18/data/cfoste18/knn-construction/data/knn-gooaq-N-3001496-k-32-int32.bin";
    load_gt_google_qa(gt_filename, gt_graph, dataset_size, gt_k);
    printf(" * gt_k=%u\n", (uint) gt_k);

    /* Initialize graph */
    printf("Initializing random graph...\n");
    printf(" * num_neighbors=%u\n", num_neighbors);
    init_random_graph();

    /* create a random ordering for the nodes */
    srand(4);
    std::vector<uint> random_ordering(dataset_size);
    std::iota(random_ordering.begin(), random_ordering.end(), 0);
    std::random_shuffle(random_ordering.begin(), random_ordering.end());

    /* create random start nodes */
    srand(7);
    std::vector<uint> random_starts(dataset_size);
    std::iota(random_starts.begin(), random_starts.end(), 0);
    std::random_shuffle(random_starts.begin(), random_starts.end());

    /* update in batches */
    uint batch_size = 10000;
    std::vector<std::vector<std::pair<float,uint>>> batch_neighbors(batch_size, std::vector<std::pair<float,uint>>(num_neighbors, {HUGE_VALF, 0}));

    /* update in batches */
    printf("Batch graph updates...\n");
    printf("Iteration, Time (s), Accuracy, Peak Memory (MB)\n");
    auto tStart = std::chrono::high_resolution_clock::now();
    uint batch_begin = 0;
    while (batch_begin < dataset_size) {
        uint batch_end = batch_begin + batch_size;
        if (batch_end > dataset_size) batch_end = dataset_size;

        /* perform the searches in parallel */
        #pragma omp parallel for num_threads(NUM_CORES)
        for (uint q = batch_begin; q < batch_end; q++) {
            uint qid = q - batch_begin;
            uint query_node = random_ordering[q];
            float* query_ptr = data_pointer + query_node*dimension;
            
            /* perform a beam search */
            uint rand_start = random_starts[q];
            auto res = internal_beam_search(query_ptr, rand_start, beam_size);
            while (res.size() > num_neighbors) res.pop();

            /* save the neighbors */
            queue_to_reverse_vector(res, batch_neighbors[qid]);
        }

        /* batch update of the graph */
        #pragma omp parallel for num_threads(NUM_CORES)
        for (uint q = batch_begin; q < batch_end; q++) {
            uint qid = q - batch_begin;
            uint query_node = random_ordering[q];
            update_graph_parallel(query_node, batch_neighbors[qid]);
        }

        /* output stats */
        double accuracy = measure_graph_quality(gt_graph, graph, k);
        auto tEnd = std::chrono::high_resolution_clock::now();
        double time = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd - tStart).count();
        size_t peak_memory = getPeakRSS();
        size_t current_memory = getCurrentRSS();
        printf("%u, %.2f, %.5f, %.3f\n", batch_end, time, 100 * accuracy, (double) peak_memory / 1024 / 1024);
        fflush(stdout);

        /* setup next batch */
        batch_begin = batch_end;
    }

    delete space;
    if (data_pointer != nullptr) delete[] data_pointer;
}


/* initialize with a random graph */
void init_random_graph() {
    graph.clear();
    graph.resize(dataset_size);

    srand(3);
    for (uint x = 0; x < dataset_size; x++) {
        std::vector<std::pair<float,uint>> neighbors(num_neighbors);
        for (uint m = 0; m < num_neighbors; m++) {
            uint node = rand() % dataset_size;
            float dist = HUGE_VALF;
            neighbors[m] = {dist, node};
        }
        graph[x] = neighbors;
    }

    #pragma omp parallel for num_threads(NUM_CORES)
    for (uint x = 0; x < dataset_size; x++) {

        // compute distances
        for (uint m = 0; m < num_neighbors; m++) {
            uint node = graph[x][m].second;
            graph[x][m].first = compute_distance(data_pointer+x*dimension, data_pointer+node*dimension);
        }

        // sort
        std::sort(graph[x].begin(), graph[x].end());
    }
}


/* Normal beam search: start from given start nodes */
std::priority_queue<std::pair<float, uint>> internal_beam_search(float* query_ptr, uint start_node, uint beam_size, uint max_iterations) {

    // get the visited list
    VisitedList* vl = visitedListPool_->getFreeVisitedList();
    vl_type* visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    // initializing the beam with the given start nodes
    std::priority_queue<std::pair<float, uint>> topCandidates;  // top k points found (largest first)
    std::priority_queue<std::pair<float, uint>> candidateSet;   // points to explore (smallest first)
    {
        float dist = compute_distance(query_ptr, data_pointer+start_node*dimension);
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
        const std::vector<std::pair<float,uint>> &neighbors = graph[candidate_node];
        for (int i = 0; i < num_neighbors; i++) {
            uint neighbor_node = neighbors[i].second;

            // skip if already visisted
            if (visited_array[neighbor_node] == visited_array_tag) continue;
            visited_array[neighbor_node] = visited_array_tag;

            // compute distance
            float const dist = compute_distance(query_ptr, data_pointer+neighbor_node*dimension);

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

/* update the graph with the neighbors of this query */
void update_graph(uint node, std::priority_queue<std::pair<float,uint>>& new_neighbors) {
    UniquePriorityQueue pq;
    
    // add existing neighbors
    for (auto val : graph[node]) {
        pq.push(val);
    }

    // add new neighbors
    while (new_neighbors.size() > 0) {
        pq.push(new_neighbors.top());
        new_neighbors.pop();
    }
    while (pq.size() > num_neighbors) pq.pop_fast();

    // update graph
    std::vector<std::pair<float,uint>> neighbors(num_neighbors);
    for (int m = num_neighbors - 1; m >= 0; m--) {
        neighbors[m] = pq.top();
        pq.pop_fast();
    }
    graph[node] = neighbors;
}

/* update the graph with the neighbors of this query */
void update_graph1(uint node, std::priority_queue<std::pair<float,uint>>& new_neighbors) {
    UniquePriorityQueue pq;
    
    // add existing neighbors
    for (auto val : graph[node]) {
        pq.push(val);
    }

    // add new neighbors
    while (new_neighbors.size() > 0) {
        auto val = new_neighbors.top();
        pq.push(val);
        new_neighbors.pop();

        // update the respective knn list
        uint neighbor_node = val.second;
        float neighbor_dist = val.first;
        
        /* find place to insert into neighbors' neighbors */
        std::vector<std::pair<float,uint>>& neighbors_neighbors = graph[neighbor_node];
        int insert_place = 10000;
        for (int m = num_neighbors; m > 0; m--) {
            if (node == neighbors_neighbors[m-1].second) {
                insert_place = 10000;
                break;
            }
            if (neighbor_dist < neighbors_neighbors[m-1].first) {
                insert_place = m - 1;
            }
        }

        /* insert into neighbors' neighbors */
        if (insert_place < num_neighbors) {
            neighbors_neighbors.pop_back(); // remove last element first to avoid resizing vec
            neighbors_neighbors.insert(neighbors_neighbors.begin() + insert_place, {neighbor_dist, node});
        }

    }
    while (pq.size() > num_neighbors) pq.pop_fast();

    // update graph
    std::vector<std::pair<float,uint>> neighbors(num_neighbors);
    for (int m = num_neighbors - 1; m >= 0; m--) {
        neighbors[m] = pq.top();
        pq.pop_fast();
    }
    graph[node] = neighbors;
}

/* this is done in parallel */
void update_graph_parallel(uint node, const std::vector<std::pair<float,uint>>& new_neighbors) {
    
    /* Update the graph with outgoing links */
    {
        // lock the node --> maybe not needed this early
        // std::lock_guard<std::mutex> lock(node_mutexes[node]); 

        /* Add the new neighbors*/
        UniquePriorityQueue pq;
        for (auto val : new_neighbors) {
            if (pq.size() < num_neighbors || val.first < pq.top().first) {
                pq.push(val);
                if (pq.size() > num_neighbors) pq.pop_fast();
            }
        }
        while (pq.size() > num_neighbors) pq.pop_fast();

        /* Add the existing neighbors */
        std::lock_guard<std::mutex> lock(node_mutexes[node]); 
        for (auto val : graph[node]) {
            if (pq.size() < num_neighbors || val.first < pq.top().first) {
                pq.push(val);
                if (pq.size() > num_neighbors) pq.pop_fast();
            }
        }
        while (pq.size() > num_neighbors) pq.pop_fast();

        /* Update the graph with these outgoing neighbors */
        std::vector<std::pair<float,uint>> neighbors(num_neighbors);
        for (int m = num_neighbors - 1; m >= 0; m--) {
            neighbors[m] = pq.top();
            pq.pop_fast();
        }
        graph[node] = neighbors;
    }

    /* Update the graph with ingoing links */
    for (auto val : new_neighbors) {
        uint neighbor_node = val.second;
        float neighbor_dist = val.first;
        std::lock_guard<std::mutex> lock(node_mutexes[neighbor_node]); 
        
        /* find place to insert into neighbors' neighbors */
        std::vector<std::pair<float,uint>>& neighbors_neighbors = graph[neighbor_node];
        int insert_place = 10000;
        for (int m = num_neighbors; m > 0; m--) {
            if (node == neighbors_neighbors[m-1].second) {
                insert_place = 10000;
                break;
            }
            if (neighbor_dist < neighbors_neighbors[m-1].first) {
                insert_place = m - 1;
            }
        }

        /* insert into neighbors' neighbors */
        if (insert_place < num_neighbors) {     
            neighbors_neighbors.pop_back(); // remove last element first to avoid resizing vec; 
            neighbors_neighbors.insert(neighbors_neighbors.begin() + insert_place, {neighbor_dist, node});
        }
    }
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
            if (std::find(gt_neighbors.begin()+1, gt_neighbors.end()+k+1, est_node) != gt_neighbors.end()+k+1) {
                total_num_correct++;
            }
        }
    }
    double accuracy = (double)total_num_correct / (dataset_size * (size_t)k);

    return accuracy;
}

void queue_to_reverse_vector(std::priority_queue<std::pair<float, uint>>& pq, std::vector<std::pair<float, uint>>& vec) {
    vec.clear();
    while (!pq.empty()) {
        vec.push_back(pq.top());
        pq.pop();
    }
    std::reverse(vec.begin(), vec.end());
}