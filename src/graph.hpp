#pragma once
// #define _RECORD_METRICS
// #ifndef NUM_CORES
//     #define NUM_CORES 1
// #endif

#include <immintrin.h>
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
#include "unique-priority-queue.hpp"
#include "visited-list-pool.h"

#define NUM_BITS 4

// // Packs a vector of integers (each with num_bits bits) into a byte array
// inline void pack_bits(const std::vector<uint8_t>& values, uint8_t* out_bytes, uint num_bits, uint dimension) {
//     memset(out_bytes, 0, dimension*num_bits/8);
//     size_t bit_offset = 0;
//     for (size_t d = 0; d < dimension; ++d) {
//         uint8_t value = values[d] & ((1 << num_bits) - 1); // mask to num_bits
//         size_t byte_idx = bit_offset / 8;
//         size_t bit_idx = bit_offset % 8;

//         // Write the lower bits of value into the current byte
//         out_bytes[byte_idx] |= value << bit_idx;

//         // If value spans two bytes, write the upper bits into the next byte
//         if (bit_idx + num_bits > 8) {
//             out_bytes[byte_idx + 1] |= value >> (8 - bit_idx);
//         }

//         bit_offset += num_bits;
//     }
// }

// // Unpacks a byte array into a vector of integers (each with num_bits bits)
// inline void unpack_bits(std::vector<uint8_t>& out_values, const uint8_t* in_bytes, uint num_bits, uint dimension) {
//     out_values.resize(dimension);
//     size_t bit_offset = 0;
//     for (size_t d = 0; d < dimension; ++d) {
//         size_t byte_idx = bit_offset / 8;
//         size_t bit_idx = bit_offset % 8;

//         // Read the bits for this value
//         uint16_t val = in_bytes[byte_idx] >> bit_idx;
//         if (bit_idx + num_bits > 8) {
//             // Value spans two bytes
//             val |= (uint16_t(in_bytes[byte_idx + 1]) << (8 - bit_idx));
//         }
//         out_values[d] = uint8_t(val & ((1 << num_bits) - 1));
//         bit_offset += num_bits;
//     }
// }

// Packs a vector of integers (each with num_bits bits) into a byte array
inline void pack_bits(const std::vector<uint8_t>& values, uint8_t* out_bytes, uint num_bits) {
    std::fill(out_bytes, out_bytes + ((values.size() * num_bits + 7) / 8), 0);  // zero-out

    size_t bit_offset = 0;
    for (size_t i = 0; i < values.size(); ++i) {
        uint8_t value = values[i] & ((1 << num_bits) - 1);  // mask to num_bits
        size_t byte_idx = bit_offset / 8;
        size_t bit_idx = bit_offset % 8;

        out_bytes[byte_idx] |= value << bit_idx;

        if (bit_idx + num_bits > 8) {
            out_bytes[byte_idx + 1] |= value >> (8 - bit_idx);
        }

        bit_offset += num_bits;
    }
}

// Unpacks a byte array into a vector of integers (each with num_bits bits)
// inline void unpack_bits(std::vector<uint8_t>& out_values, const uint8_t* in_bytes, uint num_bits, uint dimension) {
// inline void unpack_bits(std::vector<uint8_t>& out_values, const uint8_t* in_bytes, uint num_bits, size_t num_values)
// {
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

/* unpacks 4bit only, faster */
inline void unpack_bits(std::vector<uint8_t>& out_values, const uint8_t* in_bytes, uint num_bits, size_t num_values)
{
    out_values.resize(num_values);
    size_t i = 0, j = 0;
    #pragma unroll
    for (; i + 1 < num_values; i += 2, ++j) {
        uint8_t byte = in_bytes[j];
        out_values[i] = byte & 0x0F;
        out_values[i + 1] = (byte >> 4) & 0x0F;
    }
    if (i < num_values) {
        out_values[i] = in_bytes[j] & 0x0F;
    }
}


typedef uint32_t uint;
float MAX_FLOAT = std::numeric_limits<float>::max();
uint MAX_UINT = std::numeric_limits<uint>::max();

class Graph {
   public:
    bool verbose_ = false;

    // data
    // bool data_owner_ = false;
    // char* data_pointer_{nullptr};  // pointer to the data in the graph
    uint dataset_size_{0};    // max size of dataset
    uint dimension_{0};       // dimensionality of the dataset
    uint max_neighbors_{64};  // number of bottom layer neighbors

    // data: storing as --> x: representation(x), neighbors(x)
    char* data_{nullptr};         // Storing data and graph together
    size_t sizeElementLinks_{0};  // memory for the graph per node: max_neighbors_*uint
    size_t sizeElementData_{0};   // memory for each data vector: dimension * sizeof(DATA_T) [UNLESS QUANTIZED]
    size_t sizePerElement_{0};    // total memory for each node: sizeElementLinks_ + sizeElementData_
    std::vector<uint> node_labels_{};
    std::unordered_map<uint, uint> node_labels_map_{};
    std::atomic<size_t> current_element_{0};
    std::vector<uint> start_nodes_{};

    // top layer graph
    std::vector<uint> top_layer_nodes_{};
    std::unique_ptr<Graph> top_layer_graph_{nullptr};  // graph with neighbors only

    // parallelism
    uint num_cores_ = 1;
    std::vector<std::mutex> node_neighbors_locks_;  // locks for parallel neighbor updates

    // distances
    distances::SpaceInterface<float>* space_;
    DISTFUNC<float> distFunc_;
    void* distFuncParam_{nullptr};
    size_t data_size_{0};

    /* quantization */
    bool quantization_ = false;
    bool is_trained = false;
    const uint num_bits_ = NUM_BITS;    // number of bits per dimension
    uint num_levels_ = 0;  // number of quantization levels
    uint bytes_per_vector_ = 0;
    std::vector<float> quant_lower_bounds_;
    std::vector<float> quant_upper_bounds_;
    std::vector<float> quant_deltas_;
    float quant_lower_global_ = -1.0f;  // lower bound for quantization
    float quant_upper_global_ = 1.0f;   // upper bound for quantization
    float quant_delta_global_;
    std::vector<float> quant_values_;

    // search params
    std::unique_ptr<VisitedListPool> visitedListPool_{nullptr};

    // metrics
    std::atomic<size_t> metrics_hops_{0};
    std::atomic<size_t> metrics_distance_computations_{0};

    //--------------- MARK: CONSTRUCTOR
    Graph(uint dataset_size, uint dimension, uint max_neighbors, distances::SpaceInterface<float>* space)
        : dataset_size_(dataset_size), node_neighbors_locks_(dataset_size) {
        // initializing the space (from hnswlib)
        space_ = space;
        dimension_ = dimension;
        distFunc_ = space_->get_dist_func();
        distFuncParam_ = space_->get_dist_func_param();
        sizeElementData_ = dimension_ * sizeof(float);
        // printf("No Quantization\n");

        // initialize data, storing graph + dataset together (from hnswlib)
        max_neighbors_ = max_neighbors;
        if (max_neighbors_ == 0) printf(" * max_neighbors_ = 0, not storing graph with dataset\n");
        sizeElementLinks_ = sizeof(uint) + max_neighbors_ * sizeof(uint);
        sizePerElement_ = sizeElementData_ + sizeElementLinks_;
        data_ = (char*)malloc(dataset_size_ * sizePerElement_);
        if (data_ == nullptr) throw std::runtime_error("Not enough memory");
        node_labels_.resize(dataset_size_, 0);  // initialize labels

        // initializing the visited list for beam search
        visitedListPool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, dataset_size_));
    }

    /* quantize dataset */
    Graph(uint dataset_size, uint dimension, uint max_neighbors, distances::SpaceInterface<float>* space, uint num_bits)
        : dataset_size_(dataset_size), node_neighbors_locks_(dataset_size) {
        // initializing the space (from hnswlib)
        space_ = space;
        dimension_ = dimension;
        distFunc_ = space_->get_dist_func();
        distFuncParam_ = space_->get_dist_func_param();

        /* quantization */
        quantization_ = true;
        // num_bits_ = num_bits; 4
        num_levels_ = (1u << num_bits_);                       // number of quantization levels
        bytes_per_vector_ = (num_bits_ * dimension_ + 7) / 8;  // number of bytes per vector
        printf("Quantization: %u bits per dimension, %u levels, %u bytes per vector\n", num_bits,
               (unsigned int)num_levels_, bytes_per_vector_);
        sizeElementData_ = bytes_per_vector_;
        is_trained = false;
        quant_lower_bounds_.resize(dimension_, -1.0f);
        quant_upper_bounds_.resize(dimension_, 1.0f);
        quant_deltas_.resize(dimension_, (2) / ((float)num_levels_ - 1));
        quant_lower_global_ = -1.0f;  // lower bound for quantization
        quant_upper_global_ = 1.0f;   // upper bound for quantization
        quant_delta_global_ = (2) / ((float)num_levels_ - 1);

        // initialize data, storing graph + dataset together (from hnswlib)
        max_neighbors_ = max_neighbors;
        if (max_neighbors_ == 0) printf(" * max_neighbors_ = 0, not storing graph with dataset\n");
        sizeElementLinks_ = sizeof(uint) + max_neighbors_ * sizeof(uint);
        sizePerElement_ = sizeElementData_ + sizeElementLinks_;
        data_ = (char*)malloc(dataset_size_ * sizePerElement_);
        if (data_ == nullptr) throw std::runtime_error("Not enough memory");
        node_labels_.resize(dataset_size_, 0);  // initialize labels
        // initializing the visited list for beam search
        visitedListPool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, dataset_size_));
    }
    ~Graph() {
        if (data_ != nullptr) free(data_);
        data_ = nullptr;
    };
    void set_num_cores(uint num_cores) { num_cores_ = num_cores; }

    /* load each data point separately, for management in data_ */
    void add_point(const float* element_ptr, uint element_label) {
        uint element_id = current_element_.fetch_add(1);
        if (element_id >= dataset_size_) throw std::runtime_error("Element ID exceeded dataset_size_");
        node_labels_[element_id] = element_label;  // set the label for the element
        memset(data_ + element_id * sizePerElement_, 0, sizePerElement_);
        if (quantization_) {
            std::vector<uint8_t> compressed_vec(dimension_, 0);
            quantize_vector(element_ptr, compressed_vec.data());
            memcpy(data_ + element_id * sizePerElement_ + sizeElementLinks_, compressed_vec.data(), sizeElementData_);
        } else {
            memcpy(data_ + element_id * sizePerElement_ + sizeElementLinks_, element_ptr, sizeElementData_);
        }
    }

    void init_node_label_lookup() {
        node_labels_map_.clear();
        node_labels_map_.reserve(dataset_size_);
        for (uint i = 0; i < dataset_size_; i++) {
            uint label = node_labels_[i];
            node_labels_map_[label] = i;
        }
    }

    //--------------- MARK: DISTANCES
    inline const char* getDataByInternalId(uint index) const {
        return (data_ + index * sizePerElement_ + sizeElementLinks_);  // data is offset by links
    }
    float compute_distance(const float* index1_ptr, const float* index2_ptr) const {
        return distFunc_(index1_ptr, index2_ptr, distFuncParam_);
    }
    float compute_distance(const float* index1_ptr, uint index2) const {
        std::vector<float> tmp;
        return distFunc_(index1_ptr, getRepresentation(index2, tmp), distFuncParam_);
    }
    float compute_distance(uint index1, const float* index2_ptr) const {
        std::vector<float> tmp;
        return distFunc_(getRepresentation(index1, tmp), index2_ptr, distFuncParam_);
    }
    float compute_distance(uint index1, uint index2) const {
        std::vector<float> tmp1;
        std::vector<float> tmp2;
        return distFunc_(getRepresentation(index1, tmp1), getRepresentation(index2, tmp2), distFuncParam_);
    }

    //--------------- MARK: QUANTIZATION
    void train_quantizer(float* data, uint num_elements) {
        // if (!quantization_) throw std::runtime_error("Quantization is not enabled");
        if (!is_trained) {
            is_trained = true;
            quant_lower_bounds_.clear();
            quant_lower_bounds_.resize(dimension_, MAX_FLOAT);
            quant_upper_bounds_.clear();
            quant_upper_bounds_.resize(dimension_, -MAX_FLOAT);
            quant_lower_global_ = MAX_FLOAT;
            quant_upper_global_ = -MAX_FLOAT;
        }

        // Calculate lower and upper bounds for each dimension
        for (uint i = 0; i < num_elements; i++) {
            for (uint d = 0; d < dimension_; d++) {
                float value = data[i * dimension_ + d];
                if (value < quant_lower_bounds_[d]) quant_lower_bounds_[d] = value;
                if (value > quant_upper_bounds_[d]) quant_upper_bounds_[d] = value;
                if (value < quant_lower_global_) quant_lower_global_ = value;
                if (value > quant_upper_global_) quant_upper_global_ = value;
            }
        }

        // for (uint d = 0; d < dimension_; d++) {
        //     quant_lower_bounds_[d] /= 1.2;
        //     quant_upper_bounds_[d] /= 1.2;
        // }



        // NEED TO ACCOUNT FOR ZERO
        quant_deltas_.resize(dimension_);
        for (uint d = 0; d < dimension_; d++) {
            if (quant_upper_bounds_[d] - quant_lower_bounds_[d] < 0.01) {
                float ave = (quant_upper_bounds_[d] + quant_lower_bounds_[d]) / 2.0f;
                quant_lower_bounds_[d] = ave - 0.01f;
                quant_upper_bounds_[d] = ave + 0.01f;
            }
            quant_deltas_[d] = (quant_upper_bounds_[d] - quant_lower_bounds_[d]) / ((float)num_levels_ - 1);
        }

        quant_delta_global_ = (quant_upper_global_ - quant_lower_global_) / ((float)num_levels_ - 1);
        quant_values_.resize(num_levels_);
        for (uint i = 0; i < num_levels_; i++) {
            quant_values_[i] = (float)i * quant_delta_global_ + quant_lower_global_;
        }

        return;
    }

    void quantize_vector(const float* element_ptr, uint8_t* out_ptr) const {
        std::vector<uint8_t> quantized_vector(dimension_, 0);
        for (size_t d = 0; d < dimension_; d++) {
            float value = element_ptr[d];
            uint8_t quantized_idx =
                static_cast<uint8_t>(std::floor((value - quant_lower_bounds_[d]) / quant_deltas_[d] + 0.5));
            // uint8_t quantized_idx = static_cast<uint8_t>(std::floor((value - quant_lower_global_) /
            // quant_delta_global_ + 0.5));
            if (quantized_idx < 0) quantized_idx = 0;
            if (quantized_idx >= num_levels_) quantized_idx = num_levels_ - 1;
            quantized_vector[d] = quantized_idx;
        }
        pack_bits(quantized_vector, out_ptr, num_bits_);
        // pack_4bit(quantized_vector, out_ptr);
    }

    inline const float* getRepresentation(uint index, std::vector<float>& representation) const {
        const char* data_ptr = getDataByInternalId(index);
        if (quantization_) {
            representation.clear();
            representation.resize(dimension_, 0);
            std::vector<uint8_t> quantized_vector(dimension_, 0);
            // unpack_bits(quantized_vector, num_bits_, reinterpret_cast<const uint8_t*> (data_ptr), (size_t)
            // dimension_);
            unpack_bits(quantized_vector, reinterpret_cast<const uint8_t*>(data_ptr), num_bits_, dimension_);
            // unpack_4bit(quantized_vector, reinterpret_cast<const uint8_t*> (data_ptr), dimension_);
            for (uint d = 0; d < dimension_; d++) {
                float quantized_value = quantized_vector[d];
                representation[d] = quantized_value * quant_deltas_[d] + quant_lower_bounds_[d];
            }
            return representation.data();
        }
        // if not quantized, return the float representation
        else {  // MAKE THIS FASTER
            // representation.clear();
            // representation.resize(dimension_, 0);
            // memcpy(representation.data(), data_ptr, dimension_ * sizeof(float));
            return reinterpret_cast<const float*>(data_ptr);
        }
    }

    //--------------- MARK: NEIGHBORS
    // get the linked list of index for the top/bottom graph
    uint* get_linkedList(uint index) const { return (uint*)(data_ + index * sizePerElement_); }
    uint get_linkedListCount(uint* ptr) const { return *(ptr); }
    void set_linkedListCount(uint* ptr, uint count) const { *(ptr) = count; }
    // set the neighbors, given a vector
    void set_neighbors(uint index, const std::vector<uint>& neighbors) {
        uint num_neighbors = (uint)neighbors.size();
        if (num_neighbors > max_neighbors_) num_neighbors = max_neighbors_;
        uint* index_data = get_linkedList(index);
        set_linkedListCount(index_data, num_neighbors);

        uint* index_neighbors = (uint*)(index_data + 1);
        for (uint i = 0; i < num_neighbors; i++) {
            index_neighbors[i] = neighbors[i];
        }
    }
    uint* get_neighbors(uint index, uint& num_neighbors) const {
        uint* indexData = get_linkedList(index);
        num_neighbors = get_linkedListCount(indexData);
        return (uint*)(indexData + 1);
    }

    //--------------- MARK: I/O

    void save_graph(std::string filename) {
        if (verbose_) printf("Saving Graph to: %s\n", filename.c_str());
        std::ofstream outputFileStream(filename, std::ios::binary);
        if (!outputFileStream.is_open()) {
            printf("Open File Error\n");
            exit(0);
        }

        // output the datasetSize, graphNeighborsTop, graphNeighborsBottom, sizeElementLinks_
        outputFileStream.write((char*)&dataset_size_, sizeof(uint));
        outputFileStream.write((char*)&max_neighbors_, sizeof(uint));
        outputFileStream.write((char*)&sizeElementLinks_, sizeof(size_t));

        // write the top/bottom linked lists for each element
        for (uint index = 0; index < dataset_size_; index++) {
            outputFileStream.write((char*)data_ + index * sizePerElement_, sizeElementLinks_);
        }

        outputFileStream.close();
        return;
    }

    void load_graph(std::string filename) {
        if (verbose_) printf("Loading graph from: %s\n", filename.c_str());
        std::ifstream inputFileStream(filename, std::ios::binary);
        if (!inputFileStream.is_open()) {
            printf("Open File Error\n");
            exit(0);
        }

        // ensure match with the datasetSize, sizeElementLinks
        {
            uint dataset_size;
            inputFileStream.read((char*)&dataset_size, sizeof(uint));
            if (dataset_size != dataset_size_) throw std::runtime_error("dataset_size mismatch");

            uint max_neighbors;
            inputFileStream.read((char*)&max_neighbors, sizeof(uint));
            if (max_neighbors != max_neighbors_) throw std::runtime_error("max_neighbors mismatch");

            size_t sizeElementLinks;
            inputFileStream.read((char*)&sizeElementLinks, sizeof(size_t));
            if (sizeElementLinks != sizeElementLinks_) throw std::runtime_error("sizeElementLinks mismatch");
        }

        // read the top/bottom linked lists for each element
        for (uint index = 0; index < dataset_size_; index++) {
            inputFileStream.read((char*)data_ + index * sizePerElement_, sizeElementLinks_);
        }

        inputFileStream.close();

        /* initialize start nodes for search on the graph */
        start_nodes_.clear();
        for (uint m = 0; m < max_neighbors_; m++) {
            start_nodes_.push_back(rand() % dataset_size_);
        }
        return;
    }

    // print graph stats
    void print_graph_stats() const {
        printf("Graph Stats:\n");
        printf(" * dataset_size_ = %u\n", dataset_size_);
        printf(" * dimension_ = %u\n", dimension_);
        printf(" * max_neighbors_ = %u\n", max_neighbors_);
        uint max_num_neighbors = 0;
        uint min_num_neighbors = MAX_UINT;
        uint total_num_neighbors = 0;
        for (uint i = 0; i < dataset_size_; i++) {
            uint num_neighbors;
            uint* neighbors = get_neighbors(i, num_neighbors);
            if (num_neighbors > max_num_neighbors) max_num_neighbors = num_neighbors;
            if (num_neighbors < min_num_neighbors) min_num_neighbors = num_neighbors;
            total_num_neighbors += num_neighbors;
        }
        printf(" * num neighbors: min = %u, max = %u, avg = %.2f\n", min_num_neighbors, max_num_neighbors,
               (float)total_num_neighbors / dataset_size_);
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
    std::priority_queue<std::pair<float, uint>> internal_beam_search(const float* query_ptr,
                                                                     const std::vector<uint>& start_nodes,
                                                                     uint beam_size, uint max_hops = 1000) const {
        // get the visited list
        VisitedList* vl = visitedListPool_->getFreeVisitedList();
        vl_type* visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        // initializing the beam with the given start node(s)
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
        uint num_hops = 0;
        while (!candidateSet.empty()) {
            uint candidate_node = candidateSet.top().second;
            float candidate_distance = -candidateSet.top().first;

            // if we have explored all points in our beam, stop
            if ((candidate_distance > lower_bound) && (topCandidates.size() >= beam_size)) break;
            // if (candidate_distance > lower_bound) break;
            candidateSet.pop();

            // setting max iterations
            if (num_hops++ > max_hops) break;

            // iterate through node neighbors
            uint num_neighbors;
            uint* neighbors = get_neighbors(candidate_node, num_neighbors);
            for (int i = 0; i < num_neighbors; i++) {
                uint neighbor_node = neighbors[i];

                // compute distance, skip if already visisted
                if (visited_array[neighbor_node] == visited_array_tag) continue;
                visited_array[neighbor_node] = visited_array_tag;
                float dist = compute_distance(query_ptr, neighbor_node);

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
    uint internal_greedy_search(const float* query_ptr, std::vector<uint> start_nodes, uint max_hops = 1000) {
        uint candidate_node = 0;
        float candidate_distance = MAX_FLOAT;
        for (uint start_node : start_nodes) {
            float dist = compute_distance(query_ptr, candidate_node);
            if (dist < candidate_distance) {
                candidate_node = start_node;
                candidate_distance = dist;
            }
        }

        /* greedy search loop */
        uint num_hops = 0;
        bool flag_continue = true;
        while (flag_continue) {
            flag_continue = false;
            if (num_hops++ >= max_hops) break;

            // iterate through node neighbors
            uint num_neighbors;
            uint* neighbors = get_neighbors(candidate_node, num_neighbors);
            for (int m = 0; m < num_neighbors; m++) {
                uint neighbor_node = neighbors[m];
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
    //                             MARK: GRAPH CONSTRUCTION
    //
    //
    //
    //
    //====================================================================================================
    int random_seed_ = 0;

    /* construct the fixed degree hsp graph by brute force */
    void brute_force_construction() {
        init_node_label_lookup();

        /* initialize start nodes for search on the graph */
        start_nodes_.clear();
        for (uint m = 0; m < max_neighbors_; m++) {
            start_nodes_.push_back(rand() % dataset_size_);
        }

        std::vector<uint> candidates(dataset_size_);
        std::iota(candidates.begin(), candidates.end(), 0);  // fill with 0, 1, ..., dataset_size_ - 1

#pragma omp parallel for
        for (uint node = 0; node < dataset_size_; node++) {
            std::vector<float> tmp;
            const float* node_ptr = getRepresentation(node, tmp);

            std::vector<std::pair<float, uint>> candidates(dataset_size_);
            for (uint x = 0; x < dataset_size_; x++) {
                float dist = compute_distance(node_ptr, x);
                candidates[x] = std::make_pair(dist, x);
            }

            std::vector<uint> neighbors;
            fixed_hsp_test(node, candidates, neighbors, max_neighbors_);
            set_neighbors(node, neighbors);
        }
        return;
    }

    /* initialize a random graph */
    void init_random_graph() {
        init_node_label_lookup();
        if (verbose_) printf("Begin random graph initialization...\n");
        random_seed_ = 0;
        srand(random_seed_);

        /* assign random neighbors for all nodes */
        if (verbose_) printf(" * random neighbor assignment\n");
        for (uint x = 0; x < dataset_size_; x++) {
            std::vector<uint> neighbors;
            for (uint m = 0; m < max_neighbors_; m++) {
                uint node = rand() % dataset_size_;
                neighbors.push_back(node);
            }
            set_neighbors(x, neighbors);
        }
    }

    /* initialize a random graph */
    void init_empty_but_top_layer(uint num_nodes) {
        init_node_label_lookup();
        if (verbose_) printf("Begin empty but top layer graph initialization...\n");
        random_seed_ = 0;
        srand(random_seed_);

        printf("init top layer\n");
        init_top_layer_graph(num_nodes, 32, 1);  // example parameters for top layer graph
        for (uint node : top_layer_nodes_) {
            std::vector<uint> neighbors;
            top_layer_graph_->return_neighbors_labels(node, neighbors);
            set_neighbors(node, neighbors);
        }

        // check that all these neighbors are within the top_layer_nodes_
        for (uint node : top_layer_nodes_) {
            uint num_neighbors;
            uint* neighbors = get_neighbors(node, num_neighbors);
            for (uint i = 0; i < num_neighbors; i++) {
                if (std::find(top_layer_nodes_.begin(), top_layer_nodes_.end(), neighbors[i]) == top_layer_nodes_.end()) {
                    printf(" * ERROR: neighbor %u of node %u is not in top layer nodes\n", neighbors[i], node);
                }
            }
        }
    }

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
            std::vector<float> tmp;
            const float* node_ptr = getRepresentation(node, tmp);
            top_layer_graph_->add_point(node_ptr, node);
        }

        /* brute force construction of top layer graph */
        if (num_nodes < 2000 || num_iterations == 0) {
            top_layer_graph_->brute_force_construction();
        }
        /* refinement based construction of top layer graph */
        else {
            /* initialize the random graph*/
            top_layer_graph_->init_random_graph();

            /* refinement based graph construction */
            for (uint i = 0; i < num_iterations; i++) {
                top_layer_graph_->graph_refinement_iteration(num_neighbors, num_neighbors);
            }
        }

        return;
    }

    void checkStuff() {
        std::unordered_set<uint> label_check(node_labels_.begin(), node_labels_.end());
        if (label_check.size() != dataset_size_) {
            printf(" * ERROR: node_labels_ size mismatch with dataset_size_ (%u vs %zu)\n", (unsigned int)dataset_size_,
                   label_check.size());
        } else {
            printf("node labels good?\n");
        }
        for (uint x = 0; x < dataset_size_; x++) {
            uint node_id = x;
            uint node_label = node_labels_[node_id];
            uint node_id_mapped = node_labels_map_[node_label];
            if (node_id != node_id_mapped) {
                printf(" * ERROR: node_id %u does not match mapped id %u for label %u\n", node_id, node_id_mapped,
                       node_label);
            }
        }
    }

    /* refinement-based navigation graph construction */
    void graph_refinement_iteration(uint num_candidates, uint num_hops = 1000) {
        if (verbose_) printf(" * begin new iteration of graph refinement...\n");

        /* initialize start nodes for search on the graph */
        start_nodes_.clear();
        if (top_layer_graph_ == nullptr) {
            for (uint m = 0; m < max_neighbors_; m++) {
                start_nodes_.push_back(rand() % dataset_size_);
            }
        } else {
            for (uint m = 0; m < max_neighbors_; m++) {
                uint rand_start = top_layer_nodes_[rand() % top_layer_nodes_.size()];
                start_nodes_.push_back(rand_start);
            }
        }

        /* create a random ordering for the nodes, to the dataset */
        std::vector<uint> random_ordering(dataset_size_);
        std::iota(random_ordering.begin(), random_ordering.end(), 0);
        std::random_shuffle(random_ordering.begin(), random_ordering.end());

        double time_search = 0;
        double time_update = 0;

        /* construction in batches */
        if (verbose_) printf(" * refining the graph in batches...\n");
        uint batch_size = 10000;
        std::vector<std::priority_queue<std::pair<float, uint>>> batch_neighbors(batch_size);
        uint batch_begin = 0, last_printed = 0;
        while (batch_begin < dataset_size_) {
            uint batch_end = batch_begin + batch_size;
            if (batch_end > dataset_size_) batch_end = dataset_size_;

            /* perform the searches in parallel */
            auto tStart = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
            for (uint qid = 0; qid < batch_end - batch_begin; qid++) {
                uint query_node = random_ordering[batch_begin + qid];
                std::vector<float> tmp;
                const float* query_ptr = getRepresentation(query_node, tmp);

                /* perform the beam search to collect candidates */
                if (top_layer_graph_ == nullptr) {
                    batch_neighbors[qid] = internal_beam_search(query_ptr, start_nodes_, num_candidates, num_hops);
                } else {
                    // if top layer graph exists, use it to find the start node
                    // uint start_node = top_layer_graph_->search_start_node(query_ptr);
                    // batch_neighbors[qid] = internal_beam_search(query_ptr, {start_node}, num_candidates, num_hops);

                    auto res1 = top_layer_graph_->search((float*)(query_ptr), num_candidates, num_candidates, num_hops);
                    batch_neighbors[qid] = internal_beam_search(query_ptr, {res1[0].second}, num_candidates, num_hops);
                    for (auto val : res1) {
                        batch_neighbors[qid].emplace(val.first, val.second);
                    }
                }
            }
            auto tEnd = std::chrono::high_resolution_clock::now();
            time_search += std::chrono::duration_cast<std::chrono::duration<double>>(tEnd - tStart).count();
            // if (verbose_) printf("done searching\n");

            /* batch update of the graph */
            tStart = std::chrono::high_resolution_clock::now();
            #pragma omp parallel for
            for (uint qid = 0; qid < batch_end - batch_begin; qid++) {
                uint query_node = random_ordering[batch_begin + qid];
                std::vector<float> tmp;
                const float* query_ptr = getRepresentation(query_node, tmp);

                // get the new candidate nodes
                std::vector<std::pair<float, uint>> candidate_pairs;
                while (!batch_neighbors[qid].empty()) {
                    candidate_pairs.push_back(batch_neighbors[qid].top());
                    batch_neighbors[qid].pop();
                }

                // update the neighbors of query_node
                std::vector<uint> new_neighbors;
                {
                    // add candidates from search
                    std::vector<std::pair<float, uint>> candidates;
                    std::unordered_set<uint> new_neighbors_set;  // to avoid duplicates
                    for (auto val : candidate_pairs) {
                        uint idx = val.second;
                        if (new_neighbors_set.insert(idx).second) {
                            float dist = val.first;
                            candidates.push_back({dist, idx});
                        }
                    }

                    // add the existing neighbors
                    std::lock_guard<std::mutex> lock(node_neighbors_locks_[query_node]);
                    uint num_neighbors;
                    uint* current_neighbors = get_neighbors(query_node, num_neighbors);
                    for (uint m = 0; m < num_neighbors; m++) {
                        uint neighbor = current_neighbors[m];
                        if (new_neighbors_set.insert(neighbor).second) {
                            float dist = compute_distance(query_ptr, neighbor);
                            candidates.push_back({dist, neighbor});
                        }
                    }

                    // fixed hsp test to update the graph
                    hsp_test(query_node, candidates, new_neighbors, max_neighbors_);
                    // fixed_hsp_test(query_node, candidates, new_neighbors, max_neighbors_);
                    // test_knn(query_node, candidates, new_neighbors, max_neighbors_);
                    set_neighbors(query_node, new_neighbors);
                }            
                // if (verbose_) printf("updated own... \n");


                // update the reverse links of query_node
                for (uint neighbor : new_neighbors) {

                    // check if already there
                    uint num_neighbors;
                    uint* neighbor_neighbors = get_neighbors(neighbor, num_neighbors);
                    bool flag_update = true;
                    for (uint m = 0; m < num_neighbors; m++) {
                        if (neighbor_neighbors[m] == query_node) {
                            flag_update = false;
                            break;
                        }
                    }
                    if (!flag_update) continue;

                    // add automatically if there's room
                    if (num_neighbors < max_neighbors_) {
                        neighbor_neighbors[num_neighbors] = query_node;
                        set_linkedListCount(get_linkedList(neighbor), num_neighbors + 1);
                        continue;  // no need to do the hsp test
                    }

                    // only update if distance is closer than any of the others...
                    float query_dist = compute_distance(query_ptr, neighbor);
                    float max_dist = compute_distance(neighbor, neighbor_neighbors[num_neighbors - 1]);
                    if (query_dist > max_dist) {
                        continue;  
                    }

                    std::lock_guard<std::mutex> lock(node_neighbors_locks_[neighbor]);
                    std::vector<std::pair<float, uint>> candidates = {{query_dist, query_node}};
                    for (uint m = 0; m < num_neighbors; m++) {
                        float distance = compute_distance(query_ptr, neighbor_neighbors[m]);
                        candidates.push_back({distance, neighbor_neighbors[m]});
                    }

                    // fixed hsp test to update the graph
                    std::vector<uint> neighbor_new_neighbors;
                    // hsp_test(neighbor, candidates, neighbor_new_neighbors, max_neighbors_);
                    fixed_hsp_test(neighbor, candidates, neighbor_new_neighbors, max_neighbors_);
                    // test_knn(neighbor, candidates, neighbor_new_neighbors, max_neighbors_);
                    set_neighbors(neighbor, neighbor_new_neighbors);
                }
            }
            tEnd = std::chrono::high_resolution_clock::now();
            time_update += std::chrono::duration_cast<std::chrono::duration<double>>(tEnd - tStart).count();            
            // if (verbose_) printf("updated others... \n");

            /* setup next batch */
            if (batch_end / 100000 > last_printed) {
                if (verbose_) printf(" %u/%u\n", batch_end, dataset_size_);
                last_printed = batch_end / 100000;
            }
            batch_begin = batch_end;
        }

        if (verbose_)
            printf(" * graph refinement done, time search = %.2f s, time update = %.2f s\n", time_search, time_update);

        return;
    }

    void trim_graph_hsp() {
        if (verbose_) printf(" * begin graph trimming...\n");

// iterate through all nodes
#pragma omp parallel for
        for (uint node = 0; node < dataset_size_; node++) {
            std::vector<float> tmp;
            const float* node_ptr = getRepresentation(node, tmp);

            // get the neighbors of the node
            uint num_neighbors;
            uint* neighbors = get_neighbors(node, num_neighbors);

            // compute distances to neighbors
            std::vector<std::pair<float, uint>> candidates;
            for (uint m = 0; m < num_neighbors; m++) {
                float dist = compute_distance(node_ptr, neighbors[m]);
                candidates.push_back(std::make_pair(dist, neighbors[m]));
            }

            // perform hsp test to find the new neighbors
            std::vector<uint> new_neighbors;
            hsp_test(node, candidates, new_neighbors, max_neighbors_);
            set_neighbors(node, new_neighbors);
        }
        return;
    }

    //------------------- MARK: HSP REFINEMENT

    // use maxK to constrain the hsp test to the maxK closest elements in the set
    void hsp_test(uint query, const std::vector<std::pair<float, uint>>& candidates, std::vector<uint>& neighbors,
                  uint m = 0) const {
        neighbors.clear();

        // - initialize the active list A
        std::vector<std::pair<float, uint>> active_list{};
        active_list.reserve(candidates.size());

        // - initialize the list with all points and distances, find nearest neighbor
        uint index1;
        float distance_Q1 = HUGE_VAL;
        for (auto val : candidates) {
            uint index = val.second;
            float distance = val.first;
            if (index == query) continue;
            if (distance < distance_Q1) {
                distance_Q1 = distance;
                index1 = index;
            }
            active_list.push_back(std::make_pair(distance, index));
        }

        // - perform the hsp loop
        while (active_list.size() > 0) {
            // if (m > 0 && neighbors.size() >= m) break;

            // - next neighbor as closest valid point
            neighbors.push_back(index1);
            std::vector<float> tmp1;
            const float* index1_ptr = getRepresentation(index1, tmp1);

            // - set up for the next hsp neighbor
            uint index1_next;
            float distance_Q1_next = HUGE_VAL;

            // - initialize the active_list for next iteration
            // - make new list: push_back O(1) faster than deletion O(N)
            std::vector<std::pair<float, uint>> active_list_copy = active_list;
            active_list.clear();

            // - check each point for elimination
            for (int it2 = 0; it2 < (int)active_list_copy.size(); it2++) {
                uint index2 = active_list_copy[it2].second;
                float distance_Q2 = active_list_copy[it2].first;
                if (index2 == index1) continue;
                float distance_12 = compute_distance(index1_ptr, index2);

                // - check the hsp inequalities: add if not satisfied
                if (distance_Q1 >= distance_Q2 || distance_12 >= distance_Q2) {
                    active_list.emplace_back(distance_Q2, index2);

                    // - update neighbor for next iteration
                    if (distance_Q2 < distance_Q1_next) {
                        distance_Q1_next = distance_Q2;
                        index1_next = index2;
                    }
                }
            }

            // - setup the next hsp neighbor
            index1 = index1_next;
            distance_Q1 = distance_Q1_next;
        }

        return;
    }

    void fixed_hsp_test(uint x, const std::vector<std::pair<float, uint>>& candidates, std::vector<uint>& neighbors,
                        uint m) const {
        // initialize the list
        std::vector<std::tuple<float, uint, int>> active_list;
        for (auto val : candidates) {
            uint index = val.second;
            float distance = val.first;
            if (index == x) continue;  // skip self
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
            std::vector<float> tmp;
            const float* index1_ptr = getRepresentation(index1, tmp);
            neighbors.push_back(index1);
            std::get<2>(active_list[it1]) = 2000000;  // invalidate

            // iterate through all nodes (further away) to tally invalidation
            for (uint it2 = it1 + 1; it2 < (uint)active_list.size(); it2++) {
                uint index2 = std::get<1>(active_list[it2]);
                float distance_Q2 = std::get<0>(active_list[it2]);
                float distance_12 = compute_distance(index1_ptr, index2);
                if (distance_12 < distance_Q2) {
                    std::get<2>(active_list[it2])++;
                }
            }
        }

        return;
    }

    void test_knn(uint x, const std::vector<std::pair<float, uint>>& candidates, std::vector<uint>& neighbors,
                  uint m) const {
        auto active_list = candidates;
        std::sort(active_list.begin(), active_list.end());

        // get knn
        neighbors.clear();
        uint it1 = 0;
        while (neighbors.size() < m && it1 < active_list.size()) {
            uint candidate = active_list[it1].second;
            if (std::find(neighbors.begin(), neighbors.end(), candidate) == neighbors.end()) {
                neighbors.push_back(candidate);
            }
            it1++;
        }

        // if we have less than m neighbors, fill with random
        while (neighbors.size() < m) {
            uint random_node = rand() % dataset_size_;
            neighbors.push_back(random_node);
        }

        return;
    }

    //-------------------------- MARK: SEARCH
    std::vector<std::pair<float, uint>> search(const float* query_ptr, uint k, uint beam_size, uint num_hops = 1000) {
        std::vector<uint> start_nodes = start_nodes_;
        if (top_layer_graph_ != nullptr) {
            start_nodes = {top_layer_graph_->search_start_node(query_ptr)};
        }
        auto res = internal_beam_search(query_ptr, start_nodes, beam_size, num_hops);
        while (res.size() > k) res.pop();

        // convert to vector of size k
        std::vector<std::pair<float, uint>> top_candidates(k);
        for (int i = (int)k - 1; i >= 0; i--) {
            float distance = res.top().first;
            uint element_id = res.top().second;
            res.pop();

            uint element_label = node_labels_[element_id];  // get the label id
            top_candidates[i] = {distance, element_label};
        }
        return top_candidates;
    }

    /* fast greedy search for a start node */
    uint search_start_node(const float* query_ptr) {
        uint node_id = internal_greedy_search(query_ptr, start_nodes_);
        return node_labels_[node_id];
    }

    std::vector<std::pair<float, uint>> search_brute_force(const float* query_ptr, uint k) {

        std::priority_queue<std::pair<float, uint>> pq;
        for (uint i = 0; i < dataset_size_; i++) {
            float distance = compute_distance(query_ptr, i);

            if (pq.size() < k || distance < pq.top().first) {
                pq.emplace(distance, i);
                if (pq.size() > k) pq.pop();
            }
        }

        // convert to vector of size k
        std::vector<std::pair<float, uint>> top_candidates(k);
        for (int i = (int)k - 1; i >= 0; i--) {
            float distance = pq.top().first;
            uint element_id = pq.top().second;
            pq.pop();

            uint element_label = node_labels_[element_id];  // get the label id
            top_candidates[i] = {distance, element_label};
        }
        return top_candidates;
    }

    void return_neighbors_labels(uint node_label, std::vector<uint>& neighbors) {
        uint node_id = node_labels_map_[node_label];
        uint num_neighbors;
        uint* neighbor_ids = get_neighbors(node_id, num_neighbors);
        neighbors.clear();
        for (uint i = 0; i < num_neighbors; i++) {
            uint neighbor_label = node_labels_[neighbor_ids[i]];
            neighbors.push_back(neighbor_label);
        }
    }

};
