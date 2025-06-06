#pragma once

#include <sys/stat.h>
#include <queue>

#include <fstream>
#include <queue>
#include <vector>

namespace Utils {

inline bool fileExists(std::string filename) {
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0);
}

uint vecMax(std::vector<uint> const& vec) {
    uint max_val = 0;
    for (uint val : vec) {
        if (val > max_val) max_val = val;
    }
    return max_val;
}

void queue_to_reverse_vector(std::priority_queue<std::pair<float, uint>>& pq, std::vector<std::pair<float, uint>>& vec) {
    vec.clear();
    while (!pq.empty()) {
        vec.push_back(pq.top());
        pq.pop();
    }
    std::reverse(vec.begin(), vec.end());
}

// Save a graph to a binary file. Saves as format
//      D, N,
//      num_neighbors1, neighbor_1, neighbor_2, .... , neighbor_n
//      num_neighbors2, neighbor_1, neighbor_2, .... , neighbor_n
//      ...
//      num_neighborsN, neighbor_1, neighbor_2, .... , neighbor_n
//
template <typename T>
void saveGraph(std::string filename, std::vector<std::vector<T>> const& graph, unsigned int const dimension,
               unsigned int const dataset_size) {
    printf("Saving Graph to: %s\n", filename.c_str());
    std::ofstream outputFileStream(filename, std::ios::binary);
    if (!outputFileStream.is_open()) {
        printf("Open File Error\n");
        exit(0);
    }
    if ((unsigned int)graph.size() != dataset_size) {
        printf("Graph Size Does Not Match Given Dataset Size\n");
        exit(0);
    }

    // output the dimension and dataset size
    outputFileStream.write((char*)&dimension, sizeof(unsigned int));
    outputFileStream.write((char*)&dataset_size, sizeof(unsigned int));

    // save the neighbors of each element in the dataset
    //      num_neighbors, n1, n2, ..., nn
    for (unsigned int index = 0; index < dataset_size; index++) {
        unsigned int num_neighbors = (unsigned int)graph[index].size();
        outputFileStream.write((char*)&num_neighbors, sizeof(unsigned int));

        // change to use vector.data()
        for (unsigned int j = 0; j < num_neighbors; j++) {
            T neighbor_id = graph[index][j];
            outputFileStream.write((char*)&neighbor_id, sizeof(T));
        }
    }

    // done!
    outputFileStream.close();
    return;
}

// Load a graph from a binary file. All uints. Must be in format
//      D, N,
//      num_neighbors1, neighbor_1, neighbor_2, .... , neighbor_n
//      num_neighbors2, neighbor_1, neighbor_2, .... , neighbor_n
//      ...
//      num_neighborsN, neighbor_1, neighbor_2, .... , neighbor_n
//
template <typename T>
void loadGraph(std::string filename, std::vector<std::vector<T>>& graph, unsigned int const dimension,
               unsigned int const dataset_size) {
    printf("Loading Graph from: %s\n", filename.c_str());
    std::ifstream inputFileStream(filename, std::ios::binary);
    if (!inputFileStream.is_open()) {
        printf("Open File Error\n");
        exit(0);
    }

    // first, read the dimension and dataset sizeof the dataset
    unsigned int graph_dimension;
    unsigned int graph_dataset_size;
    inputFileStream.read((char*)&graph_dimension, sizeof(unsigned int));
    inputFileStream.read((char*)&graph_dataset_size, sizeof(unsigned int));
    if (graph_dimension != dimension) {
        printf("Graph dimension not consistent with dataset dimension!\n");
        exit(0);
    }
    if (graph_dataset_size != dataset_size) {
        printf("Graph dataset size not consistent with dataset dataset size!\n");
        exit(0);
    }

    // initialize the graph for all members of the dataset
    graph.resize(dataset_size);

    // get all neighbors
    for (unsigned int index = 0; index < dataset_size; index++) {
        unsigned int num_neighbors;
        inputFileStream.read((char*)&num_neighbors, sizeof(unsigned int));
        graph[index].resize(num_neighbors);
        for (unsigned int j = 0; j < num_neighbors; j++) {
            inputFileStream.read((char*)&graph[index][j], sizeof(T));
        }
    }

    // done!
    inputFileStream.close();
    return;
}

template <typename TT>
void writeArrayToFile(const std::vector<TT>& array, const std::string& filename) {
    // Open the file in write mode
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        throw std::ios_base::failure("Failed to open the file for writing.");
    }

    // Write the elements to the file
    for (size_t i = 0; i < array.size(); ++i) {
        outfile << array[i];
        if (i < array.size() - 1) {
            outfile << "\n";
        }
    }

    outfile.close();
}

template <typename TT>
std::vector<TT> readArrayFromFile(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile) {
        throw std::ios_base::failure("Failed to open file: " + filename);
    }

    std::vector<TT> array;
    TT value;

    while (infile >> value) {
        array.push_back(value);
    }

    if (infile.bad()) {
        throw std::ios_base::failure("Error reading file: " + filename);
    }

    return array;
}

template <typename TT>
void writeVectorToBinary(const std::string& filename, const std::vector<TT>& vec) {
    std::ofstream outFile(filename, std::ios::binary | std::ios::out);
    if (!outFile) {
        throw std::ios_base::failure("Failed to open file for writing");
    }

    // Write the size of the vector
    size_t size = vec.size();
    outFile.write(reinterpret_cast<const char*>(&size), sizeof(size));

    // Write the vector elements
    if (!vec.empty()) {
        outFile.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(TT));
    }

    outFile.close();
}

template <typename TT>
std::vector<TT> readVectorFromBinary(const std::string& filename) {
    std::ifstream inFile(filename, std::ios::binary | std::ios::in);
    if (!inFile) {
        throw std::ios_base::failure("Failed to open file for reading");
    }

    // Read the size of the vector
    size_t size = 0;
    inFile.read(reinterpret_cast<char*>(&size), sizeof(size));

    // Read the vector elements
    std::vector<TT> vec(size);
    if (size > 0) {
        inFile.read(reinterpret_cast<char*>(vec.data()), size * sizeof(TT));
    }

    inFile.close();
    return vec;
}

};  // namespace Utils