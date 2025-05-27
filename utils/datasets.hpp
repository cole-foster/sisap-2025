#pragma once

/**
 * @file datasets.hpp
 * @date 2024-10-26
 *
 * Getting datasets downloaded from big-ann
 * for cvpr 2024
 *
 */
#include <sys/stat.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <typeinfo>
#include <vector>
#include <algorithm>

#ifndef CPU_ONLY_
#include <cuda_runtime.h>
#endif

namespace Datasets {

// MARK: DECLARATIONS
inline bool file_exists(std::string filename);
template <typename DATA_T = float>
void read_fbin(const std::string& filename, DATA_T*& data_pointer, size_t& dimension, size_t& dataset_size);
template <typename DATA_T = int8_t>
void read_i8bin(const std::string& filename, DATA_T*& data_pointer, size_t& dimension, size_t& dataset_size);
template <typename DATA_T = uint8_t>
void read_u8bin(const std::string& filename, DATA_T*& data_pointer, size_t& dimension, size_t& dataset_size);
std::vector<std::vector<uint32_t>> read_groundtruth(const std::string& filename, int k);
void read_groundtruth(const std::string&, int&, std::vector<std::vector<uint32_t>>&, std::vector<std::vector<float>>&);
// void glove(float*& data_pointer, size_t& dimension, size_t& dataset_size, float*& test_pointer, size_t& testset_size,
// std::vector<std::vector<uint32_t>>& gt_knn, int k);

/* casting dataset to type DATA_T*/
template <typename DATA_T>
struct Dataset {
   public:
    std::string dataset;
    std::string metric;
    std::string data_type;
    size_t dimension;
    size_t dataset_size;
    size_t testset_size;
    DATA_T* data_pointer = nullptr;
    DATA_T* test_pointer = nullptr;
    int total_k = 1;
    std::vector<std::vector<uint32_t>> neighbors;
    std::vector<std::vector<float>> distances;

    // constructor: initialize dataset
    Dataset(std::string dataset) {
        this->dataset = dataset;
        data_type = typeid(DATA_T).name();
        // if (dataset == "glove") {
        //     GLOVE();
        // }
        // } else if (dataset == "SIFT_10M") {
        //     SIFT_10M();
        // } else if (dataset == "SIFT_100M") {
        //     SIFT_100M();
        // } else if (dataset == "DEEP_10M") {
        //     DEEP_10M();
        // } else if (dataset == "DEEP_100M") {
        //     DEEP_100M();
        // } else
        if (dataset == "SPACEV_1M") {
            SPACEV_1M();
        } else if (dataset == "SPACEV_10M") {
            SPACEV_10M();
        } else if (dataset == "SPACEV_100M") {
            SPACEV_100M();
        }
        // else if (dataset == "TURING_10M") {
        //     TURING_10M();
        // } else if (dataset == "Text2Image_10M") {
        //     Text2Image_10M();
        // }
        else {
            printf("Dataset Unrecognized: %s\n", dataset.c_str());
            exit(0);
        }
    }

    // destructor: deletes pointers
    ~Dataset() {
// if (data_pointer) delete[] data_pointer;
// if (test_pointer) delete[] test_pointer;
#ifndef CPU_ONLY_
        if (data_pointer) cudaFreeHost(data_pointer);
        if (test_pointer) cudaFreeHost(test_pointer);
#else
        if (data_pointer) delete[] data_pointer;
        if (test_pointer) delete[] test_pointer;
#endif
    }

    template <typename T>
    T max(T a, T b) {
        return (a > b) ? a : b;
    }
    template <typename T>
    T min(T a, T b) {
        return (a < b) ? a : b;
    }

    /* compute distances between members of the dataset*/
    float compute_distance_elements(uint32_t index1, uint32_t index2) {
        float distance = HUGE_VALF;
        if (metric == "l2") {
            distance = 0;
            for (uint d = 0; d < dimension; d++) {
                float diff = static_cast<float>(data_pointer[index1 * dimension + d]) -
                             static_cast<float>(data_pointer[index2 * dimension + d]);
                distance += diff * diff;
            }
            distance = std::sqrt(distance);
        } else if (metric == "ip") {
            distance = 0;
            for (uint d = 0; d < dimension; d++) {
                distance += static_cast<float>(data_pointer[index1 * dimension + d]) *
                            static_cast<float>(data_pointer[index2 * dimension + d]);
            }
            distance = -distance;
        } else if (metric == "cos") {
            distance = 0;
            float norm1 = 0, norm2 = 0;
            for (uint d = 0; d < dimension; d++) {
                distance += static_cast<float>(data_pointer[index1 * dimension + d]) *
                            static_cast<float>(data_pointer[index2 * dimension + d]);
                norm1 += static_cast<float>(data_pointer[index1 * dimension + d]) *
                         static_cast<float>(data_pointer[index1 * dimension + d]);
                norm2 += static_cast<float>(data_pointer[index2 * dimension + d]) *
                         static_cast<float>(data_pointer[index2 * dimension + d]);
            }
            distance = distance / (std::sqrt(norm1) * std::sqrt(norm2));
            distance = 1 - distance;
        }
        return distance;
    }

    bool check_duplicate(uint32_t index1, uint32_t index2) {
        float distance = 0;
        for (uint d = 0; d < dimension; d++) {
            float diff = static_cast<float>(data_pointer[index1 * dimension + d]) -
                         static_cast<float>(data_pointer[index2 * dimension + d]);
            distance += diff * diff;
        }
        distance = std::sqrt(distance);
        if (distance < 1e-5) {
            return true;
        }
        return false;
    }

    /* measure recall */
    double measure_recall(const std::vector<std::vector<std::pair<float, uint>>>& est_knn, int k,
                          bool flag_duplicates = true) {
        uint num_queries = est_knn.size();
        if (num_queries > testset_size) return 0;
        if (k > total_k) return 0;

        // measure the recall
        double recall = 0.0f;
        for (uint q = 0; q < num_queries; q++) {
            // check the knn of q
            for (uint i = 0; i < k; i++) {
                if (i >= est_knn[q].size()) break;
                uint est_neighbor = est_knn[q][i].second;

                // see if this is one of the true knn
                if (std::find(neighbors[q].begin(), neighbors[q].begin() + k, est_neighbor) !=
                    neighbors[q].begin() + k) {
                    recall += (1.0f);
                } else if (flag_duplicates) {
                    if (est_neighbor >= dataset_size) continue;

                    /* check if it is a duplicate of any elements */
                    double epsilon = 1e-5;
                    for (uint j = 0; j < k; j++) {
                        uint true_neighbor = neighbors[q][j];
                        if (check_duplicate(est_neighbor, true_neighbor)) {
                            recall += (1.0f);
                            break;
                        }
                    }
                }
            }
        }
        recall /= (double)(num_queries * k);
        return recall;
    }





    // NORMALIZE OF TESTSET SHOULD BE DURING SEARCH
    // void normalize() {
    //     for (uint i = 0; i < dataset_size; i++) {
    //         float norm = 0;
    //         for (uint d = 0; d < dimension; d++) {
    //             norm += (data_pointer[i * dimension + d] * data_pointer[i * dimension + d]);
    //         }
    //         norm = std::sqrt(norm);
    //         for (uint d = 0; d < dimension; d++) {
    //             data_pointer[i * dimension + d] /= norm;
    //         }
    //     }
    //     for (uint i = 0; i < testset_size; i++) {
    //         float norm = 0;
    //         for (uint d = 0; d < dimension; d++) {
    //             norm += (test_pointer[i * dimension + d] * test_pointer[i * dimension + d]);
    //         }
    //         norm = std::sqrt(norm);
    //         for (uint d = 0; d < dimension; d++) {
    //             test_pointer[i * dimension + d] /= norm;
    //         }
    //     }
    // }

   private:
    // MARK: GLOVE
    // void GLOVE() {
    //     metric = "cos";
    //     data_type = "float";
    //     if (typeid(DATA_T) != typeid(float)) {
    //         throw std::runtime_error("Dataset Class: Glove dataset is of type float");
    //     }
    //     glove(data_pointer, dimension, dataset_size, test_pointer, testset_size, neighbors, k);
    //     return;
    // }

    // MARK: SIFT / BIGANN
    // void SIFT_10M() {
    //     metric = "l2";

    //     std::string dataset_path = "/users/cfoste18/scratch/datasets/bigann/base.1B.u8bin.crop_nb_10000000";
    //     read_u8bin(dataset_path, this->data_pointer, this->dimension, this->dataset_size);

    //     std::string query_path = "/users/cfoste18/scratch/datasets/bigann/query.public.10K.u8bin";
    //     read_u8bin(query_path, this->test_pointer, this->dimension, this->testset_size);

    //     std::string gt_path = "/users/cfoste18/scratch/datasets/bigann/bigann-10M";
    //     read_groundtruth(gt_path, this->k, this->neighbors, this->distances);

    //     return;
    // }
    // void SIFT_100M() {
    //     metric = "l2";

    //     std::string dataset_path = "/users/cfoste18/scratch/datasets/bigann/base.1B.u8bin.crop_nb_100000000";
    //     read_u8bin(dataset_path, this->data_pointer, this->dimension, this->dataset_size);

    //     std::string query_path = "/users/cfoste18/scratch/datasets/bigann/query.public.10K.u8bin";
    //     read_u8bin(query_path, this->test_pointer, this->dimension, this->testset_size);

    //     std::string gt_path = "/users/cfoste18/scratch/datasets/bigann/bigann-100M";
    //     read_groundtruth(gt_path, this->k, this->neighbors, this->distances);

    //     return;
    // }

    // MARK: DEEP
    // void DEEP_10M() {
    //     metric = "l2";

    //     std::string dataset_path = "/users/cfoste18/scratch/datasets/deep1b/base.1B.fbin.crop_nb_10000000";
    //     read_fbin(dataset_path, this->data_pointer, this->dimension, this->dataset_size);

    //     std::string query_path = "/users/cfoste18/scratch/datasets/deep1b/query.public.10K.fbin";
    //     read_fbin(query_path, this->test_pointer, this->dimension, this->testset_size);

    //     std::string gt_path = "/users/cfoste18/scratch/datasets/deep1b/deep-10M";
    //     read_groundtruth(gt_path, this->k, this->neighbors, this->distances);

    //     return;
    // }
    // void DEEP_100M() {
    //     metric = "l2";

    //     std::string dataset_path = "/users/cfoste18/scratch/datasets/deep1b/base.1B.fbin.crop_nb_100000000";
    //     read_fbin(dataset_path, this->data_pointer, this->dimension, this->dataset_size);

    //     std::string query_path = "/users/cfoste18/scratch/datasets/deep1b/query.public.10K.fbin";
    //     read_fbin(query_path, this->test_pointer, this->dimension, this->testset_size);

    //     std::string gt_path = "/users/cfoste18/scratch/datasets/deep1b/deep-100M";
    //     read_groundtruth(gt_path, this->k, this->neighbors, this->distances);

    //     return;
    // }

    // MARK: SPACEV
    void SPACEV_1M() {
        metric = "l2";
        if (data_type != typeid(int8_t).name()) {
            printf("Loading data as type %s, original is type int8_t\n", typeid(DATA_T).name());
        }

        std::string dataset_path = "/users/cfoste18/scratch/datasets/MSSPACEV1B/spacev1b_base.i8bin.crop_nb_1000000";
        // std::string dataset_path =
        // "/users/cfoste18/scratch/datasets/MSSPACEV1B/spacev1b_base.i8bin.crop_nb_1000000.deduplicated";
        read_i8bin(dataset_path, this->data_pointer, this->dimension, this->dataset_size);

        std::string query_path = "/users/cfoste18/scratch/datasets/MSSPACEV1B/query.i8bin";
        read_i8bin(query_path, this->test_pointer, this->dimension, this->testset_size);

        std::string gt_path = "/users/cfoste18/scratch/datasets/MSSPACEV1B/msspacev-gt-1M";
        // std::string gt_path = "/users/cfoste18/scratch/datasets/MSSPACEV1B/msspacev-gt-1M.deduplicated";
        read_groundtruth(gt_path, this->total_k, this->neighbors, this->distances);

        return;
    }
    void SPACEV_10M() {
        metric = "l2";
        if (data_type != typeid(int8_t).name()) {
            printf("Loading data as type %s, original is type int8_t\n", typeid(DATA_T).name());
        }

        std::string dataset_path = "/users/cfoste18/scratch/datasets/MSSPACEV1B/spacev1b_base.i8bin.crop_nb_10000000";
        // std::string dataset_path =
        // "/users/cfoste18/scratch/datasets/MSSPACEV1B/spacev1b_base.i8bin.crop_nb_10000000.deduplicated";
        read_i8bin(dataset_path, this->data_pointer, this->dimension, this->dataset_size);

        std::string query_path = "/users/cfoste18/scratch/datasets/MSSPACEV1B/query.i8bin";
        read_i8bin(query_path, this->test_pointer, this->dimension, this->testset_size);

        std::string gt_path = "/users/cfoste18/scratch/datasets/MSSPACEV1B/msspacev-gt-10M";
        // std::string gt_path = "/users/cfoste18/scratch/datasets/MSSPACEV1B/msspacev-gt-10M.deduplicated";
        read_groundtruth(gt_path, this->total_k, this->neighbors, this->distances);

        return;
    }
    void SPACEV_100M() {
        metric = "l2";
        if (data_type != typeid(int8_t).name()) {
            printf("Loading data as type %s, original is type int8_t\n", typeid(DATA_T).name());
        }

        std::string dataset_path = "/users/cfoste18/scratch/datasets/MSSPACEV1B/spacev1b_base.i8bin.crop_nb_100000000";
        // std::string dataset_path =
        // "/users/cfoste18/scratch/datasets/MSSPACEV1B/spacev1b_base.i8bin.crop_nb_100000000.deduplicated";
        read_i8bin(dataset_path, this->data_pointer, this->dimension, this->dataset_size);

        std::string query_path = "/users/cfoste18/scratch/datasets/MSSPACEV1B/query.i8bin";
        read_i8bin(query_path, this->test_pointer, this->dimension, this->testset_size);

        std::string gt_path = "/users/cfoste18/scratch/datasets/MSSPACEV1B/msspacev-gt-100M";
        // std::string gt_path = "/users/cfoste18/scratch/datasets/MSSPACEV1B/msspacev-gt-100M.deduplicated";
        read_groundtruth(gt_path, this->total_k, this->neighbors, this->distances);

        return;
    }

    // MARK: MSTuringANNS
    // void TURING_10M() {
    //     metric = "l2";

    //     std::string dataset_path = "/users/cfoste18/scratch/datasets/MSTuringANNS/base1b.fbin.crop_nb_10000000";
    //     read_fbin(dataset_path, this->data_pointer, this->dimension, this->dataset_size);

    //     std::string query_path = "/users/cfoste18/scratch/datasets/MSTuringANNS/query100K.fbin";
    //     read_fbin(query_path, this->test_pointer, this->dimension, this->testset_size);

    //     std::string gt_path = "/users/cfoste18/scratch/datasets/MSTuringANNS/msturing-gt-10M";
    //     read_groundtruth(gt_path, this->k, this->neighbors, this->distances);

    //     return;
    // }

    // MARK: Text2Image
    // void Text2Image_10M() {
    //     metric = "ip";

    //     std::string dataset_path = "/users/cfoste18/scratch/datasets/text2image1B/base.1B.fbin.crop_nb_10000000";
    //     read_fbin(dataset_path, this->data_pointer, this->dimension, this->dataset_size);

    //     std::string query_path = "/users/cfoste18/scratch/datasets/text2image1B/query.public.100K.fbin";
    //     read_fbin(query_path, this->test_pointer, this->dimension, this->testset_size);

    //     std::string gt_path = "/users/cfoste18/scratch/datasets/text2image1B/text2image-10M";
    //     read_groundtruth(gt_path, this->k, this->neighbors, this->distances);

    //     return;
    // }
};

// sift-10m dataset... metric == euclidean
// void SIFT_10M(float*& data_pointer, size_t& dimension, size_t& dataset_size, float*& test_pointer, size_t&
// testset_size, std::vector<std::vector<uint>>& gt_knn, int k) {
//     std::string dataset_path = "/users/cfoste18/scratch/datasets/bigann/base.1B.u8bin.crop_nb_10000000";
//     read_u8bin(dataset_path, data_pointer, dimension, dataset_size);

//     std::string query_path = "/users/cfoste18/scratch/datasets/bigann/query.public.10K.u8bin";
//     read_u8bin(query_path, test_pointer, dimension, testset_size);

//     std::string gt_path = "/users/cfoste18/scratch/datasets/bigann/bigann-10M";
//     gt_knn = read_groundtruth(gt_path, k);
//     return;
// }

// // deep-10m dataset... metric == euclidean
// void DEEP_10M(float*& data_pointer, size_t& dimension, size_t& dataset_size, float*& test_pointer, size_t&
// testset_size, std::vector<std::vector<uint>>& gt_knn, int k) {
//     std::string dataset_path = "/users/cfoste18/scratch/datasets/deep1b/base.1B.fbin.crop_nb_10000000";
//     read_fbin(dataset_path, data_pointer, dimension, dataset_size);

//     std::string query_path = "/users/cfoste18/scratch/datasets/deep1b/query.public.10K.fbin";
//     read_fbin(query_path, test_pointer, dimension, testset_size);

//     std::string gt_path = "/users/cfoste18/scratch/datasets/deep1b/deep-10M";
//     gt_knn = read_groundtruth(gt_path, k);
//     return;
// }

// dataset was originally stored as an HDF5 file
// loading from hdf5 requires some annoying dependencies
// dataset was converted to bin files with python

// void glove(float*& data_pointer, size_t& dimension, size_t& dataset_size, float*& test_pointer, size_t& testset_size,
// std::vector<std::vector<uint>>& gt_knn, int k) {
//     // std::string h5_path = "/users/cfoste18/scratch/datasets/glove/glove-100-angular.hdf5";
//     std::string dataset_directory = "/users/cfoste18/scratch/datasets/glove/";
//     std::string dataset_path = dataset_directory + "dataset-fp32.bin";
//     std::string testset_path = dataset_directory + "testset-fp32.bin";
//     std::string neighbors_path = dataset_directory + "neighbors-uint32.bin";
//     std::string distances_path = dataset_directory + "distances-fp32.bin";

//     // read the dataset
//     {
//         std::ifstream file(dataset_path, std::ios::binary);
//         if (!file.is_open()) throw std::runtime_error("Could not open the file.");

//         // Read the dimensions of the array
//         uint rows, cols;
//         file.read(reinterpret_cast<char*>(&rows), sizeof(uint));
//         file.read(reinterpret_cast<char*>(&cols), sizeof(uint));
//         dataset_size = (size_t) rows;
//         dimension = (size_t) cols;

//         // prepare the dataset
//         data_pointer = new float[dataset_size * dimension];
//         for (size_t i = 0; i < dataset_size; i++) {
//             file.read(reinterpret_cast<char*>(data_pointer + i*dimension), dimension * sizeof(float));
//         }

//         file.close();
//     }

//     // read the testset
//     {
//         std::ifstream file(testset_path, std::ios::binary);
//         if (!file.is_open()) throw std::runtime_error("Could not open the file.");

//         // Read the dimensions of the array
//         uint rows, cols;
//         file.read(reinterpret_cast<char*>(&rows), sizeof(uint));
//         file.read(reinterpret_cast<char*>(&cols), sizeof(uint));
//         testset_size = (size_t) rows;

//         // prepare the dataset
//         test_pointer = new float[testset_size * dimension];
//         for (size_t i = 0; i < testset_size; i++) {
//             file.read(reinterpret_cast<char*>(test_pointer + i*dimension), dimension * sizeof(float));
//         }

//         file.close();
//     }

//     // read the neighbors
//     {
//         std::ifstream file(neighbors_path, std::ios::binary);
//         if (!file.is_open()) throw std::runtime_error("Could not open the file.");

//         // Read the dimensions of the array
//         uint rows, cols;
//         file.read(reinterpret_cast<char*>(&rows), sizeof(uint));
//         file.read(reinterpret_cast<char*>(&cols), sizeof(uint));
//         size_t read_k = (size_t) cols;

//         // prepare the gt neighbors
//         gt_knn.clear();
//         gt_knn.resize(testset_size);

//         // prepare the dataset
//         for (size_t i = 0; i < testset_size; i++) {
//             gt_knn[i].resize(read_k);
//             file.read(reinterpret_cast<char*>(gt_knn[i].data()), read_k * sizeof(uint));
//             if (k > 0 && k < read_k) gt_knn[i].resize(k);
//         }

//         file.close();
//     }

//     return;
// }

// // deep-10m dataset... metric == euclidean
// void glove(float*& data_pointer, size_t& dimension, size_t& dataset_size, float*& test_pointer, size_t& testset_size,
// std::vector<std::vector<uint>>& gt_knn, int k) {
//     std::string h5_path = "/users/cfoste18/scratch/datasets/glove/glove-100-angular.hdf5";

//     try {
//         // Open the HDF5 file
//         HighFive::File file(h5_path, HighFive::File::ReadOnly);

//         // get the dataset
//         {
//             auto dataset = file.getDataSet("train");
//             std::vector<size_t> dims = dataset.getDimensions();
//             dataset_size = dims[0];
//             dimension = dims[1];

//             std::vector<std::vector<float>> data_vector(dataset_size, std::vector<float>(dimension));
//             dataset.read(data_vector);

//             // convert to data pointer
//             data_pointer = new float[dims[0] * dims[1]];
//             for (size_t i = 0; i < dims[0]; i++) {
//                 memcpy(data_pointer + i*dimension, data_vector[i].data(), dimension*sizeof(float));
//             }
//         }

//         // get the testset
//         {
//             auto dataset = file.getDataSet("test");
//             std::vector<size_t> dims = dataset.getDimensions();
//             testset_size = dims[0];
//             if (dimension != dims[1]) throw std::runtime_error("issue!");

//             std::vector<std::vector<float>> data_vector(testset_size, std::vector<float>(dimension));
//             dataset.read(data_vector);

//             // convert to data pointer
//             test_pointer = new float[dims[0] * dims[1]];
//             for (size_t i = 0; i < dims[0]; i++) {
//                 memcpy(test_pointer + i*dimension, data_vector[i].data(), dimension*sizeof(float));
//             }
//         }

//         // get the neighbors
//         {
//             auto dataset = file.getDataSet("neighbors");
//             std::vector<size_t> dims = dataset.getDimensions();
//             if (testset_size != dims[0]) throw std::runtime_error("issue!");
//             size_t read_k = dims[1];

//             // std::vector<std::vector<uint>> data_vector(testset_size, std::vector<float>(read_k));
//             dataset.read(gt_knn);

//             // convert to data pointer
//             for (uint i = 0; i < testset_size; i++) {
//                 gt_knn[i].resize(k);
//             }
//         }

//     } catch (const HighFive::Exception &err) {
//         std::cerr << "Error: " << err.what() << std::endl;
//         return;
//     }

//     return;
// }

//
//
//
//          HELPER FUNCTIONS
//
//
//

// returns true if file exists
inline bool file_exists(std::string filename) {
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0);
}

template <typename DATA_T>
void read_fbin(const std::string& filename, DATA_T*& data_pointer, size_t& dimension, size_t& dataset_size) {
    std::ifstream input_file(filename, std::ios::binary);
    if (!input_file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    // Read the dataset size (first 4 bytes, i.e., an int)
    int read_num_vectors;
    input_file.read(reinterpret_cast<char*>(&read_num_vectors), sizeof(int));
    dataset_size = (size_t)read_num_vectors;

    // Read the dimension (first 4 bytes, i.e., an int)
    int read_dimension;
    input_file.read(reinterpret_cast<char*>(&read_dimension), sizeof(int));
    dimension = (size_t)read_dimension;

    // Get the file size to determine how many vectors are in the file
    input_file.seekg(0, std::ios::end);
    size_t file_size = input_file.tellg();
    input_file.seekg(2 * sizeof(int), std::ios::beg);  // Go back to just after the dimension
    if (dataset_size != (file_size) / (dimension * sizeof(float))) {
        throw std::runtime_error("Incorrect num vectors / filesize");
    }

// Create a container for the vectors
#ifndef CPU_ONLY_
    cudaMallocHost((void**)&data_pointer, sizeof(DATA_T) * dataset_size * dimension);
#else
    data_pointer = new DATA_T[dataset_size * dimension];
#endif

    // Read all the vectors, converting float to DATA_T
    std::vector<float> temp_vec(dimension);
    for (size_t i = 0; i < dataset_size; ++i) {
        input_file.read(reinterpret_cast<char*>(temp_vec.data()), dimension * sizeof(float));
        for (size_t d = 0; d < dimension; d++) {
            data_pointer[i * dimension + d] = static_cast<DATA_T>(temp_vec[d]);
        }
    }

    input_file.close();
    return;
}

template <typename DATA_T>
void read_i8bin(const std::string& filename, DATA_T*& data_pointer, size_t& dimension, size_t& dataset_size) {
    std::ifstream input_file(filename, std::ios::binary);
    if (!input_file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    // Read the dataset size (first 4 bytes, i.e., an int)
    int read_num_vectors;
    input_file.read(reinterpret_cast<char*>(&read_num_vectors), sizeof(int));
    dataset_size = (size_t)read_num_vectors;

    // Read the dimension (first 4 bytes, i.e., an int)
    int read_dimension;
    input_file.read(reinterpret_cast<char*>(&read_dimension), sizeof(int));
    dimension = (size_t)read_dimension;

    // Get the file size to determine how many vectors are in the file
    input_file.seekg(0, std::ios::end);
    size_t file_size = input_file.tellg();
    input_file.seekg(2 * sizeof(int), std::ios::beg);  // Go back to just after the dimension
    if (dataset_size != (file_size) / (dimension * sizeof(int8_t))) {
        throw std::runtime_error("Incorrect num vectors / filesize");
    }

// Initialize the dataset
#ifndef CPU_ONLY_
    cudaMallocHost((void**)&data_pointer, sizeof(DATA_T) * dataset_size * dimension);
#else
    data_pointer = new DATA_T[dataset_size * dimension];
#endif

    // Read all the vectors, converting int8_t to DATA_T
    std::vector<int8_t> temp_vec(dimension);
    for (size_t i = 0; i < dataset_size; ++i) {
        input_file.read(reinterpret_cast<char*>(temp_vec.data()), dimension * sizeof(int8_t));
        for (size_t d = 0; d < dimension; d++) {
            data_pointer[i * dimension + d] = static_cast<DATA_T>(temp_vec[d]);
        }
    }

    input_file.close();
    return;
}

template <typename DATA_T>
void read_u8bin(const std::string& filename, DATA_T*& data_pointer, size_t& dimension, size_t& dataset_size) {
    std::ifstream input_file(filename, std::ios::binary);
    if (!input_file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    // Read the dataset size (first 4 bytes, i.e., an int)
    int read_num_vectors;
    input_file.read(reinterpret_cast<char*>(&read_num_vectors), sizeof(int));
    dataset_size = (size_t)read_num_vectors;

    // Read the dimension (first 4 bytes, i.e., an int)
    int read_dimension;
    input_file.read(reinterpret_cast<char*>(&read_dimension), sizeof(int));
    dimension = (size_t)read_dimension;

    // Get the file size to determine how many vectors are in the file
    input_file.seekg(0, std::ios::end);
    size_t file_size = input_file.tellg();
    input_file.seekg(2 * sizeof(int), std::ios::beg);  // Go back to just after the dimension
    if (dataset_size != (file_size) / (dimension * sizeof(uint8_t))) {
        throw std::runtime_error("Incorrect num vectors / filesize");
    }

// Create a container for the vectors
#ifndef CPU_ONLY_
    cudaMallocHost((void**)&data_pointer, sizeof(DATA_T) * dataset_size * dimension);
#else
    data_pointer = new DATA_T[dataset_size * dimension];
#endif

    // Read all the vectors, converting uint8_t to DATA_T
    std::vector<uint8_t> temp_vec(dimension);
    for (size_t i = 0; i < dataset_size; ++i) {
        input_file.read(reinterpret_cast<char*>(temp_vec.data()), dimension * sizeof(uint8_t));
        for (size_t d = 0; d < dimension; d++) {
            data_pointer[i * dimension + d] = static_cast<DATA_T>(temp_vec[d]);
        }
    }

    input_file.close();
    return;
}

std::vector<std::vector<uint32_t>> read_groundtruth(const std::string& filename, int k = 0) {
    std::ifstream input_file(filename, std::ios::binary);
    if (!input_file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    // Read the dataset size (first 4 bytes, i.e., an int)
    int read_queries;
    input_file.read(reinterpret_cast<char*>(&read_queries), sizeof(int));

    // Read the dimension (first 4 bytes, i.e., an int)
    int read_k;
    input_file.read(reinterpret_cast<char*>(&read_k), sizeof(int));

    // Get the file size to determine how many vectors are in the file
    input_file.seekg(0, std::ios::end);
    size_t file_size = input_file.tellg();
    input_file.seekg(2 * sizeof(int), std::ios::beg);  // Go back to just after the dimension
    size_t ideal_file_size =
        (size_t)read_queries * (size_t)read_k * (sizeof(float) + sizeof(int));  // both index + distance
    if (file_size - 2 * sizeof(int) != ideal_file_size) {
        std::cout << file_size << " " << ideal_file_size << std::endl;
        throw std::runtime_error("Incorrect num vectors / filesize");
    }

    // Create a container for the vectors
    std::vector<std::vector<uint32_t>> neighbors(read_queries, std::vector<uint32_t>(read_k));
    std::vector<std::vector<float>> distances(read_queries, std::vector<float>(read_k));

    // Read all the vectors
    std::vector<int> temp_vec_int(read_k);
    for (int i = 0; i < read_queries; ++i) {
        input_file.read(reinterpret_cast<char*>(temp_vec_int.data()), read_k * sizeof(int));
        for (size_t d = 0; d < read_k; d++) {
            neighbors[i][d] = (uint32_t)(temp_vec_int[d]);
        }
    }

    // Read all the vectors
    std::vector<float> temp_vec_float(read_k);
    for (int i = 0; i < read_queries; ++i) {
        input_file.read(reinterpret_cast<char*>(temp_vec_float.data()), read_k * sizeof(float));
        for (size_t d = 0; d < read_k; d++) {
            distances[i][d] = (float)(temp_vec_float[d]);
        }
    }

    // trimming to just top k
    if (k > 0) {
        for (int i = 0; i < read_queries; i++) {
            neighbors[i].resize(k);
        }
    }

    input_file.close();
    return neighbors;
}

void read_groundtruth(const std::string& filename, int& read_k, std::vector<std::vector<uint32_t>>& neighbors,
                      std::vector<std::vector<float>>& distances) {
    // printf("Fetching gt file: %s\n", filename.c_str());
    std::ifstream input_file(filename, std::ios::binary);
    if (!input_file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    // Read the dataset size (first 4 bytes, i.e., an int)
    int read_queries;
    input_file.read(reinterpret_cast<char*>(&read_queries), sizeof(int));

    // Read the dimension (first 4 bytes, i.e., an int)
    input_file.read(reinterpret_cast<char*>(&read_k), sizeof(int));

    // Get the file size to determine how many vectors are in the file
    input_file.seekg(0, std::ios::end);
    size_t file_size = input_file.tellg();
    input_file.seekg(2 * sizeof(int), std::ios::beg);  // Go back to just after the dimension
    size_t ideal_file_size =
        (size_t)read_queries * (size_t)read_k * (sizeof(float) + sizeof(int));  // both index + distance
    if (file_size - 2 * sizeof(int) != ideal_file_size) {
        std::cout << file_size << " " << ideal_file_size << std::endl;
        throw std::runtime_error("Incorrect num vectors / filesize");
    }

    // Create a container for the vectors
    neighbors.resize(read_queries, std::vector<uint32_t>(read_k));
    distances.resize(read_queries, std::vector<float>(read_k));

    // Read all the vectors
    std::vector<int> temp_vec_int(read_k);
    for (int i = 0; i < read_queries; ++i) {
        input_file.read(reinterpret_cast<char*>(temp_vec_int.data()), read_k * sizeof(int));
        for (size_t d = 0; d < read_k; d++) {
            neighbors[i][d] = (uint32_t)(temp_vec_int[d]);
        }
    }

    // Read all the vectors
    std::vector<float> temp_vec_float(read_k);
    for (int i = 0; i < read_queries; ++i) {
        input_file.read(reinterpret_cast<char*>(temp_vec_float.data()), read_k * sizeof(float));
        for (size_t d = 0; d < read_k; d++) {
            distances[i][d] = (float)(temp_vec_float[d]);
        }
    }

    input_file.close();
    return;
}

};  // namespace Datasets