#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#include "knn.hpp"
#include "graph.hpp"

inline void get_input_array_shapes(const py::buffer_info& buffer, size_t* rows, size_t* features) {
    if (buffer.ndim != 2 && buffer.ndim != 1) {
        char msg[256];
        snprintf(msg, sizeof(msg),
                 "Input vector data wrong shape. Number of dimensions %d. Data must be a 1D or 2D array.", buffer.ndim);
        throw std::runtime_error(msg);
    }
    if (buffer.ndim == 2) {
        *rows = buffer.shape[0];
        *features = buffer.shape[1];
    } else {
        *rows = 1;
        *features = buffer.shape[0];
    }
}


// MARK: TASK 1
class Task1 {
   public:
    uint dataset_size_;
    uint element_count_{0};
    uint dimension_;
    uint num_neighbors_;
    distances::SpaceInterface<float>* space_{nullptr};
    std::string space_name;
    uint num_threads_ = 1;
    Graph* alg_{nullptr};
    uint num_cores_ = 1;
    uint num_bits_ = 0;  // number of bits for quantization

    Task1(uint dataset_size, uint dimension, uint num_neighbors, uint num_bits = 0, const std::string& space_name = "ip") {
        printf("Initializing class for task 1\n");
        dataset_size_ = dataset_size;
        dimension_ = dimension;
        num_neighbors_ = num_neighbors;
        num_bits_ = num_bits;
        if (space_name == "ip") {
            space_ = new distances::InnerProductSpace(dimension);
        } else {
            throw std::runtime_error("Space name must be ip, sorry");
        }
        if (num_bits_ > 0) {
            // only 4 bits
            num_bits_ = 4;
            alg_ = new Graph(dataset_size_, dimension_, num_neighbors_, space_, num_bits_);
        } else {
            alg_ = new Graph(dataset_size_, dimension_, num_neighbors_, space_);
        }
        element_count_ = 0;
        // num_cores_ = omp_get_num_procs();
        // alg_->set_num_cores(num_cores_);
        alg_->verbose_ = true;  // enable verbose output
    }
    ~Task1() {
        if (alg_ != nullptr) delete alg_;
        if (space_ != nullptr) delete space_;
    }

    void train_quantizer(py::object input) {
        py::array_t<float, py::array::c_style | py::array::forcecast> items(input);
        auto buffer = items.request();

        // check the dimensions of the input
        size_t num_rows, num_cols;
        get_input_array_shapes(buffer, &num_rows, &num_cols);
        if (num_cols != dimension_) throw std::runtime_error("Wrong dimensionality of the vectors");

        // train the quantizer
        alg_->train_quantizer((float*)items.data(0), num_rows);
    }

    /* add items to the index, useful for not loading entire dataset in memory */
    void addItems(py::object input) {
        py::array_t<float, py::array::c_style | py::array::forcecast> items(input);
        auto buffer = items.request();

        // check the dimensions of the input
        size_t num_rows, num_cols;
        get_input_array_shapes(buffer, &num_rows, &num_cols);
        if (num_cols != dimension_) throw std::runtime_error("Wrong dimensionality of the vectors");
        {
            uint start_id = element_count_;
            for (uint id = 0; id < num_rows; id++) {
                uint element_id = start_id + id;
                alg_->add_point((float*)items.data(id), element_id);
            }
            element_count_ += num_rows;
        }
    }

    /* perform iterative construction */
    void build(uint num_candidates, uint num_hops = 100, uint num_iterations = 1) {
        alg_->init_random_graph();
        for (uint i = 0; i < num_iterations; i++) {
            printf(" * iteration %u/%u\n", i + 1, num_iterations);
            alg_->init_top_layer_graph(4000, 32, 1);
            alg_->graph_refinement_iteration(num_candidates, num_hops);
            alg_->trim_graph_hsp();  // trim the graph to keep only the top 10% of neighbors
            alg_->print_graph_stats();
        }
    }

    /* perform search over a batch of queries*/
    py::object search(py::object input, uint k, uint beam_size, uint num_hops = 1000) {
        py::array_t <float, py::array::c_style | py::array::forcecast > items(input);
        auto buffer = items.request();
        size_t num_rows, num_cols;
        uint* neighbors_ptr;
        float* distances_ptr;

        // scope for parallelism
        {
            py::gil_scoped_release l;
            get_input_array_shapes(buffer, &num_rows, &num_cols);
            neighbors_ptr = new uint[num_rows * k];
            distances_ptr = new float[num_rows * k];

            #pragma omp parallel for
            for (uint q = 0; q < num_rows; q++) {
                auto res = alg_->search((float*)items.data(q), k, beam_size, num_hops);
                for (uint i = 0; i < k; i++) {
                    if (i < res.size()) {
                        neighbors_ptr[q * k + i] = res[i].second;
                        distances_ptr[q * k + i] = res[i].first;
                    } else {
                        neighbors_ptr[q * k + i] = 0;  // fill with dummy values
                        distances_ptr[q * k + i] = 100000.0f;  // fill with dummy values
                    }
                }
            }
        }

        py::capsule free_when_done_neighbors(neighbors_ptr, [](void* ptr) { delete[] ptr; });
        py::capsule free_when_done_distances(distances_ptr, [](void* ptr) { delete[] ptr; });

        /* save as np object  */
        return py::make_tuple(
            py::array_t<uint>({(uint) num_rows, k},                // shape
                              {k * sizeof(uint), sizeof(uint)},  // C-style contiguous strides for each index
                              neighbors_ptr, free_when_done_neighbors),
            py::array_t<float>({(uint) num_rows, k},                  // shape
                               {k * sizeof(float), sizeof(float)},  // C-style contiguous strides for each index
                               distances_ptr, free_when_done_distances));
    }

};











// MARK: TASK 2
class Task2 {
   public:
    uint dataset_size_;
    uint element_count_{0};
    uint dimension_;
    distances::SpaceInterface<float>* space_{nullptr};
    std::string space_name;
    uint num_threads_ = 1;
    KNN* alg_{nullptr};
    uint num_cores_ = 1;

    Task2(uint dataset_size, uint dimension, const std::string& space_name = "ip") {
        printf("Initializing KNN class for index...\n");
        dataset_size_ = dataset_size;
        dimension_ = dimension;
        if (space_name == "ip") {
            space_ = new distances::InnerProductSpace(dimension);
        } else {
            throw std::runtime_error("Space name must be ip, sorry");
        }
        alg_ = new KNN(dataset_size_, dimension_, space_);
        alg_->init_dataset();
        element_count_ = 0;
        num_cores_ = omp_get_num_procs();
        alg_->set_num_cores(num_cores_);
    }
    ~Task2() {
        if (alg_ != nullptr) delete alg_;
        if (space_ != nullptr) delete space_;
    }

    /* add items to the index, useful for not loading entire dataset in memory */
    void addItems(py::object input) {
        py::array_t<float, py::array::c_style | py::array::forcecast> items(input);
        auto buffer = items.request();

        // check the dimensions of the input
        size_t num_rows, num_cols;
        get_input_array_shapes(buffer, &num_rows, &num_cols);
        if (num_cols != dimension_) throw std::runtime_error("Wrong dimensionality of the vectors");
        {
            // py::gil_scoped_release l;
            uint start_id = element_count_;
            for (uint id = 0; id < num_rows; id++) {
                uint element_id = start_id + id;
                alg_->add_point((float*)items.data(id), element_id);
            }
            element_count_ += num_rows;
        }
    }

    /* add items to the index, useful for not loading entire dataset in memory */
    py::object create_knn(uint k, uint num_neighbors, uint num_hops, uint omap_size, uint num_iterations) {
        uint omap_neighbors = 32;
        {
            py::gil_scoped_release l;
            for (uint i = 0; i < num_iterations; i++) {
                printf(" * iteration %d/%d\n", i + 1, num_iterations);
                int random_seed = i * 71;  // use a different seed for each iteration
                alg_->set_omap_params(omap_size, omap_neighbors);
                alg_->iterate_knn_refinement(num_neighbors, num_hops, random_seed);
            }
        }

        return return_knn(k);
    }

    py::object return_knn(uint k) {

        /* retrieve the knn graph from memory (duplication) */
        uint* neighbors_ptr = new uint[dataset_size_ * k];
        float* distances_ptr = new float[dataset_size_ * k];
        {
            py::gil_scoped_release l;
            #pragma omp parallel for
            for (uint index = 0; index < dataset_size_; index++) {
                const auto& edges = (*alg_->graph_)[index];
                for (uint m = 0; m < k; m++) {
                    if (m >= edges.size()) {
                        neighbors_ptr[index * k + m] = 0;
                        distances_ptr[index * k + m] = 0;
                    } else {
                        neighbors_ptr[index * k + m] = edges[m].second;
                        distances_ptr[index * k + m] = edges[m].first;
                    }
                }
            }
        }
        py::capsule free_when_done_neighbors(neighbors_ptr, [](void* ptr) { delete[] ptr; });
        py::capsule free_when_done_distances(distances_ptr, [](void* ptr) { delete[] ptr; });

        /* save as np object  */
        return py::make_tuple(
            py::array_t<uint>({dataset_size_, k},                // shape
                              {k * sizeof(uint), sizeof(uint)},  // C-style contiguous strides for each index
                              neighbors_ptr, free_when_done_neighbors),
            py::array_t<float>({dataset_size_, k},                  // shape
                               {k * sizeof(float), sizeof(float)},  // C-style contiguous strides for each index
                               distances_ptr, free_when_done_distances));
    }
};

PYBIND11_MODULE(Submission, m) {
    py::class_<Task2>(m, "Task2")
        .def(py::init<const uint, const uint, const std::string&>(), py::arg("dataset_size"), py::arg("dimension"), 
             py::arg("space") = "ip")
        .def("add_items", &Task2::addItems, py::arg("data"))
        .def("create_knn", &Task2::create_knn, py::arg("k"), py::arg("num_neighbors"), py::arg("num_hops"),
             py::arg("omap_size"), py::arg("num_iterations"));

    
    py::class_<Task1>(m, "Task1")
        .def(py::init<const uint, const uint, const uint, const uint,  const std::string&>(), py::arg("dataset_size"), py::arg("dimension"), 
            py::arg("num_neighbors"), py::arg("num_bits"), py::arg("space") = "ip")
        .def("train", &Task1::train_quantizer, py::arg("data"))
        .def("add_items", &Task1::addItems, py::arg("data"))
        .def("build", &Task1::build, py::arg("num_candidates"), py::arg("num_hops"), py::arg("num_iterations") = 1)
        .def("search", &Task1::search, py::arg("data"), py::arg("k"), py::arg("beam_size") = 1, py::arg("num_hops") = 1000);
}