#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

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

// MARK: GRAPH
class Task2 {
   public:
    uint dataset_size_;
    uint element_count_{0};
    uint dimension_;
    distances::SpaceInterface<float>* space_{nullptr};
    std::string space_name;
    uint num_threads_ = 1;
    Graph* alg_{nullptr};
    uint num_cores_ = 1;

    Task2(uint dataset_size, uint dimension, const std::string& space_name = "ip") {
        printf("Initializing Graph Index...\n");
        dataset_size_ = dataset_size;
        dimension_ = dimension;
        if (space_name == "ip") {
            space_ = new distances::InnerProductSpace(dimension);
        } else {
            throw std::runtime_error("Space name must be ip, sorry");
        }
        alg_ = new Graph(dataset_size_, dimension_, space_);
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
            py::gil_scoped_release l;
            uint start_id = element_count_;
#pragma omp parallel for num_threads(num_cores_)
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
                const auto& edges = (*alg_->graph_with_distances_)[index];
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
}