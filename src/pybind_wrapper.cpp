#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

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
class SubmissionGraph {
   public:
    uint dataset_size_;
    uint element_count_{0};
    uint dimension_;
    distances::SpaceInterface<float>* space_{nullptr};
    std::string space_name;
    uint num_threads_ = 1;
    Graph* alg_{nullptr};
    uint num_cores_ = 1;

    SubmissionGraph(uint dataset_size, uint dimension, const std::string& space_name = "ip") {
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
    ~SubmissionGraph() {
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
        printf("Adding %zu items to the index\n", num_rows);

        // add the elements to the index
        {
            py::gil_scoped_release l;
            uint start_id = element_count_;

            #pragma omp parallel for num_threads(num_cores_)
            for (uint id = 0; id < num_rows; id++) {
                uint element_id = start_id + id;
                float* row_ptr = static_cast<float*>(buffer.ptr) + id * num_cols;
                alg_->add_point(row_ptr, element_id);
            }
            element_count_ += num_rows;
        }
    }

    /* add items to the index, useful for not loading entire dataset in memory */
    void create_knn() {
        printf("Begin refinement-based knn graph construction\n");
        uint num_iterations = 1;
        uint k = 15;
        uint num_neighbors = 48;
        uint num_hops = 15;
        uint omap_size = 1000;
        uint omap_neighbors = 32;
        {
            py::gil_scoped_release l;
            for (uint i = 0; i < num_iterations; i++) {
                printf(" * iteration %d/%d\n", i+1, num_iterations);
                int random_seed = i * 71; // use a different seed for each iteration
                alg_->set_omap_params(omap_size, omap_neighbors);
                alg_->iterate_knn_refinement(num_neighbors, num_hops, random_seed);
            }
        }

        printf(" * done with knn graph construction\n");
    }

    py::object return_knn(uint k) {

        /* retrieve the knn graph from memory (duplication) */
        uint32_t* neighbors_ptr = new uint32_t[dataset_size_*k];
        float* distances_ptr = new float[dataset_size_*k];
        {
            py::gil_scoped_release l;
            #pragma omp parallel for 
            for (uint32_t index = 0; index < dataset_size_; index++) {
                const auto& edges = (*alg_->graph_with_distances_)[index];
                for (uint32_t m = 0; m < k; m++) {
                    if (m >= edges.size()) {
                        neighbors_ptr[index*k+m] = 0;
                        distances_ptr[index*k+m] = 0;
                    } else {
                        neighbors_ptr[index*k+m] = edges[m].second;
                        distances_ptr[index*k+m] = edges[m].first;
                    }
                }
            }
        }
        printf(" * done with knn graph retrieval\n");

        py::capsule free_when_done_neighbors(neighbors_ptr, [](void* ptr) {
            delete[] ptr;
        });
        py::capsule free_when_done_distances(distances_ptr, [](void* ptr) {
            delete[] ptr;
        });

        printf(" why the seg fault?\n");
        
        /* save as np object  */
        return py::make_tuple(
            py::array_t<uint32_t>(
                { dataset_size_, k },  // shape
                { k * sizeof(uint32_t), sizeof(uint32_t) },  // C-style contiguous strides for each index
                neighbors_ptr,
                free_when_done_neighbors),
            py::array_t<float>(
                { dataset_size_, k },  // shape
                { k * sizeof(float), sizeof(float) },  // C-style contiguous strides for each index
                distances_ptr, 
                free_when_done_distances)
        );
    }
};




PYBIND11_MODULE(Submission, m) {
    py::class_<SubmissionGraph>(m, "Graph")
        .def(py::init<const uint, const uint, const std::string&>(), 
            py::arg("dataset_size"), 
            py::arg("dimension"), 
            py::arg("space") = "ip")
        .def("add_items", &SubmissionGraph::addItems, 
            py::arg("data"))
        .def("create_knn", &SubmissionGraph::create_knn)
        .def("return_knn", &SubmissionGraph::return_knn,
            py::arg("k"));
}