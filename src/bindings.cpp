#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // This magically converts Python lists to std::vector!
#include <pybind11/operators.h>

// Include your actual engine headers
#include <LKTorch.hpp>
#include <vector>

namespace py = pybind11;

// "tensor_engine" is the name you will type when you `import` it in Python
PYBIND11_MODULE(lktorch, m) {
    m.doc() = "My custom C++ Deep Learning Engine";

    // 1. Bind the Tensor class
    py::class_<Tensor>(m, "Tensor")
        // Bind the constructor (e.g., one that takes a shape vector)
        .def(py::init<std::vector<int>>())

        // Bind regular methods
        .def("Reshape", &Tensor::Reshape)

        // pybind11 lets you bind C++ operators natively to Python operators!
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self);

    m.def("Reshape", &Reshape, "This shape the tensor from shape A to shape B, given that the two shape looks similar");

    // 2. Bind the LinearLayer (You must tell Pybind about the Module base class too!)
    py::class_<Module, std::shared_ptr<Module>>(m, "Module");

    py::class_<LinearLayer, Module, std::shared_ptr<LinearLayer>>(m, "LinearLayer")
        // Constructor takes in_features, out_features
        .def(py::init<int, int>())
        // Bind the forward method
        .def("forward", &LinearLayer::forward);
}