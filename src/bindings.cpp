#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // This magically converts Python lists to std::vector!
#include <pybind11/numpy.h> // This magically converts Python lists to std::vector!
#include <pybind11/operators.h>
#include <pybind11/functional.h> 

// Include your actual engine headers
#include <LKTorch.hpp>
#include <vector>

namespace py = pybind11;
typedef std::function<std::vector<float>(std::vector<float>)> VectorFunction;


// "tensor_engine" is the name you will type when you `import` it in Python
PYBIND11_MODULE(lktorch, m) {
    m.doc() = "My C++ Deep Learning Engine";


    // ========================================================================
    // 1. TENSOR  CLASS
    // ========================================================================
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<std::vector<int>>(),
            py::arg("dimension"),
            "Initialize a Tensor with a given shape/dimension configuration.")
        .def(py::init<std::vector<int>, std::vector<float>>(),
            py::arg("dimension"), py::arg("data"),
            "Initialize a Tensor with a given shape and populate it with a flat list of float values.")
        .def(py::init([](py::array_t<float, py::array::c_style | py::array::forcecast> input_array) {
                py::buffer_info buf = input_array.request();
                std::vector<int> shape;
                for (auto dim : buf.shape) 
                    shape.push_back(dim);
                float* ptr = static_cast<float*>(buf.ptr);

                std::vector<float> flat_data(ptr, ptr + buf.size);
                return Tensor(shape, flat_data);
                }), py::arg("data"), "Initialize a Tensor from any nested Python list or NumPy array.")
        
        .def("size", &Tensor::size,
            "Returns the total number of elements flattened inside the tensor.")
        .def("dimension", &Tensor::dimension,
            "Returns a list containing the shape/dimensions of the tensor.")
        .def("empty", &Tensor::empty,
            "Returns True if the underlying tensor storage is empty or uninitialized.")

        .def("backward", &Tensor::backward,
            py::arg("is_root") = true,
            "Triggers the reverse-mode automatic differentiation (backpropagation) DAG from this node.")
        .def("clear", &Tensor::clear,
            "Clears out underlying data or gradients safely.")

        .def("accessA", &Tensor::accessA,
            py::arg("indices"),
            "Access the forward-pass value at a specific coordinate vector (e.g., [0, 2, 5]).")
        .def("accessGA", &Tensor::accessGA,
            py::arg("indices"),
            "Access the gradient value (grad) at a specific coordinate vector.")

        .def("Permute", &Tensor::Permute, py::arg("target_dimension"), py::arg("target_permutation"),
            "Permuting the element in the tensor however you like")
        .def("PermuteDimension", &Tensor::PermuteDimension, py::arg("target_dimension"),
            "Swapping the dimensions around.")
        .def("Reshape", &Tensor::Reshape, py::arg("new_shape"),
            "Reshapes the tensor to a new dimension mapping without altering underlying memory size.")
        .def("Transpose", &Tensor::Transpose,
            "Swap the top most 2 dimensions of the tensor.")
        .def("Slice", &Tensor::Slice, py::arg("start_indices"), py::arg("end_indices"),
            "Slices a sub-region out of the tensor using coordinate boundaries.")
        .def("Merge", &Tensor::Merge, py::arg("other"),
            "Merges/concatenates two tensors together along their matching dimensional parameters.")
        .def("ValueMultiply", &Tensor::ValueMultiply, py::arg("other"),
            "Performs element-wise multiplication matching the structural bounds.")

        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self + float())
        .def(py::self - float())
        .def(py::self * float())
        .def(py::self / float())
        .def(py::self += py::self)
        .def(py::self -= py::self)
        .def(py::self *= py::self)
        .def(py::self += float())
        .def(py::self -= float())
        .def(py::self *= float())
        .def(py::self /= float())
        
        .def("__str__", [](const Tensor& t) {
            std::stringstream ss;
            ss << t; 
            return ss.str();
        })
        .def("__repr__", [](const Tensor& t) {
            std::stringstream ss;
            ss << t; 
            return ss.str();
        });
    // --- GLOBAL SCOPE FUNCTIONS ---

    
    m.def("Permute", &Permute, py::arg("tensor"), py::arg("target_dimension"), py::arg("target_permutation"),
        "Permuting the element in the tensor however you like");
    m.def("PermuteDimension", &PermuteDimension, py::arg("tensor"), py::arg("target_dimension"),
        "Swapping the dimensions around.");
    m.def("Transpose", &Transpose, py::arg("tensor"), "Swap the top most 2 dimensions of the tensor.");
    m.def("Reshape", &Reshape, py::arg("tensor"), py::arg("new_shape"), "Reshapes the tensor to a new dimension mapping without altering underlying memory size.");
    m.def("Slice", &Tensor::Slice, py::arg("tensor"), py::arg("start_indices"), py::arg("end_indices"), "Slices a sub-region out of the tensor using coordinate boundaries.");
    m.def("Merge", &Tensor::Merge, py::arg("tensor1"), py::arg("tensor2"), "Merges/concatenates two tensors together along their matching dimensional parameters.");
    m.def("ValueMultiply", &Tensor::ValueMultiply, py::arg("tensor1"), py::arg("tensor2"), "Performs element-wise multiplication matching the structural bounds.");

    m.def("Sum", &Sum, py::arg("x"), "Get the sum of the outermost layer of the tensor");
    m.def("Mean", &Mean, py::arg("x"), "Get the mean of the outermost layer of the tensor");
    m.def("SoftMax", &SoftMax, py::arg("x"), "Get the SoftMax of the outermost layer of the tensor");

    m.def("SumAll", &SumAll, py::arg("x"), "Get the sum of the entire tensor");
    m.def("MeanAll", &MeanAll, py::arg("x"), "Get the mean of the entire tensor");
    m.def("Flatten", &Flatten, py::arg("x"), "Flatten the entire tensor");
    m.def("Huber", &Huber, py::arg("x"), py::arg("epsilon"), "Get the huber loss of the entire tensor");

    m.def("ScalarMultiply", &ScalarMultiply, py::arg("tensor"), py::arg("y"), "Scalar Multiply");
    m.def("ScalarDivide", &ScalarDivide, py::arg("tensor"), py::arg("y"), "Scalar Divison");
    m.def("ScalarAdd", &ScalarAdd, py::arg("tensor"), py::arg("y"), "Scalar Add");
    m.def("ScalarSubtract", &ScalarSubtract, py::arg("tensor"), py::arg("y"), "Scalar Subtraction");
    

    // ========================================================================
    // 2. TENSOR FUNCTION CLASS
    // ========================================================================
    py::class_<TensorFunction>(m, "TensorFunction", "Base class for differentiable activations.")
        .def(py::init<>())
        .def(py::init<VectorFunction, VectorFunction>(),
            py::arg("f"), py::arg("dF"),
            "Initialize a TensorFunction by passing the forward and derivative vector functions.")
        .def("forward", static_cast<Tensor(TensorFunction::*)(Tensor)>(&TensorFunction::forward),
            py::arg("x"),
            "Run the forward pass of this mathematical function on a Tensor.")
        .def("__call__", static_cast<Tensor(TensorFunction::*)(Tensor)>(&TensorFunction::operator()),
            py::arg("x"),
            "Allows the instance to be called directly like a function: activation(tensor)");

    // ========================================================================
    // 3. TENSOR INIT (Initialization Functions)
    // ========================================================================
    m.def("Zeros", &Zeros, py::arg("dimension"), "Create a tensor filled with 0.0");
    m.def("Ones", &Ones, py::arg("dimension"), "Create a tensor filled with 1.0");
    m.def("UniformValue", &UniformValue, py::arg("dimension"), py::arg("x"),
        "Create a tensor filled with a specific scalar value.");
    m.def("UniformRandom", &UniformRandom, py::arg("dimension"), py::arg("l"), py::arg("r"),
        "Create a tensor filled with random numbers from a uniform distribution [l, r].");
    m.def("NormalRandom", &NormalRandom, py::arg("dimension"), py::arg("mean"), py::arg("deviation"),
        "Create a tensor filled with random numbers from a Gaussian/Normal distribution.");
    m.def("CustomRandom", &CustomRandom, py::arg("dimension"), py::arg("init_function"),
        "Initialize a tensor using a custom Python lambda function.");

    m.def("CreateTensor", [](py::array_t<float, py::array::c_style | py::array::forcecast> input_array) {
        py::buffer_info buf = input_array.request();
        std::vector<int> shape;
        for (auto dim : buf.shape)
            shape.push_back(dim);
        float* ptr = static_cast<float*>(buf.ptr);

        std::vector<float> flat_data(ptr, ptr + buf.size);
        return Tensor(shape, flat_data);
        }, py::arg("data"),
        "Initialize a tensor by giving it a multidimensional array of float.");

    // ========================================================================
    // 4. GLOBAL ACTIVATION FUNCTIONS (The Externs)
    // ========================================================================
    m.attr("reLU") = py::cast(reLU);
    m.attr("Sigmoid") = py::cast(Sigmoid);
    m.attr("Abs") = py::cast(Abs);
    m.attr("Sqr") = py::cast(Sqr);
    m.attr("Sqrt") = py::cast(Sqrt);
    m.attr("Log") = py::cast(Log);
    m.attr("Tanh") = py::cast(Tanh);
    m.attr("Inverse") = py::cast(Inverse);
    m.attr("Min") = py::cast(Min);
    m.attr("Max") = py::cast(Max);




    // ========================================================================
    // 5. MODULE class
    // ========================================================================
    py::class_<Module, std::shared_ptr<Module>>(m, "Module", "Abstract base class for all neural network modules.")
        .def("get_parameters", &Module::GetParameters, "Returns a list of trainable parameter Tensors.")
        .def("get_state_dict", &Module::GetStateDict, "Extracts raw float data arrays representing module weights.")
        .def("load_state_dict", &Module::LoadStateDict, py::arg("state_dict"), "Restores weights from raw float data matrix layouts.")

        .def("forward", &Module::forward, py::arg("x"), "Virtual forward pass logic execution node.")
        .def("__call__", [](Module& self, Tensor x) { return self(x); }, py::arg("x"), "Allows calling the module like a function instance.");


    // ========================================================================
    // 6. LINEAR LAYER class
    // ========================================================================
    py::class_<LinearLayer, Module, std::shared_ptr<LinearLayer>>(m, "LinearLayer")
        .def(py::init<>())
        .def(py::init<int, int>(), py::arg("in_features"), py::arg("out_features"), "Fully Connected linear transformation layer.");

    py::class_<Conv2D, Module, std::shared_ptr<Conv2D>>(m, "Conv2D")
        .def(py::init<>())
        .def(py::init<std::vector<int>, std::vector<int>>(), py::arg("in_features"), py::arg("out_features") = std::vector<int>{ 1 }, "Convolutional Layers.");

    // ========================================================================
    // 7. OTHER LAYER class
    // ========================================================================
    py::class_<Functional_Layer, Module, std::shared_ptr<Functional_Layer>>(m, "Functional_Layer")
        .def(py::init<TensorFunction>(), py::arg("f"), "Wraps a custom differentiable TensorFunction global wrapper.");

    py::class_<reLU_Layer, Module, std::shared_ptr<reLU_Layer>>(m, "reLU_Layer").def(py::init<>());
    py::class_<Sigmoid_Layer, Module, std::shared_ptr<Sigmoid_Layer>>(m, "Sigmoid_Layer").def(py::init<>());
    py::class_<Abs_Layer, Module, std::shared_ptr<Abs_Layer>>(m, "Abs_Layer").def(py::init<>());
    py::class_<Sqr_Layer, Module, std::shared_ptr<Sqr_Layer>>(m, "Sqr_Layer").def(py::init<>());
    py::class_<Sqrt_Layer, Module, std::shared_ptr<Sqrt_Layer>>(m, "Sqrt_Layer").def(py::init<>());
    py::class_<Log_Layer, Module, std::shared_ptr<Log_Layer>>(m, "Log_Layer").def(py::init<>());
    py::class_<Tanh_Layer, Module, std::shared_ptr<Tanh_Layer>>(m, "Tanh_Layer").def(py::init<>());
    py::class_<Inverse_Layer, Module, std::shared_ptr<Inverse_Layer>>(m, "Inverse_Layer").def(py::init<>());
    py::class_<Min_Layer, Module, std::shared_ptr<Min_Layer>>(m, "Min_Layer").def(py::init<>());
    py::class_<Max_Layer, Module, std::shared_ptr<Max_Layer>>(m, "Max_Layer").def(py::init<>());
    py::class_<Transpose_Layer, Module, std::shared_ptr<Transpose_Layer>>(m, "Transpose_Layer").def(py::init<>());
    py::class_<Sum_Layer, Module, std::shared_ptr<Sum_Layer>>(m, "Sum_Layer").def(py::init<>());
    py::class_<Flatten_Layer, Module, std::shared_ptr<Flatten_Layer>>(m, "Flatten_Layer").def(py::init<>());
    py::class_<Mean_Layer, Module, std::shared_ptr<Mean_Layer>>(m, "Mean_Layer").def(py::init<>());
    py::class_<SoftMax_Layer, Module, std::shared_ptr<SoftMax_Layer>>(m, "SoftMax_Layer").def(py::init<>());

    py::class_<Huber_Layer, Module, std::shared_ptr<Huber_Layer>>(m, "Huber_Layer")
        .def(py::init<float>(), py::arg("epsilon"));
    py::class_<Reshape_Layer, Module, std::shared_ptr<Reshape_Layer>>(m, "Reshape_Layer")
        .def(py::init<std::vector<int>>(), py::arg("new_shape"));
    py::class_<Slice_Layer, Module, std::shared_ptr<Slice_Layer>>(m, "Slice_Layer")
        .def(py::init<std::vector<int>, std::vector<int>>(), py::arg("start_indices"), py::arg("end_indices"));
    py::class_<Unfold_Layer, Module, std::shared_ptr<Unfold_Layer>>(m, "Unfold_Layer")
        .def(py::init<int, int>(), py::arg("height"), py::arg("col"));

    py::class_<ScalarMultiply_Layer, Module, std::shared_ptr<ScalarMultiply_Layer>>(m, "ScalarMultiply_Layer")
        .def(py::init<float>(), py::arg("scalar"));
    py::class_<ScalarDivide_Layer, Module, std::shared_ptr<ScalarDivide_Layer>>(m, "ScalarDivide_Layer")
        .def(py::init<float>(), py::arg("scalar"));
    py::class_<ScalarAdd_Layer, Module, std::shared_ptr<ScalarAdd_Layer>>(m, "ScalarAdd_Layer")
        .def(py::init<float>(), py::arg("scalar"));
    py::class_<ScalarSubtract_Layer, Module, std::shared_ptr<ScalarSubtract_Layer>>(m, "ScalarSubtract_Layer")
        .def(py::init<float>(), py::arg("scalar"));


    // ========================================================================
    // 8. SEQUENTIAL class
    // ========================================================================
    py::class_<Sequential, Module, std::shared_ptr<Sequential>>(m, "Sequential", "Chains modules into linear structural pipelines.")
        .def(py::init<>())
        .def(py::init<std::vector<std::shared_ptr<Module>>>(), py::arg("modules"), "Construct sequential chains from array lists.")
        .def("push_back", &Sequential::push_back, py::arg("module"), "Appends a module block wrapper link.")
        .def("__getitem__", [](Sequential& self, int index) {
        return self[index];
            }, py::arg("index"), "Retrieves reference pointers via block placement locations.");

    // ========================================================================
    // 9. LOSS
    // ========================================================================
    m.def("MSELoss", &MSELoss, py::arg("a"), py::arg("b"), "Mean Squared Error.");
    m.def("MAELoss", &MAELoss, py::arg("a"), py::arg("b"), "Mean Absolute Error.");
    m.def("HuberLoss", &HuberLoss, py::arg("a"), py::arg("b"), py::arg("epsilon"), "Huber Loss with epsilon smoothing.");
    m.def("RMSELoss", &RMSELoss, py::arg("a"), py::arg("b"), "Root Mean Squared Error.");
    m.def("BCELoss", &BCELoss, py::arg("target"), py::arg("prediction"), "Binary Cross Entropy Loss.");
    m.def("HingeLoss", &HingeLoss, py::arg("target"), py::arg("prediction"), "Hinge Loss for SVMs.");
    m.def("KL_Divergence", &KL_Divergence, py::arg("prediction"), py::arg("target"), "Kullback-Leibler Divergence.");
    m.def("CrossEntropyLoss", &CrossEntropyLoss, py::arg("target"), py::arg("prediction"), "Cross Entropy Loss for multi-class classification.");

    // ========================================================================
    // 10. Optimizer
    // ========================================================================
    py::class_<GradientDescent, std::shared_ptr<GradientDescent>>(m, "GradientDescent", "Standard Stochastic Gradient Descent.")
        .def(py::init<float>(), py::arg("learning_rate") = 0.0f)
        .def("get_learning_rate", &GradientDescent::get_learning_rate)
        .def("set_learning_rate", &GradientDescent::set_learning_rate, py::arg("lr"))
        .def("step", &GradientDescent::step, "Update all registered parameters using calculated gradients.")
        .def("zero_gradient", &GradientDescent::zero_gradient, "Wipes the gradients from all tracked tensors.")

        .def("add_parameter", static_cast<void (GradientDescent::*)(Tensor)>(&GradientDescent::add_parameter), py::arg("ts"))
        .def("add_parameter", static_cast<void (GradientDescent::*)(std::vector<Tensor>)>(&GradientDescent::add_parameter), py::arg("ts"));


    py::class_<GradientMomentum, GradientDescent, std::shared_ptr<GradientMomentum>>(m, "GradientMomentum", "SGD with Momentum tracking.")
        .def(py::init<float, float>(), py::arg("learning_rate") = 0.0f, py::arg("glide") = 0.0f)
        .def("get_glide", &GradientMomentum::get_glide)
        .def("set_glide", &GradientMomentum::set_glide, py::arg("gl"))
        .def("multiply_gradient", &GradientMomentum::multiply_gradient);


    py::class_<DecayDescent, GradientMomentum, std::shared_ptr<DecayDescent>>(m, "DecayDescent", "Momentum with LR Decay.")
        .def(py::init<float, float, float>(), py::arg("learning_rate") = 0.0f, py::arg("glide") = 0.0f, py::arg("decay") = 0.0f)
        .def("get_decay", &DecayDescent::get_decay)
        .def("set_decay", &DecayDescent::set_decay, py::arg("dc"))
        .def("decay_learning_rate", &DecayDescent::decay_learning_rate);


    py::class_<AdaGrad, GradientDescent, std::shared_ptr<AdaGrad>>(m, "AdaGrad", "Adaptive Gradient Algorithm.")
        .def(py::init<float>(), py::arg("learning_rate") = 0.0f)
        .def("add_parameter", static_cast<void (AdaGrad::*)(Tensor)>(&AdaGrad::add_parameter), py::arg("ts"))
        .def("add_parameter", static_cast<void (AdaGrad::*)(std::vector<Tensor>)>(&AdaGrad::add_parameter), py::arg("ts"));


    py::class_<RMSProp, AdaGrad, std::shared_ptr<RMSProp>>(m, "RMSProp", "Root Mean Square Propagation.")
        .def(py::init<float, float>(), py::arg("learning_rate") = 0.0f, py::arg("decay_rate") = 0.9f);


    py::class_<Adam, AdaGrad, std::shared_ptr<Adam>>(m, "Adam", "Adaptive Moment Estimation.")
        .def(py::init<float, float, float>(), py::arg("learning_rate") = 0.0f, py::arg("glide") = 0.99f, py::arg("decay") = 0.99f)
        .def("add_parameter", static_cast<void (Adam::*)(Tensor)>(&Adam::add_parameter), py::arg("ts"))
        .def("add_parameter", static_cast<void (Adam::*)(std::vector<Tensor>)>(&Adam::add_parameter), py::arg("ts"));


    // ========================================================================
    // 11. File read
    // ========================================================================
    m.def("ReadFile", &IOHandle::read, py::arg("path"), "Read from a file.");
    m.def("WriteFile", &IOHandle::write, py::arg("content"), py::arg("path"), "Write to a file.");

    // ========================================================================
    // 12. Singleton backprop
    // ========================================================================
    m.def("activate_backprop", &activate_backprop, "The functionality of this function is extremely complex, thus I'd rather not discuss it");
    m.def("deactivate_backprop", &deactivate_backprop, "The functionality of this function is extremely complex, thus I'd rather not discuss it");
    m.def("toggle_backprop", &toggle_backprop, "The functionality of this function is extremely complex, thus I'd rather not discuss it");
    m.def("set_backprop_status", &set_backprop_status, py::arg("flag"), "The functionality of this function is extremely complex, thus I'd rather not discuss it");
    m.def("backprop_activated", &backprop_activated, "The functionality of this function is extremely complex, thus I'd rather not discuss it");
}