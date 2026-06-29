#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // This magically converts Python lists to std::vector!
#include <pybind11/numpy.h> // This magically converts Python lists to std::vector!
#include <pybind11/operators.h>
#include <pybind11/functional.h> 

// Include your actual engine headers
#include <LKTorch.hpp>
#include <vector>

namespace py = pybind11;
typedef std::function<std::vector<float>(std::vector<float>, int)> VectorFunction;


#define VERY_SCARY "The functionality of this function is extremely complex, thus I'd rather not discuss it"

// "tensor_engine" is the name you will type when you `import` it in Python
PYBIND11_MODULE(lktorch, m) {
    m.doc() = "My C++ Deep Learning Engine";


    // ========================================================================
    // 1. TENSOR  CLASS
    // ========================================================================
    py::class_<Tensor>(m, "Tensor")
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
        .def("numpy", [](Tensor& self) {
                std::vector<int> dim = self.dimension();
                std::vector<ssize_t> shape(dim.begin(), dim.end());
                py::array_t<float> result(shape);

                py::buffer_info buf = result.request();
                float* ptr = static_cast<float*>(buf.ptr);

                std::vector<float> tensor_data = self.get_data();

                std::memcpy(ptr, tensor_data.data(), tensor_data.size() * sizeof(float));

                return result;
            }, "Convert the Tensor back into a standard NumPy array.")
        
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

        .def("MapValue", &Tensor::MapValue, py::arg("mapping"), py::arg("stride"), py::arg("new_dimension"),
            "Map the value in the tensor however you like")
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


    m.def("MapValue", &MapValue, py::arg("tensor"), py::arg("mapping"), py::arg("stride"), py::arg("new_dimension"),
        "Map the value in the tensor however you like");
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
    m.def("MaxPool", &MaxPool, py::arg("x"), "Get the max of the outermost layer of the tensor");
    m.def("MinPool", &MinPool, py::arg("x"), "Get the min of the outermost layer of the tensor");
    m.def("MaxIndex", &MaxIndex, py::arg("x"), "Get the max index of the outermost layer of the tensor (backprop)");
    m.def("MinIndex", &MinIndex, py::arg("x"), "Get the min index of the outermost layer of the tensor (backprop)");
    m.def("LinearNormalize", &LinearNormalize, py::arg("x"), "Linear normalize the outermost layer of the tensor");
    m.def("ExpNormalize", &ExpNormalize, py::arg("x"), "Exponential normalize the outermost layer of the tensor");
    m.def("Dropout", &Dropout, py::arg("x"), py::arg("rate"), "Drop out rate of the tensor");
    m.def("Unfold", &Unfold, py::arg("x"), py::arg("stride_d"), 
        py::arg("stride") = std::vector<int>(0), py::arg("offset") = std::vector<int>(0), "Unfold the tensor");

    m.def("SumAll", &SumAll, py::arg("x"), "Get the sum of the entire tensor");
    m.def("MeanAll", &MeanAll, py::arg("x"), "Get the mean of the entire tensor");
    m.def("Flatten", &Flatten, py::arg("x"), py::arg("dim") = 0, "Flatten the entire tensor");
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
#define bindTF(NAME) m.attr(#NAME) = py::cast(NAME);
    bindTF(reLU); bindTF(Sigmoid); bindTF(Abs);
    bindTF(Min); bindTF(Max);
    bindTF(Sqr);bindTF(Sqrt);bindTF(Cube);bindTF(Cbrt);
    bindTF(Log);bindTF(Inverse)bindTF(Exp);
    bindTF(Sinh);bindTF(Cosh);bindTF(Tanh);
    bindTF(Sin);bindTF(Cos);bindTF(Tan);bindTF(Asin)bindTF(Acos);bindTF(Atan);
#undef bindTF


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

#define bindSimpleLayer(LAYER_NAME) py::class_<LAYER_NAME, Module, std::shared_ptr<LAYER_NAME>>(m, #LAYER_NAME).def(py::init<>());
    bindSimpleLayer(reLU_Layer);
    bindSimpleLayer(Sigmoid_Layer);
    bindSimpleLayer(Abs_Layer);

    bindSimpleLayer(Sqr_Layer);
    bindSimpleLayer(Sqrt_Layer);
    bindSimpleLayer(Cube_Layer);
    bindSimpleLayer(Cbrt_Layer);

    bindSimpleLayer(Log_Layer);
    bindSimpleLayer(Inverse_Layer);
    bindSimpleLayer(Exp_Layer);

    bindSimpleLayer(Sinh_Layer);
    bindSimpleLayer(Cosh_Layer);
    bindSimpleLayer(Tanh_Layer);

    bindSimpleLayer(Min_Layer);
    bindSimpleLayer(Max_Layer);
    bindSimpleLayer(MinPool_Layer);
    bindSimpleLayer(MaxPool_Layer);

    bindSimpleLayer(Sum_Layer);
    bindSimpleLayer(Mean_Layer);
    bindSimpleLayer(LinearNormalize_Layer);
    bindSimpleLayer(ExpNormalize_Layer);
    bindSimpleLayer(Transpose_Layer);
#undef bindSimpleLayer

#define bindSinglyLayer(LAYER_NAME, VAR_TYPE, VAR) \
py::class_<LAYER_NAME, Module, std::shared_ptr<LAYER_NAME>>(m, #LAYER_NAME) \
    .def(py::init<VAR_TYPE>(), VAR); 

    bindSinglyLayer(PermuteDimension_Layer, std::vector<int>, py::arg("dim"));
    bindSinglyLayer(Flatten_Layer, int, py::arg("dim") = 0);
    bindSinglyLayer(Huber_Layer, float, py::arg("epsilon"))
    bindSinglyLayer(Dropout_Layer, float, py::arg("rate"));
    bindSinglyLayer(Reshape_Layer, std::vector<int>, py::arg("new_shape"));
    bindSinglyLayer(ScalarAdd_Layer, float, py::arg("scalar"));
    bindSinglyLayer(ScalarSubtract_Layer, float, py::arg("scalar"));
    bindSinglyLayer(ScalarMultiply_Layer, float, py::arg("scalar"));
    bindSinglyLayer(ScalarDivide_Layer, float, py::arg("scalar"));

    py::class_<Slice_Layer, Module, std::shared_ptr<Slice_Layer>>(m, "Slice_Layer")
        .def(py::init<std::vector<int>, std::vector<int>>(), py::arg("start_indices"), py::arg("end_indices"));
    py::class_<Unfold_Layer, Module, std::shared_ptr<Unfold_Layer>>(m, "Unfold_Layer")
        .def(py::init<std::vector<int>, std::vector<int>, std::vector<int>>(), py::arg("stride_d")
            , py::arg("stride") = std::vector<int>(0), py::arg("offset") = std::vector<int>(0));

#undef bindSinglyLayer



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
#define bindLossFunction(NAME, DESCRIPTION) m.def(#NAME, &NAME, py::arg("prediction"), py::arg("target"), DESCRIPTION);
    bindLossFunction(MSELoss, "Mean Squared Error.");
    bindLossFunction(MAELoss, "Mean Absolute Error.");
    bindLossFunction(RMSELoss, "Root Mean Squared Error.");
    bindLossFunction(BCELoss, "Binary Cross Entropy Loss.");
    bindLossFunction(HingeLoss, "Hinge Loss for SVMs.");
    bindLossFunction(KL_Divergence, "Kullback-Leibler Divergence.");
    bindLossFunction(CrossEntropyLoss, "Cross Entropy Loss for multi-class classification.");
#undef bindLossFunction
    m.def("HuberLoss", &HuberLoss, py::arg("prediction"), py::arg("target"), py::arg("epsilon"), "Huber Loss with epsilon smoothing.");


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
    m.def("activate_backprop", &activate_backprop, VERY_SCARY);
    m.def("deactivate_backprop", &deactivate_backprop, VERY_SCARY);
    m.def("toggle_backprop", &toggle_backprop, VERY_SCARY);
    m.def("set_backprop_status", &set_backprop_status, py::arg("flag"), VERY_SCARY);
    m.def("backprop_activated", &backprop_activated, VERY_SCARY);
}