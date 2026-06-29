#ifndef LAYER_HPP
#define LAYER_HPP

#include <Tensor/Tensor.hpp>
#include <TensorManip/Module.hpp>
#include <vector>

class LinearLayer : public Module {
public:
	LinearLayer() { n = m = 0; W = Zeros(std::vector<int>{0}); B = Zeros(std::vector<int>{0});}
	LinearLayer(int x, int y) {
		n = x, m = y;
		W = UniformRandom(std::vector<int>{x, y}, -1, 1);
		B = UniformRandom(std::vector<int>{y}, -1, 1);
		RegisterParameter(W);
		RegisterParameter(B);
	}
	Tensor forward (Tensor x) override {return x * W + B;}
private:
	int n, m;
	Tensor W, B;
};

class Conv2D : public Module {
public:
	Conv2D() { len_in = len_out = 0; }
	Conv2D(std::vector<int> input, std::vector<int> output = std::vector<int>{ 1 }) {
		len_in = len_out = 1;
		for (int i : input) len_in *= i;
		for (int i : output) len_out *= i;
		li = LinearLayer(len_in, len_out);
		RegisterParameter(li);
	}
	Tensor forward(Tensor x) override { return li(x.Reshape(std::vector<int>{len_in})).Reshape(output); }
private:
	int len_in, len_out;
	std::vector<int> output;
	LinearLayer li;
};


class Functional_Layer : public Module {
public:
	Functional_Layer(TensorFunction f) : f(f) {}
	Tensor forward(Tensor x) override { return f(x); }
private:
	TensorFunction f;
};



#define constructSimpleLayer(LAYER_NAME, FUNCTION) \
class LAYER_NAME : public Module { \
public: \
	Tensor forward(Tensor x) override { return FUNCTION(x); } \
};

constructSimpleLayer(reLU_Layer, reLU);
constructSimpleLayer(Sigmoid_Layer, Sigmoid);
constructSimpleLayer(Abs_Layer, Abs);

constructSimpleLayer(Sqr_Layer, Sqr);
constructSimpleLayer(Sqrt_Layer, Sqrt);
constructSimpleLayer(Cube_Layer, Cube);
constructSimpleLayer(Cbrt_Layer, Cbrt);

constructSimpleLayer(Log_Layer, Log);
constructSimpleLayer(Inverse_Layer, Inverse);
constructSimpleLayer(Exp_Layer, Exp);

constructSimpleLayer(Sinh_Layer, Sinh);
constructSimpleLayer(Cosh_Layer, Cosh);
constructSimpleLayer(Tanh_Layer, Tanh);

constructSimpleLayer(Min_Layer, Min);
constructSimpleLayer(Max_Layer, Max);
constructSimpleLayer(MinPool_Layer, MinPool);
constructSimpleLayer(MaxPool_Layer, MaxPool);

constructSimpleLayer(Sum_Layer, Sum);
constructSimpleLayer(Mean_Layer, Mean);
constructSimpleLayer(LinearNormalize_Layer, LinearNormalize);
constructSimpleLayer(ExpNormalize_Layer, ExpNormalize);



#define constructSinglyLayer(LAYER_NAME, FUNCTION, VARIABLE_TYPE, VARIABLE_NAME) \
class LAYER_NAME : public Module { \
public: \
	LAYER_NAME(VARIABLE_TYPE VARIABLE_NAME): VARIABLE_NAME(VARIABLE_NAME){}\
	Tensor forward(Tensor x) override { return FUNCTION(x, VARIABLE_NAME); } \
private:\
	VARIABLE_TYPE VARIABLE_NAME;\
};

constructSinglyLayer(Huber_Layer, Huber, float, epsilon);
constructSinglyLayer(PermuteDimension_Layer, PermuteDimension, std::vector<int>, y);
constructSinglyLayer(Reshape_Layer, Reshape, std::vector<int>, y);
constructSimpleLayer(Transpose_Layer, Transpose);
constructSinglyLayer(Flatten_Layer, Flatten, int, dim);

constructSinglyLayer(ScalarAdd_Layer, ScalarAdd, int, y);
constructSinglyLayer(ScalarSubtract_Layer, ScalarSubtract, int, y);
constructSinglyLayer(ScalarMultiply_Layer, ScalarMultiply, int, y);
constructSinglyLayer(ScalarDivide_Layer, ScalarDivide, int, y);

constructSinglyLayer(Dropout_Layer, Dropout, float, rate);

#undef constructSinglyLayer
#undef constructSimpleLayer

class Slice_Layer : public Module {
public:
	Slice_Layer(std::vector<int> l, std::vector<int> r) : l(l), r(r){}
	Tensor forward(Tensor x) override { return Slice(x, l, r); }
private:
	std::vector<int> l, r;
};

class Unfold_Layer : public Module {
public:
	Unfold_Layer(std::vector<int> slider_d, std::vector<int> stride = std::vector<int>(0), std::vector<int> modulo = std::vector<int>(0)) 
		: slider_d(slider_d), stride(stride), modulo(modulo){}
	Tensor forward(Tensor x) override { return Unfold(x, slider_d, stride, modulo); }
private:
	std::vector<int> slider_d, stride, modulo;
};


#endif