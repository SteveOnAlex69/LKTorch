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

class reLU_Layer: public Module {
public:
	Tensor forward(Tensor x) override { return reLU(x); }
};
class Sigmoid_Layer : public Module {
public:
	Tensor forward(Tensor x) override { return Sigmoid(x); }
};
class Abs_Layer : public Module {
public:
	Tensor forward(Tensor x) override { return Abs(x); }
};
class Sqr_Layer : public Module {
public:
	Tensor forward(Tensor x) override { return Sqr(x); }
};
class Huber_Layer : public Module {
public:
	Huber_Layer(float epsilon) : epsilon(epsilon) {}
	Tensor forward(Tensor x) override { return Huber(x, epsilon); }
private: 
	float epsilon;
};
class Sqrt_Layer : public Module {
public:
	Tensor forward(Tensor x) override { return Sqrt(x); }
};
class Log_Layer : public Module {
public:
	Tensor forward(Tensor x) override { return Log(x); }
};
class Tanh_Layer : public Module {
public:
	Tensor forward(Tensor x) override { return Tanh(x); }
};
class Inverse_Layer : public Module {
public:
	Tensor forward(Tensor x) override { return Inverse(x); }
};
class Min_Layer : public Module {
public:
	Tensor forward(Tensor x) override { return Min(x); }
};
class Max_Layer : public Module {
public:
	Tensor forward(Tensor x) override { return Max(x); }
};
class Reshape_Layer : public Module {
public:
	Reshape_Layer(std::vector<int> y): y(y){}
	Tensor forward(Tensor x) override { return Reshape(x, y); }
private: 
	std::vector<int> y;
};
class Transpose_Layer : public Module {
public:
	Tensor forward(Tensor x) override { return Transpose(x); }
};
class Slice_Layer : public Module {
public:
	Slice_Layer(std::vector<int> l, std::vector<int> r) : l(l), r(r){}
	Tensor forward(Tensor x) override { return Slice(x, l, r); }
private:
	std::vector<int> l, r;
};
class Sum_Layer : public Module {
public:
	Tensor forward(Tensor x) override { return Sum(x); }
};
class Flatten_Layer : public Module {
public:
	Tensor forward(Tensor x) override { return Flatten(x); }
};
class Mean_Layer : public Module {
public:
	Tensor forward(Tensor x) override { return Mean(x); }
};
class SoftMax_Layer : public Module {
public:
	Tensor forward(Tensor x) override { return SoftMax(x); }
};
class ScalarMultiply_Layer : public Module {
public:
	ScalarMultiply_Layer(float y) : y(y) {}
	Tensor forward(Tensor x) override { return ScalarMultiply(x, y); }
private:
	float y;
};
class ScalarDivide_Layer : public Module {
public:
	ScalarDivide_Layer(float y) : y(y) {}
	Tensor forward(Tensor x) override { return ScalarDivide(x, y); }
private:
	float y;
};
class ScalarAdd_Layer : public Module {
public:
	ScalarAdd_Layer(float y) : y(y) {}
	Tensor forward(Tensor x) override { return ScalarAdd(x, y); }
private:
	float y;
};
class ScalarSubtract_Layer : public Module {
public:
	ScalarSubtract_Layer(float y) : y(y) {}
	Tensor forward(Tensor x) override { return ScalarSubtract(x, y); }
private:
	float y;
};
class Unfold_Layer : public Module {
public:
	Unfold_Layer(int a, int b) : a(a), b(b) {}
	Tensor forward(Tensor x) override { return Unfold(x, a, b); }
private:
	int a, b;
};

#endif