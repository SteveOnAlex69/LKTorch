#ifndef BASETENSOR_HPP
#define BASETENSOR_HPP

#include <Tensor/RawTensor.hpp>

struct Tensor {
	std::shared_ptr<RawTensor> ts;
	Tensor(std::shared_ptr<RawTensor> t = nullptr);
	Tensor(StaticIntVector dimension);
	Tensor(StaticIntVector dimension, std::vector<float> x);
	Tensor(StaticIntVector dimension, StaticFloatVector x);

	int get_tensor_size();
	StaticIntVector get_tensor_dimension();

	void backward(bool is_root = true);
	void clear();
	bool empty();

	void gradient_descent(float lr);
	void zero_gradient();
	void multiply_gradient(float x);

	std::shared_ptr<StaticFloatVector>& A();
	std::shared_ptr<StaticFloatVector>& gA();

	float& accessA(StaticIntVector x) { return ts->accessA(x); }
	float& accessGA(StaticIntVector x) { return ts->accessGA(x);}
	float& accessA(std::vector<int> x) { return ts->accessA(x); }
	float& accessGA(std::vector<int> x) { return ts->accessGA(x); }

	Tensor Transpose();
	Tensor Reshape(StaticIntVector y);
	Tensor Slice(StaticIntVector l, StaticIntVector r);
	Tensor Merge(Tensor y);
};

Tensor operator + (Tensor x, Tensor y);
Tensor operator - (Tensor x, Tensor y);
Tensor operator * (Tensor x, Tensor y);
Tensor Reshape(Tensor x, StaticIntVector y);
Tensor Transpose(Tensor x);
Tensor Slice(Tensor x, StaticIntVector l, StaticIntVector r);
Tensor Merge(Tensor x, Tensor y);

#endif