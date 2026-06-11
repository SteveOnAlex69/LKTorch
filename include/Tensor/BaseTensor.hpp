#ifndef BASETENSOR_HPP
#define BASETENSOR_HPP

#include <Tensor/RawTensor.hpp>

struct Tensor {
	std::shared_ptr<RawTensor> ts;
	Tensor(std::shared_ptr<RawTensor> t = nullptr);
	Tensor(std::vector<int> dimension);
	Tensor(std::vector<int> dimension, std::vector<float> x);

	int get_tensor_size();
	std::vector<int> get_tensor_dimension();

	void backward(bool is_root = true);
	void clear();
	bool empty();

	std::shared_ptr<StaticFloatVector>& A();
	std::shared_ptr<StaticFloatVector>& gA();

	float& accessA(std::vector<int> x) { return ts->accessA(x); }
	float& accessGA(std::vector<int> x) { return ts->accessGA(x); }

	Tensor Permute(std::vector<int> new_d, std::vector<int> new_perm);
	Tensor PermuteDimension(std::vector<int> perm);
	Tensor Transpose();
	Tensor Reshape(std::vector<int> y);
	Tensor Slice(std::vector<int> l, std::vector<int> r);
	Tensor Merge(Tensor y);
	Tensor ValueMultiply(Tensor y);
};

Tensor operator + (Tensor x, Tensor y);
Tensor operator - (Tensor x, Tensor y);
Tensor operator * (Tensor x, Tensor y);

#endif