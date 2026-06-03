#ifndef RAWTENSOR_HPP
#define RAWTENSOR_HPP


#include <Helper/StaticVector.hpp>
#include <vector>
#include <functional>


enum TransformType {
	NOTHING = 0,

	ADD = 1,
	SUBTRACT = 2,
	MULTIPLY = 3,
	CUSTOM_FUNCTION = 4,
	TRANSPOSE = 5,
	SLICE = 6,
	MERGE = 7
};

// Defining a tensor, which do a shit tons of things
struct RawTensor {
public:
	RawTensor(StaticIntVector dimension);
	RawTensor(StaticIntVector dimension, std::shared_ptr<StaticFloatVector> preA, std::shared_ptr<StaticFloatVector> pregA);
	RawTensor(StaticIntVector dimension, std::vector<float> x);
	RawTensor(StaticIntVector dimension, StaticFloatVector x);

	~RawTensor();

	int get_tensor_size();
	StaticIntVector get_tensor_dimension();

	int get_reference_count();
	void increase_reference_count();
	void decrease_reference_count();

	void set_trans_type(TransformType trans);
	TransformType get_trans_type() const;

	void set_start_dimension(StaticIntVector d);


	void setDF(std::function<StaticFloatVector(StaticFloatVector&)> dF);

	void add_parent(std::shared_ptr<RawTensor> p);
	std::vector<std::shared_ptr<RawTensor>> get_parent() const;

	void backward(bool is_root = false);

	void clear();
	bool empty();

	// these methods are faster, so you should use.
	// but use with caution and care, because if the tensor is init without a value, these guys would be null
	std::shared_ptr<StaticFloatVector>& A();
	std::shared_ptr<StaticFloatVector>& gA();

	// disclaimer: these two methods are slow. It's best to just fetch the entire array A() or gA()
	float& accessA(StaticIntVector x);
	float& accessGA(StaticIntVector x);

	void gradient_descent(float lr);
	void zero_gradient();
	void multiply_gradient(float x);

private:
	int tensor_size;
	std::shared_ptr<StaticFloatVector> value, gradient_value;
	StaticIntVector dimension, start_dimension;
	int reference_counter;

	TransformType t_type = NOTHING;
	std::vector<std::shared_ptr<RawTensor>> parents;
	std::function<StaticFloatVector(StaticFloatVector&)> dF;
};

std::shared_ptr<RawTensor> operator + (std::shared_ptr<RawTensor> x, std::shared_ptr<RawTensor> y);
std::shared_ptr<RawTensor> operator - (std::shared_ptr<RawTensor> x, std::shared_ptr<RawTensor> y);

// This will collapse the middle layer
std::shared_ptr<RawTensor> operator * (std::shared_ptr<RawTensor> x, std::shared_ptr<RawTensor> y);
std::shared_ptr<RawTensor> transpose(std::shared_ptr<RawTensor> x);
std::shared_ptr<RawTensor> reshape(std::shared_ptr<RawTensor> x, StaticIntVector y);
std::shared_ptr<RawTensor> slice(std::shared_ptr<RawTensor> x, StaticIntVector l, StaticIntVector r);
std::shared_ptr<RawTensor> merge(std::shared_ptr<RawTensor> x, std::shared_ptr<RawTensor> y);

#endif