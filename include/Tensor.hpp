#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <StaticVector.hpp>
#include <vector>


enum TransformType {
	NOTHING = 0,
	ADD = 1,
	SUBTRACT = 2,
	MULTIPLY = 3,


	RELU = 4,
	SIGMOID = 5,
};

// Defining a tensor, which do a shit tons of things
struct RawTensor {
public:
	RawTensor(StaticIntVector dimension);
	RawTensor(StaticIntVector dimension, std::vector<float> x);
	RawTensor(StaticIntVector dimension, StaticFloatVector x);

	int get_tensor_size();
	StaticIntVector get_tensor_dimension();

	int get_reference_count();
	void increase_reference_count();
	void decrease_reference_count();

	void set_trans_type(TransformType trans);
	TransformType get_trans_type() const;

	void add_parent(std::shared_ptr<RawTensor> p);
	std::vector<std::shared_ptr<RawTensor>> get_parent() const;

	void backward();

	void clear();
	bool empty();

	std::shared_ptr<StaticFloatVector>& A();
	std::shared_ptr<StaticFloatVector>& gA();
	float& accessA(int x);
	float& accessGA(int x);

	void gradient_descent(float lr);
	void zero_gradient();
	void multiply_gradient(float x);
private:
	int tensor_size;
	std::shared_ptr<StaticFloatVector> value, gradient_value;
	StaticIntVector dimension;
	int reference_counter;
	
	TransformType t_type = NOTHING;
	std::vector<std::shared_ptr<RawTensor>> parents;
};

std::shared_ptr<RawTensor> operator + (std::shared_ptr<RawTensor> x, std::shared_ptr<RawTensor> y);
std::shared_ptr<RawTensor> operator - (std::shared_ptr<RawTensor> x, std::shared_ptr<RawTensor> y);
std::shared_ptr<RawTensor> operator * (std::shared_ptr<RawTensor> x, std::shared_ptr<RawTensor> y);
std::shared_ptr<RawTensor> change_dimension (std::shared_ptr<RawTensor> x, StaticIntVector y);


std::shared_ptr<RawTensor> reLU (std::shared_ptr<RawTensor> x);
std::shared_ptr<RawTensor> sigmoid (std::shared_ptr<RawTensor> x);



struct Tensor {
	std::shared_ptr<RawTensor> ts;
	Tensor(std::shared_ptr<RawTensor> t = nullptr);
	Tensor(StaticIntVector dimension);
	Tensor(StaticIntVector dimension, std::vector<float> x);
	Tensor(StaticIntVector dimension, StaticFloatVector x);

	int get_tensor_size();
	StaticIntVector get_tensor_dimension();

	void backward();
	void clear();
	bool empty();


	void gradient_descent(float lr);
	void zero_gradient();
	void multiply_gradient(float x);

	float& accessA(int x);
	float& accessGA(int x);

	std::shared_ptr<StaticFloatVector>& A();
	std::shared_ptr<StaticFloatVector>& gA();
};


Tensor operator + (Tensor  x, Tensor  y);
Tensor operator - (Tensor  x, Tensor  y);
Tensor operator * (Tensor  x, Tensor  y);
Tensor change_dimension(Tensor x, StaticIntVector y);


Tensor reLU(Tensor x);
Tensor sigmoid(Tensor x);

std::ostream& operator << (std::ostream& os, Tensor x);

#endif