#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <DebugAssist.hpp>
#include <LinearAlg.hpp>
#include <Layer.hpp>


enum TransformType {
	ADD = 0,
	SUBTRACT = 1,
	LINEAR = 2,
	RELU = 3,
	SIGMOID = 4
};

// Defining a tensor, which do a shit tons of things


struct Tensor {
public:
	Tensor(int n);
	Tensor(Line x);
	void set_trans_type(TransformType trans);
	TransformType get_trans_type() const;
	void set_parent1(std::shared_ptr<Tensor> p);
	std::shared_ptr<Tensor> get_parent1() const;
	void set_parent2(std::shared_ptr<Tensor> p);
	std::shared_ptr<Tensor> get_parent2() const;

	
	void set_linear_layer(void* l);
	void* get_linear_layer() const;

	Line getA() const;

	Tensor forward(LinearLayer& ll);
	void backward(Line dL);
private:
	int n;
	Line A;
	
	TransformType trans_type;
	std::shared_ptr<Tensor> parent1, parent2;

	void* lilirlaler = nullptr;
};


float reLU(float x);
float DreLU(float x);

float sigmoid(float x);
float Dsigmoid(float x);


Tensor reLU(const Tensor& x);
Tensor sigmoid(const Tensor& x);

Tensor operator + (const Tensor& x, const Tensor& y);
Tensor operator - (const Tensor& x, const Tensor& y);

#endif