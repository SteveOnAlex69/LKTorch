#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <vector>
#include <Tensor.hpp>

class SGD { // Stochastic Gradient Descent
public:
	SGD(float learning_rate = 0);
	float get_learning_rate() const;
	void set_learning_rate(float lr);
	void add_parameter(Tensor ts);
	
	void step();
	void zero_gradient();
private:
	float learning_rate;
	std::vector<Tensor> parameters;
};


#endif