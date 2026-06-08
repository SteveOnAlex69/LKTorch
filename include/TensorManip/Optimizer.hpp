#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <vector>
#include <Tensor/Tensor.hpp>

class GradientDescent { // Gradient Descent
public:
	GradientDescent(float learning_rate = 0);
	float get_learning_rate() const;
	void set_learning_rate(float lr);
	void add_parameter(Tensor ts);
	void add_parameter(std::vector<Tensor> ts);
	
	void step();
	void zero_gradient();
protected :
	float learning_rate;
	std::vector<Tensor> parameters;
};

 
class GradientMomentum : public GradientDescent { // Gradient Descent with Momentum
public:
	GradientMomentum(float learning_rate = 0, float glide = 0);
	float get_glide() const ;
	void set_glide(float gl);

	void step();
	void multiply_gradient();
protected:
	float glide;
};


class DecayDescent : public GradientMomentum { // Gradient Descent with Decaying learning rate
public:
	DecayDescent(float learning_rate = 0, float glide = 0, float decay = 0);
	float get_decay() const;
	void set_decay(float dc);

	void decay_learning_rate();
protected:
	float decay;
};

class AdaGrad : public GradientDescent { // Adagrad
public:
	AdaGrad(float learning_rate = 0);
	void step();


	void add_parameter(Tensor ts);
	void add_parameter(std::vector<Tensor> ts);
protected:
	std::vector<std::shared_ptr<StaticFloatVector>> G;
};


class RMSProp : public AdaGrad { // Adagrad but the G parameter decay, fixing the fault of the network stopping learning after a while
public:
	RMSProp(float learning_rate = 0, float decay_rate = 0.9);
	void step();
protected:
	float decay_rate;
};


// RMSProp + Momentum = Adam
class Adam : public AdaGrad { // Adagrad but you can choose the decay rate
public:
	Adam(float learning_rate = 0, float glide = 0.99, float decay = 0.99);

	void add_parameter(Tensor ts);
	void add_parameter(std::vector<Tensor> ts);
	void step();
protected:
	float glide, decay_rate;
	std::vector<std::shared_ptr<StaticFloatVector>> V;
	int step_count;
};


#endif