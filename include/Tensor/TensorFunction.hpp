#ifndef TENSORFUNCTION_HPP
#define TENSORFUNCTION_HPP

#include <functional>
#include <Tensor/BaseTensor.hpp>

// This is a general class to run the function through the entire tensor

typedef std::function<StaticFloatVector(StaticFloatVector&)> VectorFunction;


class TensorFunction { 
public:
	TensorFunction(VectorFunction f, VectorFunction dF);
	std::shared_ptr<RawTensor> forward(std::shared_ptr<RawTensor> x);
	Tensor forward(Tensor x);

	std::shared_ptr<RawTensor> operator () (std::shared_ptr<RawTensor> x);
	Tensor operator () (Tensor x);


	VectorFunction f, dF;
};




#endif