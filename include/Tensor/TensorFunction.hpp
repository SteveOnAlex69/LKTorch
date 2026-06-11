#ifndef TENSORFUNCTION_HPP
#define TENSORFUNCTION_HPP

#include <functional>
#include <Tensor/BaseTensor.hpp>

// This is a general class to run the function through the entire tensor

typedef std::function<std::vector<float>(std::vector<float>)> VectorFunction;


class TensorFunction { 
public:
	TensorFunction();
	TensorFunction(VectorFunction f, VectorFunction dF);
	std::shared_ptr<RawTensor> forward(std::shared_ptr<RawTensor> x);
	Tensor forward(Tensor x);

	std::shared_ptr<RawTensor> operator () (std::shared_ptr<RawTensor> x);
	Tensor operator () (Tensor x);


	VectorFunction f, dF;
};




#endif