#ifndef MODULE_HPP
#define MODULE_HPP

#include <Tensor/Tensor.hpp>
#include <vector>

class Module {
public:
	virtual ~Module() = default;
	virtual Tensor forward(Tensor x) = 0;

	Tensor operator () (Tensor x) { return forward(x); }
	std::vector<Tensor> get_parameters() { return parameters; }
protected:
	std::vector<Tensor> parameters;
	void register_parameter(Tensor param) { parameters.push_back(param); }
	void register_parameter(std::vector<Tensor> param) { 
		for(auto i: param)
			parameters.push_back(i); 
	}
	void register_parameter(Module &x) {
		for (auto i : x.get_parameters())
			parameters.push_back(i);
	}
};


#endif