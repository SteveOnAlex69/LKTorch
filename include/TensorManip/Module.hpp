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


	std::vector<std::vector<float>> get_state_dict() {
		std::vector<std::vector<float>> ans;
		for (auto i : parameters) {
			StaticFloatVector& cur = *i.A();
			ans.push_back(cast_to_vector(cur));
		}
		return ans;
	}
	void load_state_dict(std::vector<std::vector<float>> bruh) {
		int n = bruh.size();
		if (n != parameters.size()) throw_error("In load state dict: parameter mismatched");
		for (int i = 0; i < n; ++i) {
			if (parameters[i].get_tensor_size() != bruh[i].size()) throw_error("In load state dict: parameter mismatched");
			StaticFloatVector& cur = *parameters[i].A();
			for (int j = 0; j < bruh[i].size(); ++j)
				cur[j] = bruh[i][j];
		}
	}
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