#ifndef SEQUENTIAL_HPP
#define SEQUENTIAL_HPP

#include <Tensor/Tensor.hpp>
#include <vector>

class Sequential : public Module {
public:
	Sequential() {}
	Sequential(std::vector<std::shared_ptr<Module>> a) {
		value = a;
		for (auto i : value) RegisterParameter(*i);
	}
	Tensor forward(Tensor x) override { 
		for (auto i : value) {
			x = i->forward(x);
		}
		return x;
	}
	std::shared_ptr<Module>& operator [] (int x) { return *(value.data() + x); }

	void push_back(std::shared_ptr<Module> x) { value.push_back(x); }
private:
	std::vector<std::shared_ptr<Module>> value;
};


#endif