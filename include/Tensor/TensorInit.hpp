#ifndef TENSORINIT_HPP
#define TENSORINIT_HPP

#include <Tensor/BaseTensor.hpp>
#include <type_traits>

Tensor Zeros(std::vector<int> dimension);
Tensor Ones(std::vector<int> dimension);
Tensor UniformValue(std::vector<int> dimension, float x);
Tensor UniformRandom(std::vector<int> dimension, float l, float r);
Tensor NormalRandom(std::vector<int> dimension, float mean, float deviation);
Tensor CustomRandom(std::vector<int> dimension, std::function<float(int)> init_function);



namespace tensor_utils {
	template<class T>
	constexpr bool is_vector = false;

	template<class T, class Alloc>
	constexpr bool is_vector<std::vector<T, Alloc>> = true;

	template<class T>
	std::vector<int> get_dimension(T a) {
		if constexpr (is_vector<T> == true) {
			std::vector<int> ans = get_dimension(a[0]);
			ans.insert(ans.begin(), a.size());
			return ans;
		}
		else return std::vector<int>(0);
	}

	template<class T>
	std::vector<float> get_data(T a) {
		if constexpr (is_vector<T> == true) {
			std::vector<float> ans;
			for (auto i : a) {
				std::vector<float> cur = get_data(i);
				for (float j : cur) ans.push_back(j);
			}
			return ans;
		}
		else {
			return std::vector<float>(1, a);
		}
	}
}

template<class T> 
Tensor CreateTensor(T ans) {
	return Tensor(tensor_utils::get_dimension(ans), tensor_utils::get_data(ans));
}


#endif