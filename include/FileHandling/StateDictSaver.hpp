#ifndef STATEDICTSAVER_HPP
#define STATEDICTSAVER_HPP

#include <Helper/StaticVector.hpp>
#include <string>
#include <vector>
#include <fstream>

namespace IOHandle {
	unsigned int to_int_form(float x) {
		int ans = *((int*)(&x));
		return ans;
	}

	float to_float_form(unsigned int x) {
		float ans = *((float*)(&x));
		return ans;
	}

	void write(std::vector<StaticFloatVector> content, std::string path) {
		std::ofstream output(path.c_str());
		
		// first, write the number of parameter
		output << "Parameter_count: " << content.size() << std::endl;
		
		// Now, write the content for each parameter
		for (auto i : content) {
			output << "Tensor_size: " << i.size() << "\n";
			output << "Tensor_content: ";
			for (int j = 0; j < i.size(); ++j) output << to_int_form(i[j]) << " ";
			output << "\n";
		}
	}


	std::vector<StaticFloatVector> read(std::string path) {
		std::ifstream input(path.c_str());

		// first, read the number of parameter;
		std::string dummy = "";
		int sz;
		input >> dummy >> sz;

		std::vector<StaticFloatVector> ans;
		// Now, read the content for each parameter
		for (int i = 0; i < sz; ++i) {
			int m;
			input >> dummy >> m;

			input >> dummy;
			StaticFloatVector cur(m);
			for (int j = 0; j < m; ++j) {
				unsigned int num; input >> num;
				cur[j] = to_float_form(num);
			}
			ans.push_back(cur);
		}
		return ans;
	}
}


#endif