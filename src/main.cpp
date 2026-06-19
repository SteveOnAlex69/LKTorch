#include <iostream>
#include <iomanip>

#include <LKTorch.hpp>

#include <math.h>
#include <random>
#include <algorithm>
#include <functional>

// Now we will try to extrapolate the function f(x, y) = e^-(x^2 + y^2)

std::mt19937_64 rd(69);


Tensor join_info(Tensor public_state, Tensor hidden_state) {
	std::vector<int> dim = public_state.dimension();
	int sz = dim.size();

	// Create the perm array: [sz-1, 0, 1, ..., sz-2]
	std::vector<int> perm;
	perm.push_back(sz - 1);
	for (int i = 0; i < sz - 1; ++i) {
		perm.push_back(i);
	}

	public_state = public_state.PermuteDimension(perm);
	hidden_state = hidden_state.PermuteDimension(perm);

	// Assuming Merge is a member function of Tensor based on your bindings
	Tensor state = public_state.Merge(hidden_state);

	// Create the invperm array: [1, 2, ..., sz-1, 0]
	std::vector<int> invperm;
	for (int i = 0; i < sz - 1; ++i) {
		invperm.push_back(i + 1);
	}
	invperm.push_back(0);

	return state.PermuteDimension(invperm);
}

Tensor extract_public(Tensor tensor) {
	return tensor.Slice({ 0 }, { 0 });
}

Tensor extract_hidden(Tensor tensor) {
	return tensor.Slice({ 1 }, { 3 });
}

void solve() {
	Sequential model = Sequential({
		std::make_shared<LinearLayer>(4, 8),
		std::make_shared<reLU_Layer>(reLU_Layer()),
		std::make_shared<LinearLayer>(8, 8),
		std::make_shared<reLU_Layer>(),
		std::make_shared<LinearLayer>(8, 4),
		std::make_shared<Sigmoid_Layer>()
	});

	const float PI = std::atan(1) * 4;
	std::vector<std::vector<float>> public_test;
	for (int i = 0; i < 100; ++i) {
		std::vector<float> cur;
		for (int j = 0; j < 10; ++j)
			cur.push_back(std::sin(PI * (i + j) / 50));
		public_test.push_back(cur);
	}

	Tensor shit = Ones(std::vector<int>{4});
	if (true) {
		std::cout << "one\n";
		Tensor Hentai = model(shit);
		Hentai = model(Hentai);
		std::cout << "two\n";
	}
	std::cout << "three\n";
}



void debug_zone() {
	std::vector<std::vector<std::vector<float>>> a(2, std::vector<std::vector<float>>(2, std::vector<float>(2, 1)));
	Tensor gaming = CreateTensor(a);
	Tensor bulul = gaming * 3;
	Tensor halal = bulul * 6;
	std::cout << gaming << "\n" << bulul << "\n" << halal << "\n\n";
	halal.backward();
	std::cout << gaming << "\n" << bulul << "\n";
}

int main(int argv, char* args[]) {
	std::cout << "booted" << std::endl;

	std::cout << std::fixed << std::setprecision(9);

	if (false)
		debug_zone();
	else solve();

	std::cout << "Exit success" << std::endl;
	return EXIT_SUCCESS;
}