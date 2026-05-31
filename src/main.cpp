#include <iostream>
#include <iomanip>
#include <Tensor.hpp>
#include <GeneralMath.hpp>
#include <Optimizer.hpp>
#include <Loss.hpp>
#include <Optimizer.hpp>
#include <Layer.hpp>
#include <math.h>
#include <random>
#include <algorithm>

// Now we will try to extrapolate the function f(x, y) = e^-(x^2 + y^2)

std::mt19937_64 rd(69);

struct Dataset {
	std::vector<float> input;
	std::vector<float> output;

	Dataset();
	Dataset(std::vector<float> i, std::vector<float> o) : input(i), output(o) {}
};

float f(float x, float y) { return std::exp(-sqr(x) - sqr(y)); }

std::vector<Dataset> private_test, public_test;
void prepare_dataset() {
	
	for (int i = -20; i <= 20; ++i)
		for (int j = -20; j <= 20; ++j) {
			std::vector<float> input{ (float)i / 2, (float)j / 2 };
			std::vector<float> output{ f(input[0], input[1]) };
			public_test.push_back(Dataset(input, output));
		}
		


	for (int i = 0; i <= 0; ++i)
		for (int j = 0; j <= 0; ++j) {
			std::vector<float> input{ (float)i / 2, (float)j / 2 };
			std::vector<float> output{ f(input[0], input[1]) };
			public_test.push_back(Dataset(input, output));
		}
	for (int i = -100; i <= 100; ++i)
		for (int j = -100; j <= 100; ++j) {
			std::vector<float> input{ (float)i / 10, (float)j / 10 };
			std::vector<float> output{ f(input[0], input[1]) };
			private_test.push_back(Dataset(input, output));
		}

	std::shuffle(public_test.begin(), public_test.end(), rd);
}

LinearLayer layer1(2, 16), layer2(16, 16), layer3(16, 1);

// load up a pair of input and output
std::pair<Tensor, Tensor> load_dataset(std::vector<Dataset>& test, int l, int r) {
	r = std::min(r, (int)test.size());
	std::vector<float> input, output;
	int input_dimension = -1, output_dimension = -1, test_count = 0;
	for (int i = l; i < r; ++i) {
		Dataset cur = test[i];
		input_dimension = cur.input.size();
		output_dimension = cur.output.size();
		test_count++;
		for (float j : cur.input) input.push_back(j);
		for (float j : cur.output) output.push_back(j);
	}
	Tensor input_tensor(std::vector<int>{test_count, input_dimension}, input);
	Tensor output_tensor(std::vector<int>{test_count, output_dimension}, output);
	return std::make_pair(input_tensor, output_tensor);
}

float calculate_loss(std::vector<Dataset>& test) {
	std::pair<Tensor, Tensor> data = load_dataset(test, 0, test.size());

	Tensor tenso = data.first;
	tenso = layer1(tenso);
	tenso = reLU(tenso);
	tenso = layer2(tenso);
	tenso = reLU(tenso);
	tenso = layer3(tenso);

	tenso = sigmoid(tenso);
	Tensor final_loss = MSELoss(tenso, data.second);

	return final_loss.accessA(0) / test.size();
}

void solve() {
	layer1.randomize_weight();
	layer2.randomize_weight();
	layer3.randomize_weight();
	
	float lr = 0.01;
	SGD optimizer(lr);
	
	for (Tensor i : layer1.get_parameters()) optimizer.add_parameter(i);
	for (Tensor i : layer2.get_parameters()) optimizer.add_parameter(i);
	for (Tensor i : layer3.get_parameters()) optimizer.add_parameter(i);
	
	clock_t cock = clock();
	std::vector<int> lmao{ 1, 5, 10, 20, 50, 100, 200, 500, 1000};
	for (int it = 1; it <= 1000; ++it) {
		const int BATCH = 100;
		for(int i = 0; i < (int) public_test.size(); i += BATCH){
			optimizer.zero_gradient();

			std::pair<Tensor, Tensor> data = load_dataset(public_test, i, i + BATCH);
			Tensor tenso = data.first;
			tenso = layer1(tenso);
			tenso = reLU(tenso);
			tenso = layer2(tenso);
			tenso = reLU(tenso);
			tenso = layer3(tenso);
			tenso = sigmoid(tenso);

			tenso = MSELoss(tenso, data.second);
			tenso.accessGA(0) = (float)1;
			tenso.backward();

			optimizer.step();
		}

		if (std::binary_search(lmao.begin(), lmao.end(), it)) {
			std::cout << "After " << it << " rounds of training\n";
			std::cout << "--> Time elapsed: " << clock() - cock << "ms!\n";
			std::cout << "--> On public test: " << calculate_loss(public_test) << "\n";
			std::cout << "--> On private test: " << calculate_loss(private_test) << "\n";
		}
	}

	
}

int main(int argv, char* args[]) {
	std::cout << "booted" << std::endl;

	std::cout << std::fixed << std::setprecision(9);

	prepare_dataset();
	solve();

	return EXIT_SUCCESS;
}