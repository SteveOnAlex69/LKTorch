#include <iostream>
#include <iomanip>

#include <TensorManip/TensorManip.hpp>
#include <Tensor/Tensor.hpp>

#include <Helper/GeneralMath.hpp>
#include <Helper/StaticVector.hpp>

#include <FileHandling/StateDictSaver.hpp>

#include <math.h>
#include <random>
#include <algorithm>
#include <functional>

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
	for (int i = -100; i <= 100; ++i)
		for (int j = -100; j <= 100; ++j) {
			std::vector<float> input{ (float)i / 10, (float)j / 10 };
			std::vector<float> output{ f(input[0], input[1]) };
			private_test.push_back(Dataset(input, output));
		}

	std::shuffle(public_test.begin(), public_test.end(), rd);
}

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

float calculate_loss(std::vector<Dataset>& test, Module& your_nn, std::function<Tensor(Tensor, Tensor)> Loss = MSELoss) {
	std::pair<Tensor, Tensor> data = load_dataset(test, 0, test.size());

	Tensor tenso = your_nn(data.first);
	Tensor final_loss = Loss(tenso, data.second);

	return MeanAll(final_loss).accessA(std::vector<int>{});
}

class MyNeuralNetwork: public Module {
public:
	MyNeuralNetwork() {
		layer1 = LinearLayer(2, 16);
		layer2 = LinearLayer(16, 16);
		layer3 = LinearLayer(16, 1);
		register_parameter(layer1);
		register_parameter(layer2);
		register_parameter(layer3);
	}

	Tensor forward(Tensor x) override {
		x = reLU(layer1(x));
		x = reLU(layer2(x));
		x = Sigmoid(layer3(x));
		return x;
	}
private:
	LinearLayer layer1, layer2, layer3;
};

void solve() {
	//MyNeuralNetwork my_nn;
	Sequential my_nn = Sequential({
		std::make_shared<LinearLayer>(2, 16),
		std::make_shared<reLU_Layer>(),
		std::make_shared<LinearLayer>(16, 16),
		std::make_shared<reLU_Layer>(),
		std::make_shared<LinearLayer>(16, 1),
		std::make_shared<Sigmoid_Layer>(),
	});


	my_nn.load_state_dict(IOHandle::read(std::string(PROJECT_DIR) + "//asset//tmp.txt"));


	std::cout << "--> MSE Loss:\n";
	std::cout << "----> On public test: " << calculate_loss(public_test, my_nn) << "\n";
	std::cout << "----> On private test: " << calculate_loss(private_test, my_nn) << "\n";

	std::cout << "--> MAE Loss:\n";
	std::cout << "----> On public test: " << calculate_loss(public_test, my_nn, MAELoss) << "\n";
	std::cout << "----> On private test: " << calculate_loss(private_test, my_nn, MAELoss) << "\n";
	/*
	float lr = 0.001;
	Adam optimizer1(lr);
	optimizer1.add_parameter(my_nn.get_parameters());

	
	clock_t cock = clock();
	std::pair<Tensor, Tensor> data = load_dataset(public_test, 0, public_test.size());
	std::vector<int> lmao{ 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000};
	for (int it = 1; it <= 1000; ++it) {
		const int BATCH = 101;
		// This is redundant, but by good design I should still do this
		optimizer1.zero_gradient();
		for(int i = 0; i < (int) public_test.size(); i += BATCH){
			Tensor tenso = data.first.Slice(std::vector<int>{i, 0}, 
				std::vector<int>{(int)min(i + BATCH, public_test.size()) - 1, 1});

			tenso = my_nn(tenso);
			tenso = RMSELoss(tenso, data.second.Slice(std::vector<int>{i, 0},
				std::vector<int>{(int)min(i + BATCH, public_test.size()) - 1, 0}));
			tenso = MeanAll(tenso);
			tenso.backward();

			// This automatically reset the gradient
			optimizer1.step();
		}
		//optimizer.decay_learning_rate();

		if (std::binary_search(lmao.begin(), lmao.end(), it)) {
			std::cout << "After " << it << " rounds of training\n";
			std::cout << "--> Time elapsed: " << clock() - cock << "ms!\n";

			std::cout << "--> MSE Loss:\n";
			std::cout << "----> On public test: " << calculate_loss(public_test, my_nn) << "\n";
			std::cout << "----> On private test: " << calculate_loss(private_test, my_nn) << "\n";

			std::cout << "--> MAE Loss:\n";
			std::cout << "----> On public test: " << calculate_loss(public_test, my_nn, MAELoss) << "\n";
			std::cout << "----> On private test: " << calculate_loss(private_test, my_nn, MAELoss) << "\n";
		}
	}

	IOHandle::write(my_nn.get_state_dict(), std::string(PROJECT_DIR) + "//asset//tmp.txt");*/
}



void debug_zone() {
	
}

int main(int argv, char* args[]) {
	std::cout << "booted" << std::endl;

	std::cout << std::fixed << std::setprecision(9);

	prepare_dataset();

	if (false)
		debug_zone();
	else solve();

	std::cout << "Exit success" << std::endl;
	return EXIT_SUCCESS;
}