#ifndef RAWTENSOR_HPP
#define RAWTENSOR_HPP

#include <memory>
#include <Helper/StaticVector.hpp>
#include <vector>
#include <functional>
#include <Tensor/SingletonTensor.hpp>


enum TransformType {
	NOTHING = 0,

	ADD = 1,
	SUBTRACT = 2,
	MULTIPLY = 3,
	VALUE_MULTIPLY = 4,
	CUSTOM_FUNCTION = 5,
	CUSTOM_PERMUTE = 6,
	MERGE = 7
};

// Defining a tensor, which do a shit tons of things
struct RawTensor {
public:
	RawTensor(StaticIntVector dimension, bool nograd = false);
	RawTensor(StaticIntVector dimension, std::shared_ptr<StaticFloatVector> preA, std::shared_ptr<StaticFloatVector> pregA);
	RawTensor(StaticIntVector dimension, std::vector<float> x, bool nograd = false);
	RawTensor(StaticIntVector dimension, StaticFloatVector x, bool nograd = false);

	~RawTensor();

	int get_tensor_size();
	StaticIntVector get_tensor_dimension();
	int get_row();

	int get_reference_count();
	void increase_reference_count();
	void decrease_reference_count();

	void set_trans_type(TransformType trans);
	TransformType get_trans_type() const;

	void set_df(std::function<std::vector<float>(std::vector<float>, int)>  dF);
	void set_permutation(std::vector<std::pair<int,int>> perm, std::pair<int,int> stride);

	void add_parent(std::shared_ptr<RawTensor> p);
	std::vector<std::shared_ptr<RawTensor>> get_parent() const;

	void backward(bool is_root = false);

	void clear();
	bool empty();

	// these methods are faster, so you should use.
	std::shared_ptr<StaticFloatVector>& A();
	std::shared_ptr<StaticFloatVector>& gA();

	// disclaimer: these two methods are slow. It's best to just fetch the entire array A() or gA()
	float& access_A(StaticIntVector x);
	float& access_GA(StaticIntVector x);

	void set_name(std::string s);
	std::string get_name();
private:
	int tensor_size;
	std::shared_ptr<StaticFloatVector> value, gradient_value;
	StaticIntVector dimension;
	int reference_counter;

	TransformType t_type = NOTHING;
	std::vector<std::shared_ptr<RawTensor>> parents;
	std::function<std::vector<float>(std::vector<float>, int)> dF;

	std::vector<std::pair<int, int>> perm;
	std::pair<int, int> stride;

	std::string name;
};

std::shared_ptr<RawTensor> operator + (std::shared_ptr<RawTensor> x, std::shared_ptr<RawTensor> y);
std::shared_ptr<RawTensor> operator - (std::shared_ptr<RawTensor> x, std::shared_ptr<RawTensor> y);

// This will collapse the middle layer
std::shared_ptr<RawTensor> operator * (std::shared_ptr<RawTensor> x, std::shared_ptr<RawTensor> y);

std::shared_ptr<RawTensor> map_value(std::shared_ptr<RawTensor> x, std::vector<std::pair<int, int>> perm, std::pair<int, int> stride, StaticIntVector new_dimension);

std::shared_ptr<RawTensor> value_multiply (std::shared_ptr<RawTensor> x, std::shared_ptr<RawTensor> y);
std::shared_ptr<RawTensor> reshape(std::shared_ptr<RawTensor> x, StaticIntVector y);
std::shared_ptr<RawTensor> slice(std::shared_ptr<RawTensor> x, StaticIntVector l, StaticIntVector r);
std::shared_ptr<RawTensor> merge(std::shared_ptr<RawTensor> x, std::shared_ptr<RawTensor> y);

std::shared_ptr<RawTensor> permute(std::shared_ptr<RawTensor> x, StaticIntVector d, StaticIntVector perm);
std::shared_ptr<RawTensor> permute_dimension(std::shared_ptr<RawTensor> x, StaticIntVector d);
std::shared_ptr<RawTensor> transpose(std::shared_ptr<RawTensor> x);

#endif