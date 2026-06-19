#include <Tensor/DebugTensor.hpp>
#include <string>
#include <map>


std::ostream& operator << (std::ostream& os, Tensor x) {
	if (x.empty()) return os << "empty";
	StaticIntVector d = x.dimension();
	StaticFloatVector& A = *(x.A());
	if (A.size() != x.size()) throw_error("Internal Array Size and Tensor Size Mismatched");
	for (int i = d.size() - 2; i >= 0; --i) d[i] *= d[i + 1];
	for (int i = 0; i < x.size(); ++i) {
		for (int j = 0; j < d.size(); ++j) if (i % d[j] == 0) os << "(";
		os << A[i];
		int count = 0;
		for (int j = 0; j < d.size(); ++j) if ((i + 1) % d[j] == 0) os << ")";
		if (i + 1 < x.size())
			os << ", ";
	}
	return os;
}



std::string TransformName[] = {
	"NOTHING",
	"ADD",
	"SUBTRACT",
	"MULTPLY",
	"CUSTOM_FUNCTION",
	"TRANSPOSE",
	"SLICE",
	"MERGE"
};

void get_node(std::shared_ptr<RawTensor> ptr, std::map<std::shared_ptr<RawTensor>, int> &t,
	std::map<std::shared_ptr<RawTensor>, std::vector<std::shared_ptr<RawTensor>>> &p) {
	if (t.count(ptr)) return;
	t[ptr] = ptr->get_trans_type();
	p[ptr] = ptr->get_parent();

	for (auto i : ptr->get_parent())
		get_node(i, t, p);
}

void print_structure(Tensor x) {
	std::map<std::shared_ptr<RawTensor>, int> transformation_type;
	std::map<std::shared_ptr<RawTensor>, std::vector<std::shared_ptr<RawTensor>>> parents;

	get_node(x.ts, transformation_type, parents);

	for (auto i : transformation_type) {
		std::cout << i.first << " " << TransformName[i.second]<< " " << parents[i.first] << "\n";
	}
}
