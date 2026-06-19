#include <TensorManip/Loss.hpp>
#include <DebugAssist/DebugAssist.hpp>
#include <Helper/GeneralMath.hpp>

Tensor MSELoss(Tensor x, Tensor y) {
	int x_size = x.size(), y_size = y.size();
	if (x_size != y_size) throw_error("Size mismatched when calculating loss");
	return Mean(Sqr(x - y));
}

Tensor MAELoss(Tensor x, Tensor y) {
	int x_size = x.size(), y_size = y.size();
	if (x_size != y_size) throw_error("Size mismatched when calculating loss");
	return Mean(Abs(x - y));
}

Tensor HuberLoss(Tensor x, Tensor y, float z) {
	int x_size = x.size(), y_size = y.size();
	if (x_size != y_size) throw_error("Size mismatched when calculating loss");
	return Mean(Huber(x - y, z));
}


Tensor RMSELoss(Tensor a, Tensor b) {
	return Sqrt(MSELoss(a, b));
}


Tensor BCELoss(Tensor A, Tensor B) { // a = expected distribution, b = your distribution
	if (A.dimension() != B.dimension()) throw_error("Size mismatched when calculating loss");

	Tensor nA = (A - 1) * (-1);
	Tensor nB = (B - 1) * (-1);

	return Mean(ValueMultiply(A, Log(B)) + ValueMultiply(nA, Log(nB))) * (-1);
}

Tensor CrossEntropyLoss(Tensor A, Tensor B) {
	if (A.dimension() != B.dimension()) throw_error("Size mismatched when calculating loss");
	A = SoftMax(A);
	B = SoftMax(B);
	return Sum(ValueMultiply(A, Log(B))) * -1;
}

Tensor HingeLoss(Tensor A, Tensor B) {
	if (A.dimension() != B.dimension()) throw_error("Size mismatched when calculating loss");
	return Mean(reLU((ValueMultiply(A, B) - 1) * (-1)));
}

Tensor KL_Divergence(Tensor A, Tensor B) {
	if (A.dimension() != B.dimension()) throw_error("Size mismatched when calculating loss");
	A = SoftMax(A);
	B = SoftMax(B);

	Tensor C = Log(ValueMultiply(A, Inverse(B)));
	return Sum(Abs(ValueMultiply(A, C)));
}