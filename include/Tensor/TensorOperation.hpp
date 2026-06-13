#ifndef TENSOROPERATION_HPP
#define TENSOROPERATION_HPP

#include <Tensor/BaseTensor.hpp>
#include <Tensor/TensorFunction.hpp>

extern TensorFunction reLU, Sigmoid, Abs, Sqr, Sqrt, Log, Tanh, Inverse, Min, Max;


Tensor Permute(Tensor x, std::vector<int> new_d, std::vector<int> new_p);
Tensor PermuteDimension(Tensor x, std::vector<int> perm);
Tensor Reshape(Tensor x, std::vector<int> y);
Tensor Transpose(Tensor x);
Tensor Slice(Tensor x, std::vector<int> l, std::vector<int> r);
Tensor Merge(Tensor x, Tensor y);
Tensor ValueMultiply(Tensor x, Tensor y);

// Only Transform The Innermost Layer
Tensor Sum(Tensor x);
Tensor Mean(Tensor x);
Tensor SoftMax(Tensor x);

Tensor SumAll(Tensor x);
Tensor MeanAll(Tensor x);

Tensor Flatten(Tensor x);
Tensor Huber(Tensor x, float epsilon);

Tensor operator * (Tensor x, float y);
Tensor operator / (Tensor x, float y);
Tensor operator + (Tensor x, float y);
Tensor operator - (Tensor x, float y);

Tensor ScalarMultiply(Tensor x, float y);
Tensor ScalarDivide(Tensor x, float y);
Tensor ScalarAdd(Tensor x, float y);
Tensor ScalarSubtract(Tensor x, float y);

Tensor Unfold(Tensor x, int a, int b);

#endif