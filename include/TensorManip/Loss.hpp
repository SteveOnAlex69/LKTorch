#ifndef LOSS_HPP
#define LOSS_HPP

#include <Tensor/Tensor.hpp>

Tensor MSELoss(Tensor a, Tensor b);
Tensor MAELoss(Tensor a, Tensor b);
Tensor HuberLoss(Tensor a, Tensor b, float epsilon);
Tensor RMSELoss(Tensor a, Tensor b);

Tensor BCELoss(Tensor target, Tensor prediction);
Tensor HingeLoss(Tensor target, Tensor prediction);
Tensor KL_Divergence(Tensor prediction, Tensor target);
Tensor CrossEntropyLoss(Tensor target, Tensor prediction);



#endif