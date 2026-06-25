#ifndef LOSS_HPP
#define LOSS_HPP

#include <Tensor/Tensor.hpp>

Tensor MSELoss(Tensor a, Tensor b);
Tensor MAELoss(Tensor a, Tensor b);
Tensor HuberLoss(Tensor a, Tensor b, float epsilon);
Tensor RMSELoss(Tensor a, Tensor b);

Tensor BCELoss(Tensor prediction, Tensor target);
Tensor HingeLoss(Tensor prediction, Tensor target);
Tensor KL_Divergence(Tensor prediction, Tensor target);
Tensor CrossEntropyLoss(Tensor prediction, Tensor target);



#endif