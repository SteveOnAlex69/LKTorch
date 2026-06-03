#ifndef DEBUGTENSOR_HPP
#define DEBUGTENSOR_HPP

#include <DebugAssist/DebugAssist.hpp>
#include <Tensor/BaseTensor.hpp>


std::ostream& operator << (std::ostream& os, Tensor x);
void print_structure(Tensor x);

#endif