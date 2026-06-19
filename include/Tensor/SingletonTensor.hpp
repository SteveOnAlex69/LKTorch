#ifndef SINGLETON_TENSOR_HPP
#define SINGLETON_TENSOR_HPP

void activate_backprop();
void deactivate_backprop();
void toggle_backprop();
void set_backprop_status(bool f);
bool backprop_activated();


void activate_debug();
void deactivate_debug();
bool debug_activated();

#endif