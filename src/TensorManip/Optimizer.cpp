#include <TensorManip/Optimizer.hpp>
#include <DebugAssist/DebugAssist.hpp>
#include <math.h>


GradientDescent::GradientDescent(float lr) {
	learning_rate = lr;
}
float GradientDescent::get_learning_rate() const { return learning_rate; }
void GradientDescent::set_learning_rate(float lr) { learning_rate = lr; }

void GradientDescent::add_parameter(Tensor ts) { parameters.push_back(ts); }
void GradientDescent::add_parameter(std::vector<Tensor> ts) {
	for (Tensor i : ts)
		add_parameter(i);
}

void GradientDescent::step() {
	for (auto i : parameters) {
		StaticFloatVector& A = *(i.A());
		StaticFloatVector& gA = *(i.gA());

		A -= gA * learning_rate;
	}
	zero_gradient();
}
void GradientDescent::zero_gradient() {
	for (auto i : parameters) {
		StaticFloatVector& gA = *(i.gA());
		gA *= 0;
	}
}


GradientMomentum::GradientMomentum(float lr, float gl) {
	learning_rate = lr;
	glide = gl;
	if (glide > 1 || glide < 0) throw_error("In Momentum Optimizer Initilization: Glide not in range [0, 1]");
}
float GradientMomentum::get_glide() const { return glide; }
void GradientMomentum::set_glide(float gl) { glide = gl; }

void GradientMomentum::step() {
	for (auto i : parameters) {
		StaticFloatVector& A = *(i.A());
		StaticFloatVector& gA = *(i.gA());

		A -= gA * learning_rate;
	}
	multiply_gradient();
}
void GradientMomentum::multiply_gradient() {
	for (auto i : parameters) {
		StaticFloatVector& gA = *(i.gA());
		gA *= glide;
	}
}


DecayDescent::DecayDescent(float lr, float gl, float dc) {
	learning_rate = lr;
	glide = gl;
	decay = dc;	
	if (glide > 1 || glide < 0)  throw_error("In Decaying Descent Optimizer Initilization: Glide not in range [0, 1]");
	if (decay > 1 || decay < 0) throw_error("In Decaying Descent Optimizer Initilization: Decay not in range [0, 1]");
}
float DecayDescent::get_decay() const { return decay; }
void DecayDescent::set_decay(float dc) { decay = dc; }

void DecayDescent::decay_learning_rate() {
	learning_rate *= decay;
}


const float EPS = 1e-8;

AdaGrad::AdaGrad(float lr) {
	learning_rate = lr;
}
void AdaGrad::add_parameter(Tensor ts) { 
	parameters.push_back(ts); 
	G.push_back(std::make_shared<StaticFloatVector>(ts.size()));
}
void AdaGrad::add_parameter(std::vector<Tensor> ts) {
	for (Tensor i : ts)
		add_parameter(i);
}
void AdaGrad::step() {
	for(int i = 0; i < (int) parameters.size(); ++i) {
		StaticFloatVector& A = *(parameters[i].A());
		StaticFloatVector& gA = *(parameters[i].gA());

		StaticFloatVector& curG = *G[i];
		for (int j = 0; j < curG.size(); ++j) {
			curG[j] += gA[j] * gA[j];
			A[j] -= gA[j] * learning_rate / sqrt(curG[j] + EPS);
		}
	}
	zero_gradient();
}

RMSProp::RMSProp(float lr, float dc) {
	learning_rate = lr;
	decay_rate = dc;
	if (decay_rate > 1 || decay_rate < 0) throw_error("In RMSProp Initilization: Decay not in range [0, 1]");
}

void RMSProp::step() {
	for (int i = 0; i < (int)parameters.size(); ++i) {
		StaticFloatVector& A = *(parameters[i].A());
		StaticFloatVector& gA = *(parameters[i].gA());

		StaticFloatVector& curG = *G[i];
		for (int j = 0; j < curG.size(); ++j) {
			curG[j] = curG[j] * decay_rate + gA[j] * gA[j] * (1 - decay_rate);
			A[j] -= gA[j] * learning_rate / sqrt(curG[j] + EPS);
		}
	}
	zero_gradient();
}


Adam::Adam(float lr, float gl, float dc) {
	learning_rate = lr;
	glide = gl;
	decay_rate = dc;
	step_count = 0;
}
void Adam::add_parameter(Tensor ts) {
	parameters.push_back(ts);
	G.push_back(std::make_shared<StaticFloatVector>(ts.size()));
	V.push_back(std::make_shared<StaticFloatVector>(ts.size()));
}
void Adam::add_parameter(std::vector<Tensor> ts) {
	for (Tensor i : ts)
		add_parameter(i);
}
void Adam::step() {
	step_count++;
	float glide_bias = 1 - pow(glide, step_count);
	float decay_bias = 1 - pow(decay_rate, step_count);
	for (int i = 0; i < (int)parameters.size(); ++i) {
		StaticFloatVector& A = *(parameters[i].A());
		StaticFloatVector& gA = *(parameters[i].gA());

		StaticFloatVector& curG = *G[i];
		StaticFloatVector& curV = *V[i];
		int sz = curG.size();
		for (int j = 0; j < sz; ++j) {
			curG[j] = curG[j] * decay_rate + gA[j] * gA[j] * (1 - decay_rate);
			curV[j] = curV[j] * glide + gA[j] * (1 - glide);

			float momentum = curV[j] / glide_bias;
			float suppressor = curG[j] / decay_bias;
			A[j] -= momentum * learning_rate / sqrt(suppressor + EPS);
		}
	}
	zero_gradient();
}