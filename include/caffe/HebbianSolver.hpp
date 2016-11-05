#pragma once

#include <string>
#include <vector>

#include "caffe/sgd_solvers.hpp"

namespace caffe {

	template <typename Dtype>
	class HebbianSolver :
		public SGDSolver<Dtype>
	{
	public:
		explicit HebbianSolver(const SolverParameter& param) : SGDSolver<Dtype>(param) {}
		explicit HebbianSolver(const string& param_file) : SGDSolver<Dtype>(param_file) {}

		virtual inline const char* type() const { return "Hebbian"; }

		DISABLE_COPY_AND_ASSIGN(HebbianSolver);
	};

}
