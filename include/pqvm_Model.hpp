#pragma once

#include "pqvm_Tensor.hpp"
#include "pqvm_Layer.hpp"

namespace pqvm {

class Model
{
public:
	Model();

	void operator+=(Layer *layer);

	void	 set_input(TensorBase *in);
	float* get_float_output();

	int		load_weights(const char* filename);

	void  begin_session();	
	void	run();
	void  end_session();

	void  compile();

	void* qptr;
};


}
