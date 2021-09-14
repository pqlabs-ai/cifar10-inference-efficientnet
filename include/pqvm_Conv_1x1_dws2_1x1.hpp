#pragma once

#include "pqvm_Layer.hpp"
#include "pqvm_Model.hpp"

namespace pqvm   {

class Conv_1x1_dws2_1x1: public Layer
{
public:
	Conv_1x1_dws2_1x1(Model &x, int W, int H, int kSize, int C, int N_expand, int N);
};


}


