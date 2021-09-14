#pragma once

typedef unsigned char uchar;

namespace pqvm {

class TensorBase
{
public:
	virtual void   create(int w, int h, int c) = 0;
	virtual void   set(int c, int y, int x, float v) = 0;
	void* qptr;
};

template<typename T>
class Tensor : public TensorBase
{
public:
	Tensor();
	Tensor(int w, int h, int c);
	~Tensor();

	virtual void   create(int w, int h, int c);
	virtual void   set(int c, int y, int x, float v);
};


}


