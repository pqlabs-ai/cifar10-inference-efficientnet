#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <algorithm>
#include "pqvm_Model.hpp"
#include "pqvm_Layer.hpp"
#include "pqvm_Conv_3x3_dw___1x1.hpp"
#include "pqvm_Conv_1x1_dw___1x1_sc.hpp"
#include "pqvm_Conv_1x1_dws2_1x1.hpp"
#include "pqvm_Conv_1x1_avg_1x1.hpp"
#include "opencv2/opencv.hpp"

using namespace std::chrono;
using namespace pqvm;

static Tensor<uchar> images[10][1000];  // ten classes, 1000 images for each class
static Tensor<uchar> warmup_image(32,32,4);

void ReadImage(TensorBase* in, const char* filename)
{
	cv::Mat mat = cv::imread(filename, cv::IMREAD_COLOR);

	for(int i = 0; i < mat.rows; i++)
	{
			for(int j = 0; j < mat.cols; j++)
			{
					cv::Vec3b bgrPixel = mat.at<cv::Vec3b>(i, j);
					in->set(2, i, j, bgrPixel[0]);
					in->set(1, i, j, bgrPixel[1]);
					in->set(0, i, j, bgrPixel[2]);
					in->set(3, i, j, 0);
			}
	}
}

static void compile_network(Model &x)
{	
	typedef Layer* LayerPTR;
	LayerPTR stage_1n2, stage_3_1, stage_4_1, stage_5_1, stage_6_1, stage_7_1, stage_8_1, stage_9_1;

	x.set_input(&warmup_image);

	x += stage_1n2 = new Conv_3x3_dw___1x1   (x, 32,32, 5,   4,   48,  48);
	x += stage_3_1 = new Conv_1x1_dw___1x1_sc(x, 32,32, 5,  48,  144,  48);
	x += stage_4_1 = new Conv_1x1_dw___1x1_sc(x, 32,32, 5,  48,  144,  48);
	x += stage_5_1 = new Conv_1x1_dws2_1x1   (x, 32,32, 5,  48,  288,  96);
	x += stage_6_1 = new Conv_1x1_dw___1x1_sc(x, 16,16, 5,  96,  576, 128);
	x += stage_7_1 = new Conv_1x1_dws2_1x1   (x, 16,16, 5, 128,  768, 192);
	x += stage_8_1 = new Conv_1x1_dw___1x1_sc(x,  8, 8, 5, 192, 1152, 384);
	x += stage_9_1 = new Conv_1x1_avg_1x1    (x,  8, 8,		 384, 1280,  10);

	x.compile();

	printf("Loading weights\n");
	char* weightsfile = (char*)("efficientnet_pq_cifar_q+.weights");
	x.load_weights(weightsfile);
}

void cifar10_validate()
{	
	Model cnn;
	compile_network(cnn);

	char filename[256];
	int correct_cnt = 0;
	int total = 0;
	
	printf("Loading images ...\n");
	for (int k = 9; k >=0; k--)
	{
		for (int i = 1; i <= 1000; i++)
		{
			sprintf(filename, "%d/%04d.png", k, i);
			images[k][i-1].create(32,32,4);
			ReadImage(&images[k][i-1], filename);

			total++;
		}
	}
	printf("10000 images loaded\n\n");
	total = 0;
	cnn.begin_session();
	
	// Warm up
	for(int i=0; i<1000; i++) cnn.run();

	auto t0 = high_resolution_clock::now();

	for (int k = 0; k < 10; k++)
	{
		for (int i = 1; i <= 1000; i++)
		{
			cnn.set_input(&images[k][i-1]);
			cnn.run();

			float *p = cnn.get_float_output();
			int cls = std::max_element(p, p + 10) - p;
			if(cls==k) correct_cnt++;
			total++;
		}		
	}

	auto t1 = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(t1 - t0);

	cnn.end_session();

	printf("Top-1 accuracy: %.2f%% %ld us per sample\n\n", 100.0 * correct_cnt / total, duration.count() / total);
}

int main()
{
	cifar10_validate();
	return 0;
}