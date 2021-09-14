# dawnbench inference cifar10
## run inference task on cifar10

The following instructions show how to achieve the performance that we submitted to DAWNBench step by step.
1. Install dependencies
```
	sudo apt install libopencv-dev
```
2. Clone this repo and prepare test images
```
	unzip images.zip
```
3. Compile
```
	cmake .
	make
```
4. Run
```	
	./cifar10_infer
``` 
The top-1 accuracy should be about 94.55%. The average running time is 0.073 milliseconds for each image.