# Do `module load compiler/cuda/11.0/compilervars` on HPC to load the CUDA compiler.
all:nvcc -std=c++11 -o subtask4 subtask4.cu -arch=sm_35 -w -lboost_filesystem -lboost_system
	
run: ./subtask