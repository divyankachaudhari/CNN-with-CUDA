all: subtask1 subtask2 run_python_script subtask3 subtask4


subtask1: src/assignment2_subtask1.cpp
	g++ -std=c++11 -o subtask1 src/assignment2_subtask1.cpp

subtask2: src/assignment2_subtask2.cu 
	nvcc -std=c++11 -o subtask2 src/assignment2_subtask2.cu -arch=sm_35

run_python_script:
	python3 preprocessing.py

subtask3: src/assignment2_subtask3.cu
	nvcc -std=c++11 -o subtask3 src/assignment2_subtask3.cu -arch=sm_35

subtask4: src/assignment2_subtask4.cu
	nvcc -std=c++11 -o subtask4 src/assignment2_subtask4.cu -arch=sm_35 -w -lboost_filesystem -lboost_system