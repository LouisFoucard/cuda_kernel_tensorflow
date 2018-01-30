TF_INC = $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB = $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

kernel = matmul_smem_kernel

bin/$(kernel).so : $(kernel).cc obj/$(kernel).cu.o
		g++ -std=c++11 -shared -o bin/$(kernel).so $(kernel).cc obj/$(kernel).cu.o -I $(TF_INC) -I$(TF_INC)/external/nsync/public -fPIC -L$(TF_LIB) -ltensorflow_framework
obj/$(kernel).cu.o : $(kernel).cu.cc
		    nvcc -std=c++11 -o obj/$(kernel).cu.o -c $(kernel).cu.cc -I $(TF_INC) -I$(TF_INC)/external/nsync/public -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr --gpu-architecture=sm_61

clean:
	rm obj/*.o bin/*.so
