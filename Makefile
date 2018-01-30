TF_INC = $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB = $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

kernel = matmul_smem_kernel

$(kernel).so : $(kernel).cc $(kernel).cu.o
		g++ -std=c++11 -shared -o $(kernel).so $(kernel).cc $(kernel).cu.o -I $(TF_INC) -I$(TF_INC)/external/nsync/public -fPIC -L$(TF_LIB) -ltensorflow_framework
$(kernel).cu.o : $(kernel).cu.cc
		    nvcc -std=c++11 -c $(kernel).cu.cc -I $(TF_INC) -I$(TF_INC)/external/nsync/public -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr --gpu-architecture=sm_61

clean:
	rm *.o *.so
