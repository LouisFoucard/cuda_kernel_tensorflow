import tensorflow as tf
import numpy as np
import time

mod = tf.load_op_library('./matmul_smem_kernel.so')

MATRIX_SIZE = (16000, 16000)

with tf.Session() as sess:
    A = np.random.randint(10, size=MATRIX_SIZE).astype('float32')
    B = np.random.randint(10, size=MATRIX_SIZE).astype('float32')

    input_A = tf.constant(A)
    input_B = tf.constant(B)

    print("\n \n Matrix multiplication benchmark \n size of matrix: {}x{} \n starting .. \n".format(
        *MATRIX_SIZE))

    t0 = time.clock()
    mod.mat_mul_shared_mem(input_A, input_B).eval()
    t1 = time.clock()
    dt_cuda = t1 - t0
    print("tf cuda kernel time: {:.2f}s \n".format(dt_cuda))
    np_a = np.array(A)
    np_b = np.array(B)

    t2 = time.clock()
    np.matmul(np_a, np_b)
    t3 = time.clock()
    dt_np = t3 - t2
    print("numpy time: {:.2f}s \n".format(dt_np))

    print("{:.2f} fold improvement".format(dt_np / dt_cuda))
