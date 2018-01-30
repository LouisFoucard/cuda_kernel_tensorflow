// matmul_smem_kernel.cc
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
//#include "matmul_smem_kernel.h" //(for some reason some third party inclusions  (eigen, cuda) fail when
// in a header file, which why I declare the kernel launcher here.)

using namespace tensorflow;  // NOLINT(build/namespaces)

using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("MatMulSharedMem")\
.Attr("T: {float, int32}")\
.Input("input_a: T")\
.Input("input_b: T")\
.Output("output: T")\
.Doc(R"doc(
Cuda implementation of Matrix product, using shared memory.
)doc");

template <typename GPUDevice, typename T>
struct MatMulSharedMemKernelLauncher {
  void operator() (const GPUDevice& d, const T * A, const T * B, T * C, const int numARows,
                   const int numACols, const int numBRows, const int numBCols);
};

template <typename Device, typename T>
class MatMulOp : public OpKernel {
 public:
  explicit MatMulOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor_A = context->input(0);
    const Tensor& input_tensor_B = context->input(1);

    const TensorShape& input_A_shape = input_tensor_A.shape();
    const TensorShape& input_B_shape = input_tensor_B.shape();

    OP_REQUIRES(context, TensorShapeUtils::IsMatrixOrHigher(input_A_shape),
		errors::InvalidArgument(
					"input must be 2-dim, received shape: ",
					input_tensor_A.shape().DebugString()));

    OP_REQUIRES(context, TensorShapeUtils::IsMatrixOrHigher(input_B_shape),
		errors::InvalidArgument(
					"input must be 2-dim, received shape: ",
					input_tensor_B.shape().DebugString()));

    // get input dimensions
    const int numARows = input_tensor_A.matrix<T>().dimension(0);
    const int numACols = input_tensor_A.matrix<T>().dimension(1);
    const int numBRows = input_tensor_B.matrix<T>().dimension(0);
    const int numBCols = input_tensor_B.matrix<T>().dimension(1);

    OP_REQUIRES(context, numACols == numBRows,
		errors::InvalidArgument(
					"invalid matrix dimensions for multiplication, received matrices shapes: ",
                    input_tensor_A.shape().DebugString(), input_tensor_B.shape().DebugString()));

    //flatten out inputs
    auto input_A_reshaped = input_tensor_A.flat<T>();
    auto input_B_reshaped = input_tensor_B.flat<T>();
    
    // Create an output tensor
    TensorShape out_shape({numARows, numBCols});
    
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output_tensor));
    auto output = output_tensor->flat<T>();

    // Set all but the first element of the output tensor to 0.
    const int N = input_A_reshaped.size();
            
    // Call the cuda kernel launcher
    MatMulSharedMemKernelLauncher<Device, T>()(context->eigen_device<Device>(),
                                               input_A_reshaped.data(), input_B_reshaped.data(), output.data(),
                                               numARows, numACols, numBRows, numBCols);
    
  }
};

#define REGISTER_GPU(T) \
REGISTER_KERNEL_BUILDER(Name("MatMulSharedMem").Device(DEVICE_GPU).TypeConstraint<T>("T"), MatMulOp<GPUDevice, T>);
REGISTER_GPU(int32);
REGISTER_GPU(float);
