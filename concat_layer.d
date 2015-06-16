concat_layer.o: src/caffe/layers/concat_layer.cu include/caffe/layer.hpp \
 include/caffe/blob.hpp include/caffe/common.hpp \
 include/caffe/util/device_alternate.hpp \
 include/caffe/greentea/greentea.hpp /opt/AMDAPPSDK-2.9-1/include/CL/cl.h \
 /opt/AMDAPPSDK-2.9-1/include/CL/cl_platform.h \
 ../ViennaCL/viennacl/ocl/context.hpp ../ViennaCL/viennacl/ocl/forwards.h \
 ../ViennaCL/viennacl/ocl/handle.hpp ../ViennaCL/viennacl/ocl/error.hpp \
 ../ViennaCL/viennacl/ocl/kernel.hpp ../ViennaCL/viennacl/ocl/program.hpp \
 ../ViennaCL/viennacl/tools/shared_ptr.hpp \
 ../ViennaCL/viennacl/ocl/device.hpp \
 ../ViennaCL/viennacl/ocl/device_utils.hpp \
 ../ViennaCL/viennacl/forwards.h ../ViennaCL/viennacl/meta/enable_if.hpp \
 ../ViennaCL/viennacl/version.hpp ../ViennaCL/viennacl/ocl/local_mem.hpp \
 ../ViennaCL/viennacl/ocl/platform.hpp \
 ../ViennaCL/viennacl/ocl/command_queue.hpp \
 ../ViennaCL/viennacl/tools/sha1.hpp ../ViennaCL/viennacl/ocl/backend.hpp \
 ../ViennaCL/viennacl/ocl/enqueue.hpp \
 ../ViennaCL/viennacl/backend/opencl.hpp ../ViennaCL/viennacl/vector.hpp \
 ../ViennaCL/viennacl/detail/vector_def.hpp \
 ../ViennaCL/viennacl/tools/entry_proxy.hpp \
 ../ViennaCL/viennacl/scalar.hpp ../ViennaCL/viennacl/backend/memory.hpp \
 ../ViennaCL/viennacl/backend/mem_handle.hpp \
 ../ViennaCL/viennacl/backend/cpu_ram.hpp \
 ../ViennaCL/viennacl/context.hpp ../ViennaCL/viennacl/traits/handle.hpp \
 ../ViennaCL/viennacl/traits/context.hpp \
 ../ViennaCL/viennacl/backend/util.hpp \
 ../ViennaCL/viennacl/meta/result_of.hpp \
 ../ViennaCL/viennacl/linalg/scalar_operations.hpp \
 ../ViennaCL/viennacl/tools/tools.hpp \
 ../ViennaCL/viennacl/tools/adapter.hpp \
 ../ViennaCL/viennacl/meta/predicate.hpp \
 ../ViennaCL/viennacl/traits/size.hpp \
 ../ViennaCL/viennacl/traits/start.hpp \
 ../ViennaCL/viennacl/traits/stride.hpp \
 ../ViennaCL/viennacl/linalg/host_based/scalar_operations.hpp \
 ../ViennaCL/viennacl/linalg/host_based/common.hpp \
 ../ViennaCL/viennacl/linalg/opencl/scalar_operations.hpp \
 ../ViennaCL/viennacl/linalg/opencl/kernels/scalar.hpp \
 ../ViennaCL/viennacl/ocl/utils.hpp \
 ../ViennaCL/viennacl/linalg/opencl/common.hpp \
 ../ViennaCL/viennacl/linalg/opencl/kernels/matrix.hpp \
 ../ViennaCL/viennacl/scheduler/preset.hpp \
 ../ViennaCL/viennacl/device_specific/forwards.h \
 ../ViennaCL/viennacl/scheduler/io.hpp \
 ../ViennaCL/viennacl/scheduler/forwards.h \
 ../ViennaCL/viennacl/device_specific/execution_handler.hpp \
 ../ViennaCL/viennacl/device_specific/lazy_program_compiler.hpp \
 ../ViennaCL/viennacl/device_specific/templates/template_base.hpp \
 ../ViennaCL/viennacl/device_specific/mapped_objects.hpp \
 ../ViennaCL/viennacl/device_specific/utils.hpp \
 ../ViennaCL/viennacl/detail/matrix_def.hpp \
 ../ViennaCL/viennacl/traits/row_major.hpp \
 ../ViennaCL/viennacl/device_specific/tree_parsing.hpp \
 ../ViennaCL/viennacl/device_specific/builtin_database/vector_axpy.hpp \
 ../ViennaCL/viennacl/device_specific/builtin_database/devices/gpu/amd/northern_islands/barts.hpp \
 ../ViennaCL/viennacl/device_specific/templates/matrix_product_template.hpp \
 ../ViennaCL/viennacl/matrix_proxy.hpp ../ViennaCL/viennacl/range.hpp \
 ../ViennaCL/viennacl/slice.hpp \
 ../ViennaCL/viennacl/device_specific/templates/row_wise_reduction_template.hpp \
 ../ViennaCL/viennacl/device_specific/templates/utils.hpp \
 ../ViennaCL/viennacl/device_specific/templates/matrix_axpy_template.hpp \
 ../ViennaCL/viennacl/device_specific/templates/reduction_template.hpp \
 ../ViennaCL/viennacl/device_specific/templates/vector_axpy_template.hpp \
 ../ViennaCL/viennacl/device_specific/builtin_database/common.hpp \
 ../ViennaCL/viennacl/device_specific/builtin_database/devices/gpu/nvidia/fermi/tesla_c2050.hpp \
 ../ViennaCL/viennacl/device_specific/builtin_database/devices/gpu/nvidia/fermi/geforce_gtx_470.hpp \
 ../ViennaCL/viennacl/device_specific/builtin_database/devices/gpu/nvidia/maxwell/geforce_gtx_750_ti.hpp \
 ../ViennaCL/viennacl/device_specific/builtin_database/devices/gpu/amd/northern_islands/scrapper.hpp \
 ../ViennaCL/viennacl/device_specific/builtin_database/devices/gpu/nvidia/tesla/geforce_gtx_260.hpp \
 ../ViennaCL/viennacl/device_specific/builtin_database/devices/gpu/amd/southern_islands/tahiti.hpp \
 ../ViennaCL/viennacl/device_specific/builtin_database/devices/gpu/amd/northern_islands/devastator.hpp \
 ../ViennaCL/viennacl/device_specific/builtin_database/devices/gpu/nvidia/kepler/tesla_k20m.hpp \
 ../ViennaCL/viennacl/device_specific/builtin_database/devices/gpu/nvidia/fermi/geforce_gtx_580.hpp \
 ../ViennaCL/viennacl/device_specific/builtin_database/devices/gpu/amd/volcanic_islands/hawaii.hpp \
 ../ViennaCL/viennacl/device_specific/builtin_database/devices/gpu/amd/evergreen/cypress.hpp \
 ../ViennaCL/viennacl/device_specific/builtin_database/devices/gpu/amd/evergreen/cedar.hpp \
 ../ViennaCL/viennacl/device_specific/builtin_database/devices/gpu/nvidia/fermi/geforce_gt_540m.hpp \
 ../ViennaCL/viennacl/device_specific/builtin_database/devices/accelerator/fallback.hpp \
 ../ViennaCL/viennacl/device_specific/builtin_database/devices/cpu/fallback.hpp \
 ../ViennaCL/viennacl/device_specific/builtin_database/devices/gpu/fallback.hpp \
 ../ViennaCL/viennacl/device_specific/builtin_database/matrix_axpy.hpp \
 ../ViennaCL/viennacl/device_specific/builtin_database/row_wise_reduction.hpp \
 ../ViennaCL/viennacl/device_specific/builtin_database/matrix_product.hpp \
 ../ViennaCL/viennacl/linalg/detail/op_executor.hpp \
 ../ViennaCL/viennacl/linalg/vector_operations.hpp \
 ../ViennaCL/viennacl/linalg/host_based/vector_operations.hpp \
 ../ViennaCL/viennacl/linalg/detail/op_applier.hpp \
 ../ViennaCL/viennacl/linalg/opencl/vector_operations.hpp \
 ../ViennaCL/viennacl/linalg/opencl/kernels/vector.hpp \
 ../ViennaCL/viennacl/vector_proxy.hpp \
 ../ViennaCL/viennacl/device_specific/builtin_database/reduction.hpp \
 .build_release/src/caffe/proto/caffe.pb.h include/caffe/syncedmem.hpp \
 include/caffe/util/math_functions.hpp \
 include/caffe/util/mkl_alternate.hpp \
 include/caffe/greentea/greentea_math_functions.hpp \
 include/caffe/layer_factory.hpp include/caffe/vision_layers.hpp \
 include/caffe/common_layers.hpp include/caffe/data_layers.hpp \
 include/caffe/data_transformer.hpp include/caffe/filler.hpp \
 include/caffe/internal_thread.hpp include/caffe/util/db.hpp \
 include/caffe/loss_layers.hpp include/caffe/neuron_layers.hpp \
 include/caffe/greentea/greentea_im2col.hpp

include/caffe/layer.hpp:

include/caffe/blob.hpp:

include/caffe/common.hpp:

include/caffe/util/device_alternate.hpp:

include/caffe/greentea/greentea.hpp:

/opt/AMDAPPSDK-2.9-1/include/CL/cl.h:

/opt/AMDAPPSDK-2.9-1/include/CL/cl_platform.h:

../ViennaCL/viennacl/ocl/context.hpp:

../ViennaCL/viennacl/ocl/forwards.h:

../ViennaCL/viennacl/ocl/handle.hpp:

../ViennaCL/viennacl/ocl/error.hpp:

../ViennaCL/viennacl/ocl/kernel.hpp:

../ViennaCL/viennacl/ocl/program.hpp:

../ViennaCL/viennacl/tools/shared_ptr.hpp:

../ViennaCL/viennacl/ocl/device.hpp:

../ViennaCL/viennacl/ocl/device_utils.hpp:

../ViennaCL/viennacl/forwards.h:

../ViennaCL/viennacl/meta/enable_if.hpp:

../ViennaCL/viennacl/version.hpp:

../ViennaCL/viennacl/ocl/local_mem.hpp:

../ViennaCL/viennacl/ocl/platform.hpp:

../ViennaCL/viennacl/ocl/command_queue.hpp:

../ViennaCL/viennacl/tools/sha1.hpp:

../ViennaCL/viennacl/ocl/backend.hpp:

../ViennaCL/viennacl/ocl/enqueue.hpp:

../ViennaCL/viennacl/backend/opencl.hpp:

../ViennaCL/viennacl/vector.hpp:

../ViennaCL/viennacl/detail/vector_def.hpp:

../ViennaCL/viennacl/tools/entry_proxy.hpp:

../ViennaCL/viennacl/scalar.hpp:

../ViennaCL/viennacl/backend/memory.hpp:

../ViennaCL/viennacl/backend/mem_handle.hpp:

../ViennaCL/viennacl/backend/cpu_ram.hpp:

../ViennaCL/viennacl/context.hpp:

../ViennaCL/viennacl/traits/handle.hpp:

../ViennaCL/viennacl/traits/context.hpp:

../ViennaCL/viennacl/backend/util.hpp:

../ViennaCL/viennacl/meta/result_of.hpp:

../ViennaCL/viennacl/linalg/scalar_operations.hpp:

../ViennaCL/viennacl/tools/tools.hpp:

../ViennaCL/viennacl/tools/adapter.hpp:

../ViennaCL/viennacl/meta/predicate.hpp:

../ViennaCL/viennacl/traits/size.hpp:

../ViennaCL/viennacl/traits/start.hpp:

../ViennaCL/viennacl/traits/stride.hpp:

../ViennaCL/viennacl/linalg/host_based/scalar_operations.hpp:

../ViennaCL/viennacl/linalg/host_based/common.hpp:

../ViennaCL/viennacl/linalg/opencl/scalar_operations.hpp:

../ViennaCL/viennacl/linalg/opencl/kernels/scalar.hpp:

../ViennaCL/viennacl/ocl/utils.hpp:

../ViennaCL/viennacl/linalg/opencl/common.hpp:

../ViennaCL/viennacl/linalg/opencl/kernels/matrix.hpp:

../ViennaCL/viennacl/scheduler/preset.hpp:

../ViennaCL/viennacl/device_specific/forwards.h:

../ViennaCL/viennacl/scheduler/io.hpp:

../ViennaCL/viennacl/scheduler/forwards.h:

../ViennaCL/viennacl/device_specific/execution_handler.hpp:

../ViennaCL/viennacl/device_specific/lazy_program_compiler.hpp:

../ViennaCL/viennacl/device_specific/templates/template_base.hpp:

../ViennaCL/viennacl/device_specific/mapped_objects.hpp:

../ViennaCL/viennacl/device_specific/utils.hpp:

../ViennaCL/viennacl/detail/matrix_def.hpp:

../ViennaCL/viennacl/traits/row_major.hpp:

../ViennaCL/viennacl/device_specific/tree_parsing.hpp:

../ViennaCL/viennacl/device_specific/builtin_database/vector_axpy.hpp:

../ViennaCL/viennacl/device_specific/builtin_database/devices/gpu/amd/northern_islands/barts.hpp:

../ViennaCL/viennacl/device_specific/templates/matrix_product_template.hpp:

../ViennaCL/viennacl/matrix_proxy.hpp:

../ViennaCL/viennacl/range.hpp:

../ViennaCL/viennacl/slice.hpp:

../ViennaCL/viennacl/device_specific/templates/row_wise_reduction_template.hpp:

../ViennaCL/viennacl/device_specific/templates/utils.hpp:

../ViennaCL/viennacl/device_specific/templates/matrix_axpy_template.hpp:

../ViennaCL/viennacl/device_specific/templates/reduction_template.hpp:

../ViennaCL/viennacl/device_specific/templates/vector_axpy_template.hpp:

../ViennaCL/viennacl/device_specific/builtin_database/common.hpp:

../ViennaCL/viennacl/device_specific/builtin_database/devices/gpu/nvidia/fermi/tesla_c2050.hpp:

../ViennaCL/viennacl/device_specific/builtin_database/devices/gpu/nvidia/fermi/geforce_gtx_470.hpp:

../ViennaCL/viennacl/device_specific/builtin_database/devices/gpu/nvidia/maxwell/geforce_gtx_750_ti.hpp:

../ViennaCL/viennacl/device_specific/builtin_database/devices/gpu/amd/northern_islands/scrapper.hpp:

../ViennaCL/viennacl/device_specific/builtin_database/devices/gpu/nvidia/tesla/geforce_gtx_260.hpp:

../ViennaCL/viennacl/device_specific/builtin_database/devices/gpu/amd/southern_islands/tahiti.hpp:

../ViennaCL/viennacl/device_specific/builtin_database/devices/gpu/amd/northern_islands/devastator.hpp:

../ViennaCL/viennacl/device_specific/builtin_database/devices/gpu/nvidia/kepler/tesla_k20m.hpp:

../ViennaCL/viennacl/device_specific/builtin_database/devices/gpu/nvidia/fermi/geforce_gtx_580.hpp:

../ViennaCL/viennacl/device_specific/builtin_database/devices/gpu/amd/volcanic_islands/hawaii.hpp:

../ViennaCL/viennacl/device_specific/builtin_database/devices/gpu/amd/evergreen/cypress.hpp:

../ViennaCL/viennacl/device_specific/builtin_database/devices/gpu/amd/evergreen/cedar.hpp:

../ViennaCL/viennacl/device_specific/builtin_database/devices/gpu/nvidia/fermi/geforce_gt_540m.hpp:

../ViennaCL/viennacl/device_specific/builtin_database/devices/accelerator/fallback.hpp:

../ViennaCL/viennacl/device_specific/builtin_database/devices/cpu/fallback.hpp:

../ViennaCL/viennacl/device_specific/builtin_database/devices/gpu/fallback.hpp:

../ViennaCL/viennacl/device_specific/builtin_database/matrix_axpy.hpp:

../ViennaCL/viennacl/device_specific/builtin_database/row_wise_reduction.hpp:

../ViennaCL/viennacl/device_specific/builtin_database/matrix_product.hpp:

../ViennaCL/viennacl/linalg/detail/op_executor.hpp:

../ViennaCL/viennacl/linalg/vector_operations.hpp:

../ViennaCL/viennacl/linalg/host_based/vector_operations.hpp:

../ViennaCL/viennacl/linalg/detail/op_applier.hpp:

../ViennaCL/viennacl/linalg/opencl/vector_operations.hpp:

../ViennaCL/viennacl/linalg/opencl/kernels/vector.hpp:

../ViennaCL/viennacl/vector_proxy.hpp:

../ViennaCL/viennacl/device_specific/builtin_database/reduction.hpp:

.build_release/src/caffe/proto/caffe.pb.h:

include/caffe/syncedmem.hpp:

include/caffe/util/math_functions.hpp:

include/caffe/util/mkl_alternate.hpp:

include/caffe/greentea/greentea_math_functions.hpp:

include/caffe/layer_factory.hpp:

include/caffe/vision_layers.hpp:

include/caffe/common_layers.hpp:

include/caffe/data_layers.hpp:

include/caffe/data_transformer.hpp:

include/caffe/filler.hpp:

include/caffe/internal_thread.hpp:

include/caffe/util/db.hpp:

include/caffe/loss_layers.hpp:

include/caffe/neuron_layers.hpp:

include/caffe/greentea/greentea_im2col.hpp:
