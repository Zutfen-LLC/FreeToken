#include <freetoken/tensor.h>
#include <freetoken/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace d6_transport {

inline void* mapped_ptr(tvm::ffi::TensorView tensor) {
    if (tensor.device().device_type == kDLCUDA) return tensor.data_ptr();
    void* device = nullptr;
    const auto err = cudaHostGetDevicePointer(&device, tensor.data_ptr(), 0);
    host::RuntimeCheck(err == cudaSuccess,
        "D6 transport requires cudaHostAllocMapped storage: ", cudaGetErrorString(err));
    return device;
}

template <int K, int H>
__global__ void pack_active_bf16(const uint4* __restrict__ routes,
                                 uint4* __restrict__ mapped_host,
                                 const int32_t* __restrict__ active_count) {
    constexpr int vectors_per_route = H * 2 / sizeof(uint4);
    const int route = blockIdx.x;
    if (route >= active_count[0]) return;
    for (int col = threadIdx.x; col < vectors_per_route; col += blockDim.x) {
        mapped_host[route * vectors_per_route + col] = routes[route * vectors_per_route + col];
    }
}

template <int K, int H>
__global__ void scatter_active_bf16(const uint4* __restrict__ mapped_host,
                                    const int32_t* __restrict__ positions,
                                    const int32_t* __restrict__ active_count,
                                    uint4* __restrict__ reconstruction) {
    constexpr int vectors_per_route = H * 2 / sizeof(uint4);
    const int route = blockIdx.x;
    if (route >= active_count[0]) return;
    const int destination = positions[route];
    for (int col = threadIdx.x; col < vectors_per_route; col += blockDim.x) {
        reconstruction[destination * vectors_per_route + col] =
            mapped_host[route * vectors_per_route + col];
    }
}

template <int K, int H>
struct CountAwareTransport {
    static void verify_payload(tvm::ffi::TensorView payload) {
        using namespace host;
        auto dtype = SymbolicDType{};
        TensorMatcher({1, K, H}).with_dtype(dtype)
            .with_device<kDLCUDA, kDLCUDAHost, kDLCPU>().verify(payload);
        RuntimeCheck(dtype.unwrap().bits == 16 && dtype.unwrap().lanes == 1,
                     "D6 transport requires a 16-bit route contribution dtype");
    }

    static void pack(tvm::ffi::TensorView mapped_host,
                     tvm::ffi::TensorView routes,
                     tvm::ffi::TensorView active_count) {
        using namespace host;
        verify_payload(mapped_host); verify_payload(routes);
        auto device = SymbolicDevice{};
        TensorMatcher({1, K, H}).with_device<kDLCUDA>(device).verify(routes);
        TensorMatcher({}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(active_count);
        LaunchKernel(K, 256, device.unwrap())(
            pack_active_bf16<K, H>,
            static_cast<const uint4*>(routes.data_ptr()),
            static_cast<uint4*>(mapped_ptr(mapped_host)),
            static_cast<const int32_t*>(active_count.data_ptr()));
    }

    static void scatter(tvm::ffi::TensorView reconstruction,
                        tvm::ffi::TensorView mapped_host,
                        tvm::ffi::TensorView positions,
                        tvm::ffi::TensorView active_count) {
        using namespace host;
        verify_payload(reconstruction); verify_payload(mapped_host);
        auto device = SymbolicDevice{};
        TensorMatcher({1, K, H}).with_device<kDLCUDA>(device).verify(reconstruction);
        TensorMatcher({1, K}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(positions);
        TensorMatcher({}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(active_count);
        LaunchKernel(K, 256, device.unwrap())(
            scatter_active_bf16<K, H>,
            static_cast<const uint4*>(mapped_ptr(mapped_host)),
            static_cast<const int32_t*>(positions.data_ptr()),
            static_cast<const int32_t*>(active_count.data_ptr()),
            static_cast<uint4*>(reconstruction.data_ptr()));
    }
};

}  // namespace d6_transport
