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

template <int K>
__global__ void pack_active_metadata(const int32_t* __restrict__ slots,
                                     const float* __restrict__ weights,
                                     int32_t* __restrict__ mapped_slots,
                                     float* __restrict__ mapped_weights,
                                     const int32_t* __restrict__ active_count) {
    const int route = threadIdx.x;
    if (route < active_count[0]) {
        mapped_slots[route] = slots[route];
        mapped_weights[route] = weights[route];
    }
}

template <int K>
__global__ void unpack_active_metadata(int32_t* __restrict__ slots,
                                       float* __restrict__ weights,
                                       const int32_t* __restrict__ mapped_slots,
                                       const float* __restrict__ mapped_weights,
                                       const int32_t* __restrict__ active_count) {
    const int route = threadIdx.x;
    if (route < active_count[0]) {
        slots[route] = mapped_slots[route];
        weights[route] = mapped_weights[route];
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

    static void pack_metadata(tvm::ffi::TensorView mapped_slots,
                              tvm::ffi::TensorView mapped_weights,
                              tvm::ffi::TensorView slots,
                              tvm::ffi::TensorView weights,
                              tvm::ffi::TensorView active_count) {
        using namespace host;
        auto device = SymbolicDevice{};
        TensorMatcher({1, K}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(slots);
        TensorMatcher({1, K}).with_dtype<float>().with_device<kDLCUDA>(device).verify(weights);
        TensorMatcher({1, K}).with_dtype<int32_t>().with_device<kDLCUDAHost, kDLCPU>().verify(mapped_slots);
        TensorMatcher({1, K}).with_dtype<float>().with_device<kDLCUDAHost, kDLCPU>().verify(mapped_weights);
        TensorMatcher({}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(active_count);
        LaunchKernel(1, K, device.unwrap())(
            pack_active_metadata<K>, static_cast<const int32_t*>(slots.data_ptr()),
            static_cast<const float*>(weights.data_ptr()),
            static_cast<int32_t*>(mapped_ptr(mapped_slots)),
            static_cast<float*>(mapped_ptr(mapped_weights)),
            static_cast<const int32_t*>(active_count.data_ptr()));
    }

    static void unpack_metadata(tvm::ffi::TensorView slots,
                                tvm::ffi::TensorView weights,
                                tvm::ffi::TensorView mapped_slots,
                                tvm::ffi::TensorView mapped_weights,
                                tvm::ffi::TensorView active_count) {
        using namespace host;
        auto device = SymbolicDevice{};
        TensorMatcher({1, K}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(slots);
        TensorMatcher({1, K}).with_dtype<float>().with_device<kDLCUDA>(device).verify(weights);
        TensorMatcher({1, K}).with_dtype<int32_t>().with_device<kDLCUDAHost, kDLCPU>().verify(mapped_slots);
        TensorMatcher({1, K}).with_dtype<float>().with_device<kDLCUDAHost, kDLCPU>().verify(mapped_weights);
        TensorMatcher({}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(active_count);
        LaunchKernel(1, K, device.unwrap())(
            unpack_active_metadata<K>, static_cast<int32_t*>(slots.data_ptr()),
            static_cast<float*>(weights.data_ptr()),
            static_cast<const int32_t*>(mapped_ptr(mapped_slots)),
            static_cast<const float*>(mapped_ptr(mapped_weights)),
            static_cast<const int32_t*>(active_count.data_ptr()));
    }
};

}  // namespace d6_transport
