// This file contains functions that may be shared across different benchmark
// libraries, such as stream creation or error-checking macros. This should not
// be confused with the standard gpu_utilities library, which creates new
// processes to query the GPU. These functions assume a CUDA context has
// already been created!
#ifndef BENCHMARK_GPU_UTILITIES_H
#define BENCHMARK_GPU_UTILITIES_H
#ifdef __cplusplus
extern "C" {
#endif
#include <cuda_runtime.h>
#include <stdint.h>
#include "library_interface.h"

// This macro takes a cudaError_t value. It prints an error message and returns
// 0 if the cudaError_t isn't cudaSuccess. Otherwise, it returns nonzero.
#define CheckCUDAError(val) (InternalCUDAErrorCheck((val), #val, __FILE__, __LINE__))

// Prints an error message and returns 0 if the given CUDA result is an error.
// This is intended to be used only via the CheckCUDAError macro.
int InternalCUDAErrorCheck(cudaError_t result, const char *fn,
  const char *file, int line);

// This function creates a CUDA stream with the given stream priority, or, if
// the given stream_priority isn't a valid stream, it will create a stream with
// the default priority. This populates the cudaStream_t pointed to by the
// stream parameter. This will return cudaSuccess on success and a different
// value if an error occurs. Always creates a nonblocking stream. If libsmctrl
// is built-in, this function will also apply the provided TPC mask. sm_mask is
// ignored otherwise.
cudaError_t CreateCUDAStreamWithPriorityAndMask(int stream_priority,
    uint64_t sm_mask, cudaStream_t *stream);

// This returns the current CPU time, in seconds. This time will correspond
// to the CPU times obtained by runner.c.
double CurrentSeconds(void);

// Returns the value of CUDA's global nanosecond timer. This is more convoluted
// than should be necessary due to a bug in this register on the Jetson boards.
__device__ inline uint64_t GlobalTimer64(void) {
  // Due to a bug in CUDA's 64-bit globaltimer, the lower 32 bits can wrap
  // around after the upper bits have already been read. Work around this by
  // reading the high bits a second time. Use the second value to detect a
  // rollover, and set the lower bits of the 64-bit "timer reading" to 0, which
  // would be valid, it's passed over during the duration of the reading. If no
  // rollover occurred, just return the initial reading.
  volatile uint64_t first_reading;
  volatile uint32_t second_reading;
  uint32_t high_bits_first;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(first_reading));
  high_bits_first = first_reading >> 32;
  asm volatile("mov.u32 %0, %%globaltimer_hi;" : "=r"(second_reading));
  if (high_bits_first == second_reading) {
    return first_reading;
  }
  // Return the value with the updated high bits, but the low bits set to 0.
  return ((uint64_t) second_reading) << 32;
}

// Returns the ID of the SM this is executed on.
static __device__ __inline__ uint32_t GetSMID(void) {
  uint32_t to_return;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(to_return));
  return to_return;
}

// A convenience function that copies the first dimension of 3-dimensional
// block and grid dimensions from the plugin's parameters. In other words, sets
// thread_count to block_dim[0] and block_count to grid_dim[0]. Returns 0 if
// any entry in block_dim or grid_dim other than the first has been set to a
// value other than 1.  Returns 1 on success.
int GetSingleBlockAndGridDimensions(InitializationParameters *params,
    int *thread_count, int *block_count);

// Like GetSingleBlockAndGridDimensions, but only checks and obtains the first
// dimension of params->block_dim.
int GetSingleBlockDimension(InitializationParameters *params,
    int *thread_count);

#ifdef __cplusplus
} // extern "C"
#endif
#endif  // BENCHMARK_GPU_UTILITIES_H

