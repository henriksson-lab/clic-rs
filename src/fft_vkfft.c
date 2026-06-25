#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifndef VKFFT_BACKEND
#define VKFFT_BACKEND 3
#endif

#ifndef VKFFT_MAX_FFT_DIMENSIONS
#define VKFFT_MAX_FFT_DIMENSIONS 4
#endif

#include "vkFFT.h"

typedef struct ClicVkfftPlan {
  VkFFTApplication app;
  cl_command_queue queue;
  cl_mem real_mem;
  cl_mem complex_mem;
  uint64_t real_size;
  uint64_t complex_size;
} ClicVkfftPlan;

static void clic_vkfft_configure(uint64_t width,
                                 uint64_t height,
                                 uint64_t depth,
                                 VkFFTConfiguration* configuration) {
  configuration->numberBatches = 1;
  configuration->size[0] = width;
  configuration->size[1] = height;
  configuration->size[2] = depth;
  configuration->FFTdim = 1;
  if (configuration->size[1] > 1) {
    configuration->FFTdim++;
  }
  if (configuration->size[2] > 1) {
    configuration->FFTdim++;
  }

  configuration->normalize = 1;
  configuration->performR2C = 1;
  configuration->performDCT = 0;
  configuration->isInputFormatted = 1;
  configuration->inverseReturnToInputBuffer = 1;
  configuration->inputBufferStride[0] = configuration->size[0];
  configuration->inputBufferStride[1] =
      configuration->inputBufferStride[0] * configuration->size[1];
  configuration->inputBufferStride[2] =
      configuration->inputBufferStride[1] * configuration->size[2];
}

static int clic_vkfft_create_plan(cl_device_id device,
                                  cl_context context,
                                  cl_command_queue queue,
                                  cl_mem real_mem,
                                  uint64_t real_size,
                                  cl_mem complex_mem,
                                  uint64_t complex_size,
                                  uint64_t width,
                                  uint64_t height,
                                  uint64_t depth,
                                  ClicVkfftPlan** plan_out) {
  if (plan_out == NULL) {
    return VKFFT_ERROR_EMPTY_FILE;
  }
  *plan_out = NULL;

  ClicVkfftPlan* plan = (ClicVkfftPlan*)calloc(1, sizeof(ClicVkfftPlan));
  if (plan == NULL) {
    return VKFFT_ERROR_MALLOC_FAILED;
  }
  plan->queue = queue;
  plan->real_mem = real_mem;
  plan->complex_mem = complex_mem;
  plan->real_size = real_size;
  plan->complex_size = complex_size;

  VkFFTConfiguration configuration = VKFFT_ZERO_INIT;
  clic_vkfft_configure(width, height, depth, &configuration);

  configuration.bufferSize = &plan->complex_size;
  configuration.inputBufferSize = &plan->real_size;
  configuration.buffer = &plan->complex_mem;
  configuration.inputBuffer = &plan->real_mem;
  configuration.outputBuffer = &plan->complex_mem;
  configuration.device = &device;
  configuration.context = &context;
  configuration.commandQueue = &plan->queue;

  VkFFTResult res = initializeVkFFT(&plan->app, configuration);
  if (res != VKFFT_SUCCESS) {
    free(plan);
    return (int)res;
  }

  *plan_out = plan;
  return VKFFT_SUCCESS;
}

int clic_vkfft_perform_fft(cl_device_id device,
                           cl_context context,
                           cl_command_queue queue,
                           cl_mem input_mem,
                           uint64_t input_size,
                           cl_mem output_mem,
                           uint64_t output_size,
                           uint64_t width,
                           uint64_t height,
                           uint64_t depth,
                           const uint8_t* load_application_string,
                           uint64_t load_application_string_len,
                           int save_application_to_string,
                           uint8_t** saved_application_string,
                           uint64_t* saved_application_string_len) {
  VkFFTConfiguration configuration = VKFFT_ZERO_INIT;
  clic_vkfft_configure(width, height, depth, &configuration);

  configuration.bufferSize = &output_size;
  configuration.inputBufferSize = &input_size;
  configuration.buffer = &output_mem;
  configuration.inputBuffer = &input_mem;
  configuration.outputBuffer = &output_mem;
  configuration.device = &device;
  configuration.context = &context;
  configuration.commandQueue = &queue;

  if (load_application_string != NULL && load_application_string_len > 0) {
    configuration.loadApplicationFromString = 1;
    configuration.saveApplicationToString = 0;
    configuration.loadApplicationString = (void*)load_application_string;
  } else if (save_application_to_string) {
    configuration.loadApplicationFromString = 0;
    configuration.saveApplicationToString = 1;
  }

  VkFFTApplication app = VKFFT_ZERO_INIT;
  VkFFTResult res = initializeVkFFT(&app, configuration);
  if (res != VKFFT_SUCCESS) {
    return (int)res;
  }

  if (saved_application_string != NULL && saved_application_string_len != NULL) {
    *saved_application_string = NULL;
    *saved_application_string_len = 0;
    if (save_application_to_string && app.saveApplicationString != NULL &&
        app.applicationStringSize > 0) {
      *saved_application_string = (uint8_t*)malloc(app.applicationStringSize);
      if (*saved_application_string == NULL) {
        deleteVkFFT(&app);
        return VKFFT_ERROR_MALLOC_FAILED;
      }
      memcpy(*saved_application_string, app.saveApplicationString, app.applicationStringSize);
      *saved_application_string_len = app.applicationStringSize;
    }
  }

  VkFFTLaunchParams launch_params = VKFFT_ZERO_INIT;
  launch_params.commandQueue = &queue;
  res = VkFFTAppend(&app, -1, &launch_params);
  deleteVkFFT(&app);
  return (int)res;
}

int clic_vkfft_perform_ifft(cl_device_id device,
                            cl_context context,
                            cl_command_queue queue,
                            cl_mem input_mem,
                            uint64_t input_size,
                            cl_mem output_mem,
                            uint64_t output_size,
                            uint64_t width,
                            uint64_t height,
                            uint64_t depth,
                            const uint8_t* load_application_string,
                            uint64_t load_application_string_len,
                            int save_application_to_string,
                            uint8_t** saved_application_string,
                            uint64_t* saved_application_string_len) {
  VkFFTConfiguration configuration = VKFFT_ZERO_INIT;
  clic_vkfft_configure(width, height, depth, &configuration);

  configuration.bufferSize = &input_size;
  configuration.inputBufferSize = &output_size;
  configuration.buffer = &input_mem;
  configuration.inputBuffer = &output_mem;
  configuration.outputBuffer = &input_mem;
  configuration.device = &device;
  configuration.context = &context;
  configuration.commandQueue = &queue;

  if (load_application_string != NULL && load_application_string_len > 0) {
    configuration.loadApplicationFromString = 1;
    configuration.saveApplicationToString = 0;
    configuration.loadApplicationString = (void*)load_application_string;
  } else if (save_application_to_string) {
    configuration.loadApplicationFromString = 0;
    configuration.saveApplicationToString = 1;
  }

  VkFFTApplication app = VKFFT_ZERO_INIT;
  VkFFTResult res = initializeVkFFT(&app, configuration);
  if (res != VKFFT_SUCCESS) {
    return (int)res;
  }

  if (saved_application_string != NULL && saved_application_string_len != NULL) {
    *saved_application_string = NULL;
    *saved_application_string_len = 0;
    if (save_application_to_string && app.saveApplicationString != NULL &&
        app.applicationStringSize > 0) {
      *saved_application_string = (uint8_t*)malloc(app.applicationStringSize);
      if (*saved_application_string == NULL) {
        deleteVkFFT(&app);
        return VKFFT_ERROR_MALLOC_FAILED;
      }
      memcpy(*saved_application_string, app.saveApplicationString, app.applicationStringSize);
      *saved_application_string_len = app.applicationStringSize;
    }
  }

  VkFFTLaunchParams launch_params = VKFFT_ZERO_INIT;
  launch_params.commandQueue = &queue;
  res = VkFFTAppend(&app, 1, &launch_params);
  deleteVkFFT(&app);
  return (int)res;
}

const char* clic_vkfft_error_string(int result) {
  return getVkFFTErrorString((VkFFTResult)result);
}

void clic_vkfft_free(void* ptr) {
  free(ptr);
}

int clic_vkfft_create_fft_plan(cl_device_id device,
                               cl_context context,
                               cl_command_queue queue,
                               cl_mem input_mem,
                               uint64_t input_size,
                               cl_mem output_mem,
                               uint64_t output_size,
                               uint64_t width,
                               uint64_t height,
                               uint64_t depth,
                               ClicVkfftPlan** plan_out) {
  return clic_vkfft_create_plan(device,
                                context,
                                queue,
                                input_mem,
                                input_size,
                                output_mem,
                                output_size,
                                width,
                                height,
                                depth,
                                plan_out);
}

int clic_vkfft_create_ifft_plan(cl_device_id device,
                                cl_context context,
                                cl_command_queue queue,
                                cl_mem input_mem,
                                uint64_t input_size,
                                cl_mem output_mem,
                                uint64_t output_size,
                                uint64_t width,
                                uint64_t height,
                                uint64_t depth,
                                ClicVkfftPlan** plan_out) {
  return clic_vkfft_create_plan(device,
                                context,
                                queue,
                                output_mem,
                                output_size,
                                input_mem,
                                input_size,
                                width,
                                height,
                                depth,
                                plan_out);
}

int clic_vkfft_append_fft(ClicVkfftPlan* plan, cl_mem input_mem) {
  if (plan == NULL) {
    return VKFFT_ERROR_EMPTY_FILE;
  }
  plan->real_mem = input_mem;
  VkFFTLaunchParams launch_params = VKFFT_ZERO_INIT;
  launch_params.commandQueue = &plan->queue;
  return (int)VkFFTAppend(&plan->app, -1, &launch_params);
}

int clic_vkfft_append_ifft(ClicVkfftPlan* plan) {
  if (plan == NULL) {
    return VKFFT_ERROR_EMPTY_FILE;
  }
  VkFFTLaunchParams launch_params = VKFFT_ZERO_INIT;
  launch_params.commandQueue = &plan->queue;
  return (int)VkFFTAppend(&plan->app, 1, &launch_params);
}

void clic_vkfft_delete_plan(ClicVkfftPlan* plan) {
  if (plan != NULL) {
    deleteVkFFT(&plan->app);
    free(plan);
  }
}
