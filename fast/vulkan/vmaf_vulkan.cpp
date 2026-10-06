// Derived from libvmaf (https://github.com/Netflix/vmaf), whose CUDA feature
// extractors are Copyright 2016-2023 Netflix, Inc. and Copyright 2021 NVIDIA
// Corporation, licensed under the BSD+Patent License
// (https://opensource.org/licenses/BSDplusPatent); see
// LICENSE at the root of this repository.
//
// VMAF's features on any GPU with Vulkan: VIF, ADM and motion, the three
// that VMAF v0.6.1 and VMAF NEG are predicted from.
//
// A port of libvmaf's CUDA feature extractors (libvmaf/src/feature/cuda:
// integer_vif_cuda.c, integer_adm_cuda.c, integer_motion_cuda.c and their
// kernels, at the commit and with the pull requests that
// README.md lists). The compute shaders in shaders/ give
// the integer sums the CUDA kernels give; the functions below marked "as
// libvmaf" turn them into feature scores with libvmaf's own expressions, in
// the same types and order, so a feature is the same double as CUDA's.
//
// What differs from the CUDA code is the structure, not the numbers:
//   - VMAF and VMAF NEG come from one pass over the frames. Their features
//     differ only in the enhancement gain limit (100 and 1): motion, VIF's
//     filters, ADM's wavelet transform and denominator are shared, and only
//     the limited parts are calculated per limit. libvmaf runs VIF and ADM
//     twice.
//   - CUDA's float and double steps are exact integer arithmetic on the GPU
//     (see shaders/common.slang).
//   - Sums are kept as pairs of 32-bit words (64-bit atomics are optional).
//
// The library scores features only. The caller predicts VMAF from them with
// libvmaf (vmaf_import_feature_score), which also keeps the model code one.
//
// Built by fast/scripts/build_vmaf_vulkan.ps1 with MSVC and the static C runtime,
// as the bundled libvmaf is: powf(), pow() and log2() below then are the
// same functions as in libvmaf's own conversion of the sums.
#define _USE_MATH_DEFINES
#define VK_NO_PROTOTYPES
#define VK_USE_PLATFORM_WIN32_KHR
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan.h>

#include "shaders_spv.h"
#include "v1_host.h"
#include "v1_speed.h"
#include "cambi_lut.h"

#define VV_EXPORT extern "C" __declspec(dllexport)

// ------------------------------------------------------------------ Vulkan

#define VK_INSTANCE_FUNCTIONS(X) \
    X(vkDestroyInstance) X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties) \
    X(vkGetPhysicalDeviceFeatures) X(vkGetPhysicalDeviceMemoryProperties) \
    X(vkGetPhysicalDeviceQueueFamilyProperties) X(vkCreateDevice) X(vkGetDeviceProcAddr) \
    X(vkEnumerateDeviceExtensionProperties) X(vkGetPhysicalDeviceProperties2)

#define VK_DEVICE_FUNCTIONS(X) \
    X(vkDestroyDevice) X(vkGetDeviceQueue) X(vkCreateBuffer) X(vkDestroyBuffer) \
    X(vkGetBufferMemoryRequirements) X(vkAllocateMemory) X(vkFreeMemory) X(vkBindBufferMemory) \
    X(vkMapMemory) X(vkCreateShaderModule) X(vkDestroyShaderModule) X(vkCreateDescriptorSetLayout) \
    X(vkDestroyDescriptorSetLayout) X(vkCreatePipelineLayout) X(vkDestroyPipelineLayout) \
    X(vkCreateComputePipelines) X(vkDestroyPipeline) X(vkCreateDescriptorPool) X(vkDestroyDescriptorPool) \
    X(vkAllocateDescriptorSets) X(vkUpdateDescriptorSets) X(vkCreateCommandPool) X(vkDestroyCommandPool) \
    X(vkAllocateCommandBuffers) X(vkBeginCommandBuffer) X(vkEndCommandBuffer) X(vkResetCommandBuffer) \
    X(vkCmdBindPipeline) X(vkCmdBindDescriptorSets) X(vkCmdPushConstants) X(vkCmdDispatch) X(vkCmdDispatchIndirect) \
    X(vkCmdPipelineBarrier) X(vkCmdCopyBuffer) X(vkCmdFillBuffer) X(vkCreateFence) X(vkDestroyFence) \
    X(vkWaitForFences) X(vkResetFences) X(vkQueueSubmit) X(vkDeviceWaitIdle) \
    X(vkCreateQueryPool) X(vkDestroyQueryPool) X(vkCmdResetQueryPool) X(vkCmdWriteTimestamp) \
    X(vkGetQueryPoolResults)

namespace {

struct InstanceApi {
    HMODULE library = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
#define X(name) PFN_##name name = nullptr;
    VK_INSTANCE_FUNCTIONS(X)
#undef X
};

struct DeviceApi {
#define X(name) PFN_##name name = nullptr;
    VK_DEVICE_FUNCTIONS(X)
#undef X
};

thread_local std::string g_error;

int fail(int code, const std::string &message)
{
    g_error = message;
    return code;
}

// One instance per process, made on first use and kept: vulkan-1.dll is
// loaded by the first call, never inside DllMain.
InstanceApi *instance_api()
{
    static InstanceApi api;
    static bool tried = false;
    if (tried)
        return api.instance ? &api : nullptr;
    tried = true;
    api.library = LoadLibraryW(L"vulkan-1.dll");
    if (!api.library) {
        g_error = "vulkan-1.dll was not found (no Vulkan driver)";
        return nullptr;
    }
    api.vkGetInstanceProcAddr =
        (PFN_vkGetInstanceProcAddr)GetProcAddress(api.library, "vkGetInstanceProcAddr");
    auto create = (PFN_vkCreateInstance)api.vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkCreateInstance");
    if (!create) {
        g_error = "vulkan-1.dll has no vkCreateInstance";
        return nullptr;
    }
    VkApplicationInfo application = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    application.pApplicationName = "libvmaf-fast";
    application.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo info = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    info.pApplicationInfo = &application;
    VkInstance instance;
    VkResult result = create(&info, nullptr, &instance);
    if (result != VK_SUCCESS) {
        g_error = "vkCreateInstance failed (" + std::to_string(result) + ")";
        return nullptr;
    }
#define X(name) api.name = (PFN_##name)api.vkGetInstanceProcAddr(instance, #name);
    VK_INSTANCE_FUNCTIONS(X)
#undef X
    api.instance = instance;
    return &api;
}

std::vector<VkPhysicalDevice> physical_devices(InstanceApi *api)
{
    uint32_t count = 0;
    api->vkEnumeratePhysicalDevices(api->instance, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    if (count)
        api->vkEnumeratePhysicalDevices(api->instance, &count, devices.data());
    devices.resize(count);
    return devices;
}

// --------------------------------------------------------------- features

enum { kScales = 4, kVifSums = 9 };
// The sums' slots (64 bits each) in the accumulator buffer.
enum {
    kSlotSad = 0,
    kSlotVif = 1,                               // [scale][kVifSums]
    kSlotCsfDen = kSlotVif + kScales * kVifSums, // [scale][band]
    kSlotCm = kSlotCsfDen + kScales * 3,         // [limit][scale][band]
    // VMAF v1's CAMBI (vv_v1_cambi), per scale: the top k's sum above its
    // k-th largest value (units of 2^-24), that value's bits, how many of it
    // are in the top k, and whether a value was not a whole number of units.
    kSlotCambi = kSlotCm + 2 * kScales * 3,
    kSlotCambiInvalid = kSlotCambi + V1_CAMBI_SCALES * 4,  // a sample above its bit depth's largest
    kSlots = kSlotCambiInvalid + 1
};

// What vv_features returns per frame.
enum {
    kVifScale0 = 0,      // 0..3: VIF per scale, gain limit 100 (VMAF)
    kAdm2 = 4,
    kMotion2 = 5,
    kVifScale0Neg = 6,   // 6..9: VIF per scale, gain limit 1 (VMAF NEG)
    kAdm2Neg = 10,
    kMotion = 11,
    kFeatures = 12
};

struct FrameSums {
    bool scored = false;
    uint64_t slots[kSlots] = {};
    double cambiPooled[V1_CAMBI_SCALES] = {};  // spatial_pooling on the CPU, where the GPU's top k was not exact
    unsigned cambiPooledScales = 0;           // which scales those are (bits)
};

// As libvmaf: write_scores() of integer_vif_cuda.c.
void vif_scores(const uint64_t *slots, int limit, double out[kScales])
{
    for (int scale = 0; scale < kScales; ++scale) {
        const int64_t *sums = (const int64_t *)&slots[kSlotVif + scale * kVifSums];
        const int64_t x = sums[0], num_x = sums[1], den_log = sums[2], num_non_log = sums[3],
                      den_non_log = sums[4], x2 = sums[5 + 2 * limit], num_log = sums[6 + 2 * limit];
        float num = num_log / 2048.0 + x2 + (den_non_log - ((num_non_log) / 16384.0) / (65025.0));
        float den = den_log / 2048.0 - (x + (num_x * 17)) + den_non_log;
        out[scale] = num / den;
    }
}

struct dwt_model_params {
    float a;
    float k;
    float f0;
    float g[4];
};

const dwt_model_params dwt_7_9_YCbCr_threshold[3] = {
    { 0.495f, 0.466f, 0.401f, { 1.501f, 1.0f, 0.534f, 1.0f } },
    { 1.633f, 0.353f, 0.209f, { 1.520f, 1.0f, 0.502f, 1.0f } },
    { 0.944f, 0.521f, 0.404f, { 1.868f, 1.0f, 0.516f, 1.0f } }
};

const float dwt_7_9_basis_function_amplitudes[6][4] = {
    { 0.62171, 0.67234, 0.72709, 0.67234 },
    { 0.34537, 0.41317, 0.49428, 0.41317 },
    { 0.18004, 0.22727, 0.28688, 0.22727 },
    { 0.091401, 0.11792, 0.15214, 0.11792 },
    { 0.045943, 0.059758, 0.077727, 0.059758 },
    { 0.023013, 0.030018, 0.039156, 0.030018 }
};

#define ADM_BORDER_FACTOR (0.1)
const double kAdmNormViewDist = 3.0;    // DEFAULT_ADM_NORM_VIEW_DIST
const int kAdmRefDisplayHeight = 1080;  // DEFAULT_ADM_REF_DISPLAY_HEIGHT

// As libvmaf: dwt_quant_step() of integer_adm_cuda.c.
float dwt_quant_step(const dwt_model_params *params, int lambda, int theta,
                     double adm_norm_view_dist, int adm_ref_display_height)
{
    float r = adm_norm_view_dist * adm_ref_display_height * M_PI / 180.0;
    float temp = log10(pow(2.0, lambda + 1) * params->f0 * params->g[theta] / r);
    float Q = 2.0 * params->a * pow(10.0, params->k * temp * temp) /
              dwt_7_9_basis_function_amplitudes[lambda][theta];
    return Q;
}

// As libvmaf: the i_rfactor table of integer_compute_adm_cuda(), for the
// default viewing distance and display height (the only ones VMAF's models
// use).
void adm_rfactors(uint32_t i_rfactor[12])
{
    const double pow2_32 = pow(2, 32);
    for (unsigned scale = 0; scale < 4; ++scale) {
        float factor1 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 1, kAdmNormViewDist, kAdmRefDisplayHeight);
        float factor2 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 2, kAdmNormViewDist, kAdmRefDisplayHeight);
        float rfactor[3] = { 1.0f / factor1, 1.0f / factor1, 1.0f / factor2 };
        if (scale == 0) {
            i_rfactor[0] = 36453;
            i_rfactor[1] = 36453;
            i_rfactor[2] = 49417;
        } else {
            i_rfactor[scale * 3] = (uint32_t)(rfactor[0] * pow2_32);
            i_rfactor[scale * 3 + 1] = (uint32_t)(rfactor[1] * pow2_32);
            i_rfactor[scale * 3 + 2] = (uint32_t)(rfactor[2] * pow2_32);
        }
    }
}

// As libvmaf: conclude_adm_cm() of integer_adm_cuda.c.
void conclude_adm_cm(const int64_t *accum, int h, int w, int scale, float *result)
{
    int left = w * ADM_BORDER_FACTOR - 0.5;
    int top = h * ADM_BORDER_FACTOR - 0.5;
    int right = w - left;
    int bottom = h - top;
    const uint32_t shift_inner_accum = (uint32_t)ceil(log2(h));

    const uint32_t shift_xcub[3] = { (uint32_t)ceil(log2(w) - 4), (uint32_t)ceil(log2(w) - 4),
                                     (uint32_t)ceil(log2(w) - 3) };
    int constant_offset[3] = { 52, 52, 57 };

    uint32_t shift_cub = (uint32_t)ceil(log2(w));
    float final_shift[3] = { powf(2, (45 - shift_cub - shift_inner_accum)),
                             powf(2, (39 - shift_cub - shift_inner_accum)),
                             powf(2, (36 - shift_cub - shift_inner_accum)) };
    float powf_add = powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);

    float f_accum;
    *result = 0;
    for (int i = 0; i < 3; ++i) {
        if (scale == 0) {
            f_accum = (float)(accum[i] / pow(2, (constant_offset[i] - shift_xcub[i] - shift_inner_accum)));
        } else {
            f_accum = (float)(accum[i] / final_shift[scale - 1]);
        }
        *result += powf(f_accum, 1.0f / 3.0f) + powf_add;
    }
}

// As libvmaf: conclude_adm_csf_den() of integer_adm_cuda.c.
void conclude_adm_csf_den(const uint64_t *accum, int h, int w, int scale, float *result,
                          float adm_norm_view_dist, float adm_ref_display_height)
{
    const int left = w * ADM_BORDER_FACTOR - 0.5;
    const int top = h * ADM_BORDER_FACTOR - 0.5;
    const int right = w - left;
    const int bottom = h - top;
    const float factor1 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 1, adm_norm_view_dist,
                                         adm_ref_display_height);
    const float factor2 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 2, adm_norm_view_dist,
                                         adm_ref_display_height);
    const float rfactor[3] = { 1.0f / factor1, 1.0f / factor1, 1.0f / factor2 };
    const uint32_t accum_convert_float[4] = { 18, 32, 27, 23 };

    int32_t shift_accum;
    double shift_csf;
    if (scale == 0) {
        shift_accum = (int32_t)ceil(log2((bottom - top) * (right - left)) - 20);
        shift_accum = shift_accum > 0 ? shift_accum : 0;
        shift_csf = pow(2, (accum_convert_float[scale] - shift_accum));
    } else {
        shift_accum = (int32_t)ceil(log2(bottom - top));
        const uint32_t shift_cub = (uint32_t)ceil(log2(right - left));
        shift_csf = pow(2, (accum_convert_float[scale] - shift_accum - shift_cub));
    }
    const float powf_add = powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);

    *result = 0;
    for (int i = 0; i < 3; ++i) {
        const double csf = (double)(accum[i] / shift_csf) * pow(rfactor[i], 3);
        *result += powf(csf, 1.0f / 3.0f) + powf_add;
    }
}

// As libvmaf: write_scores() of integer_adm_cuda.c.
double adm_score(const uint64_t *slots, int limit, unsigned w, unsigned h)
{
    double num = 0;
    double den = 0;
    const int64_t *adm_cm = (const int64_t *)&slots[kSlotCm + limit * kScales * 3];
    const uint64_t *adm_csf = &slots[kSlotCsfDen];
    float num_scale;
    float den_scale;
    for (unsigned scale = 0; scale < 4; ++scale) {
        w = (w + 1) / 2;
        h = (h + 1) / 2;
        conclude_adm_cm(&adm_cm[scale * 3], h, w, scale, &num_scale);
        conclude_adm_csf_den(&adm_csf[scale * 3], h, w, scale, &den_scale, kAdmNormViewDist,
                             kAdmRefDisplayHeight);
        num += num_scale;
        den += den_scale;
    }
    const double numden_limit = 1e-10 * (w * h) / (1920.0 * 1080.0);
    num = num < numden_limit ? 0 : num;
    den = den < numden_limit ? 0 : den;
    if (den == 0.0)
        return 1.0f;
    return num / den;
}

// As libvmaf: normalize_and_scale_sad() of integer_motion_cuda.c.
double normalize_and_scale_sad(uint64_t sad, unsigned w, unsigned h)
{
    return (float)(sad / 256.) / (w * h);
}

float float_from_bits(uint32_t bits)
{
    float value;
    memcpy(&value, &bits, sizeof value);
    return value;
}

// log_generate() of vif_statistics.cuh, (uint16_t)roundf(log2f(i) * 2048),
// as the GPU calculates it: CUDA's log2f() is the polynomial below (read from
// the kernel's PTX), which is not the host's log2f() to the last bit.
uint32_t cuda_log_generate(uint32_t i)
{
    const float value = (float)i;
    uint32_t bits;
    memcpy(&bits, &value, sizeof bits);
    const uint32_t exponent = (bits - 0x3F3504F3u) & 0xFF800000u;
    const float m = float_from_bits(bits - exponent);
    const float e = fmaf((float)(int32_t)exponent, float_from_bits(0x34000000u), 0.0f);
    const float f = m + -1.0f;
    float p = fmaf(f, float_from_bits(0x3DC6B27Fu), float_from_bits(0xBE2C7F30u));
    static const uint32_t coefficients[8] = { 0x3E2FCF2Au, 0xBE374E43u, 0x3E520BF4u, 0xBE763C8Bu,
                                              0x3E93BF99u, 0xBEB8AA49u, 0x3EF6384Au, 0xBF38AA3Bu };
    for (uint32_t coefficient : coefficients)
        p = fmaf(p, f, float_from_bits(coefficient));
    const float square = f * (f * p);
    const float log2 = e + fmaf(f, float_from_bits(0x3FB8AA3Bu), square);
    return (uint32_t)(log2 * 2048.0f + 0.5f) & 0xFFFFu;
}

// ------------------------------------------------------------------ engine

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    VkDeviceSize allocation = 0;  // of its memory, which another API imports by that size
    uint32_t memoryType = 0;      // which another API on Vulkan imports it as
    void *mapped = nullptr;
};

struct Pipeline {
    VkShaderModule module = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    uint32_t bindings = 0;
};

struct Pass {
    int shader;
    VkDescriptorSet set;
    uint32_t constants[32];
    uint32_t constantBytes;
    uint32_t groups[3];
    VkBuffer indirect;                // dispatched with the sizes the GPU wrote here, when set
    VkDeviceSize indirectOffset;
    // Per slot, where the pass reads the slot's own staging buffer (VMAF v1's
    // passes that read the pictures): the set for each slot, in place of `set`.
    std::vector<VkDescriptorSet> slotSets;
    // Independent of the pass after it (neither writes what the other reads
    // or writes): no barrier between them, so that the GPU runs them together
    // (but when profiling, which times each).
    bool overlapNext = false;
};

// A buffer a pass binds, or part of one (at a multiple of the storage
// buffers' offset alignment).
struct Bound {
    Buffer *buffer;
    VkDeviceSize offset = 0, range = VK_WHOLE_SIZE;
    Bound(Buffer *b) : buffer(b) {}
    Bound(Buffer *b, VkDeviceSize o, VkDeviceSize r) : buffer(b), offset(o), range(r) {}
};

struct Slot {
    Buffer staging, result;
    Buffer cambiKeep;                 // VMAF v1's CAMBI: the c-values of scales the CPU pools (host-visible)
    std::vector<Pass> cambiKeepPasses;
    Buffer speedChroma;               // VMAF v1's SpEED: the pair's chroma planes (host-visible)
    // With frames from a decoder (shared): SpEED reads the chroma planes from
    // `staging` (sharedChroma), unless the frame's came through speedChroma
    // (vv_chroma_staging): then speedPassesHost.
    bool chromaFromHost = false;
    std::vector<Pass> speedPassesHost;
    Buffer speedKeep;                 // and the filtered planes est_params reads (host-visible)
    std::vector<Pass> speedPasses;
    VkCommandBuffer commands = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool busy = false;
    bool scored = false;
    unsigned index = 0;
};

enum { kPushBytes = 128, kMaxBindings = 9 };

// VMAF v1's options for ADM3 and motion3 (the model's feature_opts_dicts).
struct V1Options {
    double viewDistance = 3.0;   // adm_norm_view_dist
    int displayHeight = 1080;    // adm_ref_display_height
    int csfMode = 0;             // adm_csf_mode: 0 Watson97, 2 Barten-Watson blend
    double noiseWeight = 0.03125;
    double dlmWeight = 1.0;
    double minValue = 0.0;       // adm_min_val
    double motionMax = 10000.0;  // motion_max_val
    bool fiveFrameWindow = false;
    bool movingAverage = false;
};

// SpEED's est_params and get_speed_score (v1_speed.c) on worker threads, a
// frame's four filtered planes a job, each thread with its own libvmaf
// state: the same code on the same values whichever thread takes a frame.
// A few milliseconds a frame at 4K, which the thread that submits the
// frames would otherwise spend.
class SpeedPool {
public:
    ~SpeedPool() { stop(); }

    bool start(const V1SpeedOptions &options, unsigned w, unsigned h, size_t planeFloats, int count)
    {
        floats = planeFloats * 4;
        for (int i = 0; i < count; ++i) {
            V1Speed *state = v1_speed_new(&options, w, h);
            if (!state)
                return false;
            states.push_back(state);
        }
        try {
            for (V1Speed *state : states)
                threads.emplace_back([this, state] { run(state); });
        } catch (...) {
            return false;
        }
        return true;
    }

    // A frame's filtered planes (ref U, ref V, dis U, dis V), copied.
    void submit(unsigned index, const float *planes)
    {
        std::unique_lock<std::mutex> lock(mutex);
        done.wait(lock, [this] { return queue.size() < kMaxQueued; });
        std::vector<float> buffer;
        if (!spare.empty()) {
            buffer = std::move(spare.back());
            spare.pop_back();
        }
        buffer.assign(planes, planes + floats);
        queue.emplace_back(index, std::move(buffer));
        ++pending;
        work.notify_one();
    }

    void wait()
    {
        std::unique_lock<std::mutex> lock(mutex);
        done.wait(lock, [this] { return pending == 0; });
    }

    // A frame's u, v and uv scores; false for a frame that was not scored.
    bool result(unsigned index, double out[3])
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto found = results.find(index);
        if (found == results.end())
            return false;
        memcpy(out, found->second.data(), 3 * sizeof(double));
        return true;
    }

    double busyNs = 0.0;  // the workers' time, summed (the profile)
    uint64_t jobs = 0;

private:
    enum { kMaxQueued = 64 };

    void run(V1Speed *state)
    {
        for (;;) {
            std::pair<unsigned, std::vector<float>> job;
            {
                std::unique_lock<std::mutex> lock(mutex);
                work.wait(lock, [this] { return stopping || !queue.empty(); });
                if (queue.empty())
                    return;
                job = std::move(queue.front());
                queue.pop_front();
            }
            const auto started = std::chrono::steady_clock::now();
            const size_t n = floats / 4;
            float *data = job.second.data();
            float *planes[4] = { data, data + 2 * n, data + n, data + 3 * n };  // ref U, dis U, ref V, dis V
            std::array<double, 3> scores;
            v1_speed_chroma(state, planes, scores.data());
            const double ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - started).count();
            std::lock_guard<std::mutex> lock(mutex);
            results[job.first] = scores;
            spare.push_back(std::move(job.second));
            busyNs += ns;
            ++jobs;
            --pending;
            done.notify_all();
        }
    }

    void stop()
    {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
            queue.clear();
        }
        work.notify_all();
        for (std::thread &thread : threads)
            thread.join();
        threads.clear();
        for (V1Speed *state : states)
            v1_speed_free(state);
        states.clear();
    }

    std::mutex mutex;
    std::condition_variable work, done;
    std::deque<std::pair<unsigned, std::vector<float>>> queue;
    std::vector<std::vector<float>> spare;
    std::unordered_map<unsigned, std::array<double, 3>> results;
    std::vector<std::thread> threads;
    std::vector<V1Speed *> states;
    size_t floats = 0, pending = 0;
    bool stopping = false;
};

} // namespace

struct vv_context {
    InstanceApi *api = nullptr;
    DeviceApi vk;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memoryProperties = {};
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queueFamily = 0;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    std::string deviceName;

    int w = 0, h = 0, bpc = 8;
    bool nativeDouble = false;
    // VMAF v1: ADM3 and motion3 as libvmaf's CPU code calculates them.
    bool v1 = false;
    V1Options options;
    Buffer picPrev[2];
    float rfactorV1[kScales][3] = {};
    int build_passes_v1();
    // adm_fused.slang's parameters per scale (kAdmParams words each).
    enum { kAdmParams = 52, kAdmBandRows = 32, kAdmOwnColumns = 126 };  // adm_fused.slang's BAND and OWN
    Buffer admParams, admPartial;
    VkDeviceSize admPartialWords = 0;  // its int64s: every row's sums per workgroup, of the largest scale
    std::vector<int32_t> admParamValues;
    int skip = 0;  // timing tests: 1 = no motion, 2 = no VIF, 4 = no ADM
    int passLimit = 0;  // timing tests: only the first N scored passes
    int decoupleVariant = 0;  // adm_decouple_0's VARIANT (shaders/adm_decouple.slang)
    uint32_t strideBytes = 0, planeBytes = 0;
    // The frames' luma planes come from another API on this GPU (a decoder's
    // CUDA, or its own Vulkan device), which writes them into the slots'
    // staging buffers: GPU memory it imports by the handles vv_export gives,
    // in place of host memory. Vulkan imports them only on the same GPU and
    // driver (vv_shared_device).
    bool shared = false;
    PFN_vkGetMemoryWin32HandleKHR getMemoryHandle = nullptr;
    uint8_t deviceUuid[VK_UUID_SIZE] = {}, driverUuid[VK_UUID_SIZE] = {};

    Pipeline pipelines[kShaderCount];
    std::vector<Buffer *> buffers;
    Buffer picRef, picDis, blur[2], vifTmp, rdRef[2], rdDis[2], logTable, divTable;
    Buffer bandsRef[2], bandsDis[2], admR, admA, admF, acc;

    std::vector<Pass> motion[2];  // by frame parity
    std::vector<Pass> scored;     // VIF and ADM (and VMAF v1's CAMBI)

    // VMAF v1's CAMBI on the GPU (vv_v1_cambi): what cambi.c's init works
    // out, and the buffers its passes use.
    bool cambi = false;
    bool cambiPoolOnCpu = false;  // every scale pooled by v1_host (the tests of that)
    bool cambiBruteForce = false; // the c-values counted pixel by pixel (CVALUES), for the tests of CVALUES_SLIDE
    Buffer cambiArgs;             // the KEEP passes' dispatch sizes
    VkDeviceSize cambiKeepOffset[V1_CAMBI_SCALES] = {};  // each scale's place in a slot's cambiKeep (words)
    V1CambiOptions cambiOptions = {};
    V1CambiConstants cambiConst = {};
    Buffer cambiImage, cambiMaskFull, cambiReciprocal, cambiHist, cambiState;
    Buffer cambiRaw[V1_CAMBI_SCALES], cambiFiltered[V1_CAMBI_SCALES], cambiMask[V1_CAMBI_SCALES],
        cambiC[V1_CAMBI_SCALES];
    int enable_cambi(const double *values);

    // VMAF v1's SpEED chroma (vv_v1_speed): the GPU filters the four chroma
    // planes (shaders/speed.slang), v1_speed.c scores them when a frame is
    // collected.
    bool speed = false;
    V1SpeedOptions speedOptions = {};
    V1SpeedFilters speedFilters = {};
    SpeedPool speedPool;
    uint32_t chromaW = 0, chromaH = 0, chromaStrideBytes = 0, chromaPlaneBytes = 0;
    Buffer speedFilterTaps, speedOperating, speedScaling;
    int enable_speed(const double *values);

    // Profiling (vv_profile, for the speed work): the GPU's time per pass,
    // summed by shader, from timestamps around each, and the CPU's per step.
    bool profile = false;
    VkQueryPool queryPool = VK_NULL_HANDLE;
    double timestampNs = 1.0;
    enum { kQueriesPerSlot = 1024, kProfileUpload = kShaderCount, kProfileCopies, kProfileLabels };
    std::vector<std::vector<int>> slotLabels;  // per slot: what each interval between its timestamps was
    double gpuNs[kProfileLabels] = {};
    uint64_t gpuCount[kProfileLabels] = {};
    enum { kCpuCopy, kCpuRecord, kCpuWait, kCpuSpeed, kCpuCambiPool, kCpuSteps };
    double cpuNs[kCpuSteps] = {};
    uint64_t cpuCount[kCpuSteps] = {};
    double now_ns() const;
    int enable_profile();

    std::vector<Slot> slots;
    unsigned nextSlot = 0;
    unsigned frames = 0;
    std::vector<FrameSums> sums;
    bool failed = false;

    ~vv_context();
    int init(int deviceIndex, int width, int height, int bitDepth, int flags);
    // `upload`: host memory the CPU only writes and the GPU reads (frames' planes).
    int create_buffer(Buffer &buffer, VkDeviceSize size, bool hostVisible, bool exported = false, bool upload = false);
    int create_pipeline(int shader, uint32_t bindings);
    int add_pass(std::vector<Pass> &list, int shader, std::initializer_list<Bound> bound,
                 const void *constants, uint32_t constantBytes, uint32_t gx, uint32_t gy);
    VkDescriptorSet descriptor_set(int shader, const std::vector<Bound> &bound);
    // The last pass added, given a set per slot: `bound(slot)` the slot's buffers.
    template <typename F> int per_slot(std::vector<Pass> &list, F bound);
    // VMAF v1: the slots' luma planes are read where they are written (no copy).
    bool direct = false;
    uint32_t disOffset = 0;  // the distorted plane's place in a slot's staging buffer
    // VMAF v1 with a decoder's frames: the four chroma planes in a slot's
    // staging buffer too (ref U, ref V, dis U, dis V), from sharedChroma, each
    // sharedChromaSpacing bytes after the one before (vv_shared_chroma).
    uint32_t sharedChroma = 0, sharedChromaSpacing = 0;
    Bound slot_ref(Slot &slot) { return Bound(&slot.staging, 0, planeBytes); }
    Bound slot_dis(Slot &slot) { return Bound(&slot.staging, disOffset, planeBytes); }
    int upload(Buffer &target, const void *data, size_t bytes);
    int build_passes();
    int submit(const uint8_t *ref, ptrdiff_t refStride, const uint8_t *dis, ptrdiff_t disStride, bool score);
    int submit_v1(const uint8_t *const planes[6], const ptrdiff_t strides[6], bool score);
    int staging(uint8_t **ref, uint8_t **dis);
    int commit(bool score);
    Slot *pending = nullptr;
    int collect(Slot &slot);
    int flush();
};

vv_context::~vv_context()
{
    if (!device)
        return;
    vk.vkDeviceWaitIdle(device);
    if (queryPool)
        vk.vkDestroyQueryPool(device, queryPool, nullptr);
    for (Slot &slot : slots) {
        if (slot.fence)
            vk.vkDestroyFence(device, slot.fence, nullptr);
    }
    for (Buffer *buffer : buffers) {
        if (buffer->buffer)
            vk.vkDestroyBuffer(device, buffer->buffer, nullptr);
        if (buffer->memory)
            vk.vkFreeMemory(device, buffer->memory, nullptr);
    }
    for (Pipeline &pipeline : pipelines) {
        if (pipeline.pipeline) vk.vkDestroyPipeline(device, pipeline.pipeline, nullptr);
        if (pipeline.layout) vk.vkDestroyPipelineLayout(device, pipeline.layout, nullptr);
        if (pipeline.setLayout) vk.vkDestroyDescriptorSetLayout(device, pipeline.setLayout, nullptr);
        if (pipeline.module) vk.vkDestroyShaderModule(device, pipeline.module, nullptr);
    }
    if (descriptorPool) vk.vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    if (commandPool) vk.vkDestroyCommandPool(device, commandPool, nullptr);
    vk.vkDestroyDevice(device, nullptr);
}

int vv_context::create_buffer(Buffer &buffer, VkDeviceSize size, bool hostVisible, bool exported, bool upload)
{
    buffers.push_back(&buffer);
    buffer.size = (size + 3) & ~VkDeviceSize(3);
    VkBufferCreateInfo info = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    VkExternalMemoryBufferCreateInfo external = { VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO };
    external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    if (exported)
        info.pNext = &external;
    info.size = buffer.size;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vk.vkCreateBuffer(device, &info, nullptr, &buffer.buffer) != VK_SUCCESS)
        return fail(-1, "vkCreateBuffer failed");
    VkMemoryRequirements requirements;
    vk.vkGetBufferMemoryRequirements(device, buffer.buffer, &requirements);
    // Host memory: cached if there is such a type (results are read back);
    // for an upload, not cached (an APU's GPU reads cached host memory slowly,
    // through the CPU's caches): the GPU's own if it can be mapped, else the
    // host's. Device memory: the GPU's.
    const VkMemoryPropertyFlags visible = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    struct Want { VkMemoryPropertyFlags with, without; };
    std::vector<Want> wanted;
    if (upload) {
        wanted = { { visible | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_CACHED_BIT },
                   { visible, VK_MEMORY_PROPERTY_HOST_CACHED_BIT }, { visible, 0 } };
    } else if (hostVisible) {
        wanted = { { visible | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0 }, { visible, 0 } };
    } else {
        wanted = { { VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0 }, { 0, 0 } };
    }
    VkMemoryAllocateInfo allocate = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    allocate.allocationSize = requirements.size;
    // Exported: an allocation of this buffer alone, which is what the
    // importing API is told it is.
    VkExportMemoryAllocateInfo exportInfo = { VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO };
    exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    VkMemoryDedicatedAllocateInfo dedicated = { VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO };
    dedicated.buffer = buffer.buffer;
    if (exported) {
        exportInfo.pNext = &dedicated;
        allocate.pNext = &exportInfo;
    }
    // The first type of the first wish that has room (a mappable heap of the
    // GPU's own is small: 256 MB on a Radeon 780M).
    int type = -1;
    VkResult allocated = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    for (const Want &want : wanted) {
        for (uint32_t i = 0; i < memoryProperties.memoryTypeCount && allocated != VK_SUCCESS; ++i) {
            const VkMemoryPropertyFlags flags = memoryProperties.memoryTypes[i].propertyFlags;
            if (!(requirements.memoryTypeBits & (1u << i)) || (flags & want.with) != want.with || (flags & want.without))
                continue;
            type = (int)i;
            allocate.memoryTypeIndex = i;
            allocated = vk.vkAllocateMemory(device, &allocate, nullptr, &buffer.memory);
        }
        if (allocated == VK_SUCCESS)
            break;
    }
    if (type < 0)
        return fail(-1, "no suitable GPU memory type");
    if (allocated != VK_SUCCESS) {
        buffer.memory = VK_NULL_HANDLE;
        return fail(-2, "out of GPU memory (" + std::to_string(requirements.size >> 20) + " MB buffer)");
    }
    buffer.allocation = requirements.size;
    buffer.memoryType = (uint32_t)type;
    if (vk.vkBindBufferMemory(device, buffer.buffer, buffer.memory, 0) != VK_SUCCESS)
        return fail(-1, "vkBindBufferMemory failed");
    if (hostVisible && vk.vkMapMemory(device, buffer.memory, 0, VK_WHOLE_SIZE, 0, &buffer.mapped) != VK_SUCCESS)
        return fail(-1, "vkMapMemory failed");
    return 0;
}

int vv_context::create_pipeline(int shader, uint32_t bindings)
{
    Pipeline &pipeline = pipelines[shader];
    if (pipeline.pipeline)
        return 0;
    pipeline.bindings = bindings;
    VkShaderModuleCreateInfo module = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    module.codeSize = kShaders[shader].bytes;
    module.pCode = kShaders[shader].code;
    if (vk.vkCreateShaderModule(device, &module, nullptr, &pipeline.module) != VK_SUCCESS)
        return fail(-1, std::string("vkCreateShaderModule failed for ") + kShaders[shader].name);
    VkDescriptorSetLayoutBinding layoutBindings[kMaxBindings] = {};
    for (uint32_t i = 0; i < bindings; ++i) {
        layoutBindings[i].binding = i;
        layoutBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[i].descriptorCount = 1;
        layoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo setLayout = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    setLayout.bindingCount = bindings;
    setLayout.pBindings = layoutBindings;
    if (vk.vkCreateDescriptorSetLayout(device, &setLayout, nullptr, &pipeline.setLayout) != VK_SUCCESS)
        return fail(-1, "vkCreateDescriptorSetLayout failed");
    VkPushConstantRange range = { VK_SHADER_STAGE_COMPUTE_BIT, 0, kPushBytes };
    VkPipelineLayoutCreateInfo layout = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    layout.setLayoutCount = 1;
    layout.pSetLayouts = &pipeline.setLayout;
    layout.pushConstantRangeCount = 1;
    layout.pPushConstantRanges = &range;
    if (vk.vkCreatePipelineLayout(device, &layout, nullptr, &pipeline.layout) != VK_SUCCESS)
        return fail(-1, "vkCreatePipelineLayout failed");
    VkComputePipelineCreateInfo info = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module = pipeline.module;
    info.stage.pName = "main";
    info.layout = pipeline.layout;
    if (vk.vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline.pipeline) != VK_SUCCESS)
        return fail(-1, std::string("the GPU driver could not compile the shader ") + kShaders[shader].name);
    return 0;
}

VkDescriptorSet vv_context::descriptor_set(int shader, const std::vector<Bound> &bound)
{
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkDescriptorSetAllocateInfo allocate = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    allocate.descriptorPool = descriptorPool;
    allocate.descriptorSetCount = 1;
    allocate.pSetLayouts = &pipelines[shader].setLayout;
    if (vk.vkAllocateDescriptorSets(device, &allocate, &set) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    VkDescriptorBufferInfo infos[kMaxBindings];
    VkWriteDescriptorSet writes[kMaxBindings];
    uint32_t count = 0;
    for (const Bound &entry : bound) {
        infos[count] = { entry.buffer->buffer, entry.offset, entry.range };
        writes[count] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[count].dstSet = set;
        writes[count].dstBinding = count;
        writes[count].descriptorCount = 1;
        writes[count].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[count].pBufferInfo = &infos[count];
        ++count;
    }
    vk.vkUpdateDescriptorSets(device, count, writes, 0, nullptr);
    return set;
}

int vv_context::add_pass(std::vector<Pass> &list, int shader, std::initializer_list<Bound> bound,
                         const void *constants, uint32_t constantBytes, uint32_t gx, uint32_t gy)
{
    if (int error = create_pipeline(shader, (uint32_t)bound.size()))
        return error;
    Pass pass = {};
    pass.shader = shader;
    pass.set = descriptor_set(shader, std::vector<Bound>(bound));
    if (!pass.set)
        return fail(-1, "vkAllocateDescriptorSets failed");
    memcpy(pass.constants, constants, constantBytes);
    pass.constantBytes = constantBytes;
    pass.groups[0] = gx;
    pass.groups[1] = gy;
    pass.groups[2] = 1;
    list.push_back(pass);
    return 0;
}

template <typename F> int vv_context::per_slot(std::vector<Pass> &list, F bound)
{
    Pass &pass = list.back();
    for (Slot &slot : slots) {
        VkDescriptorSet set = descriptor_set(pass.shader, bound(slot));
        if (!set)
            return fail(-1, "vkAllocateDescriptorSets failed");
        pass.slotSets.push_back(set);
    }
    return 0;
}

namespace {

void barrier(const DeviceApi &vk, VkCommandBuffer commands, VkPipelineStageFlags from, VkPipelineStageFlags to)
{
    VkMemoryBarrier memory = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    memory.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    memory.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                           VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    vk.vkCmdPipelineBarrier(commands, from, to, 0, 1, &memory, 0, nullptr, 0, nullptr);
}

const VkPipelineStageFlags kCompute = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
const VkPipelineStageFlags kTransfer = VK_PIPELINE_STAGE_TRANSFER_BIT;

uint32_t groups(int count, int size) { return (uint32_t)((count + size - 1) / size); }

int ceil_log2(int value) { return (int)ceil(log2((double)value)); }

} // namespace

// Copies `data` into a device buffer through slot 0's staging buffer.
int vv_context::upload(Buffer &target, const void *data, size_t bytes)
{
    Buffer staging;
    if (int error = create_buffer(staging, bytes, true))
        return error;
    memcpy(staging.mapped, data, bytes);
    VkCommandBuffer commands = slots[0].commands;
    VkCommandBufferBeginInfo begin = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vk.vkBeginCommandBuffer(commands, &begin);
    VkBufferCopy region = { 0, 0, bytes };
    vk.vkCmdCopyBuffer(commands, staging.buffer, target.buffer, 1, &region);
    vk.vkEndCommandBuffer(commands);
    VkSubmitInfo submit = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &commands;
    VkResult result = vk.vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE);
    if (result == VK_SUCCESS)
        result = vk.vkDeviceWaitIdle(device);
    vk.vkDestroyBuffer(device, staging.buffer, nullptr);
    vk.vkFreeMemory(device, staging.memory, nullptr);
    buffers.pop_back();
    return result == VK_SUCCESS ? 0 : fail(-1, "uploading a table to the GPU failed");
}

int vv_context::build_passes()
{
    if (v1)
        return build_passes_v1();
    const bool deep = bpc > 8;
    const int strideWords = (int)(strideBytes / 4);
    int error = 0;

    // Motion (integer_motion_cuda.c: calculate_motion_score).
    for (int parity = 0; parity < 2 && !error && !(skip & 1); ++parity) {
        const int32_t constants[] = { w, h, strideWords, bpc, 1 << (bpc - 1), kSlotSad };
        error = add_pass(motion[parity], deep ? kShader_motion_16 : kShader_motion_8,
                         { &picRef, &blur[parity], &blur[1 - parity], &acc }, constants, sizeof constants,
                         groups(w, 16), groups(h, 16));
    }

    // VIF (integer_vif_cuda.c: filter1d_8, filter1d_16).
    if (!(skip & 2)) {
        const uint64_t epsilon = 0x3EDB7CDFD9D7BDBBull;  // 65536 * 1.0e-10
        int sw = w, sh = h, sourceStride = strideWords;
        for (int scale = 0; scale < kScales && !error; ++scale) {
            if (scale > 0) {
                sw /= 2;
                sh /= 2;
            }
            uint32_t shiftVP = 16, addVP = 32768, shiftSq = 16, addSq = 32768;
            if (scale == 0) {
                shiftVP = (uint32_t)bpc;
                addVP = 1u << (bpc - 1);
                shiftSq = (uint32_t)(bpc - 8) * 2;
                addSq = bpc == 8 ? 0 : 1u << (shiftSq - 1);
            }
            // Scale s writes the next scale's images to pair s % 2.
            Buffer *inRef = scale == 0 ? &picRef : &rdRef[(scale - 1) % 2];
            Buffer *inDis = scale == 0 ? &picDis : &rdDis[(scale - 1) % 2];
            const int nextStride = (sw + 1) / 2;
            const uint32_t vertical[] = { (uint32_t)sw, (uint32_t)sh, (uint32_t)sourceStride,
                                          shiftVP, addVP, shiftSq, addSq };
            const int verticalShader = scale == 0 ? (deep ? kShader_vif_vert_0_16 : kShader_vif_vert_0_8)
                                                  : kShader_vif_vert_1 + (scale - 1);
            error = add_pass(scored, verticalShader, { inRef, inDis, &vifTmp }, vertical, sizeof vertical,
                             groups(sw, 16), groups(sh, 8));
            if (error)
                break;
            const uint32_t horizontal[] = { (uint32_t)sw, (uint32_t)sh, (uint32_t)nextStride,
                                            (uint32_t)(kSlotVif + scale * kVifSums), 100, 1,
                                            (uint32_t)epsilon, (uint32_t)(epsilon >> 32) };
            const int horizontalShader = (nativeDouble ? kShader_vif_hori_native_0 : kShader_vif_hori_0) + scale;
            error = add_pass(scored, horizontalShader,
                             { &vifTmp, &rdRef[scale % 2], &rdDis[scale % 2], &acc, &logTable }, horizontal,
                             sizeof horizontal, groups(sw, 64), (uint32_t)sh);
            sourceStride = nextStride;
        }
    }

    // ADM (integer_adm_cuda.c: integer_compute_adm_cuda).
    uint32_t i_rfactor[12];
    adm_rfactors(i_rfactor);
    int inW = w, inH = h, inStride = strideWords;
    for (int scale = 0; scale < kScales && !error && !(skip & 4); ++scale) {
        const int set = scale % 2;
        Buffer *inRef = scale == 0 ? &picRef : &bandsRef[1 - set];
        Buffer *inDis = scale == 0 ? &picDis : &bandsDis[1 - set];
        const int bw = (inW + 1) / 2, bh = (inH + 1) / 2;
        const int bandStride = set == 0 ? (w + 1) / 2 : ((w + 1) / 2 + 1) / 2;
        const int outStride = (w + 1) / 2;  // of admR, admA, admF

        {   // dwt2_8_device / adm_dwt2_16_device / adm_dwt2_s123_combined_device
            static const int kV[4][2] = { { 0, 0 }, { 0, 0 }, { 16, 32768 }, { 16, 32768 } };
            static const int kH[4][2] = { { 16, 32768 }, { 15, 16384 }, { 16, 32768 }, { 15, 16384 } };
            const int32_t constants[] = { inW, inH, inStride, bandStride,
                                          scale == 0 ? bpc : kV[scale][0], scale == 0 ? 1 << (bpc - 1) : kV[scale][1],
                                          kH[scale][0], kH[scale][1] };
            const int shader = scale == 0 ? (deep ? kShader_adm_dwt_0_16 : kShader_adm_dwt_0_8) : kShader_adm_dwt;
            error = add_pass(scored, shader, { inRef, inDis, &bandsRef[set], &bandsDis[set] }, constants,
                             sizeof constants, groups(bw, 16), groups(bh, 8));
            if (error)
                break;
        }
        {   // adm_csf_den_scale_device / adm_csf_den_s123_device
            int left = bw * (float)(ADM_BORDER_FACTOR) - 0.5f;
            int top = bh * (float)(ADM_BORDER_FACTOR) - 0.5f;
            int right = bw - left;
            int bottom = bh - top;
            uint32_t shiftSq = 0, addSq = 0, shiftCub = 0, addCub = 0, shiftAccum, addAccum;
            if (scale == 0) {
                const int32_t shift_area = (int32_t)ceil(log2((bottom - top) * (right - left)) - 20);
                shiftAccum = shift_area > 0 ? (uint32_t)shift_area : 0;
                addAccum = shiftAccum > 0 ? 1u << (shiftAccum - 1) : 0;
            } else {
                static const uint32_t shift_sq[3] = { 31, 30, 31 };
                shiftSq = shift_sq[scale - 1];
                addSq = 1u << shiftSq;
                shiftCub = (uint32_t)ceil_log2(right - left);
                addCub = shiftCub ? 1u << (shiftCub - 1) : 0;
                shiftAccum = (uint32_t)ceil_log2(bottom - top);
                addAccum = shiftAccum ? 1u << (shiftAccum - 1) : 0;
            }
            const uint32_t constants[] = { (uint32_t)top, (uint32_t)left, (uint32_t)right, (uint32_t)bandStride,
                                           shiftSq, addSq, shiftCub, addCub, shiftAccum, addAccum,
                                           (uint32_t)(kSlotCsfDen + scale * 3) };
            error = add_pass(scored, scale == 0 ? kShader_adm_csf_den_0 : kShader_adm_csf_den,
                             { &bandsRef[set], &acc }, constants, sizeof constants,
                             groups(right - left, 1024), (uint32_t)(bottom - top));
            if (error)
                break;
        }
        // The enhancement gain limit: 100 (VMAF), then 1 (VMAF NEG).
        for (int limit = 0; limit < 2 && !error; ++limit) {
            {   // adm_decouple_device / adm_decouple_s123_device, adm_csf_device / i4_adm_csf_device
                int left = bw * (float)(ADM_BORDER_FACTOR) - 0.5f - 1;
                int top = bh * (float)(ADM_BORDER_FACTOR) - 0.5f - 1;
                int right = bw - left + 2;
                int bottom = bh - top + 2;
                if (left < 0) left = 0;
                if (right > bw) right = bw;
                if (top < 0) top = 0;
                if (bottom > bh) bottom = bh;
                const uint32_t constants[] = { (uint32_t)top, (uint32_t)bottom, (uint32_t)left, (uint32_t)right,
                                               (uint32_t)bandStride, (uint32_t)outStride,
                                               limit == 0 ? 100u : 1u,
                                               i_rfactor[scale * 3], i_rfactor[scale * 3 + 1], i_rfactor[scale * 3 + 2] };
                const int decouple0 = decoupleVariant ? kShader_adm_decouple_0_v1 + decoupleVariant - 1
                                                      : kShader_adm_decouple_0;
                error = add_pass(scored, scale == 0 ? decouple0 : kShader_adm_decouple,
                                 { &bandsRef[set], &bandsDis[set], &admR, &admA, &admF, &divTable }, constants,
                                 sizeof constants, groups(right - left, 16), groups(bottom - top, 8));
                if (error)
                    break;
            }
            {   // adm_cm_device / i4_adm_cm_device
                int left = bw * (float)(ADM_BORDER_FACTOR) - 0.5f;
                int top = bh * (float)(ADM_BORDER_FACTOR) - 0.5f;
                int right = bw - left;
                int bottom = bh - top;
                int start_col, end_col, start_row, end_row;
                if (scale == 0) {
                    start_col = std::max(0, left);
                    end_col = std::min(right, bw);
                    start_row = std::max(0, top);
                    end_row = std::min(bottom, bh);
                } else {
                    start_col = (left > 1) ? left : ((left <= 0) ? 0 : 1);
                    end_col = (right < (bw - 1)) ? right : ((right > (bw - 1)) ? bw : bw - 1);
                    start_row = (top > 1) ? top : ((top <= 0) ? 0 : 1);
                    end_row = (bottom < (bh - 1)) ? bottom : ((bottom > (bh - 1)) ? bh : bh - 1);
                }
                struct {
                    int32_t w, h, startRow, startCol, endCol, stride;
                    uint32_t rfactor[3];
                    int32_t shiftSub[3], shiftSq[3], addSq[3], shiftCub[3], addCub[3];
                    int32_t shiftInner, addInner;
                    uint32_t slot;
                } constants = {};
                constants.w = bw;
                constants.h = bh;
                constants.startRow = start_row;
                constants.startCol = start_col;
                constants.endCol = end_col;
                constants.stride = outStride;
                static const int shift_sub[3] = { 10, 10, 12 }, fixed_shift[3] = { 4, 4, 3 };
                static const int shift_xsq[3] = { 29, 29, 30 };
                for (int band = 0; band < 3; ++band) {
                    constants.rfactor[band] = i_rfactor[scale * 3 + band];
                    constants.shiftSub[band] = scale == 0 ? shift_sub[band] : 0;
                    constants.shiftSq[band] = scale == 0 ? shift_xsq[band] : 30;
                    constants.addSq[band] = 1 << (constants.shiftSq[band] - 1);
                    const double shift = scale == 0 ? ceil(log2((double)bw) - fixed_shift[band]) : ceil(log2((double)bw));
                    constants.shiftCub[band] = shift > 0 ? (int32_t)shift : 0;
                    constants.addCub[band] = constants.shiftCub[band] ? 1 << (constants.shiftCub[band] - 1) : 0;
                }
                constants.shiftInner = ceil_log2(bh);
                constants.addInner = constants.shiftInner ? 1 << (constants.shiftInner - 1) : 0;
                constants.slot = (uint32_t)(kSlotCm + limit * kScales * 3 + scale * 3);
                error = add_pass(scored, scale == 0 ? kShader_adm_cm_0 : kShader_adm_cm,
                                 { &admR, &admA, &admF, &acc }, &constants, sizeof constants, 1,
                                 (uint32_t)std::max(0, end_row - start_row));
            }
        }
        inW = bw;
        inH = bh;
        inStride = bandStride;
    }
    return error;
}

namespace {

// barten_watson_blend_csf() of libvmaf's barten_csf_tools.h: [theta][scale].
const float BLENDED_CSF_1080_3H[2][4] = { { 0.01183, 0.025026, 0.04295, 0.058621 },
                                          { 0.004302, 0.011778, 0.023918, 0.035901 } };
const float BLENDED_CSF_1080_5H[2][4] = { { 0.004212, 0.014809, 0.029642, 0.047464 },
                                          { 0.000984, 0.005852, 0.0146, 0.027574 } };
const float BLENDED_CSF_2160_3H[2][4] = { { 0.00226, 0.01183, 0.025026, 0.04295 },
                                          { 0.000479, 0.004302, 0.011778, 0.023918 } };
const float BLENDED_CSF_2160_5H[2][4] = { { 0.000092, 0.004212, 0.014809, 0.029642 },
                                          { 0.000050, 0.000984, 0.005852, 0.0146 } };

// The contrast sensitivity factors of a scale ({h, v, d}) as libvmaf's CPU
// code takes them (adm_csf and the others of integer_adm.c). False when the
// options are ones it has no table for.
bool v1_rfactors(const V1Options &o, int scale, float rfactor[3])
{
    float factor1, factor2;
    if (o.csfMode == 2) {
        const float (*table)[4] =
            (o.displayHeight == 1080 && o.viewDistance == 3.0) ? BLENDED_CSF_1080_3H :
            (o.displayHeight == 1080 && o.viewDistance == 5.0) ? BLENDED_CSF_1080_5H :
            (o.displayHeight == 2160 && o.viewDistance == 1.5) ? BLENDED_CSF_1080_3H :
            (o.displayHeight == 2160 && o.viewDistance == 3.0) ? BLENDED_CSF_2160_3H :
            (o.displayHeight == 2160 && o.viewDistance == 5.0) ? BLENDED_CSF_2160_5H : nullptr;
        if (!table)
            return false;
        factor1 = table[0][scale];
        factor2 = table[1][scale];
    } else if (o.csfMode == 0) {
        factor1 = 1.0f / dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 1, o.viewDistance, o.displayHeight);
        factor2 = 1.0f / dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 2, o.viewDistance, o.displayHeight);
    } else {
        return false;
    }
    rfactor[0] = factor1;
    rfactor[1] = factor1;
    rfactor[2] = factor2;
    return true;
}

} // namespace

// VMAF v1's passes: motion and ADM as libvmaf's CPU code calculates them
// (integer_motion.c, integer_adm.c), which is what the shaders' V1 and
// ROWWISE variants follow where the CUDA kernels differ from it.
int vv_context::build_passes_v1()
{
    const bool deep = bpc > 8;
    const int strideWords = (int)(strideBytes / 4);
    int error = 0;

    // motion_score_pipeline_8 / _16: against the previous reference frame,
    // or the one before it (the five-frame window). Frame i's reference is
    // kept in picPrev[i % 2] once the frame is done.
    for (int parity = 0; parity < 2 && !error; ++parity) {
        const int32_t constants[] = { w, h, strideWords, bpc, 1 << (bpc - 1), kSlotSad };
        Buffer *previous = &picPrev[options.fiveFrameWindow ? parity : 1 - parity];
        error = add_pass(motion[parity], deep ? kShader_motion_v1_16 : kShader_motion_v1_8,
                         { &picRef, previous, &acc }, constants, sizeof constants, groups(w, 16), groups(h, 16));
        if (!error && direct)  // the frame before's (or the one before it) reference where it is: its slot
            error = per_slot(motion[parity], [&](Slot &slot) {
                const size_t count = slots.size(), lag = options.fiveFrameWindow ? 2 : 1;
                Slot &before = slots[((size_t)(&slot - slots.data()) + count - lag) % count];
                return std::vector<Bound>{ slot_ref(slot), slot_ref(before), Bound(&acc) };
            });
    }

    const float cos_1deg_sq = cos(1.0 * M_PI / 180.0) * cos(1.0 * M_PI / 180.0);
    uint32_t cosBits;
    memcpy(&cosBits, &cos_1deg_sq, sizeof cosBits);
    const uint32_t cosMantissa = (cosBits & 0x7FFFFFu) | 0x800000u;  // cos_1deg_sq = mantissa * 2^-24

    admParamValues.assign((size_t)kScales * kAdmParams, 0);
    if (int created = create_buffer(admParams, admParamValues.size() * sizeof(int32_t), false))
        return created;
    {   // scale 0's rows and workgroups across, the most of any scale
        const VkDeviceSize rows = (VkDeviceSize)(h + 1) / 2, across = (VkDeviceSize)groups((w + 1) / 2, kAdmOwnColumns);
        if (int created = create_buffer(admPartial, rows * across * 9 * 8, false))
            return created;
    }
    int inW = w, inH = h, inStride = strideWords;
    for (int scale = 0; scale < kScales && !error; ++scale) {
        const int set = scale % 2;
        Buffer *inRef = scale == 0 ? &picRef : &bandsRef[1 - set];
        Buffer *inDis = scale == 0 ? &picDis : &bandsDis[1 - set];
        const int bw = (inW + 1) / 2, bh = (inH + 1) / 2;
        const int bandStride = set == 0 ? (w + 1) / 2 : ((w + 1) / 2 + 1) / 2;
        const int outStride = (w + 1) / 2;
        if (!v1_rfactors(options, scale, rfactorV1[scale]))
            return fail(-3, "VMAF v1: no contrast sensitivity table for this viewing distance and display height");
        const float *rfactor = rfactorV1[scale];
        uint32_t i_rfactor[3];
        if (scale == 0) {
            if (fabs(options.viewDistance * options.displayHeight - kAdmNormViewDist * kAdmRefDisplayHeight) < 1.0e-8 &&
                options.csfMode == 0) {
                i_rfactor[0] = 36453;
                i_rfactor[1] = 36453;
                i_rfactor[2] = 49417;
            } else {
                const double pow2_21 = pow(2, 21);
                const double pow2_23 = pow(2, 23);
                i_rfactor[0] = (uint16_t)(rfactor[0] * pow2_21);
                i_rfactor[1] = (uint16_t)(rfactor[1] * pow2_21);
                i_rfactor[2] = (uint16_t)(rfactor[2] * pow2_23);
            }
        } else {
            const double pow2_32 = pow(2, 32);
            for (int band = 0; band < 3; ++band)
                i_rfactor[band] = (uint32_t)(rfactor[band] * pow2_32);
        }

        static const int kV[4][2] = { { 0, 0 }, { 0, 0 }, { 16, 32768 }, { 16, 32768 } };
        static const int kH[4][2] = { { 16, 32768 }, { 15, 16384 }, { 16, 32768 }, { 15, 16384 } };
        if (scale > 0) {  // the wavelet transform, as for VMAF v0.6.1 (scale 0's: in the fused pass)
            const int32_t constants[] = { inW, inH, inStride, bandStride, kV[scale][0], kV[scale][1],
                                          kH[scale][0], kH[scale][1] };
            // Scale 1 reads scale 0's approximation bands, one int32 a position.
            error = add_pass(scored, scale == 1 ? kShader_adm_dwt_approx : kShader_adm_dwt,
                             { inRef, inDis, &bandsRef[set], &bandsDis[set] }, constants, sizeof constants,
                             groups(bw, 16), groups(bh, 8));
            if (error)
                break;
        }
        // adm_csf_den, adm_decouple, adm_csf and adm_cm of both images in
        // one pass (shaders/adm_fused.slang), its parameters in admParams.
        int32_t *p = &admParamValues[(size_t)scale * kAdmParams];
        {   // adm_csf_den_scale / adm_csf_den_s123
            const int left = bw * ADM_BORDER_FACTOR - 0.5;
            const int top = bh * ADM_BORDER_FACTOR - 0.5;
            const int right = bw - left;
            const int bottom = bh - top;
            uint32_t shiftSq = 0, addSq = 0, shiftCub = 0, addCub = 0, shiftAccum, addAccum;
            if (scale == 0) {
                int32_t shift_accum = (int32_t)ceil(log2((bottom - top) * (right - left)) - 20);
                shift_accum = shift_accum > 0 ? shift_accum : 0;
                shiftAccum = (uint32_t)shift_accum;
                addAccum = shift_accum > 0 ? (1u << (shift_accum - 1)) : 0;
            } else {
                static const uint32_t shift_sq[3] = { 31, 30, 31 };
                shiftSq = shift_sq[scale - 1];
                addSq = 1u << shiftSq;
                shiftCub = (uint32_t)ceil(log2(right - left));
                addCub = (uint32_t)pow(2, ((double)shiftCub - 1));
                shiftAccum = (uint32_t)ceil(log2(bottom - top));
                addAccum = (uint32_t)pow(2, ((double)shiftAccum - 1));
            }
            p[12] = top;
            p[13] = bottom;
            p[14] = left;
            p[15] = right;
            p[33] = (int32_t)shiftSq;
            p[34] = (int32_t)addSq;
            p[35] = (int32_t)shiftCub;
            p[36] = (int32_t)addCub;
            p[37] = (int32_t)shiftAccum;
            p[38] = (int32_t)addAccum;
            p[41] = kSlotCsfDen + scale * 3;
        }
        {   // adm_cm / i4_adm_cm: the restored image masked by the additive one,
            // then (the additive impairment measure) the additive image masked
            // by the restored one. (adm_decouple's region is this one and a
            // pixel around it.)
            const int left = bw * ADM_BORDER_FACTOR - 0.5;
            const int top = bh * ADM_BORDER_FACTOR - 0.5;
            const int right = bw - left;
            const int bottom = bh - top;
            const int start_col = (left > 1) ? left : 1;
            const int end_col = (right < (bw - 1)) ? right : (bw - 1);
            const int start_row = (top > 1) ? top : 1;
            const int end_row = (bottom < (bh - 1)) ? bottom : (bh - 1);
            p[0] = bw;
            p[1] = bh;
            p[2] = bandStride;
            p[3] = 1;  // adm_enhn_gain_limit
            for (int band = 0; band < 3; ++band)
                p[4 + band] = (int32_t)i_rfactor[band];
            p[7] = (int32_t)cosMantissa;
            p[8] = start_row;
            p[9] = end_row;
            p[10] = start_col;
            p[11] = end_col;
            static const int shift_sub[3] = { 10, 10, 12 }, fixed_shift[3] = { 4, 4, 3 };
            static const int shift_xsq[3] = { 29, 29, 30 };
            for (int band = 0; band < 3; ++band) {
                p[16 + band] = scale == 0 ? shift_sub[band] : 0;
                p[19 + band] = scale == 0 ? shift_xsq[band] : 30;
                p[22 + band] = 1 << (p[19 + band] - 1);
                const uint32_t shift = scale == 0 ? (uint32_t)ceil(log2(bw) - fixed_shift[band])
                                                  : (uint32_t)ceil(log2(bw));
                p[25 + band] = (int32_t)shift;
                p[28 + band] = (int32_t)(uint32_t)pow(2, ((double)shift - 1));
            }
            const uint32_t shift_inner_accum = (uint32_t)ceil(log2(bh));
            p[31] = (int32_t)shift_inner_accum;
            p[32] = (int32_t)(uint32_t)pow(2, ((double)shift_inner_accum - 1));
            p[39] = kSlotCm + scale * 3;
            p[40] = kSlotCm + kScales * 3 + scale * 3;
            p[42] = std::min(start_row, p[12]);
            p[43] = std::max(end_row, p[13]);
        }
        if (scale == 0) {  // the transform from the pictures, and all of the band image (its a)
            p[42] = 0;
            p[43] = bh;
            p[44] = inW;
            p[45] = inH;
            p[46] = inStride;
            p[47] = bpc;
            p[48] = 1 << (bpc - 1);
            p[49] = kH[0][0];
            p[50] = kH[0][1];
            p[51] = bandStride;
        }
        {   // SLIDE: workgroups of kAdmOwnColumns columns down bands of
            // kAdmBandRows rows, each row's sums per workgroup into admPartial;
            // then ROWSUM adds and rounds each row's.
            const int colStart = scale == 0 ? 0 : std::min(p[10], p[14]);
            const int colEnd = scale == 0 ? bw : std::max(p[11], p[15]);
            const int rows = std::max(0, p[43] - p[42]);
            const uint32_t chunks = groups(std::max(0, colEnd - colStart), kAdmOwnColumns);
            const uint32_t constants[] = { (uint32_t)(scale * kAdmParams), chunks };
            if (scale == 0) {
                error = add_pass(scored, deep ? kShader_adm_fused_0_16 : kShader_adm_fused_0_8,
                                 { inRef, inDis, &divTable, &admParams, &acc, &bandsRef[set], &bandsDis[set], &admPartial },
                                 constants, sizeof constants, chunks, groups(rows, kAdmBandRows));
                if (!error && direct)
                    error = per_slot(scored, [&](Slot &slot) {
                        return std::vector<Bound>{ slot_ref(slot), slot_dis(slot), Bound(&divTable), Bound(&admParams),
                                                   Bound(&acc), Bound(&bandsRef[set]), Bound(&bandsDis[set]),
                                                   Bound(&admPartial) };
                    });
            } else {
                error = add_pass(scored, kShader_adm_fused, { &bandsRef[set], &bandsDis[set], &divTable, &admParams, &acc,
                                 &admPartial }, constants, sizeof constants, chunks, groups(rows, kAdmBandRows));
            }
            if (!error)
                error = add_pass(scored, kShader_adm_rowsum, { &bandsRef[set], &bandsDis[set], &divTable, &admParams, &acc,
                                 &admPartial }, constants, sizeof constants, groups(rows * 9, 256), 1);
            admPartialWords = std::max<VkDeviceSize>(admPartialWords, (VkDeviceSize)rows * chunks * 9);
        }
        inW = bw;
        inH = bh;
        inStride = bandStride;
    }
    if (!error)
        error = upload(admParams, admParamValues.data(), admParamValues.size() * sizeof(int32_t));
    return error;
}

namespace {

// As libvmaf's CPU code (integer_adm.c): a scale's numerator from adm_cm /
// i4_adm_cm's sums, with the noise weight the caller gives (0 for the
// additive impairment measure).
float v1_cm(const int64_t *accum, int w, int h, int scale, double adm_noise_weight)
{
    const int left = w * ADM_BORDER_FACTOR - 0.5;
    const int top = h * ADM_BORDER_FACTOR - 0.5;
    const int right = w - left;
    const int bottom = h - top;
    const uint32_t shift_inner_accum = (uint32_t)ceil(log2(h));
    float f_accum[3];
    if (scale == 0) {
        const uint32_t shift_xhcub = (uint32_t)ceil(log2(w) - 4);
        const uint32_t shift_xdcub = (uint32_t)ceil(log2(w) - 3);
        f_accum[0] = (float)(accum[0] / pow(2, (52 - shift_xhcub - shift_inner_accum)));
        f_accum[1] = (float)(accum[1] / pow(2, (52 - shift_xhcub - shift_inner_accum)));
        f_accum[2] = (float)(accum[2] / pow(2, (57 - shift_xdcub - shift_inner_accum)));
    } else {
        const uint32_t shift_cub = (uint32_t)ceil(log2(w));
        float final_shift[3] = { (float)pow(2, (45 - shift_cub - shift_inner_accum)),
                                 (float)pow(2, (39 - shift_cub - shift_inner_accum)),
                                 (float)pow(2, (36 - shift_cub - shift_inner_accum)) };
        for (int i = 0; i < 3; ++i)
            f_accum[i] = (float)(accum[i] / final_shift[scale - 1]);
    }
    float num_scale_h = powf(f_accum[0], 1.0f / 3.0f) + powf((bottom - top) * (right - left) * adm_noise_weight, 1.0f / 3.0f);
    float num_scale_v = powf(f_accum[1], 1.0f / 3.0f) + powf((bottom - top) * (right - left) * adm_noise_weight, 1.0f / 3.0f);
    float num_scale_d = powf(f_accum[2], 1.0f / 3.0f) + powf((bottom - top) * (right - left) * adm_noise_weight, 1.0f / 3.0f);
    return (num_scale_h + num_scale_v + num_scale_d);
}

// As libvmaf's CPU code: adm_csf_den_scale / adm_csf_den_s123's conclusion.
float v1_csf_den(const uint64_t *accum, int w, int h, int scale, const float rfactor[3], double adm_noise_weight)
{
    const int left = w * ADM_BORDER_FACTOR - 0.5;
    const int top = h * ADM_BORDER_FACTOR - 0.5;
    const int right = w - left;
    const int bottom = h - top;
    double shift_csf;
    if (scale == 0) {
        int32_t shift_accum = (int32_t)ceil(log2((bottom - top) * (right - left)) - 20);
        shift_accum = shift_accum > 0 ? shift_accum : 0;
        shift_csf = pow(2, (18 - shift_accum));
    } else {
        const uint32_t accum_convert_float[3] = { 32, 27, 23 };
        uint32_t shift_cub = (uint32_t)ceil(log2(right - left));
        uint32_t shift_accum = (uint32_t)ceil(log2(bottom - top));
        shift_csf = pow(2, (accum_convert_float[scale - 1] - shift_accum - shift_cub));
    }
    double csf_h = (double)(accum[0] / shift_csf) * pow(rfactor[0], 3);
    double csf_v = (double)(accum[1] / shift_csf) * pow(rfactor[1], 3);
    double csf_d = (double)(accum[2] / shift_csf) * pow(rfactor[2], 3);
    float powf_add = powf((bottom - top) * (right - left) * adm_noise_weight, 1.0f / 3.0f);
    float den_scale_h = powf(csf_h, 1.0f / 3.0f) + powf_add;
    float den_scale_v = powf(csf_v, 1.0f / 3.0f) + powf_add;
    float den_scale_d = powf(csf_d, 1.0f / 3.0f) + powf_add;
    return (den_scale_h + den_scale_v + den_scale_d);
}

// As libvmaf's CPU code: integer_compute_adm's sums and extract()'s ADM3.
// out: adm3, aim, adm2.
void v1_adm(const vv_context &context, const uint64_t *slots, double out[3])
{
    const V1Options &o = context.options;
    int w = context.w, h = context.h;
    const double numden_limit = 1e-10 * (w * h) / (1920.0 * 1080.0);
    double num = 0, den = 0, aim_num = 0;
    for (int scale = 0; scale < 4; ++scale) {
        w = (w + 1) / 2;
        h = (h + 1) / 2;
        const int64_t *cm = (const int64_t *)&slots[kSlotCm + scale * 3];
        const int64_t *aim = (const int64_t *)&slots[kSlotCm + kScales * 3 + scale * 3];
        float den_scale = v1_csf_den(&slots[kSlotCsfDen + scale * 3], w, h, scale, context.rfactorV1[scale], o.noiseWeight);
        float num_scale = v1_cm(cm, w, h, scale, o.noiseWeight);
        float aim_num_scale = v1_cm(aim, w, h, scale, 0.0);
        num += num_scale;
        den += den_scale;
        aim_num += aim_num_scale;
    }
    num = num < numden_limit ? 0 : num;
    den = den < numden_limit ? 0 : den;
    double score, score_aim = 0;
    if (den == 0.0) {
        score = 1.0f;
    } else {
        score_aim = aim_num / den;
        score = num / den;
    }
    const double adm3 = score * o.dlmWeight + (1 - score_aim) * (1 - o.dlmWeight);
    out[0] = adm3 > o.minValue ? adm3 : o.minValue;
    out[1] = score_aim;
    out[2] = score;
}

} // namespace

int vv_context::init(int deviceIndex, int width, int height, int bitDepth, int flags)
{
    api = instance_api();
    if (!api)
        return -1;
    if (width < 32 || height < 32 || width > 16384 || height > 16384)
        return fail(-3, "unsupported picture size");
    if (bitDepth < 8 || bitDepth > 16)
        return fail(-3, "unsupported bit depth");
    w = width;
    h = height;
    bpc = bitDepth;

    std::vector<VkPhysicalDevice> devices = physical_devices(api);
    if (deviceIndex < 0 || deviceIndex >= (int)devices.size())
        return fail(-3, "no such GPU");
    physical = devices[(size_t)deviceIndex];
    VkPhysicalDeviceProperties properties;
    api->vkGetPhysicalDeviceProperties(physical, &properties);
    deviceName = properties.deviceName;
    VkPhysicalDeviceFeatures features;
    api->vkGetPhysicalDeviceFeatures(physical, &features);
    if (!features.shaderInt64)
        return fail(-4, deviceName + " has no 64-bit integers in shaders");
    nativeDouble = (flags & 1) && features.shaderFloat64;
    skip = (flags >> 16) & 7;
    passLimit = (flags >> 20) & 0xFF;
    shared = (flags >> 19) & 1;
    const char *extensions[] = { VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME };
    if (shared) {
        uint32_t count = 0;
        api->vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, nullptr);
        std::vector<VkExtensionProperties> listed(count);
        if (count)
            api->vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, listed.data());
        bool found = false;
        for (uint32_t i = 0; i < count; ++i)
            found = found || !strcmp(listed[i].extensionName, extensions[0]);
        if (!found)
            return fail(-4, deviceName + " cannot share its memory with a decoder");
    }
    if (shared) {
        VkPhysicalDeviceIDProperties ids = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES };
        VkPhysicalDeviceProperties2 properties2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
        properties2.pNext = &ids;
        api->vkGetPhysicalDeviceProperties2(physical, &properties2);
        memcpy(deviceUuid, ids.deviceUUID, VK_UUID_SIZE);
        memcpy(driverUuid, ids.driverUUID, VK_UUID_SIZE);
    }
    decoupleVariant = (flags >> 28) & 7;
    api->vkGetPhysicalDeviceMemoryProperties(physical, &memoryProperties);

    uint32_t familyCount = 0;
    api->vkGetPhysicalDeviceQueueFamilyProperties(physical, &familyCount, nullptr);
    std::vector<VkQueueFamilyProperties> families(familyCount);
    api->vkGetPhysicalDeviceQueueFamilyProperties(physical, &familyCount, families.data());
    int family = -1;
    for (uint32_t i = 0; i < familyCount; ++i) {
        if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            family = (int)i;
            break;
        }
    }
    if (family < 0)
        return fail(-4, deviceName + " has no compute queue");
    queueFamily = (uint32_t)family;

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    queueInfo.queueFamilyIndex = queueFamily;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;
    VkPhysicalDeviceFeatures enabled = {};
    enabled.shaderInt64 = VK_TRUE;
    enabled.shaderFloat64 = nativeDouble ? VK_TRUE : VK_FALSE;
    VkDeviceCreateInfo deviceInfo = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    deviceInfo.pEnabledFeatures = &enabled;
    if (shared) {
        deviceInfo.enabledExtensionCount = 1;
        deviceInfo.ppEnabledExtensionNames = extensions;
    }
    VkResult result = api->vkCreateDevice(physical, &deviceInfo, nullptr, &device);
    if (result != VK_SUCCESS) {
        device = VK_NULL_HANDLE;
        return fail(-1, "vkCreateDevice failed (" + std::to_string(result) + ")");
    }
#define X(name) vk.name = (PFN_##name)api->vkGetDeviceProcAddr(device, #name);
    VK_DEVICE_FUNCTIONS(X)
#undef X
    vk.vkGetDeviceQueue(device, queueFamily, 0, &queue);
    if (shared) {
        getMemoryHandle = (PFN_vkGetMemoryWin32HandleKHR)api->vkGetDeviceProcAddr(device, "vkGetMemoryWin32HandleKHR");
        if (!getMemoryHandle)
            return fail(-4, deviceName + " cannot share its memory with a decoder");
    }

    VkCommandPoolCreateInfo poolInfo = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = queueFamily;
    if (vk.vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS)
        return fail(-1, "vkCreateCommandPool failed");
    // VMAF v1 with CAMBI and SpEED: about 150 passes, and 7 per slot (up to 16 slots).
    VkDescriptorPoolSize poolSize = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 512 * kMaxBindings };
    VkDescriptorPoolCreateInfo descriptorInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    descriptorInfo.maxSets = 512;
    descriptorInfo.poolSizeCount = 1;
    descriptorInfo.pPoolSizes = &poolSize;
    if (vk.vkCreateDescriptorPool(device, &descriptorInfo, nullptr, &descriptorPool) != VK_SUCCESS)
        return fail(-1, "vkCreateDescriptorPool failed");

    const uint32_t sampleBytes = bpc > 8 ? 2 : 1;
    strideBytes = ((uint32_t)w * sampleBytes + 3) & ~3u;
    planeBytes = strideBytes * (uint32_t)h;
    // The distorted plane's offset: a multiple of 256, which any storage
    // buffer offset alignment divides (VMAF v1 binds the planes there).
    disOffset = (planeBytes + 255) & ~255u;
    VkDeviceSize stagingBytes = (VkDeviceSize)disOffset + planeBytes;
    if (v1 && shared) {  // 4:2:0's chroma as libvmaf takes it, w/2 x h/2, its rows a word apart
        const uint32_t chromaStride = ((uint32_t)(w / 2) * (bpc > 8 ? 2 : 1) + 3) & ~3u;
        sharedChroma = (uint32_t)((stagingBytes + 255) & ~VkDeviceSize(255));
        sharedChromaSpacing = (chromaStride * (uint32_t)(h / 2) + 255) & ~255u;
        stagingBytes = (VkDeviceSize)sharedChroma + 4ull * sharedChromaSpacing;
    }
    // Read where they are, in the GPU's own memory, when a decoder writes them
    // (shared); host memory, which the GPU reads more slowly than its own
    // (measured on a Radeon 780M: by the passes reading the pictures, more
    // than the copy costs), is copied first.
    direct = v1 && shared;
    const VkDeviceSize pixels = (VkDeviceSize)w * h;
    const int w1 = (w + 1) / 2, h1 = (h + 1) / 2, w2 = (w1 + 1) / 2, h2 = (h1 + 1) / 2;
    const int rw1 = (w / 2 + 1) / 2, rh1 = (h / 2 + 1) / 2;
    const VkDeviceSize v0 = v1 ? 0 : 1, only1 = v1 ? 1 : 0;  // buffers one of the two uses hold 4 bytes in the other
    struct { Buffer *buffer; VkDeviceSize bytes; } sized[] = {
        { &picRef, direct ? 4 : planeBytes }, { &picDis, direct ? 4 : planeBytes },
        { &blur[0], pixels * 4 * v0 + 4 }, { &blur[1], pixels * 4 * v0 + 4 }, { &vifTmp, pixels * 32 * v0 + 4 },
        { &rdRef[0], (VkDeviceSize)w1 * h1 * 4 * v0 + 4 }, { &rdDis[0], (VkDeviceSize)w1 * h1 * 4 * v0 + 4 },
        { &rdRef[1], (VkDeviceSize)rw1 * rh1 * 4 * v0 + 4 }, { &rdDis[1], (VkDeviceSize)rw1 * rh1 * 4 * v0 + 4 },
        { &picPrev[0], planeBytes * only1 * (direct ? 0 : 1) + 4 }, { &picPrev[1], planeBytes * only1 * (direct ? 0 : 1) + 4 },
        { &logTable, 32768 * 4 }, { &divTable, 65536 * 4 },
        { &bandsRef[0], (VkDeviceSize)w1 * h1 * 16 }, { &bandsDis[0], (VkDeviceSize)w1 * h1 * 16 },
        { &bandsRef[1], (VkDeviceSize)w2 * h2 * 16 }, { &bandsDis[1], (VkDeviceSize)w2 * h2 * 16 },
        // VMAF v1 passes nothing between ADM's passes (adm_fused.slang): VMAF v0.6.1's images.
        { &admR, (VkDeviceSize)w1 * h1 * 16 * v0 + 4 }, { &admA, (VkDeviceSize)w1 * h1 * 16 * v0 + 4 },
        { &admF, (VkDeviceSize)w1 * h1 * 16 * v0 + 4 }, { &acc, kSlots * 8 },
    };
    for (auto &entry : sized) {
        if (int error = create_buffer(*entry.buffer, entry.bytes, false))
            return error;
    }

    const int depth = std::clamp((flags >> 8) & 0xFF, 1, 16);
    slots.resize((size_t)(((flags >> 8) & 0xFF) ? depth : 3));
    // Read where they are, the slots' pictures stay as long as the frames
    // after read them as their previous (staging()): a slot or two more, for
    // as many frames in flight.
    if (direct)
        slots.resize(std::min<size_t>(16, slots.size() + (options.fiveFrameWindow ? 2 : 1)));
    std::vector<VkCommandBuffer> commands(slots.size());
    VkCommandBufferAllocateInfo allocate = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    allocate.commandPool = commandPool;
    allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate.commandBufferCount = (uint32_t)slots.size();
    if (vk.vkAllocateCommandBuffers(device, &allocate, commands.data()) != VK_SUCCESS)
        return fail(-1, "vkAllocateCommandBuffers failed");
    for (size_t i = 0; i < slots.size(); ++i) {
        Slot &slot = slots[i];
        slot.commands = commands[i];
        VkFenceCreateInfo fenceInfo = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        if (vk.vkCreateFence(device, &fenceInfo, nullptr, &slot.fence) != VK_SUCCESS)
            return fail(-1, "vkCreateFence failed");
        if (int error = create_buffer(slot.staging, stagingBytes, !shared, shared, !shared))
            return error;
        if (int error = create_buffer(slot.result, kSlots * 8, true))
            return error;
    }

    // The tables, and zeros where a first frame reads before anything wrote:
    // the previous blur (as the CUDA code's memset) and the band images.
    std::vector<uint32_t> table(65536);
    for (uint32_t i = 0; i < 32768; ++i)
        table[i] = cuda_log_generate(32768 + i);
    if (int error = upload(logTable, table.data(), 32768 * 4))
        return error;
    const float div_Q_factor = 1073741824;  // 2^30
    for (int i = -32768; i < 32768; ++i) {
        // The CUDA kernel divides in float; the CPU code's table (div_lookup
        // of integer_adm.h), which VMAF v1 follows, in integers.
        table[(size_t)(i + 32768)] = i == 0 ? 0 : v1 ? (uint32_t)(1073741824 / i)
                                                    : (uint32_t)(int32_t)(div_Q_factor / float(i));
    }
    if (int error = upload(divTable, table.data(), 65536 * 4))
        return error;
    {
        VkCommandBuffer cb = slots[0].commands;
        VkCommandBufferBeginInfo begin = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vk.vkBeginCommandBuffer(cb, &begin);
        for (Buffer *buffer : { &blur[0], &blur[1], &vifTmp, &rdRef[0], &rdDis[0], &rdRef[1], &rdDis[1],
                                &bandsRef[0], &bandsDis[0], &bandsRef[1], &bandsDis[1], &admR, &admA, &admF,
                                &picPrev[0], &picPrev[1] })
            vk.vkCmdFillBuffer(cb, buffer->buffer, 0, VK_WHOLE_SIZE, 0);
        vk.vkEndCommandBuffer(cb);
        VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cb;
        if (vk.vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS ||
            vk.vkDeviceWaitIdle(device) != VK_SUCCESS)
            return fail(-1, "clearing the GPU buffers failed");
    }
    return build_passes();
}

// CAMBI's passes, after VMAF v1's (scored frames only), from the distorted
// luma (picDis): the steps of cambi_score, one shader stage each
// (shaders/cambi.slang), and spatial_pooling's top k found by a radix
// select over the c-values' bits, three levels of 12, 12 and 8 bits.
int vv_context::enable_cambi(const double *values)
{
    if (!v1)
        return fail(-3, "CAMBI is calculated in a VMAF v1 context only");
    if (frames || cambi)
        return fail(-3, "CAMBI is started before the first frame");
    V1CambiOptions &o = cambiOptions;
    o.high_res_speedup = (int)values[0];
    o.vis_lum_threshold = values[1];
    o.max_val = values[2];
    o.topk = values[3];
    o.window_size = (int)values[4];
    o.tvi_threshold = values[5];
    o.max_log_contrast = (int)values[6];
    o.eotf = (int)values[7];
    cambiPoolOnCpu = ((int)values[8] & 1) != 0;
    cambiBruteForce = ((int)values[8] & 2) != 0;
    if (int error = v1_cambi_constants(&o, (unsigned)w, (unsigned)h, &cambiConst))
        return fail(error, "CAMBI: cambi.c's init refuses this picture or these options");
    const V1CambiConstants &c = cambiConst;
    if (c.num_diffs != 4)
        return fail(-4, "CAMBI: only max_log_contrast 2 (4 contrasts) is calculated on the GPU");
    if (c.window_size * c.window_size >= CAMBI_RECIPROCAL_LUT_SIZE)
        return fail(-3, "CAMBI: the window is larger than the reciprocal table");

    const VkDeviceSize pixels = (VkDeviceSize)w * h;
    const int histWords = V1_CAMBI_SCALES * 3 * 4096;
    // The full picture's image and mask only where scale 0 is the full picture
    // (no speedup): FRONT writes scale 0's decimated ones otherwise.
    const VkDeviceSize full = c.speedup ? 1 : pixels;
    struct { Buffer *buffer; VkDeviceSize bytes; } sized[] = {
        { &cambiImage, full * 4 }, { &cambiMaskFull, full * 4 },
        { &cambiReciprocal, sizeof kCambiReciprocal }, { &cambiHist, (VkDeviceSize)histWords * 4 },
        { &cambiState, V1_CAMBI_SCALES * 2 * 4 },
    };
    for (auto &entry : sized) {
        if (int error = create_buffer(*entry.buffer, entry.bytes, false))
            return error;
    }
    for (int scale = 0; scale < V1_CAMBI_SCALES; ++scale) {
        const VkDeviceSize n = (VkDeviceSize)c.scale_w[scale] * c.scale_h[scale];
        const bool decimated = scale > 0 || c.speedup;
        if (int error = create_buffer(cambiFiltered[scale], n * 4, false))
            return error;
        if (int error = create_buffer(cambiC[scale], n * 4, false))
            return error;
        if (decimated) {
            if (int error = create_buffer(cambiRaw[scale], n * 4, false))
                return error;
            if (int error = create_buffer(cambiMask[scale], n * 4, false))
                return error;
        }
    }
    if (int error = upload(cambiReciprocal, kCambiReciprocal, sizeof kCambiReciprocal))
        return error;
    if (int error = create_buffer(cambiArgs, V1_CAMBI_SCALES * 3 * 4, false))
        return error;
    VkDeviceSize keepWords = 0;
    for (int scale = 0; scale < V1_CAMBI_SCALES; ++scale) {
        cambiKeepOffset[scale] = keepWords;
        keepWords += (VkDeviceSize)c.scale_w[scale] * c.scale_h[scale];
    }
    for (Slot &slot : slots) {
        if (int error = create_buffer(slot.cambiKeep, keepWords * 4, true))
            return error;
    }

    int error = 0;
    {
        const uint32_t constants[] = { (uint32_t)histWords };
        error = add_pass(scored, kShader_cambi_clear, { &cambiHist }, constants, sizeof constants,
                         groups(histWords, 256), 1);
    }
    if (!error) {
        // PRE, DERIV and MASK, and scale 0's DECIMATE with the speedup: FRONT.
        const uint32_t step = c.speedup ? 2 : 1;
        const uint32_t outW = c.speedup ? (uint32_t)c.scale_w[0] : (uint32_t)w;
        const uint32_t outH = c.speedup ? (uint32_t)c.scale_h[0] : (uint32_t)h;
        const uint32_t constants[] = { (uint32_t)w, (uint32_t)h, strideBytes / 4, (uint32_t)bpc,
                                       bpc < 10 ? 1u : 0u, (uint32_t)kSlotCambiInvalid, (uint32_t)c.mask_index,
                                       step, outW, outH };
        const int front = c.speedup ? (bpc > 8 ? kShader_cambi_front_16 : kShader_cambi_front_8)
                                    : (bpc > 8 ? kShader_cambi_front_16_1 : kShader_cambi_front_8_1);
        error = add_pass(scored, front,
                         { &picDis, c.speedup ? &cambiRaw[0] : &cambiImage, c.speedup ? &cambiMask[0] : &cambiMaskFull,
                           &acc }, constants, sizeof constants, groups((int)outW, 16), groups((int)outH, 16));
        if (!error && direct)
            error = per_slot(scored, [&](Slot &slot) {
                return std::vector<Bound>{ slot_dis(slot), Bound(c.speedup ? &cambiRaw[0] : &cambiImage),
                                           Bound(c.speedup ? &cambiMask[0] : &cambiMaskFull), Bound(&acc) };
            });
    }
    Buffer *image = &cambiImage, *mask = &cambiMaskFull;
    int inW = w;
    // The scales' c-values (each from its filtered image and mask) after all
    // of those, together: the small scales' few workgroups, one row after
    // another, took about as long as scale 0's, the GPU mostly idle.
    std::vector<Pass> cvalues;
    for (int scale = 0; scale < V1_CAMBI_SCALES && !error; ++scale) {
        const int sw = c.scale_w[scale], sh = c.scale_h[scale];
        if (scale == 0 && c.speedup) {  // FRONT wrote them
            image = &cambiRaw[scale];
            mask = &cambiMask[scale];
        } else if (scale > 0) {
            const uint32_t constants[] = { (uint32_t)sw, (uint32_t)sh, (uint32_t)inW };
            error = add_pass(scored, kShader_cambi_decimate, { image, mask, &cambiRaw[scale], &cambiMask[scale] },
                             constants, sizeof constants, groups(sw, 16), groups(sh, 16));
            image = &cambiRaw[scale];
            mask = &cambiMask[scale];
        }
        if (!error) {
            const uint32_t constants[] = { (uint32_t)sw, (uint32_t)sh };
            error = add_pass(scored, kShader_cambi_mode, { image, &cambiFiltered[scale] }, constants, sizeof constants,
                             groups(sw, 16), groups(sh, 16));
        }
        if (!error) {
            // The sliding counts (CVALUES_SLIDE) for windows of at most 65.
            const uint32_t rows = 16;
            uint32_t constants[15] = { (uint32_t)sw, (uint32_t)sh, (uint32_t)(c.window_size >> 1),
                                       (uint32_t)c.vlt_luma, (uint32_t)c.v_band_base, (uint32_t)c.v_band_size };
            for (int d = 0; d < 4; ++d) {
                constants[6 + d] = (uint32_t)c.tvi_for_diff[d];
                constants[10 + d] = (uint32_t)c.diff_weights[d];
            }
            constants[14] = rows;
            const bool slide = (c.window_size >> 1) <= 32 && c.v_band_size <= 32 * 56 && !cambiBruteForce;
            error = slide ? add_pass(cvalues, kShader_cambi_cvalues_slide, { &cambiFiltered[scale], mask, &cambiC[scale],
                                     &cambiReciprocal }, constants, sizeof constants, groups(sw, 64), groups(sh, (int)rows))
                          : add_pass(cvalues, kShader_cambi_cvalues, { &cambiFiltered[scale], mask, &cambiC[scale],
                                     &cambiReciprocal }, constants, sizeof constants, groups(sw, 16), groups(sh, 16));
        }
        inW = sw;
        image = &cambiFiltered[scale];
    }
    for (size_t i = 0; i < cvalues.size(); ++i) {
        cvalues[i].overlapNext = i + 1 < cvalues.size();
        scored.push_back(cvalues[i]);
    }
    // spatial_pooling's top k of the five scales together: the radix select's
    // three levels (HIST, SELECT) and the sum above the k-th largest (SUM).
    uint32_t sizes[12] = {};
    uint32_t mostGroups = 0;
    for (int scale = 0; scale < V1_CAMBI_SCALES; ++scale) {
        sizes[1 + scale] = (uint32_t)(c.scale_w[scale] * c.scale_h[scale]);
        sizes[6 + scale] = std::min<uint32_t>(groups((int)sizes[1 + scale], 256), 128);
        mostGroups = std::max(mostGroups, sizes[6 + scale]);
    }
    for (uint32_t level = 0; level < 3 && !error; ++level) {
        sizes[0] = level;
        error = add_pass(scored, kShader_cambi_hist, { &cambiC[0], &cambiC[1], &cambiC[2], &cambiC[3], &cambiC[4],
                         &cambiHist, &cambiState }, sizes, sizeof sizes, mostGroups, V1_CAMBI_SCALES);
        if (!error) {
            uint32_t select[7] = { level };
            for (int scale = 0; scale < V1_CAMBI_SCALES; ++scale)
                select[1 + scale] = (uint32_t)c.topk_elements[scale];
            select[6] = (uint32_t)kSlotCambi;
            error = add_pass(scored, kShader_cambi_select, { &cambiHist, &cambiState, &acc }, select, sizeof select, 1,
                             V1_CAMBI_SCALES);
        }
    }
    if (!error) {
        sizes[11] = (uint32_t)kSlotCambi;
        error = add_pass(scored, kShader_cambi_sum, { &cambiC[0], &cambiC[1], &cambiC[2], &cambiC[3], &cambiC[4],
                         &cambiState, &acc }, sizes, sizeof sizes, mostGroups, V1_CAMBI_SCALES);
    }
    if (!error) {
        uint32_t constants[7] = { cambiPoolOnCpu ? 1u : 0u, (uint32_t)kSlotCambi };
        for (int scale = 0; scale < V1_CAMBI_SCALES; ++scale)
            constants[2 + scale] = groups(c.scale_w[scale] * c.scale_h[scale], 256);
        error = add_pass(scored, kShader_cambi_args, { &acc, &cambiArgs }, constants, sizeof constants, 1, 1);
    }
    for (Slot &slot : slots) {
        for (int scale = 0; scale < V1_CAMBI_SCALES && !error; ++scale) {
            const uint32_t constants[] = { (uint32_t)(c.scale_w[scale] * c.scale_h[scale]),
                                           (uint32_t)cambiKeepOffset[scale] };
            error = add_pass(slot.cambiKeepPasses, kShader_cambi_keep, { &cambiC[scale], &slot.cambiKeep }, constants,
                             sizeof constants, 1, 1);
            if (!error) {
                slot.cambiKeepPasses.back().indirect = cambiArgs.buffer;
                slot.cambiKeepPasses.back().indirectOffset = (VkDeviceSize)scale * 3 * 4;
            }
        }
    }
    if (error)
        return error;
    cambi = true;
    return 0;
}

// SpEED chroma's passes, per slot (each reads its slot's chroma planes and
// writes its slot's filtered ones): DEC and BLUR of shaders/speed.slang, on
// 4:2:0's chroma, w/2 x h/2 as libvmaf's init_chroma takes it.
int vv_context::enable_speed(const double *values)
{
    if (!v1)
        return fail(-3, "SpEED is calculated in a VMAF v1 context only");
    if (frames || speed)
        return fail(-3, "SpEED is started before the first frame");
    V1SpeedOptions &o = speedOptions;
    o.kernelscale = values[0];
    o.prescale = values[1];
    o.bilinear = values[2] != 0;
    o.sigma_nn = values[3];
    o.nn_floor = values[4];
    o.weight_var_mode = (int)values[5];
    o.max_val = values[6];
    // picture_copy reads 8-bit samples but at 10, 12 and 16 bits.
    if (bpc != 8 && bpc != 10 && bpc != 12 && bpc != 16)
        return fail(-4, "SpEED: only 8, 10, 12 and 16 bits are calculated on the GPU");
    chromaW = (uint32_t)w / 2;
    chromaH = (uint32_t)h / 2;
    if (int error = v1_speed_filters(&o, chromaW, chromaH, &speedFilters))
        return fail(error, error == -4 ? "SpEED: only the bilinear prescale is calculated on the GPU"
                                       : "SpEED: speed.c's init refuses this picture or these options");
    const V1SpeedFilters &f = speedFilters;
    // The bilinear tables (a word of zeros where the plane is not rescaled).
    std::vector<uint32_t> scaling(1, 0);
    if (f.scaled) {
        const size_t sw = (size_t)f.scaled_w, sh = (size_t)f.scaled_h;
        scaling.assign(3 * sw + 3 * sh, 0);
        std::vector<int> x1(sw), x2(sw), y1(sh), y2(sh);
        std::vector<float> dx(sw), dy(sh);
        v1_speed_bilinear_tables(chromaW, chromaH, &f, x1.data(), x2.data(), dx.data(), y1.data(), y2.data(), dy.data());
        for (size_t i = 0; i < sw; ++i) {
            scaling[i] = (uint32_t)x1[i];
            scaling[sw + i] = (uint32_t)x2[i];
            memcpy(&scaling[2 * sw + i], &dx[i], 4);
        }
        for (size_t i = 0; i < sh; ++i) {
            scaling[3 * sw + i] = (uint32_t)y1[i];
            scaling[3 * sw + sh + i] = (uint32_t)y2[i];
            memcpy(&scaling[3 * sw + 2 * sh + i], &dy[i], 4);
        }
    }
    if (int error = create_buffer(speedScaling, scaling.size() * 4, false))
        return error;
    if (int error = upload(speedScaling, scaling.data(), scaling.size() * 4))
        return error;
    const uint32_t sampleBytes = bpc > 8 ? 2 : 1;
    chromaStrideBytes = (chromaW * sampleBytes + 3) & ~3u;
    chromaPlaneBytes = chromaStrideBytes * chromaH;
    const VkDeviceSize operating = (VkDeviceSize)f.operating_w * f.operating_h;
    std::vector<float> taps(f.antialias, f.antialias + f.antialias_taps);
    taps.insert(taps.end(), f.blur, f.blur + f.blur_taps);
    if (int error = create_buffer(speedFilterTaps, taps.size() * sizeof(float), false))
        return error;
    if (int error = upload(speedFilterTaps, taps.data(), taps.size() * sizeof(float)))
        return error;
    if (int error = create_buffer(speedOperating, operating * 4 * sizeof(float), false))
        return error;
    float inverse = bpc == 8 ? 1.0f : 1.0f / (float)(1u << (bpc == 10 ? 2 : bpc == 12 ? 4 : 8));
    uint32_t inverseBits;
    memcpy(&inverseBits, &inverse, sizeof inverseBits);
    for (Slot &slot : slots) {
        if (int error = create_buffer(slot.speedChroma, (VkDeviceSize)chromaPlaneBytes * 4, true, false, true))
            return error;
        if (int error = create_buffer(slot.speedKeep, operating * 4 * sizeof(float), true))
            return error;
        const uint32_t dec[] = { (uint32_t)f.scaled_w, (uint32_t)f.scaled_h, chromaStrideBytes / 4,
                                 chromaPlaneBytes / 4, (uint32_t)f.operating_w, (uint32_t)f.operating_h,
                                 (uint32_t)f.antialias_taps, (uint32_t)bpc, inverseBits, f.scaled ? 1u : 0u };
        const int perGroup = 64 / f.antialias_taps;  // outputs a workgroup (a thread per column of each)
        const uint32_t blur[] = { (uint32_t)f.operating_w, (uint32_t)f.operating_h, (uint32_t)f.antialias_taps,
                                  (uint32_t)f.blur_taps };
        // From the host buffer; with a decoder's frames, also from the staging buffer (the default then).
        for (int fromStaging = 0; fromStaging < (shared ? 2 : 1); ++fromStaging) {
            std::vector<Pass> &list = shared && !fromStaging ? slot.speedPassesHost : slot.speedPasses;
            uint32_t constants[10];
            memcpy(constants, dec, sizeof constants);
            Bound chroma = Bound(&slot.speedChroma);
            if (fromStaging) {
                chroma = Bound(&slot.staging, sharedChroma, 4ull * sharedChromaSpacing);
                constants[3] = sharedChromaSpacing / 4;  // planeWords
            }
            if (int error = add_pass(list, kShader_speed_dec, { chroma, &speedFilterTaps, &speedOperating,
                                     &speedScaling }, constants, sizeof constants,
                                     groups(f.operating_w * f.operating_h * 4, perGroup), 1))
                return error;
            if (int error = add_pass(list, kShader_speed_blur, { &speedOperating, &speedFilterTaps, &slot.speedKeep },
                                     blur, sizeof blur, groups(f.operating_w, 16), groups(f.operating_h, 16)))
                return error;
            list.back().groups[2] = 4;
        }
    }
    const int workers = (int)std::clamp(std::thread::hardware_concurrency() / 4, 1u, 4u);
    if (!speedPool.start(o, chromaW, chromaH, (size_t)operating, workers))
        return fail(-2, "SpEED: its threads could not be started");
    speed = true;
    return 0;
}

double vv_context::now_ns() const
{
    static const double period = [] {
        LARGE_INTEGER frequency;
        QueryPerformanceFrequency(&frequency);
        return 1e9 / (double)frequency.QuadPart;
    }();
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * period;
}

int vv_context::enable_profile()
{
    if (frames || profile)
        return fail(-3, "profiling is started before the first frame");
    VkPhysicalDeviceProperties properties;
    api->vkGetPhysicalDeviceProperties(physical, &properties);
    uint32_t familyCount = 0;
    api->vkGetPhysicalDeviceQueueFamilyProperties(physical, &familyCount, nullptr);
    std::vector<VkQueueFamilyProperties> families(familyCount);
    api->vkGetPhysicalDeviceQueueFamilyProperties(physical, &familyCount, families.data());
    if (!families[queueFamily].timestampValidBits || properties.limits.timestampPeriod <= 0.0f)
        return fail(-4, deviceName + " has no timestamps on its compute queue");
    timestampNs = properties.limits.timestampPeriod;
    VkQueryPoolCreateInfo info = { VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
    info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    info.queryCount = (uint32_t)(kQueriesPerSlot * slots.size());
    if (vk.vkCreateQueryPool(device, &info, nullptr, &queryPool) != VK_SUCCESS)
        return fail(-1, "vkCreateQueryPool failed");
    slotLabels.assign(slots.size(), {});
    profile = true;
    return 0;
}

int vv_context::collect(Slot &slot)
{
    if (!slot.busy)
        return 0;
    const double waitStart = profile ? now_ns() : 0.0;
    VkResult result = vk.vkWaitForFences(device, 1, &slot.fence, VK_TRUE, 60ull * 1000 * 1000 * 1000);
    if (profile) {
        cpuNs[kCpuWait] += now_ns() - waitStart;
        ++cpuCount[kCpuWait];
    }
    slot.busy = false;
    if (result != VK_SUCCESS) {
        failed = true;
        return fail(-5, result == VK_TIMEOUT ? "the GPU did not finish a frame in 60 seconds"
                                             : "the GPU failed (" + std::to_string(result) + ")");
    }
    vk.vkResetFences(device, 1, &slot.fence);
    if (profile) {
        const size_t s = (size_t)(&slot - slots.data());
        const std::vector<int> &labels = slotLabels[s];
        std::vector<uint64_t> stamps(labels.size() + 1);
        if (!labels.empty() &&
            vk.vkGetQueryPoolResults(device, queryPool, (uint32_t)(s * kQueriesPerSlot), (uint32_t)stamps.size(),
                                     stamps.size() * 8, stamps.data(), 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
            for (size_t i = 0; i < labels.size(); ++i) {
                gpuNs[labels[i]] += (double)(stamps[i + 1] - stamps[i]) * timestampNs;
                ++gpuCount[labels[i]];
            }
        }
    }
    FrameSums &frame = sums[slot.index];
    frame.scored = slot.scored;
    const uint32_t *words = (const uint32_t *)slot.result.mapped;
    for (int i = 0; i < kSlots; ++i)
        frame.slots[i] = (uint64_t)words[2 * i] | ((uint64_t)words[2 * i + 1] << 32);
    const double tailStart = profile ? now_ns() : 0.0;
    if (cambi && slot.scored) {
        const float *kept = (const float *)slot.cambiKeep.mapped;
        for (int scale = 0; scale < V1_CAMBI_SCALES; ++scale) {
            if (!(frame.slots[kSlotCambi + scale * 4 + 3] & 2))
                continue;
            const int sw = cambiConst.scale_w[scale], sh = cambiConst.scale_h[scale];
            std::vector<float> values(kept + cambiKeepOffset[scale], kept + cambiKeepOffset[scale] + (size_t)sw * sh);
            frame.cambiPooled[scale] = v1_cambi_pool(values.data(), cambiOptions.topk, (unsigned)sw, (unsigned)sh);
            frame.cambiPooledScales |= 1u << scale;
        }
    }
    if (profile) {
        cpuNs[kCpuCambiPool] += now_ns() - tailStart;
        ++cpuCount[kCpuCambiPool];
    }
    if (speed && slot.scored)  // est_params on the pool's threads, from a copy of the planes
        speedPool.submit(slot.index, (const float *)slot.speedKeep.mapped);
    return 0;
}

// The next frame pair's two luma planes in the memory the GPU copies them
// from, for a caller that can write them there itself (a decoder): rows of
// strideBytes, the reference at *ref and the distorted at *dis. commit()
// then scores what was written.
int vv_context::staging(uint8_t **ref, uint8_t **dis)
{
    if (failed)
        return fail(-5, "the GPU failed on an earlier frame");
    Slot &slot = slots[nextSlot];
    if (int error = collect(slot))
        return error;
    if (direct) {
        // The frames after the slot's last read its reference as their
        // previous one (motion): they are done too before it is written again.
        const size_t lag = options.fiveFrameWindow ? 2 : 1;
        for (size_t k = 1; k <= lag; ++k) {
            if (int error = collect(slots[(nextSlot + k) % slots.size()]))
                return error;
        }
    }
    pending = &slot;
    *ref = (uint8_t *)slot.staging.mapped;
    *dis = (uint8_t *)slot.staging.mapped + disOffset;
    return 0;
}

int vv_context::submit(const uint8_t *ref, ptrdiff_t refStride, const uint8_t *dis, ptrdiff_t disStride, bool score)
{
    uint8_t *targets[2];
    if (shared)
        return fail(-3, "this context takes its frames from GPU memory");
    if (int error = staging(&targets[0], &targets[1]))
        return error;
    const double copyStart = profile ? now_ns() : 0.0;
    const size_t rowBytes = (size_t)w * (bpc > 8 ? 2 : 1);
    const uint8_t *planes[2] = { ref, score ? dis : nullptr };
    const ptrdiff_t strides[2] = { refStride, disStride };
    for (int plane = 0; plane < 2; ++plane) {
        if (!planes[plane])
            continue;
        uint8_t *to = targets[plane];
        if ((size_t)strides[plane] == strideBytes) {
            memcpy(to, planes[plane], planeBytes - (strideBytes - rowBytes));
        } else {
            for (int y = 0; y < h; ++y)
                memcpy(to + (size_t)y * strideBytes, planes[plane] + (ptrdiff_t)y * strides[plane], rowBytes);
        }
    }
    if (profile) {
        cpuNs[kCpuCopy] += now_ns() - copyStart;
        ++cpuCount[kCpuCopy];
    }
    return commit(score);
}

// VMAF v1's way in with the chroma planes, which SpEED reads: planes and
// strides of the reference's Y, U and V and the distorted's.
int vv_context::submit_v1(const uint8_t *const planes[6], const ptrdiff_t strides[6], bool score)
{
    if (shared)
        return fail(-3, "this context takes its frames from GPU memory");
    if (!planes[0] || (score && !planes[3]))
        return fail(-3, "a frame needs its reference picture, a scored one its distorted one too");
    if (speed && score) {
        for (int plane : { 1, 2, 4, 5 }) {
            if (!planes[plane])
                return fail(-3, "SpEED needs the chroma planes");
        }
    }
    uint8_t *targets[2];
    if (int error = staging(&targets[0], &targets[1]))
        return error;
    const double copyStart = profile ? now_ns() : 0.0;
    auto copy = [](uint8_t *to, uint32_t toStride, const uint8_t *from, ptrdiff_t fromStride, size_t rowBytes,
                   uint32_t rows) {
        if (fromStride == (ptrdiff_t)toStride) {
            memcpy(to, from, (size_t)toStride * (rows - 1) + rowBytes);
        } else {
            for (uint32_t y = 0; y < rows; ++y)
                memcpy(to + (size_t)y * toStride, from + (ptrdiff_t)y * fromStride, rowBytes);
        }
    };
    const size_t sampleBytes = bpc > 8 ? 2 : 1;
    copy(targets[0], strideBytes, planes[0], strides[0], (size_t)w * sampleBytes, (uint32_t)h);
    if (score)
        copy(targets[1], strideBytes, planes[3], strides[3], (size_t)w * sampleBytes, (uint32_t)h);
    if (speed && score) {
        uint8_t *chroma = (uint8_t *)pending->speedChroma.mapped;
        const int order[4] = { 1, 2, 4, 5 };  // ref U, ref V, dis U, dis V
        for (int i = 0; i < 4; ++i)
            copy(chroma + (size_t)i * chromaPlaneBytes, chromaStrideBytes, planes[order[i]], strides[order[i]],
                 chromaW * sampleBytes, chromaH);
    }
    if (profile) {
        cpuNs[kCpuCopy] += now_ns() - copyStart;
        ++cpuCount[kCpuCopy];
    }
    return commit(score);
}

int vv_context::commit(bool score)
{
    if (!pending)
        return fail(-3, "no frame was started");
    Slot &slot = *pending;
    pending = nullptr;
    nextSlot = (nextSlot + 1) % (unsigned)slots.size();
    const double recordStart = profile ? now_ns() : 0.0;
    const size_t slotIndex = (size_t)(&slot - slots.data());
    const uint32_t queryBase = (uint32_t)(slotIndex * kQueriesPerSlot);

    const unsigned index = frames++;
    sums.emplace_back();
    slot.index = index;
    slot.scored = score;

    VkCommandBuffer cb = slot.commands;
    vk.vkResetCommandBuffer(cb, 0);
    VkCommandBufferBeginInfo begin = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vk.vkBeginCommandBuffer(cb, &begin);
    std::vector<int> *labels = profile ? &slotLabels[slotIndex] : nullptr;
    auto stamp = [&](int label) {
        if (labels && labels->size() + 1 < kQueriesPerSlot) {
            vk.vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool,
                                   queryBase + (uint32_t)labels->size() + 1);
            labels->push_back(label);
        }
    };
    if (labels) {
        labels->clear();
        vk.vkCmdResetQueryPool(cb, queryPool, queryBase, kQueriesPerSlot);
        vk.vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, queryBase);
    }
    barrier(vk, cb, kCompute | kTransfer, kTransfer);
    if (!direct) {
        VkBufferCopy copy = { 0, 0, planeBytes };
        vk.vkCmdCopyBuffer(cb, slot.staging.buffer, picRef.buffer, 1, &copy);
        if (score) {
            copy.srcOffset = disOffset;
            vk.vkCmdCopyBuffer(cb, slot.staging.buffer, picDis.buffer, 1, &copy);
        }
    }
    vk.vkCmdFillBuffer(cb, acc.buffer, 0, VK_WHOLE_SIZE, 0);
    barrier(vk, cb, kTransfer, kCompute | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT);
    stamp(kProfileUpload);
    const bool moves = !v1 || index >= (options.fiveFrameWindow ? 2u : 1u);
    const std::vector<Pass> *lists[4] = { moves ? &motion[index % 2] : nullptr, score ? &scored : nullptr,
                                          score && cambi ? &slot.cambiKeepPasses : nullptr,
                                          score && speed ? (slot.chromaFromHost ? &slot.speedPassesHost
                                                                                : &slot.speedPasses) : nullptr };
    if (shared)
        slot.chromaFromHost = false;  // the next frame's: from staging unless vv_chroma_staging says
    for (const std::vector<Pass> *list : lists) {
        if (!list)
            continue;
        int done = 0;
        for (const Pass &pass : *list) {
            if (list == &scored && passLimit && done++ >= passLimit)
                break;
            if (!pass.groups[0] || !pass.groups[1])
                continue;
            const Pipeline &pipeline = pipelines[pass.shader];
            vk.vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
            const VkDescriptorSet set = pass.slotSets.empty() ? pass.set : pass.slotSets[slotIndex];
            vk.vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout, 0, 1, &set, 0, nullptr);
            vk.vkCmdPushConstants(cb, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pass.constantBytes, pass.constants);
            if (pass.indirect)
                vk.vkCmdDispatchIndirect(cb, pass.indirect, pass.indirectOffset);
            else
                vk.vkCmdDispatch(cb, pass.groups[0], pass.groups[1], pass.groups[2]);
            if (!pass.overlapNext || profile)
                barrier(vk, cb, kCompute, kCompute | kTransfer | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT);
            stamp(pass.shader);
        }
    }
    if (v1 && !direct) {  // read from its slot otherwise (staging())
        barrier(vk, cb, kCompute, kTransfer);
        VkBufferCopy keep = { 0, 0, planeBytes };
        vk.vkCmdCopyBuffer(cb, picRef.buffer, picPrev[index % 2].buffer, 1, &keep);
    }
    VkBufferCopy back = { 0, 0, kSlots * 8 };
    vk.vkCmdCopyBuffer(cb, acc.buffer, slot.result.buffer, 1, &back);
    stamp(kProfileCopies);
    // What the shaders and the copy wrote into host-visible buffers, visible
    // to the host once the fence is signalled.
    VkMemoryBarrier toHost = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    toHost.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    toHost.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vk.vkCmdPipelineBarrier(cb, kCompute | kTransfer, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &toHost, 0, nullptr, 0,
                            nullptr);
    vk.vkEndCommandBuffer(cb);
    VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cb;
    VkResult result = vk.vkQueueSubmit(queue, 1, &submitInfo, slot.fence);
    if (result != VK_SUCCESS) {
        failed = true;
        return fail(-5, "vkQueueSubmit failed (" + std::to_string(result) + ")");
    }
    slot.busy = true;
    if (profile) {
        cpuNs[kCpuRecord] += now_ns() - recordStart;
        ++cpuCount[kCpuRecord];
    }
    return 0;
}

int vv_context::flush()
{
    int error = 0;
    for (size_t i = 0; i < slots.size(); ++i) {
        // In submission order, so the first failure is the one reported.
        Slot &slot = slots[(nextSlot + i) % slots.size()];
        if (int slotError = collect(slot))
            error = error ? error : slotError;
    }
    if (speed)
        speedPool.wait();
    return error;
}

// --------------------------------------------------------------------- API

// The GPUs Vulkan lists; the index is what vv_create takes. Returns how many
// there are, or a negative error. `flags`: 1 = usable (64-bit integers),
// 2 = has double.
VV_EXPORT int vv_device(int index, char *name, int nameBytes, uint32_t *vendor, uint32_t *type, uint32_t *flags)
{
    InstanceApi *api = instance_api();
    if (!api)
        return -1;
    std::vector<VkPhysicalDevice> devices = physical_devices(api);
    if (index >= 0 && index < (int)devices.size()) {
        VkPhysicalDeviceProperties properties;
        api->vkGetPhysicalDeviceProperties(devices[(size_t)index], &properties);
        VkPhysicalDeviceFeatures features;
        api->vkGetPhysicalDeviceFeatures(devices[(size_t)index], &features);
        if (name && nameBytes > 0) {
            strncpy(name, properties.deviceName, (size_t)nameBytes - 1);
            name[nameBytes - 1] = 0;
        }
        if (vendor) *vendor = properties.vendorID;
        if (type) *type = (uint32_t)properties.deviceType;
        if (flags) *flags = (features.shaderInt64 ? 1u : 0u) | (features.shaderFloat64 ? 2u : 0u);
    }
    return (int)devices.size();
}

// `flags`: 1 = use the GPU's double where it has one (the scores are the
// same; see shaders/vif_hori.slang); bits 8-15 = frames in flight (0: 3).
// Bits 16-18 leave out motion, VIF or ADM, and bits 20-27 all but the first
// N passes of VIF and ADM, to time the others. Bit 19: the frames come from
// GPU memory a decoder shares (vv_shared_next, vv_export). Bits 28-30: the
// scale-0 decouple shader's VARIANT, for the diagnosis.
VV_EXPORT int vv_create(vv_context **out, int device, int width, int height, int bitDepth, int flags)
{
    vv_context *context = new vv_context();
    int error = context->init(device, width, height, bitDepth, flags);
    if (error) {
        delete context;
        *out = nullptr;
        return error;
    }
    *out = context;
    return 0;
}

VV_EXPORT void vv_destroy(vv_context *context) { delete context; }

VV_EXPORT const char *vv_error() { return g_error.c_str(); }

// The next frame pair's luma planes (16-bit little-endian samples above 8
// bits). `score` 0: only motion is calculated for the frame (libvmaf's
// n_subsample), and `distorted` may be null.
VV_EXPORT int vv_submit(vv_context *context, const uint8_t *reference, ptrdiff_t referenceStride,
                        const uint8_t *distorted, ptrdiff_t distortedStride, int score)
{
    return context->submit(reference, referenceStride, distorted, distortedStride, score != 0);
}

// The memory to write the next pair's luma planes to (rows *stride bytes
// apart), and vv_commit to score them: vv_submit without its copy.
VV_EXPORT int vv_staging(vv_context *context, uint8_t **reference, uint8_t **distorted, uint32_t *stride)
{
    *stride = context->strideBytes;
    if (context->shared)
        return fail(-3, "this context takes its frames from GPU memory");
    return context->staging(reference, distorted);
}

VV_EXPORT int vv_commit(vv_context *context, int score) { return context->commit(score != 0); }

// For a context made with flag bit 19 (frames from GPU memory): the staging
// buffer the next pair's luma planes are to be written into by the API that
// imported it -- *slot says which (they take turns), the reference's rows
// start at offset 0 and the distorted's at *planeBytes, *stride bytes apart.
// Once they are written (and that API has finished writing), vv_commit.
// Returns how many slots there are.
VV_EXPORT int vv_shared_next(vv_context *context, int *slot, uint32_t *stride, uint32_t *planeBytes)
{
    if (!context->shared)
        return fail(-3, "this context takes its frames from host memory");
    uint8_t *unused[2];
    if (int error = context->staging(&unused[0], &unused[1]))
        return error;
    *slot = (int)context->nextSlot;
    *stride = context->strideBytes;
    *planeBytes = context->disOffset;  // where the distorted plane starts
    return (int)context->slots.size();
}

// A slot's staging buffer as a Win32 handle another API on this GPU imports
// (an opaque Win32 handle of a dedicated allocation of *bytes: CUDA's
// cuImportExternalMemory, or Vulkan's VkImportMemoryWin32HandleInfoKHR with
// what vv_shared_device gives). The caller closes the handle once it has
// imported it; the memory lives as long as the context.
VV_EXPORT int vv_export(vv_context *context, int slot, void **handle, uint64_t *bytes)
{
    if (!context->shared || slot < 0 || slot >= (int)context->slots.size())
        return fail(-3, "no such shared buffer");
    VkMemoryGetWin32HandleInfoKHR info = { VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR };
    info.memory = context->slots[(size_t)slot].staging.memory;
    info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    HANDLE win32 = nullptr;
    VkResult result = context->getMemoryHandle(context->device, &info, &win32);
    if (result != VK_SUCCESS)
        return fail(-1, "vkGetMemoryWin32HandleKHR failed (" + std::to_string(result) + ")");
    *handle = win32;
    *bytes = context->slots[(size_t)slot].staging.allocation;
    return 0;
}

// For a context made with flag bit 19: what another Vulkan device must match
// to import the staging buffers vv_export gives -- the GPU and driver
// (VkPhysicalDeviceIDProperties' deviceUUID and driverUUID, 16 bytes each:
// Vulkan imports an opaque handle only where both are the exporter's) -- and
// the memory type they were allocated from, the index the importer allocates
// with (the same GPU and driver list the same types).
VV_EXPORT int vv_shared_device(vv_context *context, uint8_t *deviceUuid, uint8_t *driverUuid, uint32_t *memoryType)
{
    if (!context->shared)
        return fail(-3, "this context takes its frames from host memory");
    memcpy(deviceUuid, context->deviceUuid, VK_UUID_SIZE);
    memcpy(driverUuid, context->driverUuid, VK_UUID_SIZE);
    *memoryType = context->slots[0].staging.memoryType;
    return 0;
}

// Waits for every submitted frame.
VV_EXPORT int vv_flush(vv_context *context) { return context->flush(); }

// VMAF v1's CAMBI of the distorted frames on the GPU, for a context made by
// vv_create_v1, before its first frame. `options`: cambi_high_res_speedup,
// cambi_vis_lum_threshold, cambi_max_val, the topk in effect, window_size,
// tvi_threshold, max_log_contrast, the eotf in effect (0 bt1886, 1 pq), and
// test flags: 1 pools every scale on the CPU (0: only where the GPU's sum is
// not exact in a double), 2 counts the c-values pixel by pixel (CVALUES, not
// CVALUES_SLIDE): nine doubles. -4 for options it does
// not calculate (max_log_contrast other than 2).
VV_EXPORT int vv_v1_cambi(vv_context *context, const double *options)
{
    return context->enable_cambi(options);
}

namespace {

// A c-value's bits as units of 2^-24, false when it is not a whole number of them.
bool cambi_units(uint32_t bits, uint64_t &units)
{
    if (bits == 0) { units = 0; return true; }
    const uint32_t e = bits >> 23;
    const uint64_t mantissa = (bits & 0x7FFFFFu) | 0x800000u;
    if (e < 126 || e >= 142) return false;
    units = mantissa << (e - 126);
    return true;
}

}  // namespace

// Every frame's CAMBI score (cambi.c's, with cambi_max_val), after vv_flush;
// NaN for a frame that was not scored. Returns the number of frames, or -5
// when a frame has a sample above its bit depth's largest (as cambi.c fails).
VV_EXPORT int vv_v1_cambi_scores(vv_context *context, double *out, unsigned frames)
{
    if (!context->cambi)
        return fail(-3, "CAMBI is not calculated in this context");
    const std::vector<FrameSums> &sums = context->sums;
    const V1CambiConstants &c = context->cambiConst;
    const unsigned n = (unsigned)std::min<size_t>(frames, sums.size());
    for (unsigned i = 0; i < n; ++i) {
        out[i] = NAN;
        if (!sums[i].scored)
            continue;
        const uint64_t *slots = sums[i].slots;
        if (slots[kSlotCambiInvalid] & 1)
            return fail(-5, "CAMBI: frame " + std::to_string(i) + " has a sample above its bit depth's largest");
        double scores[V1_CAMBI_SCALES];
        for (int scale = 0; scale < V1_CAMBI_SCALES; ++scale) {
            if (sums[i].cambiPooledScales & (1u << scale)) {  // pooled on the CPU when the frame was done
                scores[scale] = sums[i].cambiPooled[scale];
                continue;
            }
            const uint64_t *s = &slots[kSlotCambi + scale * 4];
            uint64_t threshold;
            const bool exact = (s[3] & 1) == 0 && cambi_units((uint32_t)s[1], threshold);
            const uint64_t total = exact ? s[0] + s[2] * threshold : 0;
            if (!exact || total >= (1ull << 53))
                return fail(-6, "CAMBI: frame " + std::to_string(i) + "'s top k is not exact in a double");
            scores[scale] = (double)total * (1.0 / 16777216.0) / c.topk_elements[scale];
        }
        out[i] = v1_cambi_score(scores, c.pixels_in_window, context->cambiOptions.max_val);
    }
    return (int)sums.size();
}

// For tests and benchmarks of a context made with flag bit 19 (frames from
// GPU memory): a slot's staging buffer filled with these two luma planes
// (rows of the context's stride, as vv_shared_next describes them), as a
// decoder would write them.
VV_EXPORT int vv_test_fill_slot(vv_context *context, int slot, const uint8_t *reference, const uint8_t *distorted)
{
    if (!context->shared || slot < 0 || slot >= (int)context->slots.size())
        return fail(-3, "no such shared buffer");
    std::vector<uint8_t> planes((size_t)context->disOffset + context->planeBytes);
    memcpy(planes.data(), reference, context->planeBytes);
    memcpy(planes.data() + context->disOffset, distorted, context->planeBytes);
    return context->upload(context->slots[(size_t)slot].staging, planes.data(), planes.size());
}

// VMAF v1's SpEED chroma on the GPU (and libvmaf's est_params on the CPU,
// v1_speed.c), for a context made by vv_create_v1, before its first frame;
// its frames then go in with vv_submit_v1 (or, from a decoder, their chroma
// with vv_chroma_staging). `options`: speed_kernelscale, speed_prescale,
// 1 if speed_prescale_method is bilinear (else 0), speed_sigma_nn,
// speed_nn_floor, speed_weight_var_mode, speed_max_val: seven doubles. -4
// for what it does not calculate (a prescale method other than bilinear, a
// bit depth picture_copy reads as 8 bits).
VV_EXPORT int vv_v1_speed(vv_context *context, const double *options)
{
    return context->enable_speed(options);
}

// The next frame pair, with its chroma: `planes` and `strides` of the
// reference's Y, U and V and of the distorted's (4:2:0; samples as
// vv_submit's). `score` as vv_submit's (the distorted's planes may then be null).
VV_EXPORT int vv_submit_v1(vv_context *context, const uint8_t *const *planes, const ptrdiff_t *strides, int score)
{
    return context->submit_v1(planes, strides, score != 0);
}

// For a frame begun with vv_staging or vv_shared_next, in a context with
// SpEED: where its four chroma planes go before vv_commit -- the reference's
// U and V, then the distorted's, at planes[0..3], rows *stride bytes apart
// (w/2 x h/2 samples: 4:2:0's chroma as libvmaf takes it).
VV_EXPORT int vv_chroma_staging(vv_context *context, uint8_t **planes, uint32_t *stride)
{
    if (!context->speed)
        return fail(-3, "SpEED is not calculated in this context");
    if (!context->pending)
        return fail(-3, "no frame was started");
    uint8_t *chroma = (uint8_t *)context->pending->speedChroma.mapped;
    for (int i = 0; i < 4; ++i)
        planes[i] = chroma + (size_t)i * context->chromaPlaneBytes;
    *stride = context->chromaStrideBytes;
    if (context->shared)
        context->pending->chromaFromHost = true;  // SpEED reads this frame's from here
    return 0;
}

// For a context made with flag bit 19 (frames from GPU memory) with SpEED:
// where in each slot's staging buffer (the memory vv_export gives) the four
// chroma planes go -- the reference's U and V, then the distorted's, from
// *offset, *spacing bytes apart, rows *stride bytes apart (w/2 x h/2
// samples). A decoder writes them there on the GPU, as the lumas; or the
// frame's go through vv_chroma_staging instead.
VV_EXPORT int vv_shared_chroma(vv_context *context, uint64_t *offset, uint32_t *spacing, uint32_t *stride)
{
    if (!context->shared || !context->speed)
        return fail(-3, "no chroma in the shared buffers");
    *offset = context->sharedChroma;
    *spacing = context->sharedChromaSpacing;
    *stride = context->chromaStrideBytes;
    return 0;
}

// Every frame's SpEED chroma scores, after vv_flush: three doubles per frame,
// libvmaf's speed_chroma_u_score, _v_score and _uv_score (with
// speed_chroma_max_val); NaN for a frame that was not scored. Returns the
// number of frames.
VV_EXPORT int vv_v1_speed_scores(vv_context *context, double *out, unsigned frames)
{
    if (!context->speed)
        return fail(-3, "SpEED is not calculated in this context");
    context->speedPool.wait();
    const std::vector<FrameSums> &sums = context->sums;
    const unsigned n = (unsigned)std::min<size_t>(frames, sums.size());
    for (unsigned i = 0; i < n; ++i) {
        if (!context->speedPool.result(i, out + 3 * (size_t)i))
            out[3 * (size_t)i] = out[3 * (size_t)i + 1] = out[3 * (size_t)i + 2] = NAN;
    }
    return (int)sums.size();
}

// Profiling, for the speed work: started before the first frame, it times
// every pass on the GPU (timestamps; the passes run one after the other) and
// the CPU's steps. vv_profile_text writes what was summed so far, a line
// each: the name, the milliseconds in all, how many times.
VV_EXPORT int vv_profile(vv_context *context) { return context->enable_profile(); }

VV_EXPORT int vv_profile_text(vv_context *context, char *out, int bytes)
{
    std::string text;
    char line[160];
    for (int i = 0; i < vv_context::kProfileLabels; ++i) {
        if (!context->gpuCount[i])
            continue;
        const char *name = i == vv_context::kProfileUpload ? "(upload)"
                         : i == vv_context::kProfileCopies ? "(copies)" : kShaders[i].name;
        snprintf(line, sizeof line, "gpu %s %.3f %llu\n", name, context->gpuNs[i] / 1e6,
                 (unsigned long long)context->gpuCount[i]);
        text += line;
    }
    context->cpuNs[vv_context::kCpuSpeed] = context->speedPool.busyNs;  // on its threads
    context->cpuCount[vv_context::kCpuSpeed] = context->speedPool.jobs;
    static const char *const steps[vv_context::kCpuSteps] = { "copy", "record", "wait", "speed_threads", "cambi_pool" };
    for (int i = 0; i < vv_context::kCpuSteps; ++i) {
        snprintf(line, sizeof line, "cpu %s %.3f %llu\n", steps[i], context->cpuNs[i] / 1e6,
                 (unsigned long long)context->cpuCount[i]);
        text += line;
    }
    if (out && bytes > 0) {
        strncpy(out, text.c_str(), (size_t)bytes - 1);
        out[bytes - 1] = 0;
    }
    return (int)text.size() + 1;
}

// A frame's features (kFeatures doubles), after vv_flush. Returns 1 when
// the frame was scored, 0 when only its motion was (the others are 0 then).
VV_EXPORT int vv_features(vv_context *context, unsigned index, double *out)
{
    if (index >= context->sums.size())
        return fail(-3, "no such frame");
    const std::vector<FrameSums> &sums = context->sums;
    const unsigned w = (unsigned)context->w, h = (unsigned)context->h;
    for (int i = 0; i < kFeatures; ++i)
        out[i] = 0.0;
    // As libvmaf (integer_motion_cuda.c): the first frame's motion is 0, and
    // motion2 is the lesser of a frame's motion and the next frame's.
    auto motion = [&](unsigned frame) {
        return frame == 0 ? 0.0 : normalize_and_scale_sad(sums[frame].slots[kSlotSad], w, h);
    };
    const double own = motion(index);
    out[kMotion] = own;
    if (index == 0) {
        out[kMotion2] = 0.0;
    } else if (index + 1 < sums.size()) {
        const double next = motion(index + 1);
        out[kMotion2] = own < next ? own : next;
    } else {
        out[kMotion2] = own;
    }
    const FrameSums &frame = sums[index];
    if (!frame.scored)
        return 0;
    vif_scores(frame.slots, 0, &out[kVifScale0]);
    vif_scores(frame.slots, 1, &out[kVifScale0Neg]);
    out[kAdm2] = adm_score(frame.slots, 0, w, h);
    out[kAdm2Neg] = adm_score(frame.slots, 1, w, h);
    return 1;
}

// A frame's raw sums (kSlots 64-bit values), for tests.
VV_EXPORT int vv_sums(vv_context *context, unsigned index, uint64_t *out, int count)
{
    if (index >= context->sums.size())
        return fail(-3, "no such frame");
    memcpy(out, context->sums[index].slots, sizeof(uint64_t) * (size_t)std::min(count, (int)kSlots));
    return kSlots;
}

VV_EXPORT const char *vv_device_name(vv_context *context) { return context->deviceName.c_str(); }

// A device buffer's first `bytes` bytes, after vv_flush; for tests. `which`:
// 0 admR, 1 admA, 2 admF, 3 bandsRef[0], 4 bandsDis[0], 5 bandsRef[1], 6 bandsDis[1], 7 vifTmp,
// 8 the division table, 9 the logarithm table.
VV_EXPORT int vv_read_buffer(vv_context *context, int which, void *out, uint64_t bytes)
{
    Buffer *all[] = { &context->admR, &context->admA, &context->admF, &context->bandsRef[0], &context->bandsDis[0],
                      &context->bandsRef[1], &context->bandsDis[1], &context->vifTmp, &context->divTable,
                      &context->logTable };
    if (which < 0 || which >= (int)(sizeof all / sizeof all[0]))
        return fail(-3, "no such buffer");
    Buffer *source = all[which];
    bytes = std::min<uint64_t>(bytes, source->size);
    Buffer staging;
    if (int error = context->create_buffer(staging, bytes, true))
        return error;
    const DeviceApi &vk = context->vk;
    VkCommandBuffer cb = context->slots[0].commands;
    VkCommandBufferBeginInfo begin = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vk.vkResetCommandBuffer(cb, 0);
    vk.vkBeginCommandBuffer(cb, &begin);
    barrier(vk, cb, kCompute | kTransfer, kTransfer);
    VkBufferCopy region = { 0, 0, bytes };
    vk.vkCmdCopyBuffer(cb, source->buffer, staging.buffer, 1, &region);
    vk.vkEndCommandBuffer(cb);
    VkSubmitInfo submit = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cb;
    VkResult result = vk.vkQueueSubmit(context->queue, 1, &submit, VK_NULL_HANDLE);
    if (result == VK_SUCCESS)
        result = vk.vkDeviceWaitIdle(context->device);
    if (result == VK_SUCCESS)
        memcpy(out, staging.mapped, (size_t)bytes);
    vk.vkDestroyBuffer(context->device, staging.buffer, nullptr);
    vk.vkFreeMemory(context->device, staging.memory, nullptr);
    context->buffers.pop_back();
    return result == VK_SUCCESS ? 0 : fail(-1, "reading a GPU buffer failed");
}

// VMAF v1: a context that calculates ADM3 and motion3 as libvmaf's CPU code
// does (integer_adm.c, integer_motion.c). `options`: adm_norm_view_dist,
// adm_ref_display_height, adm_csf_mode, adm_noise_weight, adm_dlm_weight,
// adm_min_val, motion_max_val, motion_five_frame_window,
// motion_moving_average (nine doubles). Frames go in with vv_submit or
// vv_staging / vv_commit as for VMAF v0.6.1.
VV_EXPORT int vv_create_v1(vv_context **out, int device, int width, int height, int bitDepth, int flags,
                           const double *options)
{
    vv_context *context = new vv_context();
    context->v1 = true;
    V1Options &o = context->options;
    o.viewDistance = options[0];
    o.displayHeight = (int)options[1];
    o.csfMode = (int)options[2];
    o.noiseWeight = options[3];
    o.dlmWeight = options[4];
    o.minValue = options[5];
    o.motionMax = options[6];
    o.fiveFrameWindow = options[7] != 0;
    o.movingAverage = options[8] != 0;
    int error = o.viewDistance * o.displayHeight < kAdmNormViewDist * kAdmRefDisplayHeight
        ? fail(-3, "VMAF v1: the viewing distance is nearer than ADM's 16-bit pipeline takes")
        : context->init(device, width, height, bitDepth, flags);
    if (error) {
        delete context;
        *out = nullptr;
        return error;
    }
    *out = context;
    return 0;
}

// Every frame's VMAF v1 features, after vv_flush: six doubles per frame --
// adm3, aim, adm2 (0 for a frame that was not scored), motion3, motion2 and
// the motion SAD score. Motion as libvmaf's flush() of integer_motion.c
// derives it from all the frames' SAD scores. Returns the number of frames.
VV_EXPORT int vv_features_v1(vv_context *context, double *out, unsigned frames)
{
    const std::vector<FrameSums> &sums = context->sums;
    const V1Options &o = context->options;
    const unsigned n = (unsigned)std::min<size_t>(frames, sums.size());
    const unsigned w = (unsigned)context->w, h = (unsigned)context->h;
    const unsigned min_idx = o.fiveFrameWindow ? 2 : 1, stride = o.fiveFrameWindow ? 2 : 1;
    std::vector<double> sad(sums.size());
    for (size_t i = 0; i < sums.size(); ++i) {
        double score = 0.;
        if (i >= min_idx) {
            const double scaled = (double)sums[i].slots[kSlotSad] / 256. / (w * h) * 1.0;  // motion_fps_weight
            score = scaled < o.motionMax ? scaled : o.motionMax;
        }
        sad[i] = score;
    }
    // motion_blend() with the default blend factor 1: the score itself.
    double stamp_value = 0.;
    if (sums.size() > min_idx)
        stamp_value = sad[min_idx] < o.motionMax ? sad[min_idx] : o.motionMax;
    double prev_processed = 0.;
    for (size_t i = 0; i < sums.size(); ++i) {
        double motion2;
        if (i < min_idx) {
            motion2 = 0.;
        } else {
            const int lo_idx = (int)i - (int)(stride - 1);
            const size_t hi_idx = i + 1;
            if (hi_idx >= sums.size()) {
                motion2 = sad[i];
            } else if (lo_idx >= (int)min_idx) {
                motion2 = sad[(size_t)lo_idx] < sad[hi_idx] ? sad[(size_t)lo_idx] : sad[hi_idx];
            } else {
                motion2 = sad[hi_idx];
            }
        }
        double motion3;
        if (i < min_idx) {
            motion3 = stamp_value;
            prev_processed = stamp_value;
        } else {
            const double processed = motion2 < o.motionMax ? motion2 : o.motionMax;
            motion3 = o.movingAverage ? (processed + prev_processed) / 2.0 : processed;
            prev_processed = processed;
        }
        if (i < n) {
            double *row = out + i * 6;
            row[0] = row[1] = row[2] = 0.0;
            if (sums[i].scored)
                v1_adm(*context, sums[i].slots, row);
            row[3] = motion3;
            row[4] = motion2;
            row[5] = sad[i];
        }
    }
    return (int)sums.size();
}
