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
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include <vulkan/vulkan.h>

#include "shaders_spv.h"
#include "v1_host.h"
#include "v1_speed.h"
#include "cambi_lut.h"
#include "../../libvmaf/include/libvmaf/picture_gpu_plan.h"

#define VV_EXPORT extern "C" __declspec(dllexport)

// ------------------------------------------------------------------ Vulkan

#define VK_INSTANCE_FUNCTIONS(X) \
    X(vkDestroyInstance) X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties) \
    X(vkGetPhysicalDeviceFeatures) X(vkGetPhysicalDeviceMemoryProperties) \
    X(vkGetPhysicalDeviceQueueFamilyProperties) X(vkCreateDevice) X(vkGetDeviceProcAddr) \
    X(vkEnumerateDeviceExtensionProperties) X(vkGetPhysicalDeviceFeatures2) X(vkGetPhysicalDeviceProperties2)

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
    X(vkWaitForFences) X(vkResetFences) X(vkQueueSubmit) X(vkDeviceWaitIdle) X(vkGetFenceStatus) \
    X(vkCreateImage) X(vkDestroyImage) X(vkGetImageMemoryRequirements) X(vkBindImageMemory) \
    X(vkCmdCopyImageToBuffer) X(vkCreateQueryPool) X(vkDestroyQueryPool) X(vkCmdResetQueryPool) \
    X(vkCmdWriteTimestamp) X(vkGetQueryPoolResults) X(vkCreateImageView) X(vkDestroyImageView)

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
// loaded by the first call, never inside DllMain. Made once whichever
// threads ask first; why there is none is told to every caller, on its own
// thread, not only to the first.
InstanceApi *instance_api()
{
    static InstanceApi api;
    static std::string why;
    static std::once_flag once;
    std::call_once(once, [] {
        api.library = LoadLibraryW(L"vulkan-1.dll");
        if (!api.library) {
            why = "vulkan-1.dll was not found (no Vulkan driver)";
            return;
        }
        api.vkGetInstanceProcAddr =
            (PFN_vkGetInstanceProcAddr)GetProcAddress(api.library, "vkGetInstanceProcAddr");
        auto create = api.vkGetInstanceProcAddr
            ? (PFN_vkCreateInstance)api.vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkCreateInstance") : nullptr;
        if (!create) {
            why = "vulkan-1.dll has no vkCreateInstance";
            return;
        }
        VkApplicationInfo application = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
        application.pApplicationName = "libvmaf-fast";
        application.apiVersion = VK_API_VERSION_1_1;
        VkInstanceCreateInfo info = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
        info.pApplicationInfo = &application;
        VkInstance instance;
        VkResult result = create(&info, nullptr, &instance);
        if (result != VK_SUCCESS) {
            why = "vkCreateInstance failed (" + std::to_string(result) + ")";
            return;
        }
#define X(name) api.name = (PFN_##name)api.vkGetInstanceProcAddr(instance, #name);
        VK_INSTANCE_FUNCTIONS(X)
#undef X
        api.instance = instance;
    });
    if (!api.instance) {
        g_error = why;
        return nullptr;
    }
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
enum { kVifTmpWords = 5 };  // VIF_TMP_WORDS of shaders/vif_filter.slang
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
    kSlotCiede = kSlotCambiInvalid + 1,  // CIEDE2000 (vv_v1_ciede): the pixels' dE00 in units of 2^-20
    kSlots = kSlotCiede + 1
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

// What the API reads of a frame (vv_features, vv_features_v1,
// vv_v1_cambi_scores, vv_v1_ciede_scores), made from its sums when the GPU is done with it
// (collect), by the same functions on the same sums as when they were
// read later, so the sums themselves (kSlots 64-bit values, about 750
// bytes) need not be kept: only the first kRawFrames frames' are, for
// vv_sums. Motion stays as its SAD sum: a frame's motion2 needs the next
// frame's.
struct FrameRecord {
    uint64_t sad = 0;
    float vif[2][kScales] = {};  // VIF per scale, gain limit 100 and 1: vif_scores' quotients are floats
    double adm[2] = {};          // ADM2, gain limit 100 and 1
    double v1Adm[3] = {};        // VMAF v1's adm3, aim and adm2 (v1_adm)
    double cambi = 0.0;          // VMAF v1's CAMBI
    double ciede = 0.0;          // CIEDE2000 (vv_v1_ciede)
    uint8_t scored = 0;
    int8_t cambiError = 0;       // -5 or -6: vv_v1_cambi_scores fails at this frame
};
enum { kRawFrames = 256 };

// Items in blocks of kBlock, so that adding one never moves the others: a
// std::vector's growth copies them all and holds up to twice their size.
template <typename T>
class Blocks {
public:
    size_t size() const { return count; }
    T &operator[](size_t i) { return blocks[i / kBlock][i % kBlock]; }
    const T &operator[](size_t i) const { return blocks[i / kBlock][i % kBlock]; }

    // One more item, value-initialized.
    void add()
    {
        if (count % kBlock == 0)
            blocks.emplace_back(new T[kBlock]());
        ++count;
    }

private:
    enum { kBlock = 1024 };
    std::vector<std::unique_ptr<T[]>> blocks;
    size_t count = 0;
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
    VkPipeline pipeline = VK_NULL_HANDLE;  // a shader without specialization constants
    uint32_t bindings = 0;
    // A shader with specialization constants (SpecId) is compiled for each
    // set of push constants its passes use (constant k: word k).
    bool specialized = false;
    std::vector<std::pair<std::vector<uint32_t>, VkPipeline>> versions;
    uint32_t readOnly = 0;  // the bindings it only reads (NonWritable), a bit each
};

enum { kMaxSlots = 16 };

struct Pass {
    int shader;
    VkPipeline pipeline;  // the shader's, for this pass's push constants
    VkDescriptorSet set;
    uint32_t constants[32];
    uint32_t constantBytes;
    uint32_t groups[3];
    // Its buffers, and its dependency level in its list: one more than the
    // highest of the earlier passes it shares a buffer with (vv_context::
    // shares). The passes of a level run with no barrier between them.
    const Buffer *bound[9];  // kMaxBindings
    uint32_t boundCount;
    uint32_t level;
    VkBuffer indirect;                // dispatched with the sizes the GPU wrote here, when set
    VkDeviceSize indirectOffset;
    // Per slot, where the pass reads the slot's own staging buffer (VMAF v1's
    // passes that read the pictures): the set for each slot, in place of `set`.
    std::vector<VkDescriptorSet> slotSets;
    // Independent of the pass after it (neither writes what the other reads
    // or writes): no barrier between them, so that the GPU runs them together
    // (but when profiling, which times each).
    bool overlapNext = false;
    // vv_pictures: which of VMAF v1's passes that read the pictures it is,
    // their textures bound and places pushed each frame (PictureRole).
    int pictureRole = 0;
};

// vv_pictures' passes: motion (the reference and the previous one's luma),
// ADM's scale 0 (both pictures' luma), CAMBI's FRONT (the distorted luma) and
// SpEED's DEC (both pictures' chroma).
enum PictureRole { kPictureNone, kPictureMotion, kPictureAdm, kPictureCambi, kPictureSpeed };

// A buffer a pass binds, or part of one (at a multiple of the storage
// buffers' offset alignment).
struct Bound {
    Buffer *buffer;
    VkDeviceSize offset = 0, range = VK_WHOLE_SIZE;
    Bound(Buffer *b) : buffer(b) {}
    Bound(Buffer *b, VkDeviceSize o, VkDeviceSize r) : buffer(b), offset(o), range(r) {}
};
const Bound kPictureBinding(nullptr);  // a picture's texture (vv_pictures), bound each frame

struct Slot {
    Buffer staging, result;
    Buffer stagingDis;  // vv_context::direct: the distorted's plane (staging: the reference's)
    Buffer cambiKeep;                 // VMAF v1's CAMBI: the c-values of scales the CPU pools (host-visible)
    std::vector<Pass> cambiKeepPasses;
    Buffer speedChroma;               // VMAF v1's SpEED: the pair's chroma planes (host-visible)
    // With frames from a decoder (shared): SpEED reads the chroma planes from
    // `staging` (sharedChroma), unless the frame's came through speedChroma
    // (vv_chroma_staging): then speedPassesHost.
    bool chromaFromHost = false;
    std::vector<Pass> speedPassesHost;
    // With frames from a decoder (shared): a luma plane the CPU wrote instead
    // (vv_shared_host_plane: [0] the reference's, [1] the distorted's), made at
    // its first use, and whether this frame's came that way.
    Buffer hostPlane[2];
    bool planeFromHost[2] = { false, false };
    Buffer speedKeep;                 // and the filtered planes est_params reads (host-visible)
    std::vector<Pass> speedPasses;
    Buffer ciedeChroma;               // CIEDE2000: the pair's chroma planes (host-visible)
    std::vector<Pass> ciedePasses;
    // vv_pictures: the frame's pictures (indices of vv_context::pictureCache:
    // the reference, the distorted one, the reference before it for motion)
    // and their first samples' places (previous x, y, reference x, y,
    // distorted x, y: the push constants' last words, shaders/common.slang).
    int pictureRef = -1, pictureDis = -1, picturePrev = -1;
    uint32_t places[6] = {};
    VkCommandBuffer commands = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool busy = false;
    // Committed, but not yet submitted by the submit thread (vv_context::
    // submitter): its fence cannot signal before. Under submitMutex.
    bool queued = false;
    bool scored = false;
    unsigned index = 0;
};

enum { kPushBytes = 128, kMaxBindings = 9 };
// vif_fused.slang's tile rows (TH) at each scale: the build script's TH
// defines. Its tiles are 160 pixels wide.
const uint32_t kVifTileRows[4] = { 2, 2, 2, 2 };
static_assert(kMaxBindings <= sizeof(Pass::bound) / sizeof(Pass::bound[0]), "Pass::bound holds a pass's buffers");
// adm_dcm.slang's tile of contrast masking's positions (its TX x TY).
const int kDcmTile[2] = { 16, 8 };

// VMAF v1's options for ADM3 and motion3 (the model's feature_opts_dicts).
struct V1Options {
    double viewDistance = 3.0;   // adm_norm_view_dist
    int displayHeight = 1080;    // adm_ref_display_height
    int csfMode = 0;             // adm_csf_mode: 0 Watson97, 2 Barten-Watson blend, 3 its MAE fit
    bool skipScale0 = false;     // adm_skip_scale0: scale 0 only downsamples (HDR VMAF)
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
        if (index >= results.size() || !results[index].done)
            return false;
        memcpy(out, results[index].scores, 3 * sizeof(double));
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
            while (results.size() <= job.first)  // frames finish in any order
                results.add();
            memcpy(results[job.first].scores, scores.data(), sizeof results[job.first].scores);
            results[job.first].done = true;
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
    struct Result {
        double scores[3];
        bool done;
    };
    Blocks<Result> results;  // by frame
    std::vector<std::thread> threads;
    std::vector<V1Speed *> states;
    size_t floats = 0, pending = 0;
    bool stopping = false;
};

// VMAF v1's CAMBI: spatial pooling of the scales whose top k the GPU cannot
// sum exactly in a double (scale 0 of most 4K frames), on worker threads, a
// frame a job, from a copy of those scales' c-values. It is cambi.c's
// quick-select and serial double sum, about 18 ms a 4K frame, which the
// thread that submits the frames would otherwise spend while the GPU waits.
// The same code on the same values whichever thread runs it, so the scores
// are identical. The threads start with the first job: contexts whose top k
// is always exact (most below 4K) start none.
class CambiPool {
public:
    // Pools a job's scales (bits `scales`; their c-values back to back in
    // `values`, which it reorders) and works out the frame's score; returns
    // vv_context::cambi_score's error code.
    using Scorer = std::function<int(const uint64_t *sums, unsigned scales, float *values, double &out)>;

    ~CambiPool() { stop(); }

    void configure(Scorer scorer, int threadCount)
    {
        score = std::move(scorer);
        workers = threadCount;
    }

    // A buffer for a job's c-values, once fewer than kMaxQueued jobs wait, so
    // that at most kMaxQueued + workers buffers exist (each keeps its size).
    std::vector<float> buffer()
    {
        std::unique_lock<std::mutex> lock(mutex);
        done.wait(lock, [this] { return queue.size() < kMaxQueued; });
        std::vector<float> values;
        if (!spare.empty()) {
            values = std::move(spare.back());
            spare.pop_back();
        }
        values.clear();
        return values;
    }

    // A scored frame's job: its sums (kSlots) and the c-values of `scales`.
    void submit(unsigned index, const uint64_t *sums, unsigned scales, std::vector<float> values)
    {
        Job job{ index, scales, {}, std::move(values) };
        memcpy(job.sums.data(), sums, sizeof job.sums);
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (!started)
                start();
            if (!threads.empty()) {
                queue.push_back(std::move(job));
                ++pending;
                work.notify_one();
                return;
            }
        }
        run_job(job);  // no thread could be started: on this one
    }

    void wait()
    {
        std::unique_lock<std::mutex> lock(mutex);
        done.wait(lock, [this] { return pending == 0; });
    }

    // The pooled frames' scores into their records (after wait()).
    void merge(Blocks<FrameRecord> &records)
    {
        std::lock_guard<std::mutex> lock(mutex);
        for (size_t i = 0; i < results.size() && i < records.size(); ++i) {
            if (results[i].done) {
                records[i].cambi = results[i].cambi;
                records[i].cambiError = results[i].error;
            }
        }
    }

    double busyNs = 0.0;  // the workers' time, summed (the profile)
    uint64_t jobs = 0;

private:
    enum { kMaxQueued = 3 };  // a 4K job holds 33 MB
    struct Job {
        unsigned index, scales;
        std::array<uint64_t, kSlots> sums;
        std::vector<float> values;
    };

    void start()  // (under the mutex)
    {
        started = true;
        try {
            for (int i = 0; i < workers; ++i)
                threads.emplace_back([this] { run(); });
        } catch (...) {  // the threads started so far do the work
        }
    }

    void run()
    {
        for (;;) {
            Job job;
            {
                std::unique_lock<std::mutex> lock(mutex);
                work.wait(lock, [this] { return stopping || !queue.empty(); });
                if (queue.empty())
                    return;
                job = std::move(queue.front());
                queue.pop_front();
            }
            done.notify_all();  // room in the queue: a buffer() waiting goes on now, not after this job
            run_job(job);
            std::lock_guard<std::mutex> lock(mutex);
            --pending;
            done.notify_all();
        }
    }

    void run_job(Job &job)
    {
        const auto begin = std::chrono::steady_clock::now();
        double cambi = 0.0;
        const int error = score(job.sums.data(), job.scales, job.values.data(), cambi);
        const double ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - begin).count();
        std::lock_guard<std::mutex> lock(mutex);
        while (results.size() <= job.index)  // frames finish in any order
            results.add();
        Result &result = results[job.index];
        result.cambi = cambi;
        result.error = (int8_t)error;
        result.done = true;
        spare.push_back(std::move(job.values));
        busyNs += ns;
        ++jobs;
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
    }

    Scorer score;
    int workers = 1;
    std::mutex mutex;
    std::condition_variable work, done;
    std::deque<Job> queue;
    std::vector<std::vector<float>> spare;
    struct Result {
        double cambi;
        int8_t error;
        bool done;
    };
    Blocks<Result> results;  // by frame
    std::vector<std::thread> threads;
    size_t pending = 0;
    bool started = false, stopping = false;
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
    // adm_fused.slang's BAND (scale 0's; 16 at the others, which are small) and OWN.
    enum { kAdmParams = 56, kAdmBandRows = 32, kAdmBandRowsSmall = 8, kAdmOwnColumns = 126 };
    Buffer admParams, admPartial;
    std::vector<int32_t> admParamValues;
    int skip = 0;  // timing tests: 1 = no motion, 2 = no VIF, 4 = no ADM
    int passLimit = 0;  // timing tests: only the first N scored passes
    int decoupleVariant = 0;  // adm_decouple_0's VARIANT (shaders/adm_decouple.slang)
    // VIF's two passes as one (vif_fused); VV_VIF_FUSED=0 for the two, to compare.
    bool vifFused = true;
    // The GPU has subgroup arithmetic (Vulkan 1.1's, optional): motion adds
    // up its groups' distances with it (motion_8w, _16w), as vif_fused its sums.
    bool subgroupSums = true;
    // Subgroups of 32 lanes for the shaders that are faster in them (on an AMD
    // GPU, whose driver gives compute shaders 64 lanes unless asked): 32 if the
    // GPU takes a required size (VK_EXT_subgroup_size_control), else 0.
    uint32_t narrowSubgroup = 0;
    uint32_t strideBytes = 0, planeBytes = 0;
    // The first passes read each pair's frames where they were written (the
    // frame slot's staging buffers: the reference's in staging, the
    // distorted's in stagingDis, or in staging at disOffset where a decoder
    // writes them), not from picRef and picDis after a copy: on integrated
    // GPUs, discrete ones whose memory the CPU writes (barFrames), and frames
    // a decoder writes on the GPU (shared); but not for VMAF v1.
    // VV_DIRECT_FRAMES=0 for the copy, to compare.
    bool direct = true;
    // A discrete GPU whose memory the CPU can map as a whole (Resizable BAR:
    // a device-local, host-visible type on the device-local heap): the frames'
    // staging buffers there, the CPU writing them across PCIe itself, and
    // VMAF v0.6.1's first passes reading them there, so the GPU has no copy to
    // make (VMAF v1's copy reads them there). An RTX
    // 5090's 4K 10-bit pair: 1.29 -> 0.47 ms of GPU time, and the CPU's copy
    // into staging 1.85 -> 1.57 ms. VV_BAR_FRAMES=0 for staging in system
    // memory, to compare.
    bool barFrames = false;
    // The frames' luma planes come from another API on this GPU (a decoder's
    // CUDA, or its own Vulkan device), which writes them into the slots'
    // staging buffers: GPU memory it imports by the handles vv_export gives,
    // in place of host memory. Vulkan imports them only on the same GPU and
    // driver (vv_shared_device).
    bool shared = false;
    // Flag bit 1, with shared: VMAF v1's pictures to be converted
    // (vv_v1_convert), which a decoder writes for that into the shared
    // buffers (vv_convert_source); see enable_convert.
    bool convertsShared = false;
    // Flag bit 3, with shared: each slot's planes in two such buffers, the
    // reference's (staging) and the distorted's (stagingDis), for two
    // decoders that each import one (vv_export_part). A driver may import an
    // allocation only once a process: NVIDIA's on Windows refused a second
    // import of one, by CUDA or by another Vulkan device alike.
    bool split = false;
    PFN_vkGetMemoryWin32HandleKHR getMemoryHandle = nullptr;
    // A decoder's timeline semaphore (vv_import_timeline): its GPU copies into
    // the slots signal it, and a frame is submitted once the value of its own
    // (vv_commit_after) is reached, so that neither the decoder nor the
    // caller waits for the copies.
    bool timelines = false;  // the device takes timeline semaphores from other APIs
    VkSemaphore copied = VK_NULL_HANDLE;
    uint64_t copiedValue = 0;  // the next commit's, 0: none
    PFN_vkImportSemaphoreWin32HandleKHR importSemaphore = nullptr;
    PFN_vkCreateSemaphore createSemaphore = nullptr;
    PFN_vkDestroySemaphore destroySemaphore = nullptr;
    PFN_vkWaitSemaphores waitSemaphores = nullptr;
    // The submit thread, started by vv_import_timeline: it takes every
    // committed frame in order, waits on the CPU (asleep) until the decoder's
    // copies of it are done, then submits it. Not a submission that waits on
    // the GPU: NVIDIA's driver (616.92, RTX 5090) services a GPU wait on a
    // semaphore of another device with a thread of its own that polls the
    // clock, one per waiting device (1080p VMAF, NEG and v1 of a minute: the
    // run's CPU time 18.5 s with the GPU waiting, 13.3 s waited here, same
    // wall time).
    struct Submission {
        size_t slot;
        uint64_t after;  // the timeline value to wait for, 0: none
    };
    std::thread submitter;
    std::mutex submitMutex;  // the members below and the slots' `queued`
    std::condition_variable submitWork, submitDone;
    std::deque<Submission> submissions;
    bool submitStop = false;
    std::string submitError;  // the thread's failure, reported on the caller's thread
    // The queue is used from two threads once the submit thread runs:
    // vkQueueSubmit and vkDeviceWaitIdle hold this.
    std::mutex queueMutex;
    void submit_thread();
    VkResult submit_frame(Slot &slot, uint64_t after);
    bool queued(const Slot &slot);
    VkResult run_now(VkCommandBuffer commands);
    uint8_t deviceUuid[VK_UUID_SIZE] = {}, driverUuid[VK_UUID_SIZE] = {};
    // Or a decoder on this GPU writes them into Direct3D 11 textures, which
    // the context imports (vv_import_texture) and copies from on the GPU
    // (vv_commit_textures): the frames never leave the GPU's memory.
    struct Imported {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
    };
    std::vector<Imported> imported;
    PFN_vkGetMemoryWin32HandlePropertiesKHR getHandleProperties = nullptr;
    int pendingTextures[2] = { -1, -1 };
    int import_texture(void *handle, int *index);
    int commit_textures(int ref, int dis, bool score);
    unsigned completed();
    // The SIMD width each shader is compiled for (VK_EXT_subgroup_size_control:
    // 8, 16 or 32 lanes on Intel's GPUs), 0 for the driver's choice.
    uint32_t subgroupSizes[kShaderCount] = {};
    bool subgroupSizeControl = false;

    Pipeline pipelines[kShaderCount];
    std::vector<Buffer *> buffers;
    Buffer picRef, picDis, blur[2], vifTmp, rdRef[2], rdDis[2], logTable, divTable;
    // ADM's band images by scale parity: h, v, d (bandsRef, bandsDis) and a
    // (bandsARef, bandsADis), as common.slang's adm_hvd describes.
    Buffer bandsRef[2], bandsDis[2], bandsARef[2], bandsADis[2], admR, admA, admF, acc;
    // VMAF NEG's decoupled images, made beside VMAF's by one pass (adm_decouple's
    // BOTH); VV_ADM_BOTH=0 for a pass each, to compare.
    Buffer admRB, admAB, admFB;
    bool admBoth = true;
    // ADM's decouple and both limits' contrast masking as one pass a scale
    // (shaders/adm_dcm.slang), each row's masking sums in admRows until
    // adm_rows.slang rounds them: all but VMAF v1, the decouple variants and
    // pass-limited (diagnosis) runs. VV_ADM_FUSED=0 for the separate passes.
    // A buffer a scale: a scale's masking need not wait for the scale
    // before's rounding.
    Buffer admRows[kScales];
    bool admFused = true;
    // The GPU reads 16-bit integers from storage buffers (storageBuffer16BitAccess
    // and shaderInt16, both enabled): VIF scale 0 reads 16-bit samples so
    // (vif_fused_0_16s; Core Ultra 9 285K's iGPU, 4K: 7.63 -> 7.46 ms), and
    // ADM scale 0's transform stores a so (adm_dwt_0_8a, _16a).
    bool samples16 = false;
    // And 8-bit integers (VK_KHR_8bit_storage's storageBuffer8BitAccess,
    // VK_KHR_shader_float16_int8's shaderInt8): VIF scale 0 reads 8-bit
    // samples so (vif_fused_0_8s).
    bool samples8 = false;

    std::vector<Pass> motion[2];  // by frame parity
    std::vector<Pass> scored;     // VIF and ADM (and VMAF v1's CAMBI)
    size_t v1AdmEnd = 0;          // VMAF v1: scored's ADM passes, before CAMBI's

    // VMAF v1's CAMBI on the GPU (vv_v1_cambi): what cambi.c's init works
    // out, and the buffers its passes use.
    bool cambi = false;
    bool cambiPoolOnCpu = false;  // every scale pooled by v1_host (the tests of that)
    bool cambiBruteForce = false; // the c-values counted pixel by pixel (CVALUES), for the tests of CVALUES_SLIDE
    Buffer cambiArgs;             // the KEEP passes' dispatch sizes
    VkDeviceSize cambiKeepOffset[V1_CAMBI_SCALES] = {};  // each scale's place in a slot's cambiKeep (words)
    V1CambiOptions cambiOptions = {};
    V1CambiConstants cambiConst = {};
    Buffer cambiMaskFull, cambiReciprocal, cambiHist, cambiState;
    Buffer cambiFiltered[V1_CAMBI_SCALES], cambiMask[V1_CAMBI_SCALES],
        cambiC[V1_CAMBI_SCALES];
    int enable_cambi(const double *values);
    CambiPool cambiPool;  // the scales the GPU's top k could not sum exactly (after cambiConst: destroyed first)

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

    // CIEDE2000 (vv_v1_ciede, shaders/ciede.slang): from the luma the
    // features read and the 4:2:0 chroma in each slot's ciedeChroma, whole
    // planes ((w + 1) / 2 x (h + 1) / 2, as libvmaf's pictures hold them;
    // SpEED's chroma drops an odd last row and column).
    bool ciede = false;
    // Its pass reads the luma from the slots' staging, where that is the
    // GPU's own memory (an integrated GPU, barFrames) and nothing else reads
    // the pictures (flag bits 16 and 18): no copy into picRef and picDis for
    // a pass that reads each sample once.
    bool ciedeFromStaging = false;
    uint32_t ciedeW = 0, ciedeH = 0, ciedeStrideBytes = 0, ciedePlaneBytes = 0;
    int enable_ciede();
    // VMAF v1's copy of the frames into picRef and picDis: whether a pass reads them
    bool pictures_copied() const
    {
        return !(v1 && (skip & 5) == 5 && !cambi && !speed && (!ciede || ciedeFromStaging));
    }

    // The HDR models' picture conversion on the GPU (vv_v1_convert, shaders/
    // convert.slang): the source pictures go in, and the context's size at 16
    // bits comes out where the features read it -- the luma in picRef and
    // picDis, the chroma in the slot's speedChroma (or convertChroma, which
    // nothing reads, without SpEED). Per side (0 the reference, 1 the
    // distorted): its plan (libvmaf's vmaf_picture_convert_gpu_plan) followed
    // by where each of the plan's buffers is for each slot (convert.slang's
    // places), the plan on the GPU, its scratch planes, and a pass per step.
    struct ConvertSide {
        std::vector<uint32_t> words;
        uint32_t planeW[3] = {}, planeH[3] = {};
        uint32_t planeOffset[3] = {};  // the source planes' bytes into a slot's part of convertSource
        uint32_t placesAt = 0, placesWords = 0;  // slot k's places: words placesAt + k * placesWords on
        Buffer tables, scratch;
        std::vector<Pass> passes;
    };
    ConvertSide convertSides[2];
    bool converting = false;
    // The source pictures, both sides' planes, for each slot at k *
    // convertSourceBytes (host-visible, written by vv_submit_v1; in a shared
    // context the GPU's own, copied from the shared buffers). One buffer,
    // not one a slot: a slot is picked by its places, so that a pass needs no
    // descriptor set a slot.
    Buffer convertSource;
    uint32_t convertSourceBytes = 0;
    // A side's planes' bytes in a slot's part of convertSource (they follow
    // one another: the reference's, then the distorted's).
    uint32_t convert_side_bytes(int side) const
    {
        return side ? convertSourceBytes - convertSides[1].planeOffset[0] : convertSides[1].planeOffset[0];
    }
    // In a shared context, where a side's planes start in its shared buffer
    // (dis_buffer's for the distorted), laid out as in convertSource: in
    // split mode each side's own buffer from 0, else both in the one.
    uint32_t shared_source_offset(int side) const { return split ? 0 : convertSides[side].planeOffset[0]; }
    Buffer convertChroma;
    int enable_convert(const uint32_t *const plans[2], const size_t words[2]);
    int load_convert_plan(int side, const uint32_t *plan, size_t words);
    int add_convert_side(int side);
    // The steps of the sides in `sides` (bit 0 the reference, 1 the distorted),
    // a barrier after each step of both; `stamp(shader)` after each when given.
    void record_convert(VkCommandBuffer cb, size_t slotIndex, unsigned sides, const std::function<void(int)> &stamp);
    void write_convert_source(uint8_t *base, int side, const uint16_t *const planes[3], const ptrdiff_t strides[3]);
    int test_convert(int side, const uint16_t *const planes[3], const ptrdiff_t strides[3], uint16_t *const out[3],
                     const ptrdiff_t outStrides[3]);

    // Profiling (vv_profile, for the speed work): the GPU's time per pass,
    // summed by shader, from timestamps around each, and the CPU's per step.
    bool profile = false;
    VkQueryPool queryPool = VK_NULL_HANDLE;
    double timestampNs = 1.0;
    enum { kQueriesPerSlot = 1024, kProfileUpload = kShaderCount, kProfileCopies, kProfileLabels };
    std::vector<std::vector<int>> slotLabels;  // per slot: what each interval between its timestamps was
    double gpuNs[kProfileLabels] = {};
    uint64_t gpuCount[kProfileLabels] = {};
    enum { kCpuCopy, kCpuRecord, kCpuWait, kCpuSpeed, kCpuCambiCopy, kCpuCambiPool, kCpuSteps };
    double cpuNs[kCpuSteps] = {};
    uint64_t cpuCount[kCpuSteps] = {};
    double now_ns() const;
    int enable_profile();

    std::vector<Slot> slots;
    unsigned nextSlot = 0;
    unsigned frames = 0;
    Blocks<FrameRecord> records;                        // every frame's, from its commit on
    std::vector<std::array<uint64_t, kSlots>> rawSums;  // the first kRawFrames frames' sums (vv_sums)
    int cambi_score(const uint64_t *sums, const double *pooled, unsigned pooledScales, double &out) const;
    bool failed = false;

    // Timing tests (VV_GPU_TIME=<file>): the profiler's (enable_profile), its
    // report appended to the file when the context closes: each pair's time on
    // the GPU, from its first timestamp to its last, and the time from the
    // first pair's start to the last one's end (the difference: the GPU idle
    // between pairs), and each step's.
    std::string timeFile;
    double profilePairNs = 0;
    unsigned profilePairs = 0;
    uint64_t profileFirst = 0, profileLast = 0;
    std::string statsFile;  // VV_PIPELINE_STATS
    // An integrated GPU (create_buffer: frames' staging in its own mappable memory).
    bool integratedGpu = false;

    ~vv_context();
    int init(int deviceIndex, int width, int height, int bitDepth, int flags);
    // `upload`: host memory the CPU only writes and the GPU reads (frames' planes).
    int create_buffer(Buffer &buffer, VkDeviceSize size, bool hostVisible, bool exported = false, bool upload = false);
    void destroy_last_buffer(Buffer &buffer);
    // A buffer made again at another size, before any pass binds it.
    int remake_buffer(Buffer &buffer, VkDeviceSize size, bool hostVisible, bool upload);
    VkPipeline find_pipeline(int shader, const std::vector<uint32_t> &words) const;
    VkResult compile_pipeline(int shader, const std::vector<uint32_t> &words, VkPipeline *out) const;
    int compile_pipelines();
    int make_layouts(int shader, uint32_t bindings);
    void write_pipeline_stats(const char *name, VkPipeline pipeline);
    bool shares(const Pass &a, const Pass &b) const;
    int add_pass(std::vector<Pass> &list, int shader, std::initializer_list<Bound> bound,
                 const void *constants, uint32_t constantBytes, uint32_t gx, uint32_t gy);
    VkDescriptorSet descriptor_set(int shader, const std::vector<Bound> &bound);
    // The last pass added, given a set per slot: `bound(slot)` the slot's buffers.
    template <typename F> int per_slot(std::vector<Pass> &list, F bound);
    // VMAF v1 from a decoder's textures (flag bit 2, vv_pictures): the pictures
    // read where the decoder left them, not copied into the slots. Imported
    // once each (pictureCache); `lastPicture`: the reference before, motion's.
    bool pictures = false;
    struct Picture {
        HANDLE handle = nullptr;
        uint32_t width = 0, height = 0;
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView views[2] = {};  // luma, chroma (UINT)
    };
    enum { kPictureCacheSize = 64 };
    std::vector<Picture> pictureCache;
    int lastPicture = -1;
    uint32_t lastPlace[2] = {};
    PFN_vkGetMemoryWin32HandlePropertiesKHR memoryHandleProperties = nullptr;
    int import_picture(HANDLE handle, uint32_t width, uint32_t height);
    int set_pictures(HANDLE reference, uint32_t referenceW, uint32_t referenceH, int referenceX, int referenceY,
                     HANDLE distorted, uint32_t distortedW, uint32_t distortedH, int distortedX, int distortedY);
    void picture_descriptors(VkDescriptorSet set, const Pass &pass, const Slot &slot);
    void picture_barriers(VkCommandBuffer cb, const Slot &slot, bool acquire);
    // The distorted plane's place in a slot's staging buffer (in split mode,
    // in stagingDis: 0).
    uint32_t disOffset = 0;
    // VMAF v1 with a decoder's frames: the four chroma planes in a slot's
    // staging buffer too (ref U, ref V, dis U, dis V), from sharedChroma, each
    // sharedChromaSpacing bytes after the one before (vv_shared_chroma). In
    // split mode, the reference's two there and the distorted's two at the
    // same place in stagingDis.
    uint32_t sharedChroma = 0, sharedChromaSpacing = 0;
    // The buffer that holds a slot's distorted plane, at disOffset.
    Buffer &dis_buffer(Slot &slot) { return split ? slot.stagingDis : slot.staging; }
    Bound slot_ref(Slot &slot) { return Bound(&slot.staging, 0, planeBytes); }
    Bound slot_dis(Slot &slot) { return Bound(&dis_buffer(slot), disOffset, planeBytes); }
    int upload(Buffer &target, const void *data, size_t bytes);
    int build_passes();
    int submit(const uint8_t *ref, ptrdiff_t refStride, const uint8_t *dis, ptrdiff_t disStride, bool score);
    // submit's copy of the planes into staging, split by rows among
    // copyThreads threads: the caller's and copiers, started by the first
    // submit. On discrete GPUs, where the copy crosses PCIe: one thread's
    // writes do not fill it (an RTX 5090's 4K 10-bit pair, into its memory:
    // 1.57 ms with one thread, 1.04 with four). Not on integrated ones,
    // whose GPU reads the same memory the threads write (Core Ultra 9 285K's
    // iGPU: 4K 13.97 -> 14.25 ms a pair with four). VV_COPY_THREADS=N to
    // compare.
    unsigned copyThreads = 1;
    std::vector<std::thread> copiers;
    std::mutex copyMutex;
    std::condition_variable copyStart, copyDone;
    unsigned copyGeneration = 0, copyPending = 0;
    bool copyQuit = false;
    // A plane to copy: rows of rowBytes from `from` (fromStride apart) to `to`
    // (toStride apart). VMAF v0.6.1's two lumas, or VMAF v1's lumas and the
    // chroma planes SpEED reads.
    struct CopyPlane {
        uint8_t *to;
        size_t toStride;
        const uint8_t *from;
        ptrdiff_t fromStride;
        size_t rowBytes;
        uint32_t rows;
    };
    struct {
        CopyPlane planes[10];  // two luma planes, SpEED's four chroma planes and CIEDE2000's
        int count;
    } copyJob = {};
    void copy_part(unsigned part);
    void copy_planes();
    void copier(unsigned part);
    int submit_v1(const uint8_t *const planes[6], const ptrdiff_t strides[6], bool score);
    int staging(uint8_t **ref, uint8_t **dis);
    int commit(bool score);
    Slot *pending = nullptr;
    int collect(Slot &slot);
    int flush();
};

vv_context::~vv_context()
{
    if (!copiers.empty()) {
        {
            std::lock_guard<std::mutex> lock(copyMutex);
            copyQuit = true;
        }
        copyStart.notify_all();
        for (std::thread &thread : copiers)
            thread.join();
    }
    if (submitter.joinable()) {
        {
            std::lock_guard<std::mutex> lock(submitMutex);
            submitStop = true;
        }
        submitWork.notify_all();
        submitter.join();  // a wait for the decoder ends within its 100 ms
        for (const Submission &dropped : submissions)  // (a run stopped early): never submitted
            slots[dropped.slot].busy = slots[dropped.slot].queued = false;
        submissions.clear();
    }
    if (!device)
        return;
    vk.vkDeviceWaitIdle(device);
    if (profile && !timeFile.empty()) {  // VV_GPU_TIME's report (enable_profile)
        for (Slot &slot : slots)
            collect(slot);
        if (FILE *file = fopen(timeFile.c_str(), "a")) {
            const double gpu = profilePairs ? profilePairNs / profilePairs / 1e6 : 0.0;
            const double span = profileLast > profileFirst ? (double)(profileLast - profileFirst) * timestampNs : 0.0;
            fprintf(file, "%dx%d %u pairs: %.3f ms a pair on the GPU, %.3f ms a pair from first to last, busy %.1f%%\n",
                    w, h, profilePairs, gpu, profilePairs ? span / profilePairs / 1e6 : 0.0,
                    span > 0 ? 100.0 * profilePairNs / span : 0.0);
            std::vector<int> order;
            for (int i = 0; i < kProfileLabels; ++i)
                if (gpuNs[i] > 0)
                    order.push_back(i);
            std::sort(order.begin(), order.end(), [&](int a, int b) { return gpuNs[a] > gpuNs[b]; });
            for (int i : order)
                fprintf(file, "    %8.3f ms  %s\n", profilePairs ? gpuNs[i] / profilePairs / 1e6 : 0.0,
                        i < kShaderCount ? kShaders[i].name : i == kProfileUpload ? "(frames in)" : "(sums out)");
            fclose(file);
        }
    }
    if (copied)
        destroySemaphore(device, copied, nullptr);
    if (queryPool)
        vk.vkDestroyQueryPool(device, queryPool, nullptr);
    for (Slot &slot : slots) {
        if (slot.fence)
            vk.vkDestroyFence(device, slot.fence, nullptr);
    }
    for (Picture &picture : pictureCache) {
        for (VkImageView view : picture.views)
            if (view) vk.vkDestroyImageView(device, view, nullptr);
        if (picture.image) vk.vkDestroyImage(device, picture.image, nullptr);
        if (picture.memory) vk.vkFreeMemory(device, picture.memory, nullptr);
    }
    for (Buffer *buffer : buffers) {
        if (buffer->buffer)
            vk.vkDestroyBuffer(device, buffer->buffer, nullptr);
        if (buffer->memory)
            vk.vkFreeMemory(device, buffer->memory, nullptr);
    }
    for (Imported &texture : imported) {
        if (texture.image) vk.vkDestroyImage(device, texture.image, nullptr);
        if (texture.memory) vk.vkFreeMemory(device, texture.memory, nullptr);
    }
    for (Pipeline &pipeline : pipelines) {
        if (pipeline.pipeline) vk.vkDestroyPipeline(device, pipeline.pipeline, nullptr);
        for (const auto &version : pipeline.versions)
            vk.vkDestroyPipeline(device, version.second, nullptr);
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
    // Host memory: cached if there is such a type (results are read back).
    // For an upload (the frames' staging, which the CPU only writes) not
    // cached: the GPU reads uncached memory faster (Core Ultra 9 285K's iGPU,
    // reading the frames from it: 4K 15.66 -> 15.56 ms a pair; an APU's GPU
    // reads cached host memory slowly, through the CPU's caches), and the CPU
    // writes it as fast. And the GPU's own if it can be mapped, on an
    // integrated GPU (a Radeon 780M's 256 MB) and with Resizable BAR
    // (barFrames), not a discrete GPU's 256 MB window without it. Device
    // memory: the GPU's.
    const VkMemoryPropertyFlags visible = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    struct Want { VkMemoryPropertyFlags with, without; };
    std::vector<Want> wanted;
    if (upload) {
        if (barFrames || integratedGpu)
            wanted.push_back({ visible | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_CACHED_BIT });
        wanted.push_back({ visible, VK_MEMORY_PROPERTY_HOST_CACHED_BIT });
        wanted.push_back({ visible, 0 });
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

// Frees a buffer made for one call, the last create_buffer made (or began to
// make: what it got as far as). The context keeps a pointer to every buffer
// it makes, for its destructor, and such a buffer is the caller's local.
void vv_context::destroy_last_buffer(Buffer &buffer)
{
    if (buffer.buffer)
        vk.vkDestroyBuffer(device, buffer.buffer, nullptr);
    if (buffer.memory)
        vk.vkFreeMemory(device, buffer.memory, nullptr);
    buffer = Buffer();
    buffers.pop_back();
}

int vv_context::remake_buffer(Buffer &buffer, VkDeviceSize size, bool hostVisible, bool upload)
{
    if (buffer.buffer)
        vk.vkDestroyBuffer(device, buffer.buffer, nullptr);
    if (buffer.memory)
        vk.vkFreeMemory(device, buffer.memory, nullptr);
    buffer = Buffer();
    const int error = create_buffer(buffer, size, hostVisible, false, upload);
    buffers.pop_back();  // listed once already, from when it was first made
    return error;
}

// The pipeline made for `shader` and these push constants (a specialized
// shader's for exactly these; another's for any), or none yet.
VkPipeline vv_context::find_pipeline(int shader, const std::vector<uint32_t> &words) const
{
    const Pipeline &pipeline = pipelines[shader];
    if (!pipeline.specialized)
        return pipeline.pipeline;
    for (const auto &version : pipeline.versions)
        if (version.first == words)
            return version.second;
    return VK_NULL_HANDLE;
}

// Compiles `shader` for these push constants (its specialization constants,
// constant k: word k, where it has any).
VkResult vv_context::compile_pipeline(int shader, const std::vector<uint32_t> &words, VkPipeline *out) const
{
    const Pipeline &pipeline = pipelines[shader];
    VkComputePipelineCreateInfo info = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module = pipeline.module;
    info.stage.pName = "main";
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT required = {
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT };
    if (subgroupSizeControl && subgroupSizes[shader]) {
        required.requiredSubgroupSize = subgroupSizes[shader];
        info.stage.pNext = &required;
    }
    std::vector<VkSpecializationMapEntry> entries(words.size());
    for (uint32_t k = 0; k < (uint32_t)words.size(); ++k)
        entries[k] = { k, 4 * k, 4 };
    const VkSpecializationInfo specialization = { (uint32_t)words.size(), entries.data(), words.size() * 4,
                                                  words.data() };
    if (pipeline.specialized)
        info.stage.pSpecializationInfo = &specialization;
    info.layout = pipeline.layout;
    if (!statsFile.empty())
        info.flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR | VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR;
    return vk.vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, out);
}

// Every pass's pipeline: those not made yet compiled (a specialized shader
// anew for each size of picture the driver meets), then given to the passes.
// One at a time in the process: NVIDIA's driver crashed inside
// vkCreateComputePipelines (an access violation in nvoglv64.dll) while these
// were compiled a thread each, although the calls may run at once. One at a
// time costs nothing where the driver's shader cache has the pipelines, and
// up to a second for a size it has not met (a 10-bit VMAF context near 4K on
// an RTX 5090: 0.65 s instead of 0.18 s).
int vv_context::compile_pipelines()
{
    struct Job { int shader; std::vector<uint32_t> words; VkPipeline made = VK_NULL_HANDLE; VkResult result = VK_SUCCESS; };
    std::vector<Job> jobs;
    std::vector<Pass *> passes;
    for (std::vector<Pass> *list : { &motion[0], &motion[1], &scored, &convertSides[0].passes, &convertSides[1].passes })
        for (Pass &pass : *list)
            passes.push_back(&pass);
    for (Slot &slot : slots)  // VMAF v1's per slot (CAMBI's kept c-values, SpEED, CIEDE2000)
        for (std::vector<Pass> *list : { &slot.cambiKeepPasses, &slot.speedPasses, &slot.speedPassesHost,
                                         &slot.ciedePasses })
            for (Pass &pass : *list)
                passes.push_back(&pass);
    auto words_of = [&](const Pass &pass) {
        return pipelines[pass.shader].specialized
            ? std::vector<uint32_t>(pass.constants, pass.constants + pass.constantBytes / 4) : std::vector<uint32_t>();
    };
    for (const Pass *pass : passes) {
        const std::vector<uint32_t> words = words_of(*pass);
        if (find_pipeline(pass->shader, words))
            continue;
        bool queued = false;
        for (const Job &job : jobs)
            queued = queued || (job.shader == pass->shader && job.words == words);
        if (!queued)
            jobs.push_back({ pass->shader, words });
    }
    {
        static std::mutex compiling;  // every context's, in this process
        const std::lock_guard<std::mutex> lock(compiling);
        for (Job &job : jobs)
            job.result = compile_pipeline(job.shader, job.words, &job.made);
    }
    int error = 0;
    for (Job &job : jobs) {
        Pipeline &pipeline = pipelines[job.shader];
        if (job.result != VK_SUCCESS) {
            if (!error)
                error = fail(-1, std::string("the GPU driver could not compile the shader ") + kShaders[job.shader].name);
            continue;
        }
        if (pipeline.specialized)
            pipeline.versions.emplace_back(job.words, job.made);
        else
            pipeline.pipeline = job.made;
        if (!statsFile.empty())
            write_pipeline_stats(kShaders[job.shader].name, job.made);
    }
    for (Pass *pass : passes)
        pass->pipeline = find_pipeline(pass->shader, words_of(*pass));
    return error;
}

// The shader's module and layouts, and whether it has specialization
// constants: a SpecId decoration (OpDecorate, opcode 71, four words,
// decoration 1) anywhere in it.
int vv_context::make_layouts(int shader, uint32_t bindings)
{
    Pipeline &pipeline = pipelines[shader];
    pipeline.bindings = bindings;
    // And the bindings it only reads: variables decorated NonWritable (24)
    // and Binding (33), by their ids.
    const uint32_t *code = kShaders[shader].code;
    std::vector<std::pair<uint32_t, uint32_t>> bindingOf;  // id, binding
    std::vector<uint32_t> nonWritable;
    for (size_t at = 5; at < kShaders[shader].bytes / 4;) {
        const uint32_t length = code[at] >> 16, opcode = code[at] & 0xFFFF;
        if (opcode == 71 && length == 4 && code[at + 2] == 1)
            pipeline.specialized = true;
        if (opcode == 71 && length == 4 && code[at + 2] == 33)
            bindingOf.emplace_back(code[at + 1], code[at + 3]);
        if (opcode == 71 && length == 3 && code[at + 2] == 24)
            nonWritable.push_back(code[at + 1]);
        at += length ? length : 1;
    }
    for (const auto &[id, binding] : bindingOf)
        if (binding < 32 && std::find(nonWritable.begin(), nonWritable.end(), id) != nonWritable.end())
            pipeline.readOnly |= 1u << binding;
    VkShaderModuleCreateInfo module = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    module.codeSize = kShaders[shader].bytes;
    module.pCode = kShaders[shader].code;
    if (vk.vkCreateShaderModule(device, &module, nullptr, &pipeline.module) != VK_SUCCESS)
        return fail(-1, std::string("vkCreateShaderModule failed for ") + kShaders[shader].name);
    VkDescriptorSetLayoutBinding layoutBindings[kMaxBindings] = {};
    // vv_pictures' shaders: the pictures' textures (the bindings in `images`).
    uint32_t images = 0;
    if (shader == kShader_motion_v1_8_img || shader == kShader_motion_v1_16_img || shader == kShader_adm_fused_0_8_img
        || shader == kShader_adm_fused_0_16_img || shader == kShader_adm_fused_0_8_img_skip
        || shader == kShader_adm_fused_0_16_img_skip)
        images = 3;
    if (shader == kShader_cambi_front_8_img || shader == kShader_cambi_front_16_img
        || shader == kShader_cambi_front_8_1_img || shader == kShader_cambi_front_16_1_img)
        images = 1;
    if (shader == kShader_speed_dec_img)
        images = 1 | 16;
    for (uint32_t i = 0; i < bindings; ++i) {
        layoutBindings[i].binding = i;
        layoutBindings[i].descriptorType = (images >> i) & 1 ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE
                                                             : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
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
    return 0;
}

void vv_context::write_pipeline_stats(const char *name, VkPipeline pipeline)
{
    auto properties = (PFN_vkGetPipelineExecutablePropertiesKHR)api->vkGetDeviceProcAddr(
        device, "vkGetPipelineExecutablePropertiesKHR");
    auto statistics = (PFN_vkGetPipelineExecutableStatisticsKHR)api->vkGetDeviceProcAddr(
        device, "vkGetPipelineExecutableStatisticsKHR");
    FILE *file = fopen(statsFile.c_str(), "a");
    if (!properties || !statistics || !file) {
        if (file) fclose(file);
        return;
    }
    VkPipelineInfoKHR info = { VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR };
    info.pipeline = pipeline;
    uint32_t count = 0;
    properties(device, &info, &count, nullptr);
    std::vector<VkPipelineExecutablePropertiesKHR> executables(count, { VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR });
    properties(device, &info, &count, executables.data());
    for (uint32_t i = 0; i < count; ++i) {
        fprintf(file, "%s [%s] subgroup %u:", name, executables[i].name, executables[i].subgroupSize);
        VkPipelineExecutableInfoKHR executable = { VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR };
        executable.pipeline = pipeline;
        executable.executableIndex = i;
        uint32_t n = 0;
        statistics(device, &executable, &n, nullptr);
        std::vector<VkPipelineExecutableStatisticKHR> values(n, { VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR });
        statistics(device, &executable, &n, values.data());
        for (const VkPipelineExecutableStatisticKHR &value : values) {
            fprintf(file, " | %s=", value.name);
            switch (value.format) {
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR: fprintf(file, "%u", value.value.b32); break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR: fprintf(file, "%lld", (long long)value.value.i64); break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR: fprintf(file, "%llu", (unsigned long long)value.value.u64); break;
            default: fprintf(file, "%g", value.value.f64); break;
            }
        }
        fprintf(file, "\n");
        // And what the driver shows of the compiled code (its internal
        // representations), each in a file of its own beside statsFile.
        auto representations = (PFN_vkGetPipelineExecutableInternalRepresentationsKHR)api->vkGetDeviceProcAddr(
            device, "vkGetPipelineExecutableInternalRepresentationsKHR");
        uint32_t m = 0;
        if (representations && representations(device, &executable, &m, nullptr) == VK_SUCCESS && m) {
            std::vector<VkPipelineExecutableInternalRepresentationKHR> texts(
                m, { VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR });
            representations(device, &executable, &m, texts.data());
            std::vector<std::vector<char>> data(m);
            for (uint32_t k = 0; k < m; ++k) {
                data[k].resize(texts[k].dataSize + 1);
                texts[k].pData = data[k].data();
            }
            representations(device, &executable, &m, texts.data());
            for (uint32_t k = 0; k < m; ++k) {
                const std::string path = statsFile + "." + name + "." + std::to_string(k) + ".txt";
                if (FILE *out = fopen(path.c_str(), "wb")) {
                    fprintf(out, "%s: %s\n", texts[k].name, texts[k].description);
                    fwrite(data[k].data(), 1, texts[k].dataSize, out);
                    fclose(out);
                }
            }
        }
    }
    fclose(file);
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
    uint32_t count = 0, binding = 0;
    for (const Bound &entry : bound) {
        if (!entry.buffer) {  // a picture's texture (kPictureBinding): bound each frame
            ++binding;
            continue;
        }
        infos[count] = { entry.buffer->buffer, entry.offset, entry.range };
        writes[count] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[count].dstSet = set;
        writes[count].dstBinding = binding++;
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
    if (bound.size() > kMaxBindings)
        return fail(-1, std::string("too many buffers bound to the shader ") + kShaders[shader].name);
    if (!pipelines[shader].module) {  // its pipeline compiled by compile_pipelines, once every pass is in
        if (int error = make_layouts(shader, (uint32_t)bound.size()))
            return error;
    } else if (pipelines[shader].bindings != (uint32_t)bound.size()) {
        // The layout was made for the first pass's bindings: every pass of a shader binds the same number.
        return fail(-1, std::string("the shader ") + kShaders[shader].name + " is bound a different number of buffers");
    }
    Pass pass = {};
    pass.shader = shader;
    pass.set = descriptor_set(shader, std::vector<Bound>(bound));
    if (!pass.set)
        return fail(-1, "vkAllocateDescriptorSets failed");
    // VMAF v0.6.1's passes that read the frames, where they read them from
    // the frame slots' staging buffers (direct): a set for each slot. (VMAF
    // v1's give theirs with per_slot.)
    bool perSlot = false;
    for (const Bound &entry : bound)
        perSlot = perSlot || (direct && !v1 && entry.buffer && (entry.buffer == &picRef || entry.buffer == &picDis));
    for (size_t i = 0; perSlot && i < slots.size(); ++i) {
        std::vector<Bound> slotBound;
        for (const Bound &entry : bound) {
            if (entry.buffer == &picRef)
                slotBound.push_back(Bound(&slots[i].staging));
            else if (entry.buffer == &picDis)
                slotBound.push_back(shared ? Bound(&dis_buffer(slots[i]), disOffset, VK_WHOLE_SIZE)
                                           : Bound(&slots[i].stagingDis));
            else
                slotBound.push_back(entry);
        }
        const VkDescriptorSet set = descriptor_set(shader, slotBound);
        if (!set)
            return fail(-1, "vkAllocateDescriptorSets failed");
        pass.slotSets.push_back(set);
    }
    memcpy(pass.constants, constants, constantBytes);
    pass.constantBytes = constantBytes;
    pass.groups[0] = gx;
    pass.groups[1] = gy;
    pass.groups[2] = 1;
    for (const Bound &entry : bound)  // by binding (a picture's texture: none)
        if (pass.boundCount < kMaxBindings)
            pass.bound[pass.boundCount++] = entry.buffer;
    for (const Pass &earlier : list)
        if (shares(earlier, pass))
            pass.level = std::max(pass.level, earlier.level + 1);
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

// Whether two passes must not overlap: a buffer bound to both that either
// writes (binds where its shader does not mark the binding NonWritable), but
// the pictures and the division and logarithm tables, which passes only
// read, and acc, which they only add to atomically (acc_add64). Two passes
// that only read a buffer overlap (ADM's CSF denominator and masking of a
// scale). (VMAF v0.6.1's lists: VMAF v1 records its own way, commit.)
bool vv_context::shares(const Pass &a, const Pass &b) const
{
    auto writes = [&](const Pass &pass, const Buffer *buffer) {
        for (uint32_t i = 0; i < pass.boundCount; ++i)
            if (pass.bound[i] == buffer && !((pipelines[pass.shader].readOnly >> i) & 1u))
                return true;
        return false;
    };
    for (uint32_t i = 0; i < a.boundCount; ++i) {
        const Buffer *buffer = a.bound[i];
        if (!buffer || buffer == &picRef || buffer == &picDis || buffer == &divTable || buffer == &logTable
            || buffer == &acc)
            continue;
        bool both = false;
        for (uint32_t j = 0; j < b.boundCount; ++j)
            both = both || b.bound[j] == buffer;
        if (both && (writes(a, buffer) || writes(b, buffer)))
            return true;
    }
    return false;
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

// Copies `data` into a device buffer through slot 0's command buffer (and
// a staging buffer of its own): slot 0's frame, if one is in flight, first.
int vv_context::upload(Buffer &target, const void *data, size_t bytes)
{
    if (int error = collect(slots[0]))
        return error;
    Buffer staging;
    if (int error = create_buffer(staging, bytes, true)) {
        destroy_last_buffer(staging);
        return error;
    }
    memcpy(staging.mapped, data, bytes);
    VkCommandBuffer commands = slots[0].commands;
    VkCommandBufferBeginInfo begin = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vk.vkBeginCommandBuffer(commands, &begin);
    VkBufferCopy region = { 0, 0, bytes };
    vk.vkCmdCopyBuffer(commands, staging.buffer, target.buffer, 1, &region);
    vk.vkEndCommandBuffer(commands);
    const VkResult result = run_now(commands);
    destroy_last_buffer(staging);
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
        const int shader = subgroupSums ? (deep ? kShader_motion_16w : kShader_motion_8w)
                                        : (deep ? kShader_motion_16 : kShader_motion_8);
        error = add_pass(motion[parity], shader,
                         { &picRef, &blur[parity], &blur[1 - parity], &acc }, constants, sizeof constants,
                         groups(w, 64), groups(h, 8));
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
            if (vifFused && !nativeDouble) {
                // Both passes in one, the vertical pass's results in group
                // memory (shaders/vif_fused.slang).
                const uint32_t fused[] = { (uint32_t)sw, (uint32_t)sh, (uint32_t)sourceStride,
                                           shiftVP, addVP, shiftSq, addSq, (uint32_t)nextStride,
                                           (uint32_t)(kSlotVif + scale * kVifSums),
                                           (uint32_t)epsilon, (uint32_t)(epsilon >> 32) };
                // 16-bit samples read as such where the GPU can (samples16).
                const int shader = scale == 0 ? (!deep ? (samples8 ? kShader_vif_fused_0_8s : kShader_vif_fused_0_8)
                                                 : samples16 ? kShader_vif_fused_0_16s : kShader_vif_fused_0_16)
                                              : kShader_vif_fused_1 + (scale - 1);
                error = add_pass(scored, shader, { inRef, inDis, &rdRef[scale % 2], &rdDis[scale % 2], &acc, &logTable },
                                 fused, sizeof fused, groups(sw, 160), groups(sh, kVifTileRows[scale]));  // its TW x TH tiles
                sourceStride = nextStride;
                continue;
            }
            const uint32_t vertical[] = { (uint32_t)sw, (uint32_t)sh, (uint32_t)sourceStride,
                                          shiftVP, addVP, shiftSq, addSq };
            const int verticalShader = scale == 0 ? (deep ? kShader_vif_vert_0_16 : kShader_vif_vert_0_8)
                                                  : kShader_vif_vert_1 + (scale - 1);
            error = add_pass(scored, verticalShader, { inRef, inDis, &vifTmp }, vertical, sizeof vertical,
                             groups(sw, 16), groups(sh, 8));
            if (error)
                break;
            const uint32_t horizontal[] = { (uint32_t)sw, (uint32_t)sh, (uint32_t)nextStride,
                                            (uint32_t)(kSlotVif + scale * kVifSums),
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
        Buffer *inRef = scale == 0 ? &picRef : &bandsARef[1 - set];
        Buffer *inDis = scale == 0 ? &picDis : &bandsADis[1 - set];
        const int bw = (inW + 1) / 2, bh = (inH + 1) / 2;
        const int bandStride = set == 0 ? (w + 1) / 2 : ((w + 1) / 2 + 1) / 2;
        const int outStride = (w + 1) / 2;  // of admR, admA, admF
        const int aStride0 = ((w + 1) / 2 + 1) & ~1;  // scale 0's a, two a word (adm_dwt)

        {   // dwt2_8_device / adm_dwt2_16_device / adm_dwt2_s123_combined_device
            static const int kV[4][2] = { { 0, 0 }, { 0, 0 }, { 16, 32768 }, { 16, 32768 } };
            static const int kH[4][2] = { { 16, 32768 }, { 15, 16384 }, { 16, 32768 }, { 15, 16384 } };
            const int32_t constants[] = { inW, inH, inStride, bandStride,
                                          scale == 0 ? bpc : kV[scale][0], scale == 0 ? 1 << (bpc - 1) : kV[scale][1],
                                          kH[scale][0], kH[scale][1], aStride0 };
            // Scales 1-3: a shader each, with these shifts (kV, kH) built in. Scale 0
            // stores a as 16-bit values where the GPU can (samples16).
            const int shader = scale != 0 ? kShader_adm_dwt_1 + (scale - 1)
                               : samples16 ? (deep ? kShader_adm_dwt_0_16a : kShader_adm_dwt_0_8a)
                                           : (deep ? kShader_adm_dwt_0_16 : kShader_adm_dwt_0_8);
            error = add_pass(scored, shader, { inRef, inDis, &bandsRef[set], &bandsDis[set], &bandsARef[set],
                                               &bandsADis[set] }, constants,
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
        if (admFused) {  // adm_decouple and adm_cm below, in one pass, then the rows' rounding
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
            const int rows = std::max(0, end_row - start_row);
            // The gain limits (100, 1) and the shifts but shiftCub are adm_dcm's own.
            static const int fixed_shift[3] = { 4, 4, 3 };
            struct {
                int32_t w, h, inStride, startRow, endRow, startCol, endCol;
                uint32_t rfactor[3];
                int32_t shiftCub[3];
                uint32_t rowSlot;
            } constants = {};
            constants.w = bw;
            constants.h = bh;
            constants.inStride = bandStride;
            constants.startRow = start_row;
            constants.endRow = start_row + rows;
            constants.startCol = start_col;
            constants.endCol = std::max(start_col, end_col);
            for (int band = 0; band < 3; ++band) {
                constants.rfactor[band] = i_rfactor[scale * 3 + band];
                const double shift = scale == 0 ? ceil(log2((double)bw) - fixed_shift[band]) : ceil(log2((double)bw));
                constants.shiftCub[band] = shift > 0 ? (int32_t)shift : 0;
            }
            constants.rowSlot = 0;  // admRows[scale]'s first
            error = add_pass(scored, scale == 0 ? kShader_adm_dcm_0 : kShader_adm_dcm,
                             { &bandsRef[set], &bandsDis[set], &divTable, &admRows[scale] }, &constants, sizeof constants,
                             groups(constants.endCol - start_col, kDcmTile[0]), groups(rows, kDcmTile[1]));
            const uint32_t finish[] = { (uint32_t)rows, (uint32_t)ceil_log2(bh), 0,
                                        (uint32_t)(kSlotCm + scale * 3), (uint32_t)(kScales * 3) };
            if (!error)
                error = add_pass(scored, kShader_adm_rows, { &admRows[scale], &acc }, finish, sizeof finish, 6, 1);
        }
        // The enhancement gain limit: 100 (VMAF), then 1 (VMAF NEG).
        for (int limit = 0; limit < 2 && !error && !admFused; ++limit) {
            {   // adm_decouple_device / adm_decouple_s123_device, adm_csf_device / i4_adm_csf_device
                int left = bw * (float)(ADM_BORDER_FACTOR) - 0.5f - 1;
                int top = bh * (float)(ADM_BORDER_FACTOR) - 0.5f - 1;
                int right = bw - left + 2;
                int bottom = bh - top + 2;
                if (left < 0) left = 0;
                if (right > bw) right = bw;
                if (top < 0) top = 0;
                if (bottom > bh) bottom = bh;
                const bool both = admBoth && !decoupleVariant;
                if (both && limit == 0) {
                    const uint32_t constants[] = { (uint32_t)top, (uint32_t)bottom, (uint32_t)left, (uint32_t)right,
                                                   (uint32_t)bandStride, (uint32_t)outStride, 100u, 1u,
                                                   i_rfactor[scale * 3], i_rfactor[scale * 3 + 1],
                                                   i_rfactor[scale * 3 + 2] };
                    error = add_pass(scored, scale == 0 ? kShader_adm_decouple_0_both : kShader_adm_decouple_both,
                                     { &bandsRef[set], &bandsDis[set], &admR, &admA, &admF, &divTable,
                                       &admRB, &admAB, &admFB },
                                     constants, sizeof constants, groups(right - left, 16), groups(bottom - top, 8));
                } else if (!both) {
                    const uint32_t constants[] = { (uint32_t)top, (uint32_t)bottom, (uint32_t)left, (uint32_t)right,
                                                   (uint32_t)bandStride, (uint32_t)outStride,
                                                   limit == 0 ? 100u : 1u,
                                                   i_rfactor[scale * 3], i_rfactor[scale * 3 + 1],
                                                   i_rfactor[scale * 3 + 2] };
                    const int decouple0 = decoupleVariant ? kShader_adm_decouple_0_v1 + decoupleVariant - 1
                                                          : kShader_adm_decouple_0;
                    error = add_pass(scored, scale == 0 ? decouple0 : kShader_adm_decouple,
                                     { &bandsRef[set], &bandsDis[set], &admR, &admA, &admF, &divTable }, constants,
                                     sizeof constants, groups(right - left, 16), groups(bottom - top, 8));
                }
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
                const bool fromB = admBoth && !decoupleVariant && limit == 1;
                error = add_pass(scored, scale == 0 ? kShader_adm_cm_0 : kShader_adm_cm,
                                 { fromB ? &admRB : &admR, fromB ? &admAB : &admA, fromB ? &admFB : &admF, &acc },
                                 &constants, sizeof constants, 1,
                                 (uint32_t)std::max(0, end_row - start_row));
            }
        }
        inW = bw;
        inH = bh;
        inStride = scale == 0 ? aStride0 : bandStride;
    }
    return error;
}

namespace {

// The contrast sensitivity factors of a scale ({h, v, d}) as libvmaf's CPU
// code takes them (adm_csf and the others of integer_adm.c): Watson97's
// (mode 0), or the Barten-Watson blends' tables (modes 2 and 3, libvmaf's
// own: v1_host.c). False when the options are ones it has no table for.
bool v1_rfactors(const V1Options &o, int scale, float rfactor[3])
{
    float factor1, factor2;
    if (o.csfMode == 2 || o.csfMode == 3) {
        if (v1_adm_blend_factors(o.csfMode, scale, o.viewDistance, o.displayHeight, &factor1, &factor2))
            return false;
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
    for (int parity = 0; parity < 2 && !error && !(skip & 1); ++parity) {
        int32_t constants[kPushBytes / 4] = { w, h, strideWords, bpc, 1 << (bpc - 1), kSlotSad };
        Buffer *previous = &picPrev[options.fiveFrameWindow ? parity : 1 - parity];
        if (pictures) {  // the two pictures' textures, each frame
            error = add_pass(motion[parity], deep ? kShader_motion_v1_16_img : kShader_motion_v1_8_img,
                             { kPictureBinding, kPictureBinding, &acc }, constants, kPushBytes, groups(w, 64),
                             groups(h, 16));
            if (!error)
                error = per_slot(motion[parity], [&](Slot &) {
                    return std::vector<Bound>{ kPictureBinding, kPictureBinding, Bound(&acc) };
                });
            if (!error)
                motion[parity].back().pictureRole = kPictureMotion;
            continue;
        }
        error = add_pass(motion[parity], deep ? kShader_motion_v1_16 : kShader_motion_v1_8,
                         { &picRef, previous, &acc }, constants, 6 * 4, groups(w, 64), groups(h, 16));
        if (!error && direct)  // the frame before's (or the one before it) reference where it is: its slot
            error = per_slot(motion[parity], [&](Slot &slot) {
                const size_t count = slots.size(), lag = options.fiveFrameWindow ? 2 : 1;
                Slot &before = slots[((size_t)(&slot - slots.data()) + count - lag) % count];
                return std::vector<Bound>{ slot_ref(slot), slot_ref(before), Bound(&acc) };
            });
    }

    if (skip & 4) {  // no ADM (vv_create's flag bit 18: a context for CIEDE2000 or CAMBI alone)
        v1AdmEnd = 0;
        return error;
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
        // (set 0 holds scale 2's bands only: rows of its width, not scale 0's)
        const int bandStride = set == 0 ? (((w + 1) / 2 + 1) / 2 + 1) / 2 : ((w + 1) / 2 + 1) / 2;
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
            // The region adm_cm / i4_adm_cm sum: with a border of 0 (small
            // pictures), also the first and last rows and columns, which they
            // sum apart (each of the two rows rounded as one).
            p[8] = top <= 0 ? 0 : start_row;
            p[9] = bottom > bh - 1 ? bh : end_row;
            p[10] = left <= 0 ? 0 : start_col;
            p[11] = right > bw - 1 ? bw : end_col;
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
            p[42] = std::min(p[8], p[12]);
            p[43] = std::max(p[9], p[13]);
        }
        if (scale < kScales - 1) {  // the next scale's transform, of all of the band image's a
            p[42] = 0;
            p[43] = bh;
            p[51] = set == 1 ? (((w + 1) / 2 + 1) / 2 + 1) / 2 : ((w + 1) / 2 + 1) / 2;  // the next scale's bands' stride
            p[52] = kV[scale + 1][0];
            p[53] = kV[scale + 1][1];
            p[54] = kH[scale + 1][0];
            p[55] = kH[scale + 1][1];
        }
        if (scale == 0) {  // the transform from the pictures
            p[44] = inW;
            p[45] = inH;
            p[46] = inStride;
            p[47] = bpc;
            p[48] = 1 << (bpc - 1);
            p[49] = kH[0][0];
            p[50] = kH[0][1];

        }
        {   // SLIDE: workgroups of kAdmOwnColumns columns down bands of
            // kAdmBandRows rows, each row's sums per workgroup into admPartial;
            // then ROWSUM adds and rounds each row's.
            const bool next = scale < kScales - 1;  // all of the band image (the next scale's transform)
            const int colStart = next ? 0 : std::min(p[10], p[14]);
            const int colEnd = next ? bw : std::max(p[11], p[15]);
            const int rows = std::max(0, p[43] - p[42]);
            const uint32_t chunks = groups(std::max(0, colEnd - colStart), kAdmOwnColumns);
            const uint32_t constants[] = { (uint32_t)(scale * kAdmParams), chunks };
            // adm_skip_scale0: scale 0 only transformed for scale 1, nothing summed (no ROWSUM).
            const bool skip = scale == 0 && options.skipScale0;
            if (scale == 0 && pictures) {
                uint32_t pushed[kPushBytes / 4] = { constants[0], constants[1] };
                const int shader = skip ? (deep ? kShader_adm_fused_0_16_img_skip : kShader_adm_fused_0_8_img_skip)
                                        : (deep ? kShader_adm_fused_0_16_img : kShader_adm_fused_0_8_img);
                error = add_pass(scored, shader,
                                 { kPictureBinding, kPictureBinding, &divTable, &admParams, &acc, &bandsRef[1],
                                   &bandsDis[1], &admPartial },
                                 pushed, kPushBytes, chunks, groups(rows, kAdmBandRows));
                if (!error)
                    error = per_slot(scored, [&](Slot &) {
                        return std::vector<Bound>{ kPictureBinding, kPictureBinding, Bound(&divTable),
                                                   Bound(&admParams), Bound(&acc), Bound(&bandsRef[1]),
                                                   Bound(&bandsDis[1]), Bound(&admPartial) };
                    });
                if (!error)
                    scored.back().pictureRole = kPictureAdm;
            } else if (scale == 0) {
                const int shader = skip ? (deep ? kShader_adm_fused_0_16_skip : kShader_adm_fused_0_8_skip)
                                        : (deep ? kShader_adm_fused_0_16 : kShader_adm_fused_0_8);
                error = add_pass(scored, shader,
                                 { inRef, inDis, &divTable, &admParams, &acc, &bandsRef[1], &bandsDis[1], &admPartial },
                                 constants, sizeof constants, chunks, groups(rows, kAdmBandRows));
                if (!error && direct)
                    error = per_slot(scored, [&](Slot &slot) {
                        return std::vector<Bound>{ slot_ref(slot), slot_dis(slot), Bound(&divTable), Bound(&admParams),
                                                   Bound(&acc), Bound(&bandsRef[1]), Bound(&bandsDis[1]),
                                                   Bound(&admPartial) };
                    });
            } else {
                error = next ? add_pass(scored, kShader_adm_fused_next, { &bandsRef[set], &bandsDis[set], &divTable,
                                        &admParams, &acc, &admPartial, &bandsRef[1 - set], &bandsDis[1 - set] },
                                        constants, sizeof constants, chunks, groups(rows, kAdmBandRowsSmall))
                             : add_pass(scored, kShader_adm_fused, { &bandsRef[set], &bandsDis[set], &divTable, &admParams,
                                        &acc, &admPartial }, constants, sizeof constants, chunks,
                                        groups(rows, kAdmBandRowsSmall));
            }
            if (!error && !skip)
                error = add_pass(scored, kShader_adm_rowsum, { &bandsRef[set], &bandsDis[set], &divTable, &admParams, &acc,
                                 &admPartial }, constants, sizeof constants, groups(rows * 9, 256), 1);
        }
        inW = bw;
        inH = bh;
        inStride = bandStride;
    }
    v1AdmEnd = scored.size();
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
        float den_scale, num_scale, aim_num_scale;
        if (scale == 0 && o.skipScale0) {  // as libvmaf: no measure at scale 0, a denominator above 0
            den_scale = 1e-10f;
            num_scale = 0.0f;
            aim_num_scale = 0.0f;
        } else {
            den_scale = v1_csf_den(&slots[kSlotCsfDen + scale * 3], w, h, scale, context.rfactorV1[scale],
                                   o.noiseWeight);
            num_scale = v1_cm(cm, w, h, scale, o.noiseWeight);
            aim_num_scale = v1_cm(aim, w, h, scale, 0.0);
        }
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
    integratedGpu = properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
    VkPhysicalDeviceFeatures features;
    api->vkGetPhysicalDeviceFeatures(physical, &features);
    if (!features.shaderInt64)
        return fail(-4, deviceName + " has no 64-bit integers in shaders");
    nativeDouble = (flags & 1) && features.shaderFloat64;
    skip = (flags >> 16) & 7;
    passLimit = (flags >> 20) & 0xFF;
    shared = (flags >> 19) & 1;
    // A decoder's pictures to be converted (vv_v1_convert): copied out of the
    // slots for the conversion, never read where they are.
    convertsShared = shared && (flags >> 1) & 1;
    pictures = shared && !convertsShared && (flags >> 2) & 1;  // (VMAF v1 only: decided below, once the device is made)
    uint32_t listedCount = 0;
    api->vkEnumerateDeviceExtensionProperties(physical, nullptr, &listedCount, nullptr);
    std::vector<VkExtensionProperties> listed(listedCount);
    if (listedCount)
        api->vkEnumerateDeviceExtensionProperties(physical, nullptr, &listedCount, listed.data());
    auto has = [&](const char *name) {
        for (const VkExtensionProperties &extension : listed)
            if (!strcmp(extension.extensionName, name))
                return true;
        return false;
    };
    std::vector<const char *> extensions;
    // Win32 external memory, where the driver has it: for frames from a
    // decoder (shared, or Direct3D textures imported).
    const bool externalMemory = has(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
    if (shared && !externalMemory)
        return fail(-4, deviceName + " cannot share its memory with a decoder");
    if (externalMemory)
        extensions.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
    // The SIMD width per shader, where the GPU lets it be chosen.
    VkPhysicalDeviceSubgroupSizeControlFeaturesEXT sizeControl = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT };
    if (has(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME) && api->vkGetPhysicalDeviceFeatures2) {
        VkPhysicalDeviceFeatures2 query = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
        query.pNext = &sizeControl;
        api->vkGetPhysicalDeviceFeatures2(physical, &query);
        subgroupSizeControl = sizeControl.subgroupSizeControl == VK_TRUE;
        if (subgroupSizeControl)
            extensions.push_back(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
        sizeControl.computeFullSubgroups = VK_FALSE;
        sizeControl.pNext = nullptr;
    }
    // Subgroups of 32 lanes for VMAF v1's shaders faster in them, on a GPU
    // whose compute shaders get more unless asked (an AMD GPU's 64): CAMBI's
    // sliding c-values and the fused ADM (Radeon 780M; the others are as fast
    // or faster in its default 64).
    if (subgroupSizeControl && api->vkGetPhysicalDeviceProperties2) {
        VkPhysicalDeviceSubgroupSizeControlPropertiesEXT sizes = {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES_EXT };
        VkPhysicalDeviceProperties2 properties2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
        properties2.pNext = &sizes;
        api->vkGetPhysicalDeviceProperties2(physical, &properties2);
        if (sizes.minSubgroupSize <= 32 && sizes.maxSubgroupSize > 32
            && (sizes.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT))
            narrowSubgroup = 32;
    }
    if (narrowSubgroup) {
        for (int shader : { kShader_cambi_cvalues_slide_16, kShader_cambi_cvalues_slide_24,
                            kShader_cambi_cvalues_slide, kShader_adm_fused_0_8,
                            kShader_adm_fused_0_16, kShader_adm_fused, kShader_adm_fused_0_8_img,
                            kShader_adm_fused_0_16_img, kShader_adm_fused_0_8_skip, kShader_adm_fused_0_16_skip,
                            kShader_adm_fused_0_8_img_skip, kShader_adm_fused_0_16_img_skip })
            subgroupSizes[shader] = narrowSubgroup;
    }
    // A decoder's copies waited for by the context (vv_import_timeline,
    // vv_commit_after), where the driver has timeline semaphores it can share;
    // else the decoder waits for its copies itself.
    if (shared && has(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME) && has(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME)
        && api->vkGetPhysicalDeviceFeatures2) {
        VkPhysicalDeviceTimelineSemaphoreFeatures supported = {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES };
        VkPhysicalDeviceFeatures2 query = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
        query.pNext = &supported;
        api->vkGetPhysicalDeviceFeatures2(physical, &query);
        timelines = supported.timelineSemaphore == VK_TRUE;
        if (timelines) {
            extensions.push_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
            extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
        }
    }
    // Intel's GPUs run ADM's shaders fastest at 8 lanes (Core Ultra 9 285K's
    // iGPU: ADM 2.97 -> 2.42 ms a 1080p pair), but the fused decouple and
    // masking (adm_dcm) at 16 (0.64 -> 0.44 ms at scale 0), and so the CSF
    // denominator's passes (4K: scale 0 0.213 -> 0.197 ms, 1-3 0.153 -> 0.140);
    // the others at the driver's choice. Subgroup operations are only integer
    // sums here (the same in any order): the width changes no sum.
    if (subgroupSizeControl && properties.vendorID == 0x8086) {
        for (int i = 0; i < kShaderCount; ++i) {
            const char *name = kShaders[i].name;
            if (!strncmp(name, "adm_", 4))
                subgroupSizes[i] = !strncmp(name, "adm_dcm", 7) || !strcmp(name, "adm_csf_den_0")
                                   || !strcmp(name, "adm_csf_den") ? 16 : 8;
        }
    }
    if (const char *text = getenv("VV_ADM_FUSED"))
        admFused = strcmp(text, "0") != 0;
    if (properties.deviceType != VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
        copyThreads = std::clamp(std::thread::hardware_concurrency(), 1u, 4u);
    if (const char *text = getenv("VV_COPY_THREADS"))
        copyThreads = (unsigned)std::clamp(atoi(text), 1, 16);
    if (const char *text = getenv("VV_DIRECT_FRAMES"))
        direct = strcmp(text, "0") != 0;
    // Only where the staging memory is the GPU's own (an integrated GPU, or
    // barFrames): a discrete GPU's shaders would read system memory across
    // PCIe -- an RTX 5090's 4K pair took 13.6 ms of GPU time that way against
    // 1.5 with the copy.
    api->vkGetPhysicalDeviceMemoryProperties(physical, &memoryProperties);
    if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && !shared) {
        // The device-local heap's host-visible type (Resizable BAR), on a
        // heap of at least 1 GiB (without it, a 256 MB window) and with room
        // for the slots' frames many times over (VMAF v1's: their chroma
        // planes too, for SpEED).
        const VkMemoryPropertyFlags bar = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                          | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        const VkDeviceSize frameBytes = (v1 ? 3ull : 2ull) * (((uint32_t)w * (bpc > 8 ? 2 : 1) + 3) & ~3u) * (uint32_t)h;
        const VkDeviceSize needed = frameBytes * (VkDeviceSize)(((flags >> 8) & 0xFF) ? std::clamp((flags >> 8) & 0xFF, 1, 16) : 3);
        for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
            const VkMemoryType &type = memoryProperties.memoryTypes[i];
            const VkDeviceSize heap = memoryProperties.memoryHeaps[type.heapIndex].size;
            if ((type.propertyFlags & bar) == bar && heap >= (1ull << 30) && needed * 8 <= heap)
                barFrames = true;
        }
        if (const char *text = getenv("VV_BAR_FRAMES"))
            barFrames = barFrames && strcmp(text, "0") != 0;
    }
    // VMAF v1: read where they are, in the GPU's own memory, when a decoder
    // writes them (shared); host memory, which the GPU reads more slowly than
    // its own (measured on a Radeon 780M: by the passes reading the pictures,
    // more than the copy costs), is copied first -- with barFrames, from the
    // GPU's own memory (an RTX 5090's 4K 10-bit pair: the frames' copy 0.67
    // -> 0.07 ms of GPU time, SpEED's pass reading the chroma 0.24 -> 0.01).
    if (v1) {
        direct = shared && !convertsShared;
    } else {
        direct = direct && (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU || barFrames || shared);
        barFrames = barFrames && direct;
    }
    if (const char *text = getenv("VV_VIF_FUSED"))
        vifFused = strcmp(text, "0") != 0;
    // vif_fused adds up its group's sums with subgroup arithmetic (Vulkan
    // 1.1's, optional): without it, VIF's two passes.
    {
        VkPhysicalDeviceSubgroupProperties subgroup = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES };
        VkPhysicalDeviceProperties2 query = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
        query.pNext = &subgroup;
        if (api->vkGetPhysicalDeviceProperties2)
            api->vkGetPhysicalDeviceProperties2(physical, &query);
        const VkSubgroupFeatureFlags needed = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
        if (!(subgroup.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) || (subgroup.supportedOperations & needed) != needed)
            vifFused = subgroupSums = false;
    }
    if (const char *text = getenv("VV_ADM_BOTH"))
        admBoth = strcmp(text, "0") != 0;
    // For experiments: VV_SUBGROUP="shader=16,shader=8" or "*=16".
    if (const char *text = getenv("VV_SUBGROUP")) {
        std::string spec = text;
        size_t at = 0;
        while (at < spec.size()) {
            size_t end = spec.find(',', at);
            std::string item = spec.substr(at, end == std::string::npos ? std::string::npos : end - at);
            size_t equals = item.find('=');
            if (equals != std::string::npos) {
                std::string name = item.substr(0, equals);
                uint32_t size = (uint32_t)atoi(item.c_str() + equals + 1);
                for (int i = 0; i < kShaderCount; ++i)
                    if (name == "*" || name == kShaders[i].name)
                        subgroupSizes[i] = size;
            }
            if (end == std::string::npos)
                break;
            at = end + 1;
        }
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
    admFused = admFused && admBoth && !v1 && !decoupleVariant && !passLimit;

    uint32_t familyCount = 0;
    api->vkGetPhysicalDeviceQueueFamilyProperties(physical, &familyCount, nullptr);
    std::vector<VkQueueFamilyProperties> families(familyCount);
    api->vkGetPhysicalDeviceQueueFamilyProperties(physical, &familyCount, families.data());
    // A compute queue without graphics if there is one (an AMD GPU's async
    // compute): the work of others on the GPU's graphics queue (a decoder's
    // copies by Direct3D 11, say) then need not wait behind this one's.
    // VV_COMPUTE_QUEUE=0 for the first queue with compute, to compare.
    const char *computeQueue = getenv("VV_COMPUTE_QUEUE");
    const bool computeOnly = !computeQueue || strcmp(computeQueue, "0") != 0;
    int family = -1;
    for (uint32_t i = 0; i < familyCount && family < 0 && computeOnly; ++i) {
        if ((families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT))
            family = (int)i;
    }
    for (uint32_t i = 0; i < familyCount && family < 0; ++i) {
        if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
            family = (int)i;
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
    VkPhysicalDevice16BitStorageFeatures storage16 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES };
    if (api->vkGetPhysicalDeviceFeatures2) {
        VkPhysicalDeviceFeatures2 query = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
        query.pNext = &storage16;
        api->vkGetPhysicalDeviceFeatures2(physical, &query);
    }
    samples16 = storage16.storageBuffer16BitAccess && features.shaderInt16;
    if (const char *text = getenv("VV_SAMPLES16"))
        samples16 = samples16 && strcmp(text, "0") != 0;
    enabled.shaderInt16 = samples16 ? VK_TRUE : VK_FALSE;
    storage16 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES };
    storage16.storageBuffer16BitAccess = samples16 ? VK_TRUE : VK_FALSE;
    enabled.shaderFloat64 = nativeDouble ? VK_TRUE : VK_FALSE;
    VkPhysicalDevice8BitStorageFeaturesKHR storage8 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES_KHR };
    VkPhysicalDeviceShaderFloat16Int8FeaturesKHR int8 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES_KHR };
    if (has(VK_KHR_8BIT_STORAGE_EXTENSION_NAME) && has(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME)
        && api->vkGetPhysicalDeviceFeatures2) {
        VkPhysicalDeviceFeatures2 query = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
        storage8.pNext = &int8;
        query.pNext = &storage8;
        api->vkGetPhysicalDeviceFeatures2(physical, &query);
    }
    samples8 = storage8.storageBuffer8BitAccess && int8.shaderInt8;
    if (const char *text = getenv("VV_SAMPLES8"))
        samples8 = samples8 && strcmp(text, "0") != 0;
    storage8 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES_KHR };
    storage8.storageBuffer8BitAccess = VK_TRUE;
    int8 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES_KHR };
    int8.shaderInt8 = VK_TRUE;
    if (samples8) {
        extensions.push_back(VK_KHR_8BIT_STORAGE_EXTENSION_NAME);
        extensions.push_back(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
    }
    // Experiments (VV_PIPELINE_STATS=<file>): what the driver's compiler made
    // of each shader, appended to the file as its pipeline is made.
    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR executables = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR };
    if (const char *file = getenv("VV_PIPELINE_STATS"); file && *file
        && has(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME)) {
        extensions.push_back(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
        executables.pipelineExecutableInfo = VK_TRUE;
        executables.pNext = subgroupSizeControl ? &sizeControl : nullptr;
        statsFile = file;
    }
    VkDeviceCreateInfo deviceInfo = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    if (!statsFile.empty())
        deviceInfo.pNext = &executables;
    else if (subgroupSizeControl)
        deviceInfo.pNext = &sizeControl;
    if (samples16) {
        storage16.pNext = (void *)deviceInfo.pNext;
        deviceInfo.pNext = &storage16;
    }
    if (samples8) {
        int8.pNext = (void *)deviceInfo.pNext;
        storage8.pNext = &int8;
        deviceInfo.pNext = &storage8;
    }
    VkPhysicalDeviceTimelineSemaphoreFeatures timelineFeature = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES };
    timelineFeature.timelineSemaphore = VK_TRUE;
    if (timelines) {
        timelineFeature.pNext = (void *)deviceInfo.pNext;
        deviceInfo.pNext = &timelineFeature;
    }
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    deviceInfo.pEnabledFeatures = &enabled;
    deviceInfo.enabledExtensionCount = (uint32_t)extensions.size();
    deviceInfo.ppEnabledExtensionNames = extensions.empty() ? nullptr : extensions.data();
    VkResult result = api->vkCreateDevice(physical, &deviceInfo, nullptr, &device);
    if (result != VK_SUCCESS) {
        device = VK_NULL_HANDLE;
        return fail(-1, "vkCreateDevice failed (" + std::to_string(result) + ")");
    }
#define X(name) vk.name = (PFN_##name)api->vkGetDeviceProcAddr(device, #name);
    VK_DEVICE_FUNCTIONS(X)
#undef X
    if (timelines) {
        importSemaphore = (PFN_vkImportSemaphoreWin32HandleKHR)api->vkGetDeviceProcAddr(device,
                                                                                        "vkImportSemaphoreWin32HandleKHR");
        createSemaphore = (PFN_vkCreateSemaphore)api->vkGetDeviceProcAddr(device, "vkCreateSemaphore");
        destroySemaphore = (PFN_vkDestroySemaphore)api->vkGetDeviceProcAddr(device, "vkDestroySemaphore");
        waitSemaphores = (PFN_vkWaitSemaphores)api->vkGetDeviceProcAddr(device, "vkWaitSemaphores");
        timelines = importSemaphore && createSemaphore && destroySemaphore && waitSemaphores;
    }
    vk.vkGetDeviceQueue(device, queueFamily, 0, &queue);
    if (shared) {
        getMemoryHandle = (PFN_vkGetMemoryWin32HandleKHR)api->vkGetDeviceProcAddr(device, "vkGetMemoryWin32HandleKHR");
        if (!getMemoryHandle)
            return fail(-4, deviceName + " cannot share its memory with a decoder");
    }
    if (externalMemory)
        getHandleProperties = memoryHandleProperties = (PFN_vkGetMemoryWin32HandlePropertiesKHR)api->vkGetDeviceProcAddr(
            device, "vkGetMemoryWin32HandlePropertiesKHR");
    // The pictures from a decoder's textures: VMAF v1 at 8 or 10 bits (NV12,
    // P010), motion against the frame before (not the five-frame window).
    pictures = pictures && v1 && (bpc == 8 || bpc == 10) && !options.fiveFrameWindow && memoryHandleProperties;
    split = shared && !pictures && (flags >> 3) & 1;  // the pictures need no planes in the slots

    VkCommandPoolCreateInfo poolInfo = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = queueFamily;
    if (vk.vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS)
        return fail(-1, "vkCreateCommandPool failed");
    // A set a pass, and one a frame slot more for each pass that reads the
    // frames: VMAF v1 with CAMBI and SpEED about 150 passes and 7 a slot (up
    // to 16 slots); VMAF v0.6.1 fewer.
    const VkDescriptorPoolSize poolSizes[2] = { { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 512 * kMaxBindings },
                                                { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 256 } };  // (vv_pictures')
    VkDescriptorPoolCreateInfo descriptorInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    descriptorInfo.maxSets = 512;
    descriptorInfo.poolSizeCount = 2;
    descriptorInfo.pPoolSizes = poolSizes;
    if (vk.vkCreateDescriptorPool(device, &descriptorInfo, nullptr, &descriptorPool) != VK_SUCCESS)
        return fail(-1, "vkCreateDescriptorPool failed");

    const uint32_t sampleBytes = bpc > 8 ? 2 : 1;
    strideBytes = ((uint32_t)w * sampleBytes + 3) & ~3u;
    planeBytes = strideBytes * (uint32_t)h;
    // The distorted plane's offset in a slot's staging buffer when it holds
    // both planes: a multiple of 256, which any storage buffer offset
    // alignment divides (the passes bind the plane there when they read it in
    // place).
    // In split mode each of a slot's two buffers holds one video's planes,
    // laid out alike: the luma at 0, then its chroma.
    disOffset = split ? 0 : (planeBytes + 255) & ~255u;
    const bool apart = direct && !shared;  // VMAF v0.6.1's distorted plane in stagingDis
    VkDeviceSize stagingBytes = apart || split ? planeBytes : (VkDeviceSize)disOffset + planeBytes;
    if (v1 && shared) {  // 4:2:0's chroma as libvmaf takes it, w/2 x h/2, its rows a word apart
        const uint32_t chromaStride = ((uint32_t)(w / 2) * (bpc > 8 ? 2 : 1) + 3) & ~3u;
        sharedChroma = (uint32_t)((stagingBytes + 255) & ~VkDeviceSize(255));
        sharedChromaSpacing = (chromaStride * (uint32_t)(h / 2) + 255) & ~255u;
        stagingBytes = (VkDeviceSize)sharedChroma + (split ? 2ull : 4ull) * sharedChromaSpacing;
    }
    // The pictures read where a decoder left them (vv_pictures): nothing is
    // written into the slots' shared buffers, which are made only for a
    // decoder to import (at 4K they were 3 x 47 MB).
    if (pictures)
        stagingBytes = 4096;
    const VkDeviceSize pixels = (VkDeviceSize)w * h;
    const int w1 = (w + 1) / 2, h1 = (h + 1) / 2, w2 = (w1 + 1) / 2, h2 = (h1 + 1) / 2;
    const int rw1 = (w / 2 + 1) / 2, rh1 = (h / 2 + 1) / 2;
    const VkDeviceSize v0 = v1 ? 0 : 1, only1 = v1 ? 1 : 0;  // buffers one of the two uses hold 4 bytes in the other
    // VIF's vertical results: only the two-pass VIF writes them (vif_fused
    // keeps them in group memory) -- 166 MB at 4K.
    const VkDeviceSize twoPass = vifFused && !nativeDouble ? 0 : 1;
    // ADM's decoupled images: only the separate decouple and masking passes
    // write them -- adm_dcm keeps them in group memory (190 MB at 4K).
    const VkDeviceSize decoupled = admFused ? 0 : 1;
    // VMAF v1 without motion or ADM (flag bits 16 and 18): the previous
    // reference and ADM's band images are not made (at 4K 16-bit, 33 and
    // about 30 MB).
    const VkDeviceSize prevUsed = v1 && (skip & 1) ? 0 : 1, bandsUsed = v1 && (skip & 4) ? 0 : 1;
    struct { Buffer *buffer; VkDeviceSize bytes; } sized[] = {
        { &picRef, direct ? 4 : planeBytes }, { &picDis, direct ? 4 : planeBytes },
        // Motion's blur, 16-bit: two pixels a word.
        { &blur[0], (VkDeviceSize)(w + 1) / 2 * h * 4 * v0 + 4 }, { &blur[1], (VkDeviceSize)(w + 1) / 2 * h * 4 * v0 + 4 },
        { &vifTmp, pixels * 4 * kVifTmpWords * v0 * twoPass + 4 },
        // vif_fused keeps both pictures' samples in rdRef (a word each position).
        { &rdRef[0], (VkDeviceSize)w1 * h1 * 4 * v0 + 4 }, { &rdDis[0], (VkDeviceSize)w1 * h1 * 4 * v0 * twoPass + 4 },
        { &rdRef[1], (VkDeviceSize)rw1 * rh1 * 4 * v0 + 4 }, { &rdDis[1], (VkDeviceSize)rw1 * rh1 * 4 * v0 * twoPass + 4 },
        // VMAF v1's previous reference, unless read from its slot (direct).
        { &picPrev[0], planeBytes * only1 * (direct ? 0 : 1) * prevUsed + 4 },
        { &picPrev[1], planeBytes * only1 * (direct ? 0 : 1) * prevUsed + 4 },
        { &logTable, 16384 * 4 }, { &divTable, 65536 * 4 },
        // VMAF v0.6.1's: scales 0 and 2 two words a position at scale 0, 1 and 3
        // three. VMAF v1's ADM (adm_fused.slang): scale 2's bands at [0], as
        // wide as they are, and scale 1's or 3's at [1], four words a position.
        { &bandsRef[0], (v1 ? (VkDeviceSize)((w2 + 1) / 2) * ((h2 + 1) / 2) * 16 : (VkDeviceSize)w1 * h1 * 8) * bandsUsed + 4 },
        { &bandsDis[0], (v1 ? (VkDeviceSize)((w2 + 1) / 2) * ((h2 + 1) / 2) * 16 : (VkDeviceSize)w1 * h1 * 8) * bandsUsed + 4 },
        { &bandsRef[1], (VkDeviceSize)w2 * h2 * (v1 ? 16 : 12) * bandsUsed + 4 },
        { &bandsDis[1], (VkDeviceSize)w2 * h2 * (v1 ? 16 : 12) * bandsUsed + 4 },
        { &bandsARef[0], (VkDeviceSize)w1 * h1 * 4 * v0 + 4 }, { &bandsADis[0], (VkDeviceSize)w1 * h1 * 4 * v0 + 4 },
        { &bandsARef[1], (VkDeviceSize)w2 * h2 * 4 * v0 + 4 }, { &bandsADis[1], (VkDeviceSize)w2 * h2 * 4 * v0 + 4 },
        // VMAF v1 passes nothing between ADM's passes (adm_fused.slang).
        { &admR, (VkDeviceSize)w1 * h1 * 16 * v0 * decoupled + 4 },
        { &admA, (VkDeviceSize)w1 * h1 * 16 * v0 * decoupled + 4 },
        { &admF, (VkDeviceSize)w1 * h1 * 16 * v0 * decoupled + 4 }, { &acc, kSlots * 8 },
        { &admRB, (VkDeviceSize)w1 * h1 * 16 * v0 * (admBoth ? 1 : 0) * decoupled + 4 },
        { &admAB, (VkDeviceSize)w1 * h1 * 16 * v0 * (admBoth ? 1 : 0) * decoupled + 4 },
        { &admFB, (VkDeviceSize)w1 * h1 * 16 * v0 * (admBoth ? 1 : 0) * decoupled + 4 },
        // Six 64-bit sums a row of contrast masking at each scale: at most its
        // band images' rows.
        { &admRows[0], (VkDeviceSize)h1 * 6 * 8 * (admFused ? 1 : 0) + 4 },
        { &admRows[1], (VkDeviceSize)h2 * 6 * 8 * (admFused ? 1 : 0) + 4 },
        { &admRows[2], (VkDeviceSize)((h2 + 1) / 2) * 6 * 8 * (admFused ? 1 : 0) + 4 },
        { &admRows[3], (VkDeviceSize)((h2 + 3) / 4) * 6 * 8 * (admFused ? 1 : 0) + 4 },
    };
    for (auto &entry : sized) {
        if (int error = create_buffer(*entry.buffer, entry.bytes, false))
            return error;
    }

    const int depth = std::clamp((flags >> 8) & 0xFF, 1, 16);
    slots.resize((size_t)(((flags >> 8) & 0xFF) ? depth : 3));
    // VMAF v1 read where they are: the slots' pictures stay as long as the
    // frames after read them as their previous (staging()): a slot or two
    // more, for as many frames in flight.
    if (v1 && direct && !pictures)
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
        if (int error = split ? create_buffer(slot.stagingDis, stagingBytes, false, true, false)
                              : create_buffer(slot.stagingDis, apart ? planeBytes : 4, true, false, true))
            return error;
        if (int error = create_buffer(slot.result, kSlots * 8, true))
            return error;
    }
    // Timing tests (VV_GPU_TIME=<file>): the profiler's timestamps (vv_profile),
    // and its report appended to the file when the context closes (~vv_context).
    if (const char *file = getenv("VV_GPU_TIME"); file && *file) {
        timeFile = file;
        if (int error = enable_profile())
            return error;
    }
    // The tables, and zeros where a first frame reads before anything wrote:
    // the previous blur (as the CUDA code's memset) and the band images.
    std::vector<uint32_t> table(65536);
    for (uint32_t i = 0; i < 32768; ++i)
        table[i] = cuda_log_generate(32768 + i);  // log2(32768..65535) * 2048: values 30720..32768
    // Two entries a word (vif_stats.slang's log_generate).
    for (uint32_t i = 0; i < 16384; ++i)
        table[i] = table[2 * i] | (table[2 * i + 1] << 16);
    if (int error = upload(logTable, table.data(), 16384 * 4))
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
                                &bandsRef[0], &bandsDis[0], &bandsRef[1], &bandsDis[1], &bandsARef[0],
                                &bandsADis[0], &bandsARef[1], &bandsADis[1], &admR, &admA, &admF,
                                &admRB, &admAB, &admFB, &picPrev[0], &picPrev[1] })
            vk.vkCmdFillBuffer(cb, buffer->buffer, 0, VK_WHOLE_SIZE, 0);
        vk.vkEndCommandBuffer(cb);
        if (run_now(cb) != VK_SUCCESS)
            return fail(-1, "clearing the GPU buffers failed");
    }
    if (int error = build_passes())
        return error;
    return compile_pipelines();
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
    // The full picture's mask only where scale 0 is the full picture (no
    // speedup): FRONT writes scale 0's decimated one otherwise.
    const VkDeviceSize full = c.speedup ? 1 : pixels;
    struct { Buffer *buffer; VkDeviceSize bytes; } sized[] = {
        { &cambiMaskFull, full * 4 },
        { &cambiReciprocal, sizeof kCambiReciprocal }, { &cambiHist, (VkDeviceSize)histWords * 4 },
        { &cambiState, 36 * 4 },  // shaders/cambi.slang's state (STATE_*)
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
            if (int error = create_buffer(cambiMask[scale], n * 4, false))
                return error;
        }
    }
    if (int error = upload(cambiReciprocal, kCambiReciprocal, sizeof kCambiReciprocal))
        return error;
    {   // level 0's sums start at 0 (SELECT puts them back to 0 for the next frame)
        const uint32_t zeros[36] = {};
        if (int error = upload(cambiState, zeros, sizeof zeros))
            return error;
    }
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
        const int front = pictures
            ? (c.speedup ? (bpc > 8 ? kShader_cambi_front_16_img : kShader_cambi_front_8_img)
                         : (bpc > 8 ? kShader_cambi_front_16_1_img : kShader_cambi_front_8_1_img))
            : (c.speedup ? (bpc > 8 ? kShader_cambi_front_16 : kShader_cambi_front_8)
                         : (bpc > 8 ? kShader_cambi_front_16_1 : kShader_cambi_front_8_1));
        // (scale 0's MODE too: it writes scale 0's filtered image)
        uint32_t pushed[kPushBytes / 4] = {};
        memcpy(pushed, constants, sizeof constants);
        error = add_pass(scored, front,
                         { pictures ? kPictureBinding : Bound(&picDis), &cambiFiltered[0],
                           c.speedup ? &cambiMask[0] : &cambiMaskFull, &acc },
                         pushed, pictures ? kPushBytes : (uint32_t)sizeof constants, groups((int)outW, 16),
                         groups((int)outH, 16));
        if (!error && direct)
            error = per_slot(scored, [&](Slot &slot) {
                return std::vector<Bound>{ pictures ? kPictureBinding : slot_dis(slot), Bound(&cambiFiltered[0]),
                                           Bound(c.speedup ? &cambiMask[0] : &cambiMaskFull), Bound(&acc) };
            });
        if (!error && pictures)
            scored.back().pictureRole = kPictureCambi;
    }
    // Each scale's image is the one before's filtered one decimated, its mask
    // the one before's decimated: MODE reads them every other pixel and row.
    Buffer *image = &cambiFiltered[0];  // (FRONT's)
    Buffer *mask = c.speedup ? &cambiMask[0] : &cambiMaskFull;
    int inW = c.scale_w[0];
    // The scales' c-values (each from its filtered image and mask) after all
    // of those, together: the small scales' few workgroups, one row after
    // another, took about as long as scale 0's, the GPU mostly idle.
    std::vector<Pass> cvalues;
    for (int scale = 0; scale < V1_CAMBI_SCALES && !error; ++scale) {
        const int sw = c.scale_w[scale], sh = c.scale_h[scale];
        if (scale > 0) {  // (scale 0's: FRONT's)
            Buffer *const maskIn = mask;
            if (scale > 0)
                mask = &cambiMask[scale];  // MODE decimates it
            const uint32_t constants[] = { (uint32_t)sw, (uint32_t)sh, (uint32_t)inW, scale > 0 ? 2u : 1u };
            error = add_pass(scored, kShader_cambi_mode, { image, &cambiFiltered[scale], maskIn, mask },
                             constants, sizeof constants, groups(sw, 16), groups(sh, 16));
        }
        if (!error) {
            // The sliding counts (CVALUES_SLIDE) for windows of at most 65, in
            // bands of these rows a workgroup: fewer for the smaller scales,
            // whose few workgroups (each a band's rows one after another) the
            // GPU otherwise waited on -- on a Radeon 780M, of 16 at every
            // scale, 2.42 -> 2.30 ms a frame at 4K and 1.08 -> 0.89 at 1080p
            // (taller bands, with fewer rows of each band's first window
            // counted, were slower: 32 and 64 rows took 2.85 and 3.54 at 4K).
            static const uint32_t kRows[V1_CAMBI_SCALES] = { 16, 8, 8, 4, 4 };
            const uint32_t rows = kRows[scale];
            uint32_t constants[15] = { (uint32_t)sw, (uint32_t)sh, (uint32_t)(c.window_size >> 1),
                                       (uint32_t)c.vlt_luma, (uint32_t)c.v_band_base, (uint32_t)c.v_band_size };
            for (int d = 0; d < 4; ++d) {
                constants[6 + d] = (uint32_t)c.tvi_for_diff[d];
                constants[10 + d] = (uint32_t)c.diff_weights[d];
            }
            constants[14] = rows;
            const bool slide = (c.window_size >> 1) <= 32 && c.v_band_size <= 32 * 56 && !cambiBruteForce;
            // The variant of the smallest shared tile that takes the pad: HDR
            // VMAF's 5158x2160 pictures (a letterboxed 4K film scaled to the
            // model's 2160 rows) have pad 20.
            const int pad = c.window_size >> 1;
            const int slideShader = pad <= 16   ? kShader_cambi_cvalues_slide_16
                                    : pad <= 24 ? kShader_cambi_cvalues_slide_24
                                                : kShader_cambi_cvalues_slide;
            error = slide ? add_pass(cvalues, slideShader, { &cambiFiltered[scale], mask, &cambiC[scale],
                                     &cambiReciprocal }, constants, sizeof constants, groups(sw, 64), groups(sh, (int)rows))
                          : add_pass(cvalues, kShader_cambi_cvalues, { &cambiFiltered[scale], mask, &cambiC[scale],
                                     &cambiReciprocal }, constants, sizeof constants, groups(sw, 16), groups(sh, 16));
        }
        image = &cambiFiltered[scale];
        inW = sw;
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
            uint32_t select[12] = { level };
            for (int scale = 0; scale < V1_CAMBI_SCALES; ++scale) {
                select[1 + scale] = (uint32_t)c.topk_elements[scale];
                select[7 + scale] = sizes[1 + scale];  // the scale's c-values
            }
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
                // (the scales' copies, of their own buffers: side by side)
                slot.cambiKeepPasses.back().overlapNext = scale + 1 < V1_CAMBI_SCALES;
            }
        }
    }
    if (error)
        return error;
    if (int compiled = compile_pipelines())  // the passes added here
        return compiled;
    // The pool's scorer reads only what is fixed from here on (cambiConst, cambiOptions).
    cambiPool.configure([this](const uint64_t *sums, unsigned scales, float *values, double &out) {
        double pooled[V1_CAMBI_SCALES] = {};
        for (int scale = 0; scale < V1_CAMBI_SCALES; ++scale) {
            if (!(scales & (1u << scale)))
                continue;
            const unsigned sw = (unsigned)cambiConst.scale_w[scale], sh = (unsigned)cambiConst.scale_h[scale];
            pooled[scale] = v1_cambi_pool(values, cambiOptions.topk, sw, sh);
            values += (size_t)sw * sh;
        }
        return cambi_score(sums, pooled, scales, out);
    }, (int)std::clamp(std::thread::hardware_concurrency() / 4, 1u, 4u));  // as SpEED's
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
    if (converting)  // whose passes write the chroma where SpEED would not read it
        return fail(-3, "SpEED is started before the conversion (vv_v1_convert)");
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
        // (the planes from host memory, vv_chroma_staging: not with the pictures read where they are.
        // Converted ones: the conversion's, written and read on the GPU, so in its own memory. A shared
        // context has no Resizable BAR uploads, and the planes would cross PCIe twice: an RTX 5090's 4K
        // HDR pair, convert_resize_float 0.4 -> 6.7 ms, speed_dec 0.02 -> 0.30.)
        const VkDeviceSize chromaBytes = pictures ? 4 : (VkDeviceSize)chromaPlaneBytes * 4;
        if (int error = convertsShared ? create_buffer(slot.speedChroma, chromaBytes, false)
                                       : create_buffer(slot.speedChroma, chromaBytes, true, false, true))
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
        for (int fromStaging = pictures ? 1 : 0; fromStaging < (shared ? 2 : 1); ++fromStaging) {
            std::vector<Pass> &list = shared && !fromStaging ? slot.speedPassesHost : slot.speedPasses;
            uint32_t constants[kPushBytes / 4] = {};
            memcpy(constants, dec, sizeof dec);
            Bound chroma = Bound(&slot.speedChroma);
            if (fromStaging) {
                chroma = Bound(&slot.staging, sharedChroma, (split ? 2ull : 4ull) * sharedChromaSpacing);
                constants[3] = sharedChromaSpacing / 4;  // planeWords
            }
            if (fromStaging && pictures) {  // from the pictures' textures (vv_pictures)
                if (int error = add_pass(list, kShader_speed_dec_img, { kPictureBinding, &speedFilterTaps,
                                         &speedOperating, &speedScaling, kPictureBinding }, constants, kPushBytes,
                                         groups(f.operating_w * f.operating_h * 4, perGroup), 1))
                    return error;
                list.back().pictureRole = kPictureSpeed;
            } else if (fromStaging && split) {  // the distorted's two planes from its own buffer
                if (int error = add_pass(list, kShader_speed_dec_split, { chroma, &speedFilterTaps, &speedOperating,
                                         &speedScaling, Bound(&slot.stagingDis, sharedChroma,
                                                              2ull * sharedChromaSpacing) },
                                         constants, sizeof dec, groups(f.operating_w * f.operating_h * 4, perGroup), 1))
                    return error;
            } else if (int error = add_pass(list, kShader_speed_dec, { chroma, &speedFilterTaps, &speedOperating,
                                            &speedScaling }, constants, sizeof dec,
                                            groups(f.operating_w * f.operating_h * 4, perGroup), 1)) {
                return error;
            }
            if (int error = add_pass(list, kShader_speed_blur, { &speedOperating, &speedFilterTaps, &slot.speedKeep },
                                     blur, sizeof blur, groups(f.operating_w, 16), groups(f.operating_h, 16)))
                return error;
            list.back().groups[2] = 4;
        }
    }
    if (int compiled = compile_pipelines())  // the passes added here
        return compiled;
    const int workers = (int)std::clamp(std::thread::hardware_concurrency() / 4, 1u, 4u);
    if (!speedPool.start(o, chromaW, chromaH, (size_t)operating, workers))
        return fail(-2, "SpEED: its threads could not be started");
    speed = true;
    return 0;
}

// CIEDE2000's pass (shaders/ciede.slang): a thread per 2x2 luma block, the
// pair's chroma copied into the slot's ciedeChroma by submit_v1.
int vv_context::enable_ciede()
{
    if (!v1)
        return fail(-3, "CIEDE2000 is calculated in a VMAF v1 context only");
    if (frames || ciede)
        return fail(-3, "CIEDE2000 is started before the first frame");
    // A decoder's frames (their chroma is not copied where the pass reads
    // it) and converted ones (the conversion writes SpEED's chroma only):
    // the caller scores those on the CPU.
    if (shared || converting)
        return fail(-4, "CIEDE2000 on the GPU takes its frames from host memory (vv_submit_v1)");
    ciedeW = (uint32_t)(w + 1) / 2;
    ciedeH = (uint32_t)(h + 1) / 2;
    ciedeStrideBytes = (ciedeW * (bpc > 8 ? 2 : 1) + 3) & ~3u;
    ciedePlaneBytes = ciedeStrideBytes * ciedeH;
    const uint32_t constants[] = { (uint32_t)w, (uint32_t)h, strideBytes / 4, ciedeW, ciedeH, ciedeStrideBytes / 4,
                                   (uint32_t)bpc, kSlotCiede };
    const int shader = bpc > 8 ? kShader_ciede_16 : kShader_ciede_8;
    ciedeFromStaging = !direct && (skip & 5) == 5 && (integratedGpu || barFrames);
    for (Slot &slot : slots) {
        if (int error = create_buffer(slot.ciedeChroma, (VkDeviceSize)ciedePlaneBytes * 4, true, false, true))
            return error;
        // The luma where the other features read it: the slot's own (direct)
        // or the copies -- or, read by this pass alone, the slot's staging.
        const bool fromSlot = direct || ciedeFromStaging;
        const Bound ref = fromSlot ? slot_ref(slot) : Bound(&picRef);
        const Bound dis = fromSlot ? slot_dis(slot) : Bound(&picDis);
        if (int error = add_pass(slot.ciedePasses, shader, { ref, dis, &slot.ciedeChroma, &acc }, constants,
                                 sizeof constants, groups(ciedeW, 16), groups(ciedeH, 16)))
            return error;
    }
    if (int compiled = compile_pipelines())  // the passes added here
        return compiled;
    ciede = true;
    return 0;
}

// ------------------------------------------------- the HDR picture conversion

// Every plan buffer of `kind` this engine lays out: a source plane (Y, U or
// V of the 4:2:0 source), a target plane (this context's size: the features'
// luma, SpEED's chroma) or scratch.
static bool plan_buffer_fits(uint32_t kind, uint32_t plane, uint64_t elements, const uint32_t sourceW[3],
                             const uint32_t sourceH[3], int w, int h)
{
    if (kind == VMAF_GPU_PLAN_SCRATCH)
        return elements > 0 && elements < (1ull << 30);
    if (plane > 2)
        return false;
    if (kind == VMAF_GPU_PLAN_SOURCE)
        return elements == (uint64_t)sourceW[plane] * sourceH[plane];
    const uint64_t targetW = plane ? (uint64_t)(w >> 1) : (uint64_t)w, targetH = plane ? (uint64_t)(h >> 1) : (uint64_t)h;
    return kind == VMAF_GPU_PLAN_TARGET && elements == targetW * targetH;
}

// A side's plan checked, step by step, so that the shaders read and write
// only inside their buffers whatever the words (a read past a buffer's end
// can reset the GPU), and kept.
int vv_context::load_convert_plan(int side, const uint32_t *plan, size_t words)
{
    if (!plan || words < VMAF_GPU_PLAN_HEADER_WORDS || words > (1u << 26)
        || plan[VMAF_GPU_PLAN_H_MAGIC] != VMAF_GPU_PLAN_MAGIC || plan[VMAF_GPU_PLAN_H_VERSION] != VMAF_GPU_PLAN_VERSION
        || plan[VMAF_GPU_PLAN_H_WORDS] != words)
        return fail(-3, "not a picture conversion plan of this version");
    if (plan[VMAF_GPU_PLAN_H_DST_W] != (uint32_t)w || plan[VMAF_GPU_PLAN_H_DST_H] != (uint32_t)h
        || plan[VMAF_GPU_PLAN_H_DST_BPC] != 16)
        return fail(-3, "the conversion plan's target is not this context's size at 16 bits");
    const uint32_t buffers = plan[VMAF_GPU_PLAN_H_BUFFER_COUNT], steps = plan[VMAF_GPU_PLAN_H_STEP_COUNT];
    if (buffers > 64 || steps > 64 || (uint64_t)plan[VMAF_GPU_PLAN_H_BUFFERS] + (uint64_t)buffers * VMAF_GPU_PLAN_BUFFER_WORDS > words
        || (uint64_t)plan[VMAF_GPU_PLAN_H_STEPS] + (uint64_t)steps * VMAF_GPU_PLAN_STEP_WORDS > words)
        return fail(-3, "the conversion plan's buffers or steps run past its end");
    ConvertSide &c = convertSides[side];
    const uint32_t srcW = plan[VMAF_GPU_PLAN_H_SRC_W], srcH = plan[VMAF_GPU_PLAN_H_SRC_H];
    if (srcW < 2 || srcH < 2 || srcW > 16384 || srcH > 16384)
        return fail(-3, "the conversion plan's source size is not taken");
    if (plan[VMAF_GPU_PLAN_H_SRC_BPC] <= 8 || plan[VMAF_GPU_PLAN_H_SRC_BPC] > 16)  // (the shaders read 16-bit samples)
        return fail(-4, "the conversion runs on the GPU for sources of 9 to 16 bits");
    const uint32_t planeW[3] = { srcW, srcW >> 1, srcW >> 1 }, planeH[3] = { srcH, srcH >> 1, srcH >> 1 };
    std::vector<uint64_t> elements(buffers);
    std::vector<uint32_t> kinds(buffers);
    for (uint32_t b = 0; b < buffers; ++b) {
        const uint32_t *record = plan + plan[VMAF_GPU_PLAN_H_BUFFERS] + b * VMAF_GPU_PLAN_BUFFER_WORDS;
        kinds[b] = record[VMAF_GPU_PLAN_B_KIND];
        elements[b] = record[VMAF_GPU_PLAN_B_ELEMENTS];
        if (!plan_buffer_fits(kinds[b], record[VMAF_GPU_PLAN_B_PLANE], elements[b], planeW, planeH, w, h))
            return fail(-3, "a conversion plan buffer is not a plane this engine lays out");
    }
    auto table_fits = [&](uint64_t at, uint64_t count) { return at + count <= words; };
    for (uint32_t i = 0; i < steps; ++i) {
        const uint32_t *s = plan + plan[VMAF_GPU_PLAN_H_STEPS] + i * VMAF_GPU_PLAN_STEP_WORDS;
        const uint32_t kind = s[VMAF_GPU_PLAN_S_KIND], planes = s[VMAF_GPU_PLAN_S_PLANES];
        const uint32_t inW = s[VMAF_GPU_PLAN_S_IN_W], inH = s[VMAF_GPU_PLAN_S_IN_H];
        const uint32_t outW = s[VMAF_GPU_PLAN_S_OUT_W], outH = s[VMAF_GPU_PLAN_S_OUT_H];
        const bool resize = kind >= VMAF_GPU_PLAN_RESIZE_FLOAT_H && kind <= VMAF_GPU_PLAN_RESIZE_INT16_V;
        if ((!resize && kind != VMAF_GPU_PLAN_COLOUR && kind != VMAF_GPU_PLAN_CONVERT)
            || planes != (kind == VMAF_GPU_PLAN_COLOUR ? 3u : 1u) || !inW || !inH || !outW || !outH
            || inW > 16384 || inH > 16384 || outW > 16384 || outH > 16384 || (!resize && (inW != outW || inH != outH)))
            return fail(-3, "a conversion plan step is not one this engine runs");
        for (uint32_t p = 0; p < planes; ++p) {
            const uint32_t in = s[VMAF_GPU_PLAN_S_IN + p], out = s[VMAF_GPU_PLAN_S_OUT + p];
            const uint32_t read = s[VMAF_GPU_PLAN_S_READ + 4 * p], write = s[VMAF_GPU_PLAN_S_WRITE + 4 * p];
            if (in >= buffers || out >= buffers || kinds[in] == VMAF_GPU_PLAN_TARGET || kinds[out] == VMAF_GPU_PLAN_SOURCE
                || elements[in] < (uint64_t)inW * inH || elements[out] < (uint64_t)outW * outH
                || read > VMAF_GPU_PLAN_READ_W2F || write > VMAF_GPU_PLAN_WRITE_F2W || (resize && in == out))
                return fail(-3, "a conversion plan step reads or writes outside its buffers");
            // A target plane is written row by row at the engine's stride: the step makes all of it.
            if (kinds[out] == VMAF_GPU_PLAN_TARGET && elements[out] != (uint64_t)outW * outH)
                return fail(-3, "a conversion plan step writes part of a target plane");
        }
        const uint32_t table = s[VMAF_GPU_PLAN_S_TABLE], width = s[VMAF_GPU_PLAN_S_FILTER_WIDTH];
        if (resize) {
            const bool vertical = kind == VMAF_GPU_PLAN_RESIZE_FLOAT_V || kind == VMAF_GPU_PLAN_RESIZE_INT16_V;
            const uint32_t dim = vertical ? outH : outW, inDim = vertical ? inH : inW;
            if ((vertical ? outW != inW : outH != inH) || width < 1 || width > 64
                || !table_fits(table, (uint64_t)dim * (1 + width)))
                return fail(-3, "a conversion plan resize's filter runs past the plan or its picture");
            for (uint32_t k = 0; k < dim; ++k)
                if ((uint64_t)plan[table + k] + width > inDim)
                    return fail(-3, "a conversion plan resize's filter runs past the plan or its picture");
        } else if (kind == VMAF_GPU_PLAN_COLOUR) {
            if (width > 8 || !table_fits(table, 2ull * width))
                return fail(-3, "a conversion plan's colour operations run past its end");
            for (uint32_t k = 0; k < width; ++k) {
                const uint32_t op = plan[table + 2 * k], at = plan[table + 2 * k + 1];
                const uint32_t size = op == VMAF_GPU_PLAN_OP_MATRIX ? 9 : op == VMAF_GPU_PLAN_OP_TO_LINEAR ? 65537
                                    : op == VMAF_GPU_PLAN_OP_TO_GAMMA ? 65536 : 0;
                if (!size || !table_fits(at, size))
                    return fail(-3, "a conversion plan's colour operations run past its end");
            }
        }
    }
    c.words.assign(plan, plan + words);
    memcpy(c.planeW, planeW, sizeof planeW);
    memcpy(c.planeH, planeH, sizeof planeH);
    return 0;
}

// The chroma target's row and plane bytes: SpEED's chroma upload buffer's,
// or convertChroma's in the same layout.
static uint32_t chroma_stride_bytes(int w) { return ((uint32_t)(w / 2) * 2 + 3) & ~3u; }

// A side's places (convert.slang: where each plan buffer is, for each slot),
// its plan and places on the GPU, its scratch, and a pass per step.
int vv_context::add_convert_side(int side)
{
    ConvertSide &c = convertSides[side];
    const uint32_t buffers = c.words[VMAF_GPU_PLAN_H_BUFFER_COUNT];
    const uint32_t chromaStride = chroma_stride_bytes(w), chromaPlane = chromaStride * (uint32_t)(h / 2);
    std::vector<uint32_t> places;
    uint64_t scratchWords = 0;
    for (size_t k = 0; k < slots.size(); ++k) {
        scratchWords = 0;
        for (uint32_t b = 0; b < buffers; ++b) {
            const uint32_t *record = c.words.data() + c.words[VMAF_GPU_PLAN_H_BUFFERS] + b * VMAF_GPU_PLAN_BUFFER_WORDS;
            const uint32_t plane = record[VMAF_GPU_PLAN_B_PLANE];
            switch (record[VMAF_GPU_PLAN_B_KIND]) {
            case VMAF_GPU_PLAN_SOURCE:  // in slot k's part of convertSource (16-bit samples)
                places.insert(places.end(), { 0u, (uint32_t)((k * convertSourceBytes + c.planeOffset[plane]) / 2), 0u });
                break;
            case VMAF_GPU_PLAN_SCRATCH:  // the side's own, the same for every slot: commit's first two barriers
                                         // (compute -> transfer -> compute) order it after the last frame's passes
                places.insert(places.end(), { 1u, (uint32_t)scratchWords, 0u });
                scratchWords += record[VMAF_GPU_PLAN_B_ELEMENTS];
                break;
            default:  // the target's: luma at the features' stride, chroma in SpEED's layout (ref U, ref V, dis U, dis V)
                if (plane == 0)
                    places.insert(places.end(), { 2u, 0u, strideBytes / 2 });
                else
                    places.insert(places.end(),
                                  { 3u, ((uint32_t)side * 2 + plane - 1) * chromaPlane / 2, chromaStride / 2 });
                break;
            }
        }
    }
    if (scratchWords >= (1ull << 30) || (uint64_t)slots.size() * convertSourceBytes >= (1ull << 32))
        return fail(-3, "the conversion's pictures are too large");
    c.placesAt = (uint32_t)c.words.size();
    c.placesWords = 3 * buffers;
    c.words.insert(c.words.end(), places.begin(), places.end());
    const uint32_t *plan = c.words.data();
    const uint32_t placesAt = c.placesAt;
    if (int error = create_buffer(c.tables, c.words.size() * 4, false))
        return error;
    if (int error = upload(c.tables, c.words.data(), c.words.size() * 4))
        return error;
    if (int error = create_buffer(c.scratch, std::max<uint64_t>(scratchWords, 1) * 4, false))
        return error;
    Buffer *luma = side ? &picDis : &picRef;
    for (uint32_t i = 0; i < plan[VMAF_GPU_PLAN_H_STEP_COUNT]; ++i) {
        const uint32_t at = plan[VMAF_GPU_PLAN_H_STEPS] + i * VMAF_GPU_PLAN_STEP_WORDS;
        const uint32_t kind = plan[at + VMAF_GPU_PLAN_S_KIND];
        const int shader = kind == VMAF_GPU_PLAN_COLOUR ? kShader_convert_colour
                           : kind == VMAF_GPU_PLAN_CONVERT ? kShader_convert_samples
                           : kind <= VMAF_GPU_PLAN_RESIZE_FLOAT_V ? kShader_convert_resize_float
                                                                  : kShader_convert_resize_int16;
        const uint32_t vertical = kind == VMAF_GPU_PLAN_RESIZE_FLOAT_V || kind == VMAF_GPU_PLAN_RESIZE_INT16_V;
        // (the places' offset is slot 0's here: record_convert pushes the frame's slot's)
        const uint32_t constants[4] = { at, placesAt, vertical, 0 };  // (the last: 0.0f)
        const uint32_t outW = plan[at + VMAF_GPU_PLAN_S_OUT_W], outH = plan[at + VMAF_GPU_PLAN_S_OUT_H];
        // A thread makes two samples of a row (convert.slang's 16 x 8 groups).
        if (int error = add_pass(c.passes, shader, { &c.tables, &convertSource, &c.scratch, luma,
                                 speed ? &slots[0].speedChroma : &convertChroma }, constants, sizeof constants,
                                 groups((int)(outW + 1) / 2, 16), groups((int)outH, 8)))
            return error;
        // SpEED's chroma is the slot's own: a set a slot for a step that writes it.
        bool writesChroma = false;
        for (uint32_t p = 0; p < plan[at + VMAF_GPU_PLAN_S_PLANES]; ++p) {
            const uint32_t *out = plan + plan[VMAF_GPU_PLAN_H_BUFFERS]
                                  + plan[at + VMAF_GPU_PLAN_S_OUT + p] * VMAF_GPU_PLAN_BUFFER_WORDS;
            writesChroma = writesChroma || (out[VMAF_GPU_PLAN_B_KIND] == VMAF_GPU_PLAN_TARGET && out[VMAF_GPU_PLAN_B_PLANE]);
        }
        if (speed && writesChroma) {
            if (int error = per_slot(c.passes, [&](Slot &slot) {
                    return std::vector<Bound>{ &c.tables, &convertSource, &c.scratch, luma, &slot.speedChroma };
                }))
                return error;
        }
    }
    return 0;
}

// The HDR models' picture conversion on the GPU (vv_v1_convert).
int vv_context::enable_convert(const uint32_t *const plans[2], const size_t words[2])
{
    if (!v1 || pictures || direct || bpc != 16)
        return fail(-3, "the conversion runs in a VMAF v1 context of 16 bits whose frames come through vv_submit_v1 "
                        "or, made with flag bit 1, a decoder's copies into its shared buffers");
    if (frames || converting)
        return fail(-3, "the conversion is set once, before the first frame");
    if (ciede)  // whose pass reads the slots' planes, which the conversion leaves unused
        return fail(-3, "the conversion is set before CIEDE2000 (which then runs on the CPU)");
    for (int side = 0; side < 2; ++side) {
        if (int error = load_convert_plan(side, plans[side], words[side]))
            return error;
    }
    // Both sides' source planes in each slot's part of convertSource, rows
    // packed, each plane starting on a word.
    convertSourceBytes = 0;
    for (ConvertSide &c : convertSides) {
        for (int p = 0; p < 3; ++p) {
            c.planeOffset[p] = convertSourceBytes;
            convertSourceBytes += (c.planeW[p] * c.planeH[p] * 2 + 3) & ~3u;
        }
    }
    if (shared) {
        // A decoder writes the source planes into the slots' shared buffers
        // (vv_convert_source), which it imported when the context was made:
        // they must hold them, and stay as they are. commit copies them into
        // convertSource on the GPU (4K, both sides: 37 MB, tens of
        // microseconds), which stays in the GPU's memory.
        for (int side = 0; side < 2; ++side) {
            if (shared_source_offset(side) + convert_side_bytes(side) > slots[0].staging.size)
                return fail(-3, "the conversion's source pictures do not fit the shared buffers");
        }
        if (int error = create_buffer(convertSource, (VkDeviceSize)slots.size() * convertSourceBytes, false))
            return error;
    } else {
        if (int error = create_buffer(convertSource, (VkDeviceSize)slots.size() * convertSourceBytes, true, false, true))
            return error;
        // The staging buffers the conversion leaves unused (VMAF v1 copies
        // them into picRef and picDis, which the conversion writes instead):
        // 4K, two 16-bit planes, 31.6 MB a slot.
        for (Slot &slot : slots) {
            if (int error = remake_buffer(slot.staging, 4, true, true))
                return error;
        }
    }
    if (!speed) {  // the chroma's target, which only SpEED reads
        const VkDeviceSize chromaPlane = (VkDeviceSize)chroma_stride_bytes(w) * (uint32_t)(h / 2);
        if (int error = create_buffer(convertChroma, 4 * chromaPlane, false))
            return error;
    }
    for (int side = 0; side < 2; ++side) {
        if (int error = add_convert_side(side))
            return error;
    }
    if (int error = compile_pipelines())
        return error;
    converting = true;
    return 0;
}

void vv_context::record_convert(VkCommandBuffer cb, size_t slotIndex, unsigned sides,
                                const std::function<void(int)> &stamp)
{
    size_t steps = 0;
    for (int side = 0; side < 2; ++side)
        if ((sides >> side) & 1)
            steps = std::max(steps, convertSides[side].passes.size());
    for (size_t i = 0; i < steps; ++i) {
        int shader = -1;
        for (int side = 0; side < 2; ++side) {
            const std::vector<Pass> &passes = convertSides[side].passes;
            if (!((sides >> side) & 1) || i >= passes.size())
                continue;
            const Pass &pass = passes[i];
            const VkPipelineLayout layout = pipelines[pass.shader].layout;
            const VkDescriptorSet set = pass.slotSets.empty() ? pass.set : pass.slotSets[slotIndex];
            uint32_t constants[4];
            memcpy(constants, pass.constants, sizeof constants);
            constants[1] += (uint32_t)slotIndex * convertSides[side].placesWords;  // the slot's places
            vk.vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pass.pipeline);
            vk.vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
            vk.vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof constants, constants);
            vk.vkCmdDispatch(cb, pass.groups[0], pass.groups[1], 1);
            shader = pass.shader;
        }
        barrier(vk, cb, kCompute, kCompute);
        if (stamp)
            stamp(shader);
    }
}

// A side's source picture into a slot's part of convertSource (or memory laid
// out as one) at `base`, rows packed.
void vv_context::write_convert_source(uint8_t *base, int side, const uint16_t *const planes[3],
                                      const ptrdiff_t strides[3])
{
    const ConvertSide &c = convertSides[side];
    for (int p = 0; p < 3; ++p) {
        const size_t rowBytes = (size_t)c.planeW[p] * 2;
        for (uint32_t y = 0; y < c.planeH[p]; ++y)
            memcpy(base + c.planeOffset[p] + y * rowBytes, (const uint8_t *)planes[p] + y * strides[p], rowBytes);
    }
}

// One picture through a side's conversion, read back (vv_test_convert).
int vv_context::test_convert(int side, const uint16_t *const planes[3], const ptrdiff_t strides[3],
                             uint16_t *const out[3], const ptrdiff_t outStrides[3])
{
    if (!converting || side < 0 || side > 1)
        return fail(-3, "no conversion on this side (vv_v1_convert)");
    if (pending)
        return fail(-3, "a frame was started and not committed");
    for (Slot &slot : slots) {  // the frames in flight read picRef and picDis, which the conversion writes
        if (int error = collect(slot))
            return error;
    }
    // In a shared context convertSource is the GPU's own: the picture goes
    // through host memory of its own, copied into slot 0's part.
    Buffer upload;
    if (!convertSource.mapped) {
        if (int error = create_buffer(upload, convertSourceBytes, true, false, true)) {
            destroy_last_buffer(upload);
            return error;
        }
    }
    write_convert_source(upload.mapped ? (uint8_t *)upload.mapped : (uint8_t *)convertSource.mapped, side, planes,
                         strides);
    const uint32_t chromaStride = chroma_stride_bytes(w), chromaPlane = chromaStride * (uint32_t)(h / 2);
    Buffer back;
    if (int error = create_buffer(back, (VkDeviceSize)planeBytes + 2ull * chromaPlane, true)) {
        destroy_last_buffer(back);
        if (upload.mapped)
            destroy_last_buffer(upload);
        return error;
    }
    VkCommandBuffer cb = slots[0].commands;
    VkCommandBufferBeginInfo begin = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vk.vkBeginCommandBuffer(cb, &begin);
    if (upload.mapped) {
        const VkBufferCopy whole = { 0, 0, convertSourceBytes };
        vk.vkCmdCopyBuffer(cb, upload.buffer, convertSource.buffer, 1, &whole);
        barrier(vk, cb, kTransfer, kCompute);
    }
    record_convert(cb, 0, 1u << side, nullptr);
    barrier(vk, cb, kCompute, kTransfer);
    const VkBufferCopy luma = { 0, 0, planeBytes };
    vk.vkCmdCopyBuffer(cb, (side ? picDis : picRef).buffer, back.buffer, 1, &luma);
    const VkBufferCopy chroma = { (VkDeviceSize)side * 2 * chromaPlane, planeBytes, 2ull * chromaPlane };
    vk.vkCmdCopyBuffer(cb, (speed ? slots[0].speedChroma : convertChroma).buffer, back.buffer, 1, &chroma);
    VkMemoryBarrier toHost = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    toHost.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toHost.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vk.vkCmdPipelineBarrier(cb, kTransfer, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &toHost, 0, nullptr, 0, nullptr);
    vk.vkEndCommandBuffer(cb);
    const VkResult result = run_now(cb);
    if (result == VK_SUCCESS) {
        const uint8_t *from = (const uint8_t *)back.mapped;
        for (int p = 0; p < 3; ++p) {
            const uint32_t rowBytes = (p ? (uint32_t)w / 2 : (uint32_t)w) * 2, rows = p ? (uint32_t)h / 2 : (uint32_t)h;
            const uint8_t *plane = p ? from + planeBytes + (size_t)(p - 1) * chromaPlane : from;
            const uint32_t stride = p ? chromaStride : strideBytes;
            for (uint32_t y = 0; y < rows; ++y)
                memcpy((uint8_t *)out[p] + y * outStrides[p], plane + (size_t)y * stride, rowBytes);
        }
    }
    destroy_last_buffer(back);
    if (upload.mapped)
        destroy_last_buffer(upload);
    return result == VK_SUCCESS ? 0 : fail(-5, "the conversion on the GPU failed (" + std::to_string(result) + ")");
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

// A committed frame's work, its fence signalled at the end. After the
// decoder's copies of it (`after`, 0: none): on the submit thread the value
// is reached already, but the wait stays in the submission as the copies'
// memory dependency between the two devices. Reached, it costs little:
// NVIDIA's polling thread of the device then ran 0.8-1.2 G cycles a 1080p
// minute, 9-10 G while the value was still to come.
VkResult vv_context::submit_frame(Slot &slot, uint64_t after)
{
    VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &slot.commands;
    VkTimelineSemaphoreSubmitInfo timeline = { VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO };
    const VkPipelineStageFlags afterStages = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    if (copied && after) {
        timeline.waitSemaphoreValueCount = 1;
        timeline.pWaitSemaphoreValues = &after;
        submitInfo.pNext = &timeline;
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &copied;
        submitInfo.pWaitDstStageMask = &afterStages;
    }
    std::lock_guard<std::mutex> lock(queueMutex);
    return vk.vkQueueSubmit(queue, 1, &submitInfo, slot.fence);
}

// The submit thread (vv_import_timeline): the committed frames in order, each
// once the decoder's copies of it are done. After a failure it submits no
// more; the caller's next call reports it.
void vv_context::submit_thread()
{
    std::unique_lock<std::mutex> lock(submitMutex);
    for (;;) {
        submitWork.wait(lock, [&] { return submitStop || (!submissions.empty() && submitError.empty()); });
        if (submitStop)
            return;
        const Submission next = submissions.front();
        lock.unlock();
        std::string error;
        if (next.after) {
            // Asleep until the decoder's copies are done, waking every 100 ms
            // to see a stop (a run that ends before the decoder copied a frame
            // it was given).
            VkSemaphoreWaitInfo wait = { VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO };
            wait.semaphoreCount = 1;
            wait.pSemaphores = &copied;
            wait.pValues = &next.after;
            VkResult result;
            while ((result = waitSemaphores(device, &wait, 100ull * 1000 * 1000)) == VK_TIMEOUT) {
                lock.lock();
                const bool stop = submitStop;
                lock.unlock();
                if (stop)
                    return;
            }
            if (result != VK_SUCCESS)
                error = "waiting for the decoder's copies failed (" + std::to_string(result) + ")";
        }
        if (error.empty()) {
            const VkResult result = submit_frame(slots[next.slot], next.after);
            if (result != VK_SUCCESS)
                error = "vkQueueSubmit failed (" + std::to_string(result) + ")";
        }
        lock.lock();
        if (error.empty()) {
            submissions.pop_front();
            slots[next.slot].queued = false;
        } else {
            submitError = error;  // the frame stays queued: never submitted
        }
        submitDone.notify_all();
    }
}

// Whether a committed frame is still waiting for the submit thread.
bool vv_context::queued(const Slot &slot)
{
    if (!submitter.joinable())
        return false;
    std::lock_guard<std::mutex> lock(submitMutex);
    return slot.queued;
}

// One command buffer run to its end (set-up, tests): submitted, then the
// device waited for.
VkResult vv_context::run_now(VkCommandBuffer commands)
{
    VkSubmitInfo submit = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &commands;
    std::lock_guard<std::mutex> lock(queueMutex);
    const VkResult result = vk.vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE);
    return result == VK_SUCCESS ? vk.vkDeviceWaitIdle(device) : result;
}

int vv_context::collect(Slot &slot)
{
    if (!slot.busy)
        return 0;
    const double waitStart = profile ? now_ns() : 0.0;
    if (submitter.joinable()) {  // its fence signals only once the submit thread has submitted it
        std::unique_lock<std::mutex> lock(submitMutex);
        const bool submitted = submitDone.wait_for(lock, std::chrono::seconds(60), [&] {
            return !slot.queued || !submitError.empty();
        });
        if (!submitError.empty()) {
            failed = true;
            return fail(-5, submitError);
        }
        if (!submitted) {
            failed = true;
            return fail(-5, "the decoder did not copy a frame in 60 seconds");
        }
    }
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
            profilePairNs += (double)(stamps[labels.size()] - stamps[0]) * timestampNs;
            ++profilePairs;
            profileFirst = profilePairs == 1 ? stamps[0] : std::min(profileFirst, stamps[0]);
            profileLast = std::max(profileLast, stamps[labels.size()]);
            for (size_t i = 0; i < labels.size(); ++i) {
                gpuNs[labels[i]] += (double)(stamps[i + 1] - stamps[i]) * timestampNs;
                ++gpuCount[labels[i]];
            }
        }
    }
    uint64_t sums[kSlots];
    const uint32_t *words = (const uint32_t *)slot.result.mapped;
    for (int i = 0; i < kSlots; ++i)
        sums[i] = (uint64_t)words[2 * i] | ((uint64_t)words[2 * i + 1] << 32);
    if (slot.index < rawSums.size())
        memcpy(rawSums[slot.index].data(), sums, sizeof sums);
    // The scales whose top k the GPU could not sum exactly (bits): spatial_pooling
    // on the CPU, on cambiPool's threads, from a copy of their c-values; the pool
    // works out the frame's score, which flush() puts into its record.
    unsigned cambiPooledScales = 0;
    if (cambi && slot.scored) {
        for (int scale = 0; scale < V1_CAMBI_SCALES; ++scale) {
            if (sums[kSlotCambi + scale * 4 + 3] & 2)
                cambiPooledScales |= 1u << scale;
        }
    }
    if (cambiPooledScales) {
        const double copyStart = profile ? now_ns() : 0.0;
        std::vector<float> values = cambiPool.buffer();
        const float *kept = (const float *)slot.cambiKeep.mapped;
        for (int scale = 0; scale < V1_CAMBI_SCALES; ++scale) {
            if (cambiPooledScales & (1u << scale)) {
                const float *c = kept + cambiKeepOffset[scale];
                values.insert(values.end(), c, c + (size_t)cambiConst.scale_w[scale] * cambiConst.scale_h[scale]);
            }
        }
        cambiPool.submit(slot.index, sums, cambiPooledScales, std::move(values));
        if (profile) {
            cpuNs[kCpuCambiCopy] += now_ns() - copyStart;
            ++cpuCount[kCpuCambiCopy];
        }
    }
    FrameRecord &frame = records[slot.index];
    frame.sad = sums[kSlotSad];
    frame.scored = slot.scored ? 1 : 0;
    if (slot.scored) {  // the options all of this reads are fixed before the first frame
        for (int limit = 0; limit < 2; ++limit) {
            double vif[kScales];
            vif_scores(sums, limit, vif);
            for (int scale = 0; scale < kScales; ++scale)
                frame.vif[limit][scale] = (float)vif[scale];  // exact: each is a float quotient
            frame.adm[limit] = adm_score(sums, limit, (unsigned)w, (unsigned)h);
        }
        if (!(v1 && (skip & 4)))  // (VMAF v1 without ADM: no ADM passes, v1AdmEnd 0)
            v1_adm(*this, sums, frame.v1Adm);
        if (cambi && !cambiPooledScales)
            frame.cambiError = (int8_t)cambi_score(sums, nullptr, 0, frame.cambi);
        if (ciede)  // ciede.c's: 45 - 20 log10 of the mean dE00 (+inf for identical pictures, as there)
            frame.ciede = 45.0 - 20.0 * log10((double)sums[kSlotCiede] * (1.0 / 1048576.0) / ((double)w * h));
    }
    if (speed && slot.scored)  // est_params on the pool's threads, from a copy of the planes
        speedPool.submit(slot.index, (const float *)slot.speedKeep.mapped);
    return 0;
}

// The next frame pair's two luma planes in the memory the GPU reads them
// from (or copies them from: not direct), for a caller that can write them there itself (a decoder): rows of
// strideBytes, the reference at *ref and the distorted at *dis. commit()
// then scores what was written.
int vv_context::staging(uint8_t **ref, uint8_t **dis)
{
    if (failed)
        return fail(-5, "the GPU failed on an earlier frame");
    Slot &slot = slots[nextSlot];
    if (int error = collect(slot))
        return error;
    if (v1 && direct) {
        // VMAF v1's frames after the slot's last read its reference as their
        // previous one (motion): they are done too before it is written again.
        // (VMAF v0.6.1's motion keeps its own blur.)
        const size_t lag = options.fiveFrameWindow ? 2 : 1;
        for (size_t k = 1; k <= lag; ++k) {
            if (int error = collect(slots[(nextSlot + k) % slots.size()]))
                return error;
        }
    }
    if (pictures)  // given by vv_pictures before the frame's commit (which refuses it without)
        slot.pictureRef = slot.pictureDis = slot.picturePrev = -1;
    pending = &slot;
    *ref = (uint8_t *)slot.staging.mapped;
    *dis = direct && !shared && !v1 ? (uint8_t *)slot.stagingDis.mapped
           : slot.staging.mapped ? (uint8_t *)slot.staging.mapped + disOffset : nullptr;
    return 0;
}

// Rows [rows * part / copyThreads, rows * (part + 1) / copyThreads) of each
// plane of copyJob.
void vv_context::copy_part(unsigned part)
{
    for (int index = 0; index < copyJob.count; ++index) {
        const CopyPlane &plane = copyJob.planes[index];
        const uint32_t y0 = (uint32_t)((uint64_t)plane.rows * part / copyThreads);
        const uint32_t y1 = (uint32_t)((uint64_t)plane.rows * (part + 1) / copyThreads);
        if (y1 <= y0)
            continue;
        uint8_t *to = plane.to + (size_t)y0 * plane.toStride;
        const uint8_t *from = plane.from + (ptrdiff_t)y0 * plane.fromStride;
        if (plane.fromStride == (ptrdiff_t)plane.toStride) {
            // The last row without its padding, which the caller's plane need not have.
            memcpy(to, from, (size_t)(y1 - y0 - 1) * plane.toStride + (y1 == plane.rows ? plane.rowBytes
                                                                                           : plane.toStride));
        } else {
            for (uint32_t y = y0; y < y1; ++y, to += plane.toStride, from += plane.fromStride)
                memcpy(to, from, plane.rowBytes);
        }
    }
}

// copyJob's planes copied: split by rows among copyThreads threads (the
// caller's and the copiers, started the first time).
void vv_context::copy_planes()
{
    if (copiers.empty() && copyThreads > 1) {
        try {
            for (unsigned part = 1; part < copyThreads; ++part)
                copiers.emplace_back(&vv_context::copier, this, part);
        } catch (const std::system_error &) {
            copyThreads = (unsigned)copiers.size() + 1;  // the parts the threads made take
        }
    }
    if (!copiers.empty()) {
        {
            std::lock_guard<std::mutex> lock(copyMutex);
            ++copyGeneration;
            copyPending = (unsigned)copiers.size();
        }
        copyStart.notify_all();
    }
    copy_part(0);
    if (!copiers.empty()) {
        std::unique_lock<std::mutex> lock(copyMutex);
        copyDone.wait(lock, [&] { return copyPending == 0; });
    }
}

void vv_context::copier(unsigned part)
{
    unsigned seen = 0;
    std::unique_lock<std::mutex> lock(copyMutex);
    for (;;) {
        copyStart.wait(lock, [&] { return copyQuit || copyGeneration != seen; });
        if (copyQuit)
            return;
        seen = copyGeneration;
        lock.unlock();
        copy_part(part);
        lock.lock();
        if (--copyPending == 0)
            copyDone.notify_one();
    }
}

int vv_context::submit(const uint8_t *ref, ptrdiff_t refStride, const uint8_t *dis, ptrdiff_t disStride, bool score)
{
    uint8_t *targets[2];
    if (shared)
        return fail(-3, "this context takes its frames from GPU memory");
    if (converting)
        return fail(-3, "this context converts its pictures: they go in with vv_submit_v1");
    if (int error = staging(&targets[0], &targets[1]))
        return error;
    const double copyBegun = profile ? now_ns() : 0.0;
    const size_t rowBytes = (size_t)w * (bpc > 8 ? 2 : 1);
    copyJob.count = 0;
    copyJob.planes[copyJob.count++] = { targets[0], strideBytes, ref, refStride, rowBytes, (uint32_t)h };
    if (score)
        copyJob.planes[copyJob.count++] = { targets[1], strideBytes, dis, disStride, rowBytes, (uint32_t)h };
    copy_planes();
    if (profile) {
        cpuNs[kCpuCopy] += now_ns() - copyBegun;
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
    if ((speed || ciede) && score) {
        for (int plane : { 1, 2, 4, 5 }) {
            if (!planes[plane])
                return fail(-3, speed ? "SpEED needs the chroma planes" : "CIEDE2000 needs the chroma planes");
        }
    }
    if (converting) {
        for (int plane : { 1, 2, 4, 5 }) {
            if (!planes[plane] && (plane < 3 || score))
                return fail(-3, "the conversion needs the chroma planes");
        }
    }
    uint8_t *targets[2];
    if (int error = staging(&targets[0], &targets[1]))
        return error;
    const double copyStart = profile ? now_ns() : 0.0;
    // The planes copied as VMAF v0.6.1's are, on copyThreads threads (one
    // thread held VMAF v1 from system memory to its copying speed: RTX 5090,
    // 4K 10-bit, about 420 frame pairs a second).
    const size_t sampleBytes = bpc > 8 ? 2 : 1;
    copyJob.count = 0;
    if (converting) {
        // The source pictures into the slot's part of convertSource, rows
        // packed: commit converts them into picRef, picDis and the chroma.
        const size_t slotIndex = (size_t)(pending - slots.data());
        uint8_t *base = (uint8_t *)convertSource.mapped + slotIndex * convertSourceBytes;
        for (int side = 0; side < (score ? 2 : 1); ++side) {
            const ConvertSide &c = convertSides[side];
            for (int p = 0; p < 3; ++p)
                copyJob.planes[copyJob.count++] = { base + c.planeOffset[p], (uint32_t)c.planeW[p] * 2,
                                                    planes[3 * side + p], strides[3 * side + p],
                                                    (size_t)c.planeW[p] * 2, c.planeH[p] };
        }
    } else {
        copyJob.planes[copyJob.count++] = { targets[0], strideBytes, planes[0], strides[0], (size_t)w * sampleBytes,
                                            (uint32_t)h };
        if (score)
            copyJob.planes[copyJob.count++] = { targets[1], strideBytes, planes[3], strides[3],
                                                (size_t)w * sampleBytes, (uint32_t)h };
    }
    if (speed && score && !converting) {
        uint8_t *chroma = (uint8_t *)pending->speedChroma.mapped;
        const int order[4] = { 1, 2, 4, 5 };  // ref U, ref V, dis U, dis V
        for (int i = 0; i < 4; ++i)
            copyJob.planes[copyJob.count++] = { chroma + (size_t)i * chromaPlaneBytes, chromaStrideBytes,
                                                planes[order[i]], strides[order[i]], chromaW * sampleBytes,
                                                chromaH };
    }
    if (ciede && score) {
        uint8_t *chroma = (uint8_t *)pending->ciedeChroma.mapped;
        const int order[4] = { 1, 2, 4, 5 };  // ref U, ref V, dis U, dis V
        for (int i = 0; i < 4; ++i)
            copyJob.planes[copyJob.count++] = { chroma + (size_t)i * ciedePlaneBytes, ciedeStrideBytes,
                                                planes[order[i]], strides[order[i]], ciedeW * sampleBytes, ciedeH };
    }
    copy_planes();
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
    if (pictures && pending->pictureRef < 0)  // (its staging buffer has no planes: a few bytes)
        return fail(-3, "vv_pictures: the frame's pictures were not given");
    Slot &slot = *pending;
    pending = nullptr;
    nextSlot = (nextSlot + 1) % (unsigned)slots.size();
    if (pictures) {  // the frame's reference: the next frame's previous one (its motion), now that it is committed
        lastPicture = slot.pictureRef;
        lastPlace[0] = slot.places[2];
        lastPlace[1] = slot.places[3];
    }
    const double recordStart = profile ? now_ns() : 0.0;
    const size_t slotIndex = (size_t)(&slot - slots.data());
    const uint32_t queryBase = (uint32_t)(slotIndex * kQueriesPerSlot);

    const unsigned index = frames++;
    records.add();
    if (index < kRawFrames)
        rawSums.emplace_back();
    slot.index = index;
    slot.scored = score;

    VkCommandBuffer cb = slot.commands;
    vk.vkResetCommandBuffer(cb, 0);
    VkCommandBufferBeginInfo begin = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vk.vkBeginCommandBuffer(cb, &begin);
    // Profiling (vv_profile, VV_GPU_TIME): a timestamp at the start, and one
    // after each step, once the GPU is done with everything before it.
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
    if (pendingTextures[0] >= 0) {
        // From the decoder's Direct3D textures: taken over from the
        // "external" queue family for the copy and handed back after it,
        // in the general layout throughout (the memory is Direct3D's).
        const int textures[2] = { pendingTextures[0], score ? pendingTextures[1] : -1 };
        Buffer *targets[2] = { direct ? &slot.staging : &picRef,
                               direct ? (shared ? &dis_buffer(slot) : &slot.stagingDis) : &picDis };
        const VkDeviceSize offsets[2] = { 0, direct && shared ? (VkDeviceSize)disOffset : 0 };
        pendingTextures[0] = pendingTextures[1] = -1;
        for (int i = 0; i < 2; ++i) {
            if (textures[i] < 0)
                continue;
            VkImageMemoryBarrier take = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
            take.srcAccessMask = 0;
            take.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            take.oldLayout = take.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            take.srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
            take.dstQueueFamilyIndex = queueFamily;
            take.image = imported[(size_t)textures[i]].image;
            take.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            vk.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                                    nullptr, 0, nullptr, 1, &take);
            VkBufferImageCopy region = {};
            region.bufferOffset = offsets[i];
            region.bufferRowLength = strideBytes / (bpc > 8 ? 2 : 1);
            region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            region.imageExtent = { (uint32_t)w, (uint32_t)h, 1 };
            vk.vkCmdCopyImageToBuffer(cb, take.image, VK_IMAGE_LAYOUT_GENERAL, targets[i]->buffer, 1, &region);
            VkImageMemoryBarrier give = take;
            give.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            give.dstAccessMask = 0;
            give.srcQueueFamilyIndex = queueFamily;
            give.dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
            vk.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0,
                                    nullptr, 0, nullptr, 1, &give);
        }
    } else if (!direct && !converting && pictures_copied()) {  // (converting: the conversion writes them, after the barrier below)
        VkBufferCopy copy = { 0, 0, planeBytes };
        vk.vkCmdCopyBuffer(cb, slot.staging.buffer, picRef.buffer, 1, &copy);
        if (score) {
            copy.srcOffset = disOffset;
            vk.vkCmdCopyBuffer(cb, dis_buffer(slot).buffer, picDis.buffer, 1, &copy);
        }
    } else if (shared && converting) {
        // The source pictures the decoder wrote into the shared buffers
        // (vv_convert_source), into the slot's part of convertSource.
        for (int side = 0; side < (score ? 2 : 1); ++side) {
            const VkBufferCopy copy = { shared_source_offset(side),
                                        slotIndex * convertSourceBytes + convertSides[side].planeOffset[0],
                                        convert_side_bytes(side) };
            vk.vkCmdCopyBuffer(cb, (side ? dis_buffer(slot) : slot.staging).buffer, convertSource.buffer, 1, &copy);
        }
    } else if (shared) {
        // A plane the CPU wrote (vv_shared_host_plane) goes where the decoder
        // writes the other: the reference's at 0, the distorted's at disOffset.
        for (int i = 0; i < 2; ++i) {
            if (slot.planeFromHost[i] && (i == 0 || score)) {
                const VkBufferCopy copy = { 0, i ? (VkDeviceSize)disOffset : 0, planeBytes };
                vk.vkCmdCopyBuffer(cb, slot.hostPlane[i].buffer, (i ? dis_buffer(slot) : slot.staging).buffer, 1,
                                   &copy);
            }
        }
    }
    slot.planeFromHost[0] = slot.planeFromHost[1] = false;  // the next frame's: from the decoder unless it says
    vk.vkCmdFillBuffer(cb, acc.buffer, 0, VK_WHOLE_SIZE, 0);
    if (admFused && score) {
        for (const Buffer &rows : admRows)
            vk.vkCmdFillBuffer(cb, rows.buffer, 0, VK_WHOLE_SIZE, 0);
    }
    barrier(vk, cb, kTransfer, kCompute | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT);
    if (pictures)
        picture_barriers(cb, slot, true);
    stamp(kProfileUpload);
    if (converting)  // the frame's pictures converted on the GPU, ending with a barrier
        record_convert(cb, slotIndex, score ? 3u : 1u, stamp);
    const bool moves = !v1 || index >= (options.fiveFrameWindow ? 2u : 1u);
    const std::vector<Pass> *lists[5] = { moves ? &motion[index % 2] : nullptr, score ? &scored : nullptr,
                                          score && cambi ? &slot.cambiKeepPasses : nullptr,
                                          // (shared and converting: from speedChroma, which the conversion
                                          // writes; without shared, speedPasses read it)
                                          score && speed ? (slot.chromaFromHost || (shared && converting)
                                                                ? &slot.speedPassesHost
                                                                : &slot.speedPasses)
                                                         : nullptr,
                                          score && ciede ? &slot.ciedePasses : nullptr };
    if (shared)
        slot.chromaFromHost = false;  // the next frame's: from staging unless vv_chroma_staging says
    auto record = [&](const Pass &pass) {
        if (!pass.groups[0] || !pass.groups[1])
            return;
        const Pipeline &pipeline = pipelines[pass.shader];
        vk.vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pass.pipeline);
        const VkDescriptorSet set = pass.slotSets.empty() ? pass.set : pass.slotSets[slotIndex];
        if (pass.pictureRole) {  // the frame's pictures (vv_pictures), and their places
            picture_descriptors(set, pass, slot);
            uint32_t pushed[kPushBytes / 4];
            memcpy(pushed, pass.constants, kPushBytes);
            memcpy(pushed + kPushBytes / 4 - 6, slot.places, sizeof slot.places);
            vk.vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout, 0, 1, &set, 0, nullptr);
            vk.vkCmdPushConstants(cb, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, kPushBytes, pushed);
        } else {
            vk.vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout, 0, 1, &set, 0, nullptr);
            vk.vkCmdPushConstants(cb, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pass.constantBytes,
                                  pass.constants);
        }
        if (pass.indirect)
            vk.vkCmdDispatchIndirect(cb, pass.indirect, pass.indirectOffset);
        else
            vk.vkCmdDispatch(cb, pass.groups[0], pass.groups[1], pass.groups[2]);
    };
    const bool inTurns = !profile && !passLimit;
    if (v1 && inTurns) {
        // VMAF v1's features, independent of each other -- motion, ADM, CAMBI
        // (and its kept c-values), SpEED, CIEDE2000 -- each a chain of passes that wait
        // for the one before, recorded in turns: a pass (or passes that
        // overlap) of each chain, then one barrier, so that each waits for
        // its own chain's and the others' passes run beside it (the small
        // ones left the GPU idle between barriers, at 1080p most).
        std::vector<std::vector<const Pass *>> chains(5);
        if (moves)
            for (const Pass &pass : motion[index % 2]) chains[0].push_back(&pass);
        if (score) {
            for (size_t i = 0; i < v1AdmEnd && i < scored.size(); ++i) chains[1].push_back(&scored[i]);
            for (size_t i = v1AdmEnd; i < scored.size(); ++i) chains[2].push_back(&scored[i]);
            if (cambi)
                for (const Pass &pass : slot.cambiKeepPasses) chains[2].push_back(&pass);
        }
        for (int c = 3; c < 5; ++c)
            if (lists[c])
                for (const Pass &pass : *lists[c]) chains[(size_t)c].push_back(&pass);
        size_t at[5] = {};
        // The stage each chain starts at: CAMBI's first (CLEAR's), the rest after
        // it (the best of the orders tried on a Radeon 780M, by a few percent);
        // CIEDE2000's single pass with CAMBI's first.
        const int offsets[5] = { 1, 1, 0, 2, 0 };
        int stage = 0;
        for (bool waiting = true; waiting; ++stage) {
            waiting = false;
            bool recorded = false;
            for (size_t c = 0; c < chains.size(); ++c) {
                if (at[c] < chains[c].size()) waiting = true;
                if (stage < offsets[c]) continue;
                while (at[c] < chains[c].size()) {  // the chain's next pass, with those it overlaps
                    const Pass &pass = *chains[c][at[c]++];
                    record(pass);
                    recorded = true;
                    if (!pass.overlapNext)
                        break;
                }
            }
            if (recorded)
                barrier(vk, cb, kCompute, kCompute | kTransfer | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT);
        }
    } else if (inTurns) {
        // VMAF v0.6.1 and NEG level by level, a barrier after each: the
        // passes that depend on no other of the level (motion, VIF's and
        // ADM's chains) overlap, and the GPU is not drained between every
        // two. The motion and scored lists share no buffer (Pass::level is
        // within a list): their levels run together.
        uint32_t levels = 0;
        for (const std::vector<Pass> *list : lists)
            for (size_t i = 0; list && i < list->size(); ++i)
                levels = std::max(levels, (*list)[i].level + 1);
        for (uint32_t level = 0; level < levels; ++level) {
            for (const std::vector<Pass> *list : lists)
                for (size_t i = 0; list && i < list->size(); ++i)
                    if ((*list)[i].level == level)
                        record((*list)[i]);
            barrier(vk, cb, kCompute, kCompute | kTransfer);
        }
    } else {
        // Profiling and pass limits (the diagnosis): the passes in their
        // order, each finished before the next (its timestamp, its buffers).
        for (const std::vector<Pass> *list : lists) {
            if (!list)
                continue;
            int done = 0;
            for (const Pass &pass : *list) {
                if (list == &scored && passLimit && done++ >= passLimit)
                    break;
                if (!pass.groups[0] || !pass.groups[1])
                    continue;
                record(pass);
                barrier(vk, cb, kCompute, kCompute | kTransfer | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT);
                stamp(pass.shader);
            }
        }
    }
    if (v1 && !direct && !(skip & 1)) {  // read from its slot otherwise (staging()); only motion reads it
        barrier(vk, cb, kCompute, kTransfer);
        VkBufferCopy keep = { 0, 0, planeBytes };
        vk.vkCmdCopyBuffer(cb, picRef.buffer, picPrev[index % 2].buffer, 1, &keep);
    }
    if (pictures)
        picture_barriers(cb, slot, false);
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
    const uint64_t after = copiedValue;  // the decoder's copies of the frame into its slot (vv_commit_after)
    copiedValue = 0;
    if (submitter.joinable()) {  // submitted by the submit thread once they are done, in order
        {
            std::lock_guard<std::mutex> lock(submitMutex);
            if (!submitError.empty()) {
                failed = true;
                return fail(-5, submitError);
            }
            slot.queued = true;
            submissions.push_back({ slotIndex, after });
        }
        submitWork.notify_one();
    } else {
        const VkResult result = submit_frame(slot, after);
        if (result != VK_SUCCESS) {
            failed = true;
            return fail(-5, "vkQueueSubmit failed (" + std::to_string(result) + ")");
        }
    }
    slot.busy = true;
    if (profile) {
        cpuNs[kCpuRecord] += now_ns() - recordStart;
        ++cpuCount[kCpuRecord];
    }
    return 0;
}

// A Direct3D 11 texture shared by its NT handle (D3D11_RESOURCE_MISC_SHARED_
// NTHANDLE), the context's size and R16_UINT (more than 8 bits) or R8_UINT:
// one frame's luma, as a decoder writes it. The handle stays the caller's.
int vv_context::import_texture(void *handle, int *index)
{
    if (!getHandleProperties)
        return fail(-4, deviceName + " cannot import Direct3D textures");
    const VkExternalMemoryHandleTypeFlagBits type = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT;
    VkExternalMemoryImageCreateInfo external = { VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO };
    external.handleTypes = type;
    VkImageCreateInfo info = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    info.pNext = &external;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = bpc > 8 ? VK_FORMAT_R16_UINT : VK_FORMAT_R8_UINT;
    info.extent = { (uint32_t)w, (uint32_t)h, 1 };
    info.mipLevels = info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imported.emplace_back();
    Imported &texture = imported.back();
    if (vk.vkCreateImage(device, &info, nullptr, &texture.image) != VK_SUCCESS)
        return fail(-1, "vkCreateImage failed for a Direct3D texture");
    VkMemoryWin32HandlePropertiesKHR properties = { VK_STRUCTURE_TYPE_MEMORY_WIN32_HANDLE_PROPERTIES_KHR };
    VkResult result = getHandleProperties(device, type, (HANDLE)handle, &properties);
    if (result != VK_SUCCESS)
        return fail(-1, "the Direct3D texture's handle was refused (" + std::to_string(result) + ")");
    VkMemoryRequirements requirements;
    vk.vkGetImageMemoryRequirements(device, texture.image, &requirements);
    const uint32_t bits = requirements.memoryTypeBits & (properties.memoryTypeBits ? properties.memoryTypeBits : ~0u);
    if (!bits)
        return fail(-1, "no memory type takes the Direct3D texture");
    uint32_t memoryType = 0;
    while (!(bits & (1u << memoryType)))
        ++memoryType;
    VkMemoryDedicatedAllocateInfo dedicated = { VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO };
    dedicated.image = texture.image;
    VkImportMemoryWin32HandleInfoKHR import = { VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR };
    import.pNext = &dedicated;
    import.handleType = type;
    import.handle = (HANDLE)handle;
    VkMemoryAllocateInfo allocate = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    allocate.pNext = &import;
    allocate.allocationSize = requirements.size;
    allocate.memoryTypeIndex = memoryType;
    result = vk.vkAllocateMemory(device, &allocate, nullptr, &texture.memory);
    if (result != VK_SUCCESS)
        return fail(-1, "importing the Direct3D texture failed (" + std::to_string(result) + ")");
    if (vk.vkBindImageMemory(device, texture.image, texture.memory, 0) != VK_SUCCESS)
        return fail(-1, "vkBindImageMemory failed for a Direct3D texture");
    *index = (int)imported.size() - 1;
    return 0;
}

// The next frame pair from two imported textures (`dis` unused when the
// frame is not scored). The caller must not let the decoder write either
// texture again before completed() counts this frame.
int vv_context::commit_textures(int ref, int dis, bool score)
{
    if (pictures)  // (its staging buffers hold no planes: a few bytes, which the copy would overrun)
        return fail(-3, "this context reads the pictures where a decoder left them (vv_pictures)");
    const int count = (int)imported.size();
    if (ref < 0 || ref >= count || (score && (dis < 0 || dis >= count)))
        return fail(-3, "no such imported texture");
    uint8_t *unused[2];
    if (int error = staging(&unused[0], &unused[1]))
        return error;
    pendingTextures[0] = ref;
    pendingTextures[1] = score ? dis : -1;
    return commit(score);
}

// How many frames the GPU has finished, without waiting: the oldest frames'
// slots whose fences have signalled are collected.
unsigned vv_context::completed()
{
    for (size_t i = 0; i < slots.size(); ++i) {
        Slot &slot = slots[(nextSlot + i) % slots.size()];
        if (!slot.busy)
            continue;
        if (queued(slot) || vk.vkGetFenceStatus(device, slot.fence) != VK_SUCCESS || collect(slot))
            break;
    }
    unsigned busy = 0;
    for (const Slot &slot : slots)
        busy += slot.busy ? 1 : 0;
    return frames - busy;
}


// A decoder's texture (its KMT handle, `width` x `height`, NV12 or P010 as
// the context's bit depth says), imported the first time it comes: its index
// in pictureCache, or a negative error.
int vv_context::import_picture(HANDLE handle, uint32_t width, uint32_t height)
{
    for (size_t i = 0; i < pictureCache.size(); ++i)
        if (pictureCache[i].handle == handle && pictureCache[i].width == width && pictureCache[i].height == height)
            return (int)i;
    // The decoders' pools, which they reuse: a bound, not a cache that
    // evicts (a picture in flight or the next frame's previous must stay).
    if (pictureCache.size() >= kPictureCacheSize)
        return fail(-3, "vv_pictures: more than " + std::to_string(kPictureCacheSize)
                        + " different pictures (larger decoder pools than this context keeps)");
    Picture picture;
    picture.handle = handle;
    picture.width = width;
    picture.height = height;
    const bool wide = bpc > 8;
    VkExternalMemoryImageCreateInfo external = { VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO };
    external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_KMT_BIT;
    VkImageCreateInfo info = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    info.pNext = &external;
    info.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;  // its planes viewed as UINT
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = wide ? VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16 : VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
    info.extent = { width, height, 1 };
    info.mipLevels = info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    auto discard = [&](const char *text) {
        for (VkImageView view : picture.views)
            if (view) vk.vkDestroyImageView(device, view, nullptr);
        if (picture.image) vk.vkDestroyImage(device, picture.image, nullptr);
        if (picture.memory) vk.vkFreeMemory(device, picture.memory, nullptr);
        return fail(-1, text);
    };
    if (vk.vkCreateImage(device, &info, nullptr, &picture.image) != VK_SUCCESS)
        return discard("vv_pictures: Vulkan cannot take the decoder's pictures");
    VkMemoryWin32HandlePropertiesKHR properties = { VK_STRUCTURE_TYPE_MEMORY_WIN32_HANDLE_PROPERTIES_KHR };
    memoryHandleProperties(device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_KMT_BIT, handle, &properties);
    VkMemoryRequirements requirements;
    vk.vkGetImageMemoryRequirements(device, picture.image, &requirements);
    int type = -1;
    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount && type < 0; ++i)
        if ((requirements.memoryTypeBits & properties.memoryTypeBits) & (1u << i))
            type = (int)i;
    VkMemoryDedicatedAllocateInfo exclusive = { VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO };
    exclusive.image = picture.image;
    VkImportMemoryWin32HandleInfoKHR import = { VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR };
    import.pNext = &exclusive;
    import.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_KMT_BIT;
    import.handle = handle;
    VkMemoryAllocateInfo allocate = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    allocate.pNext = &import;
    allocate.allocationSize = requirements.size;
    allocate.memoryTypeIndex = (uint32_t)type;
    if (type < 0 || vk.vkAllocateMemory(device, &allocate, nullptr, &picture.memory) != VK_SUCCESS
        || vk.vkBindImageMemory(device, picture.image, picture.memory, 0) != VK_SUCCESS)
        return discard("vv_pictures: Vulkan cannot take the decoder's pictures");
    for (int plane = 0; plane < 2; ++plane) {
        VkImageViewCreateInfo view = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        view.image = picture.image;
        view.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view.format = plane == 0 ? (wide ? VK_FORMAT_R16_UINT : VK_FORMAT_R8_UINT)
                                 : (wide ? VK_FORMAT_R16G16_UINT : VK_FORMAT_R8G8_UINT);
        view.subresourceRange = { (VkImageAspectFlags)(plane == 0 ? VK_IMAGE_ASPECT_PLANE_0_BIT
                                                                  : VK_IMAGE_ASPECT_PLANE_1_BIT), 0, 1, 0, 1 };
        if (vk.vkCreateImageView(device, &view, nullptr, &picture.views[plane]) != VK_SUCCESS)
            return discard("vv_pictures: Vulkan cannot view the decoder's pictures' planes");
    }
    pictureCache.push_back(picture);
    return (int)pictureCache.size() - 1;
}

// The frame begun's pictures (vv_pictures).
int vv_context::set_pictures(HANDLE reference, uint32_t referenceW, uint32_t referenceH, int referenceX, int referenceY,
                             HANDLE distorted, uint32_t distortedW, uint32_t distortedH, int distortedX, int distortedY)
{
    if (!pictures)
        return fail(-3, "this context does not take a decoder's textures");
    if (!pending)
        return fail(-3, "no frame was started");
    auto inside = [&](uint32_t textureW, uint32_t textureH, int x, int y) {
        return x >= 0 && y >= 0 && (uint64_t)x + w <= textureW && (uint64_t)y + h <= textureH;
    };
    if (!inside(referenceW, referenceH, referenceX, referenceY) || !inside(distortedW, distortedH, distortedX, distortedY))
        return fail(-3, "vv_pictures: a picture is not inside its texture");
    const int ref = import_picture(reference, referenceW, referenceH);
    if (ref < 0)
        return ref;
    const int dis = import_picture(distorted, distortedW, distortedH);
    if (dis < 0)
        return dis;
    // The previous picture is the last committed frame's reference (commit
    // records it): given twice for one frame, or for a frame never
    // committed, this leaves the motion's previous as it was.
    Slot &slot = *pending;
    slot.pictureRef = ref;
    slot.pictureDis = dis;
    slot.picturePrev = lastPicture >= 0 ? lastPicture : ref;  // (the first frame's motion is not run)
    const uint32_t places[6] = { lastPicture >= 0 ? lastPlace[0] : (uint32_t)referenceX,
                                 lastPicture >= 0 ? lastPlace[1] : (uint32_t)referenceY,
                                 (uint32_t)referenceX, (uint32_t)referenceY, (uint32_t)distortedX, (uint32_t)distortedY };
    memcpy(slot.places, places, sizeof places);
    return 0;
}

// A pass's pictures bound in `set` (its slot's, not in use: the slot's frame
// before is done).
void vv_context::picture_descriptors(VkDescriptorSet set, const Pass &pass, const Slot &slot)
{
    VkDescriptorImageInfo infos[2];
    VkWriteDescriptorSet writes[2];
    uint32_t count = 0;
    auto add = [&](uint32_t binding, int picture, int plane) {
        infos[count] = { VK_NULL_HANDLE, pictureCache[(size_t)picture].views[plane], VK_IMAGE_LAYOUT_GENERAL };
        writes[count] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[count].dstSet = set;
        writes[count].dstBinding = binding;
        writes[count].descriptorCount = 1;
        writes[count].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        writes[count].pImageInfo = &infos[count];
        ++count;
    };
    switch (pass.pictureRole) {
    case kPictureMotion: add(0, slot.pictureRef, 0); add(1, slot.picturePrev, 0); break;
    case kPictureAdm: add(0, slot.pictureRef, 0); add(1, slot.pictureDis, 0); break;
    case kPictureCambi: add(0, slot.pictureDis, 0); break;
    case kPictureSpeed: add(0, slot.pictureRef, 1); add(4, slot.pictureDis, 1); break;
    default: break;
    }
    vk.vkUpdateDescriptorSets(device, count, writes, 0, nullptr);
}

// The frame's pictures taken from Direct3D 11 (the external queue family)
// before its passes, or given back after them.
void vv_context::picture_barriers(VkCommandBuffer cb, const Slot &slot, bool acquire)
{
    VkImageMemoryBarrier barriers[3];
    uint32_t count = 0;
    for (int picture : { slot.pictureRef, slot.pictureDis, slot.picturePrev }) {
        bool seen = picture < 0;
        for (uint32_t i = 0; i < count && !seen; ++i)
            seen = barriers[i].image == pictureCache[(size_t)picture].image;
        if (seen)
            continue;
        VkImageMemoryBarrier &b = barriers[count++];
        b = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        b.srcAccessMask = acquire ? 0 : VK_ACCESS_SHADER_READ_BIT;
        b.dstAccessMask = acquire ? VK_ACCESS_SHADER_READ_BIT : 0;
        b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = acquire ? VK_QUEUE_FAMILY_EXTERNAL : queueFamily;
        b.dstQueueFamilyIndex = acquire ? queueFamily : VK_QUEUE_FAMILY_EXTERNAL;
        b.image = pictureCache[(size_t)picture].image;
        b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    }
    if (count)
        vk.vkCmdPipelineBarrier(cb, acquire ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : kCompute,
                                acquire ? kCompute : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr,
                                count, barriers);
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
    if (cambi) {
        cambiPool.wait();
        cambiPool.merge(records);
    }
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
// GPU memory a decoder shares (vv_shared_next, vv_export); with it, bit 2:
// VMAF v1's pictures read where the decoder left them (vv_pictures, where
// the context can: vv_pictures_mode), bit 3: each slot's planes in two
// buffers, one for each decoder (vv_export_part), and bit 1: VMAF v1's
// pictures to be converted (vv_v1_convert, vv_convert_source), copied out
// of the slots rather than read there. Bits 28-30: the scale-0 decouple
// shader's VARIANT, for the diagnosis.
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

// The commit of libvmaf-fast the library was built from (as libvmaf's
// vmaf_version() gives its own), "unknown" when built outside the script.
#ifndef VV_COMMIT
#define VV_COMMIT "unknown"
#endif
VV_EXPORT const char *vv_version() { return VV_COMMIT; }

// The next frame pair's luma planes (16-bit little-endian samples above 8
// bits). `score` 0: only motion is calculated for the frame (libvmaf's
// n_subsample), and `distorted` may be null.
VV_EXPORT int vv_submit(vv_context *context, const uint8_t *reference, ptrdiff_t referenceStride,
                        const uint8_t *distorted, ptrdiff_t distortedStride, int score)
{
    return context->submit(reference, referenceStride, distorted, distortedStride, score != 0);
}

// The memory to write the next pair's luma planes to (rows *stride bytes
// apart), and vv_commit to score them: vv_submit without its copy. Only to
// write: on a discrete GPU it can be the GPU's own memory, across PCIe,
// which the CPU reads very slowly.
VV_EXPORT int vv_staging(vv_context *context, uint8_t **reference, uint8_t **distorted, uint32_t *stride)
{
    *stride = context->strideBytes;
    if (context->shared)
        return fail(-3, "this context takes its frames from GPU memory");
    if (context->converting)
        return fail(-3, "this context converts its pictures: they go in with vv_submit_v1");
    return context->staging(reference, distorted);
}

VV_EXPORT int vv_commit(vv_context *context, int score) { return context->commit(score != 0); }

// For a context made with flag bit 19: a decoder's timeline semaphore (an
// opaque Win32 handle of a VkSemaphore of the same GPU and driver, exported
// by Vulkan), which its copies into the slots signal. The caller closes the
// handle. 0, or negative (-4: the device takes no timeline semaphores; the
// decoder then waits for its copies before vv_commit, as without).
VV_EXPORT int vv_import_timeline(vv_context *context, void *handle)
{
    if (!context->shared)
        return fail(-3, "this context takes its frames from host memory");
    if (!context->timelines)
        return fail(-4, "this GPU's Vulkan takes no timeline semaphores from other APIs");
    if (context->copied)
        return fail(-3, "a timeline semaphore is already imported");
    VkSemaphoreTypeCreateInfo type = { VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO };
    type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo info = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    info.pNext = &type;
    VkSemaphore semaphore = VK_NULL_HANDLE;
    if (context->createSemaphore(context->device, &info, nullptr, &semaphore) != VK_SUCCESS)
        return fail(-1, "vkCreateSemaphore failed");
    VkImportSemaphoreWin32HandleInfoKHR import = { VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR };
    import.semaphore = semaphore;
    import.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    import.handle = handle;
    if (context->importSemaphore(context->device, &import) != VK_SUCCESS) {
        context->destroySemaphore(context->device, semaphore, nullptr);
        return fail(-4, "Vulkan cannot take the decoder's timeline semaphore");
    }
    context->copied = semaphore;
    try {
        context->submitter = std::thread(&vv_context::submit_thread, context);
    } catch (const std::system_error &) {
        context->copied = VK_NULL_HANDLE;
        context->destroySemaphore(context->device, semaphore, nullptr);
        return fail(-1, "the submit thread could not start");
    }
    return 0;
}

// The next vv_commit's frame is in its slot once the imported timeline
// semaphore reaches `value` (the decoder's last copy of it): the context's
// submit thread submits its work then, and the caller need not wait.
VV_EXPORT int vv_commit_after(vv_context *context, uint64_t value)
{
    if (!context->copied)
        return fail(-3, "no timeline semaphore is imported");
    context->copiedValue = value;
    return 0;
}

// For a context made with flag bit 19 (frames from GPU memory): the staging
// buffer the next pair's luma planes are to be written into by the API that
// imported it -- *slot says which (they take turns), the reference's rows
// start at offset 0 and the distorted's at *disOffset, *stride bytes apart
// (in split mode, flag bit 3: at *disOffset, 0, of its own buffer,
// vv_export_part's part 1). Once they are written (and that API has finished
// writing), vv_commit.
// Returns how many slots there are. A context reading a decoder's pictures
// (vv_pictures) begins each frame here too, but its buffers hold no planes
// (they are 4096 bytes): nothing may be written into them.
VV_EXPORT int vv_shared_next(vv_context *context, int *slot, uint32_t *stride, uint32_t *disOffset)
{
    if (!context->shared)
        return fail(-3, "this context takes its frames from host memory");
    uint8_t *unused[2];
    if (int error = context->staging(&unused[0], &unused[1]))
        return error;
    *slot = (int)context->nextSlot;
    *stride = context->strideBytes;
    *disOffset = context->disOffset;  // where the distorted plane starts
    return (int)context->slots.size();
}

// The frame begun's pictures where a decoder left them (a context made with
// flag bit 2 -- vv_pictures_mode says whether it takes them): each its
// texture's KMT handle (a Direct3D 11 texture shared without a mutex), the
// texture's size and its first sample's place in it, NV12 (8-bit) or P010
// (10-bit, the samples' bits at the top). Instead of the frame's planes in
// the slot's staging buffer: the pictures must stay as they are until the
// frame is done -- and the reference until the next frame is (its motion).
VV_EXPORT int vv_pictures(vv_context *context, void *reference, uint32_t referenceW, uint32_t referenceH,
                          int referenceX, int referenceY, void *distorted, uint32_t distortedW, uint32_t distortedH,
                          int distortedX, int distortedY)
{
    return context->set_pictures(reference, referenceW, referenceH, referenceX, referenceY, distorted, distortedW,
                                 distortedH, distortedX, distortedY);
}

// 1 if the context takes its frames' pictures from a decoder's textures
// (vv_pictures), else 0.
VV_EXPORT int vv_pictures_mode(vv_context *context) { return context->pictures ? 1 : 0; }

// One of a slot's staging buffers as a Win32 handle another API on this GPU
// imports (an opaque Win32 handle of a dedicated allocation of *bytes: CUDA's
// cuImportExternalMemory, or Vulkan's VkImportMemoryWin32HandleInfoKHR with
// what vv_shared_device gives): part 0 the slot's buffer, which holds both
// videos' planes; in split mode (flag bit 3) part 0 the reference's and
// part 1 the distorted's, laid out alike. The caller closes the handle once
// it has imported it; the memory lives as long as the context.
VV_EXPORT int vv_export_part(vv_context *context, int slot, int part, void **handle, uint64_t *bytes)
{
    if (!context->shared || slot < 0 || slot >= (int)context->slots.size() || part < 0
        || part > (context->split ? 1 : 0))
        return fail(-3, "no such shared buffer");
    const Slot &exported = context->slots[(size_t)slot];
    const Buffer &buffer = part ? exported.stagingDis : exported.staging;
    VkMemoryGetWin32HandleInfoKHR info = { VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR };
    info.memory = buffer.memory;
    info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    HANDLE win32 = nullptr;
    VkResult result = context->getMemoryHandle(context->device, &info, &win32);
    if (result != VK_SUCCESS)
        return fail(-1, "vkGetMemoryWin32HandleKHR failed (" + std::to_string(result) + ")");
    *handle = win32;
    *bytes = buffer.allocation;
    return 0;
}

// vv_export_part's part 0.
VV_EXPORT int vv_export(vv_context *context, int slot, void **handle, uint64_t *bytes)
{
    return vv_export_part(context, slot, 0, handle, bytes);
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
// Frames from a Direct3D 11 decoder on this GPU (Intel's, through oneVPL):
// each shared texture is imported once (*index), then each frame pair is
// committed by its two textures' indices. vv_completed gives how many
// frames the GPU has finished, so the decoder may reuse their textures.
VV_EXPORT int vv_import_texture(vv_context *context, void *handle, int *index)
{
    return context->import_texture(handle, index);
}

VV_EXPORT int vv_commit_textures(vv_context *context, int reference, int distorted, int score)
{
    return context->commit_textures(reference, distorted, score != 0);
}

VV_EXPORT int vv_completed(vv_context *context, unsigned *frames)
{
    *frames = context->completed();
    return context->failed ? fail(-5, "the GPU failed on an earlier frame") : 0;
}


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

// A scored frame's CAMBI score (cambi.c's, with cambi_max_val) from its
// sums and the scales pooled on the CPU (`pooled`, bits `pooledScales`):
// 0, or -5 when it has a sample above its bit depth's largest (as cambi.c
// fails), -6 when its top k is not exact in a double.
int vv_context::cambi_score(const uint64_t *sums, const double *pooled, unsigned pooledScales, double &out) const
{
    const V1CambiConstants &c = cambiConst;
    if (sums[kSlotCambiInvalid] & 1)
        return -5;
    double scores[V1_CAMBI_SCALES];
    for (int scale = 0; scale < V1_CAMBI_SCALES; ++scale) {
        if (pooledScales & (1u << scale)) {
            scores[scale] = pooled[scale];
            continue;
        }
        const uint64_t *s = &sums[kSlotCambi + scale * 4];
        uint64_t threshold;
        const bool exact = (s[3] & 1) == 0 && cambi_units((uint32_t)s[1], threshold);
        const uint64_t total = exact ? s[0] + s[2] * threshold : 0;
        if (!exact || total >= (1ull << 53))
            return -6;
        scores[scale] = (double)total * (1.0 / 16777216.0) / c.topk_elements[scale];
    }
    out = v1_cambi_score(scores, c.pixels_in_window, cambiOptions.max_val);
    return 0;
}

// Every frame's CAMBI score (cambi.c's, with cambi_max_val), after vv_flush;
// NaN for a frame that was not scored. Returns the number of frames, or -5
// when a frame has a sample above its bit depth's largest (as cambi.c fails).
VV_EXPORT int vv_v1_cambi_scores(vv_context *context, double *out, unsigned frames)
{
    if (!context->cambi)
        return fail(-3, "CAMBI is not calculated in this context");
    const Blocks<FrameRecord> &records = context->records;
    const unsigned n = (unsigned)std::min<size_t>(frames, records.size());
    for (unsigned i = 0; i < n; ++i) {
        out[i] = NAN;
        if (!records[i].scored)
            continue;
        if (records[i].cambiError == -5)
            return fail(-5, "CAMBI: frame " + std::to_string(i) + " has a sample above its bit depth's largest");
        if (records[i].cambiError == -6)
            return fail(-6, "CAMBI: frame " + std::to_string(i) + "'s top k is not exact in a double");
        out[i] = records[i].cambi;
    }
    return (int)records.size();
}

// For tests and benchmarks of a context made with flag bit 19 (frames from
// GPU memory): a slot's staging buffer filled with these two luma planes
// (rows of the context's stride, as vv_shared_next describes them), as a
// decoder would write them.
VV_EXPORT int vv_test_fill_slot(vv_context *context, int slot, const uint8_t *reference, const uint8_t *distorted)
{
    if (!context->shared || slot < 0 || slot >= (int)context->slots.size())
        return fail(-3, "no such shared buffer");
    if (context->pictures)  // (the staging buffers are then a few bytes)
        return fail(-3, "this context reads the pictures where a decoder left them (vv_pictures)");
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

// The HDR models' picture conversion (libvmaf's vmaf_picture_convert) on the
// GPU, for a context made by vv_create_v1 at the conversion's target size
// and 16 bits, after vv_v1_speed if SpEED is wanted and before the first
// frame: plans[0] the reference's, plans[1] the distorted's, as
// vmaf_picture_convert_gpu_plan gives them (words[i] 32-bit words each).
// Its frames then go in with vv_submit_v1 as the source pictures (4:2:0,
// more than 8 bits), or, in a context made with flag bits 19 and 1, a
// decoder copies them into the shared buffers (vv_convert_source). -4 for a plan it
// does not run (the caller then converts on the CPU).
VV_EXPORT int vv_v1_convert(vv_context *context, const uint32_t *const *plans, const size_t *words)
{
    return context->enable_convert(plans, words);
}

// To check the conversion against libvmaf's (VideoMetricsLab does on a run's
// first frame): a source picture (`side` 0 the reference, 1 the distorted)
// through the context's conversion and read back -- the luma at out[0], the
// chroma at out[1] and out[2], rows outStrides[i] bytes apart. Waits for the
// frames in flight first.
VV_EXPORT int vv_test_convert(vv_context *context, int side, const uint16_t *const *planes, const ptrdiff_t *strides,
                              uint16_t *const *out, const ptrdiff_t *outStrides)
{
    return context->test_convert(side, planes, strides, out, outStrides);
}

// For a context made with flag bit 19 (frames from GPU memory) that converts
// its pictures (vv_v1_convert): where in each slot's shared buffer a decoder
// writes a side's source plane (`side` 0 the reference, 1 the distorted;
// `plane` 0 Y, 1 U, 2 V) -- from *offset, *width x *height 16-bit samples,
// rows packed (*width * 2 bytes apart). In split mode (flag bit 3) in the
// side's own buffer (vv_export_part's part `side`), else in the one. Its
// frames are then begun with vv_shared_next and committed with vv_commit or
// vv_commit_after, as a context that does not convert.
VV_EXPORT int vv_convert_source(vv_context *context, int side, int plane, uint64_t *offset, uint32_t *width,
                                uint32_t *height)
{
    if (!context->shared || !context->converting)
        return fail(-3, "no converted pictures in the shared buffers");
    if (side < 0 || side > 1 || plane < 0 || plane > 2)
        return fail(-3, "no such plane");
    const vv_context::ConvertSide &c = context->convertSides[side];
    *offset = context->shared_source_offset(side) + c.planeOffset[plane] - c.planeOffset[0];
    *width = c.planeW[plane];
    *height = c.planeH[plane];
    return 0;
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
    if (context->pictures)
        return fail(-3, "this context reads the pictures where a decoder left them (vv_pictures)");
    if (context->converting || context->convertsShared)
        return fail(-3, "this context converts its pictures: their chroma is the conversion's");
    uint8_t *chroma = (uint8_t *)context->pending->speedChroma.mapped;
    for (int i = 0; i < 4; ++i)
        planes[i] = chroma + (size_t)i * context->chromaPlaneBytes;
    *stride = context->chromaStrideBytes;
    if (context->shared)
        context->pending->chromaFromHost = true;  // SpEED reads this frame's from here
    return 0;
}

// For a frame begun with vv_shared_next (a context made with flag bit 19):
// host memory for one of its luma planes -- `which` 0 the reference's, 1 the
// distorted's -- rows *stride bytes apart, written by the CPU instead of by
// the API that imported the slot (whose copy into the slot then carries only
// the other plane). vv_commit copies it into the slot on the GPU before the
// frame's passes. For a pair of which only one video is decoded on the GPU.
VV_EXPORT int vv_shared_host_plane(vv_context *context, int which, uint8_t **plane, uint32_t *stride)
{
    if (!context->shared)
        return fail(-3, "this context takes its frames from host memory (vv_staging)");
    if (which != 0 && which != 1)
        return fail(-3, "no such plane");
    if (!context->pending)
        return fail(-3, "no frame was started");
    if (context->pictures)
        return fail(-3, "this context reads the pictures where a decoder left them (vv_pictures)");
    if (context->converting)
        return fail(-3, "this context converts its pictures: a decoder writes them (vv_convert_source)");
    Slot &slot = *context->pending;
    if (!slot.hostPlane[which].mapped) {  // made at its first use: most pairs never need it
        if (int error = context->create_buffer(slot.hostPlane[which], context->planeBytes, true, false, true))
            return error;
    }
    slot.planeFromHost[which] = true;
    *plane = (uint8_t *)slot.hostPlane[which].mapped;
    *stride = context->strideBytes;
    return 0;
}

// For a context made with flag bit 19 (frames from GPU memory) with SpEED:
// where in each slot's staging buffer (the memory vv_export gives) the four
// chroma planes go -- the reference's U and V, then the distorted's, from
// *offset, *spacing bytes apart, rows *stride bytes apart (w/2 x h/2
// samples). In split mode (flag bit 3) each of the slot's two buffers
// (vv_export_part) holds its video's U and V there, from *offset. A decoder
// writes them there on the GPU, as the lumas; or the frame's go through
// vv_chroma_staging instead.
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
    const size_t count = context->records.size();
    const unsigned n = (unsigned)std::min<size_t>(frames, count);
    for (unsigned i = 0; i < n; ++i) {
        if (!context->speedPool.result(i, out + 3 * (size_t)i))
            out[3 * (size_t)i] = out[3 * (size_t)i + 1] = out[3 * (size_t)i + 2] = NAN;
    }
    return (int)count;
}

// CIEDE2000 (libvmaf's ciede.c) of the frame pairs on the GPU, for a context
// made by vv_create_v1, before its first frame; its frames then go in with
// vv_submit_v1, 4:2:0 (the chroma (w + 1) / 2 x (h + 1) / 2, as libvmaf's
// pictures). For CIEDE2000 alone, make the context with flag bits 16 and 18
// (no motion, no ADM). The score is ciede.c's within about 1e-5 (float
// arithmetic, shaders/ciede.slang). -4 for a context whose frames come from
// a decoder or are converted (the caller scores those on the CPU).
VV_EXPORT int vv_v1_ciede(vv_context *context) { return context->enable_ciede(); }

// Every frame's CIEDE2000 score, after vv_flush; NaN for a frame that was
// not scored, +inf for identical pictures (as ciede.c). Returns the number
// of frames.
VV_EXPORT int vv_v1_ciede_scores(vv_context *context, double *out, unsigned frames)
{
    if (!context->ciede)
        return fail(-3, "CIEDE2000 is not calculated in this context");
    const Blocks<FrameRecord> &records = context->records;
    const unsigned n = (unsigned)std::min<size_t>(frames, records.size());
    for (unsigned i = 0; i < n; ++i)
        out[i] = records[i].scored ? records[i].ciede : NAN;
    return (int)records.size();
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
    context->cpuNs[vv_context::kCpuCambiPool] = context->cambiPool.busyNs;  // on its threads
    context->cpuCount[vv_context::kCpuCambiPool] = context->cambiPool.jobs;
    static const char *const steps[vv_context::kCpuSteps] = { "copy", "record", "wait", "speed_threads", "cambi_copy",
                                                              "cambi_threads" };
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
    const Blocks<FrameRecord> &records = context->records;
    if (index >= records.size())
        return fail(-3, "no such frame");
    const unsigned w = (unsigned)context->w, h = (unsigned)context->h;
    for (int i = 0; i < kFeatures; ++i)
        out[i] = 0.0;
    // As libvmaf (integer_motion_cuda.c): the first frame's motion is 0, and
    // motion2 is the lesser of a frame's motion and the next frame's.
    auto motion = [&](unsigned frame) {
        return frame == 0 ? 0.0 : normalize_and_scale_sad(records[frame].sad, w, h);
    };
    const double own = motion(index);
    out[kMotion] = own;
    if (index == 0) {
        out[kMotion2] = 0.0;
    } else if (index + 1 < records.size()) {
        const double next = motion(index + 1);
        out[kMotion2] = own < next ? own : next;
    } else {
        out[kMotion2] = own;
    }
    const FrameRecord &frame = records[index];
    if (!frame.scored)
        return 0;
    for (int scale = 0; scale < kScales; ++scale) {
        out[kVifScale0 + scale] = frame.vif[0][scale];
        out[kVifScale0Neg + scale] = frame.vif[1][scale];
    }
    out[kAdm2] = frame.adm[0];
    out[kAdm2Neg] = frame.adm[1];
    return 1;
}

// A frame's raw sums (kSlots 64-bit values), for tests: kept for the first
// kRawFrames (256) frames only, -3 for a later one.
VV_EXPORT int vv_sums(vv_context *context, unsigned index, uint64_t *out, int count)
{
    if (index >= context->records.size())
        return fail(-3, "no such frame");
    if (index >= context->rawSums.size())
        return fail(-3, "a frame's raw sums are kept for the first " + std::to_string((int)kRawFrames) + " frames only");
    memcpy(out, context->rawSums[index].data(), sizeof(uint64_t) * (size_t)std::clamp(count, 0, (int)kSlots));
    return kSlots;
}

VV_EXPORT const char *vv_device_name(vv_context *context) { return context->deviceName.c_str(); }

// A device buffer's first `bytes` bytes, after vv_flush; for tests. `which`:
// 0 admR, 1 admA, 2 admF, 3 bandsRef[0], 4 bandsDis[0], 5 bandsRef[1], 6 bandsDis[1], 7 vifTmp,
// 8 the division table, 9 the logarithm table (two entries a word), 10 admRB, 11 admAB, 12 admFB (VMAF NEG's, BOTH),
// 13 bandsARef[0], 14 bandsADis[0], 15 bandsARef[1], 16 bandsADis[1]. 3-6 hold h, v, d (adm_hvd).
VV_EXPORT int vv_read_buffer(vv_context *context, int which, void *out, uint64_t bytes)
{
    Buffer *all[] = { &context->admR, &context->admA, &context->admF, &context->bandsRef[0], &context->bandsDis[0],
                      &context->bandsRef[1], &context->bandsDis[1], &context->vifTmp, &context->divTable,
                      &context->logTable, &context->admRB, &context->admAB, &context->admFB,
                      &context->bandsARef[0], &context->bandsADis[0], &context->bandsARef[1], &context->bandsADis[1] };
    if (which < 0 || which >= (int)(sizeof all / sizeof all[0]))
        return fail(-3, "no such buffer");
    if (int error = context->collect(context->slots[0]))  // slot 0's command buffer: not while its frame is in flight
        return error;
    Buffer *source = all[which];
    bytes = std::min<uint64_t>(bytes, source->size);
    Buffer staging;
    if (int error = context->create_buffer(staging, bytes, true)) {
        context->destroy_last_buffer(staging);
        return error;
    }
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
    const VkResult result = context->run_now(cb);
    if (result == VK_SUCCESS)
        memcpy(out, staging.mapped, (size_t)bytes);
    context->destroy_last_buffer(staging);
    return result == VK_SUCCESS ? 0 : fail(-1, "reading a GPU buffer failed");
}

// VMAF v1: a context that calculates ADM3 and motion3 as libvmaf's CPU code
// does (integer_adm.c, integer_motion.c). `options`: adm_norm_view_dist,
// adm_ref_display_height, adm_csf_mode, adm_noise_weight, adm_dlm_weight,
// adm_min_val, motion_max_val, motion_five_frame_window,
// motion_moving_average (nine doubles). Frames go in with vv_submit or
// vv_staging / vv_commit as for VMAF v0.6.1.
VV_EXPORT int vv_create_v1_ex(vv_context **out, int device, int width, int height, int bitDepth, int flags,
                              const double *options, int count);

VV_EXPORT int vv_create_v1(vv_context **out, int device, int width, int height, int bitDepth, int flags,
                           const double *options)
{
    return vv_create_v1_ex(out, device, width, height, bitDepth, flags, options, 9);
}

// vv_create_v1 with `count` options: its nine, then adm_skip_scale0 (the
// HDR VMAF models'; 0 where not given).
VV_EXPORT int vv_create_v1_ex(vv_context **out, int device, int width, int height, int bitDepth, int flags,
                              const double *options, int count)
{
    *out = nullptr;
    if (count < 9)
        return fail(-3, "VMAF v1: fewer options than its nine");
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
    o.skipScale0 = count > 9 && options[9] != 0;
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
    const Blocks<FrameRecord> &records = context->records;
    const size_t count = records.size();
    const V1Options &o = context->options;
    const unsigned n = (unsigned)std::min<size_t>(frames, count);
    const unsigned w = (unsigned)context->w, h = (unsigned)context->h;
    const unsigned min_idx = o.fiveFrameWindow ? 2 : 1, stride = o.fiveFrameWindow ? 2 : 1;
    std::vector<double> sad(count);
    for (size_t i = 0; i < count; ++i) {
        double score = 0.;
        if (i >= min_idx) {
            const double scaled = (double)records[i].sad / 256. / (w * h) * 1.0;  // motion_fps_weight
            score = scaled < o.motionMax ? scaled : o.motionMax;
        }
        sad[i] = score;
    }
    // motion_blend() with the default blend factor 1: the score itself.
    double stamp_value = 0.;
    if (count > min_idx)
        stamp_value = sad[min_idx] < o.motionMax ? sad[min_idx] : o.motionMax;
    double prev_processed = 0.;
    for (size_t i = 0; i < count; ++i) {
        double motion2;
        if (i < min_idx) {
            motion2 = 0.;
        } else {
            const int lo_idx = (int)i - (int)(stride - 1);
            const size_t hi_idx = i + 1;
            if (hi_idx >= count) {
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
            if (records[i].scored)
                memcpy(row, records[i].v1Adm, sizeof records[i].v1Adm);
            row[3] = motion3;
            row[4] = motion2;
            row[5] = sad[i];
        }
    }
    return (int)count;
}
