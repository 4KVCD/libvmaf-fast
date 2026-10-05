/**
 *
 *  Copyright 2016-2023 Netflix, Inc.
 *  Copyright 2021 NVIDIA Corporation.
 *
 *     Licensed under the BSD+Patent License (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         https://opensource.org/licenses/BSDplusPatent
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *
 */

#include "test.h"

#include "libvmaf/libvmaf_cuda.h"
#include "libvmaf/libvmaf.h"
#include "libvmaf/model.h"

static char *test_cuda_no_init()
{
    int err = 0;

    VmafConfiguration vmaf_cfg = { 0 };

    VmafContext *vmaf;
    vmaf_init(&vmaf, vmaf_cfg);
    mu_assert("problem during vmaf_init", vmaf);

    VmafModelConfig model_cfg = { 0 };
    VmafModel *model;
    vmaf_model_load(&model, &model_cfg, "vmaf_v0.6.1");
    mu_assert("problem during vmaf_model_load", model);

    err = vmaf_use_features_from_model(vmaf, model);
    mu_assert("problem during vmaf_use_features_from_model", !err);

    for (unsigned i = 0; i < 10; i++) {
        VmafPicture ref, dist;
        err = vmaf_picture_alloc(&ref, VMAF_PIX_FMT_YUV420P, 8, 1920, 1080);
        mu_assert("problem during vmaf_picture_alloc", !err);
        err = vmaf_picture_alloc(&dist, VMAF_PIX_FMT_YUV420P, 8, 1920, 1080);
        mu_assert("problem during vmaf_picture_alloc", !err);
        err = vmaf_read_pictures(vmaf, &ref, &dist, i);
        mu_assert("problem during vmaf_read_pictures", !err);
    }

    err = vmaf_read_pictures(vmaf, NULL, NULL, 0);
    mu_assert("problem during vmaf_read_pictures", !err);

    err = vmaf_close(vmaf);
    mu_assert("problem during vmaf_read_pictures", !err);

    return NULL;
}

static char *test_cuda_picture_preallocation_method_none()
{
    int err = 0;

    VmafConfiguration vmaf_cfg = { 0 };

    VmafContext *vmaf;
    vmaf_init(&vmaf, vmaf_cfg);
    mu_assert("problem during vmaf_init", vmaf);

    VmafCudaState *cu_state;
    VmafCudaConfiguration cuda_cfg = { 0 };
    err = vmaf_cuda_state_init(&cu_state, cuda_cfg);
    mu_assert("problem during vmaf_cuda_state_init", !err);
    err = vmaf_cuda_import_state(vmaf, cu_state);
    mu_assert("problem during vmaf_cuda_import_state", !err);

    VmafModelConfig model_cfg = { 0 };
    VmafModel *model;
    vmaf_model_load(&model, &model_cfg, "vmaf_v0.6.1");
    mu_assert("problem during vmaf_model_load", model);

    err = vmaf_use_features_from_model(vmaf, model);
    mu_assert("problem during vmaf_use_features_from_model", !err);

    for (unsigned i = 0; i < 10; i++) {
        VmafPicture ref, dist;
        err = vmaf_picture_alloc(&ref, VMAF_PIX_FMT_YUV420P, 8, 1920, 1080);
        mu_assert("problem during vmaf_picture_alloc", !err);
        err = vmaf_picture_alloc(&dist, VMAF_PIX_FMT_YUV420P, 8, 1920, 1080);
        mu_assert("problem during vmaf_picture_alloc", !err);
        err = vmaf_read_pictures(vmaf, &ref, &dist, i);
        mu_assert("problem during vmaf_read_pictures", !err);
    }

    err = vmaf_read_pictures(vmaf, NULL, NULL, 0);
    mu_assert("problem during vmaf_read_pictures", !err);

    err = vmaf_close(vmaf);
    mu_assert("problem during vmaf_read_pictures", !err);

    return NULL;
}

static char *test_cuda_picture_preallocation_method_host()
{
    int err = 0;

    VmafConfiguration vmaf_cfg = { 0 };

    VmafContext *vmaf;
    vmaf_init(&vmaf, vmaf_cfg);
    mu_assert("problem during vmaf_init", vmaf);

    VmafCudaState *cu_state;
    VmafCudaConfiguration cuda_cfg = { 0 };
    err = vmaf_cuda_state_init(&cu_state, cuda_cfg);
    mu_assert("problem during vmaf_cuda_state_init", !err);
    err = vmaf_cuda_import_state(vmaf, cu_state);
    mu_assert("problem during vmaf_cuda_import_state", !err);

    VmafCudaPictureConfiguration cuda_pic_cfg = {
        .pic_params = {
            .w = 1920,
            .h = 1080,
            .bpc = 8,
            .pix_fmt = VMAF_PIX_FMT_YUV420P,
        },
        .pic_prealloc_method = VMAF_CUDA_PICTURE_PREALLOCATION_METHOD_HOST,
    };

    err = vmaf_cuda_preallocate_pictures(vmaf, cuda_pic_cfg);
    mu_assert("problem during vmaf_cuda_preallocate_pictures", !err);

    VmafModelConfig model_cfg = { 0 };
    VmafModel *model;
    vmaf_model_load(&model, &model_cfg, "vmaf_v0.6.1");
    mu_assert("problem during vmaf_model_load", model);

    err = vmaf_use_features_from_model(vmaf, model);
    mu_assert("problem during vmaf_use_features_from_model", !err);

    for (unsigned i = 0; i < 10; i++) {
        VmafPicture ref, dist;
        err = vmaf_cuda_fetch_preallocated_picture(vmaf, &ref);
        mu_assert("problem during vmaf_cuda_fetch_preallocated_picture", !err);
        err = vmaf_cuda_fetch_preallocated_picture(vmaf, &dist);
        mu_assert("problem during vmaf_cuda_fetch_preallocated_picture", !err);
        err = vmaf_read_pictures(vmaf, &ref, &dist, i);
        mu_assert("problem during vmaf_read_pictures", !err);
    }

    err = vmaf_read_pictures(vmaf, NULL, NULL, 0);
    mu_assert("problem during vmaf_read_pictures", !err);

    err = vmaf_close(vmaf);
    mu_assert("problem during vmaf_read_pictures", !err);

    return NULL;
}

static char *test_cuda_picture_preallocation_method_host_pinned()
{
    int err = 0;

    VmafConfiguration vmaf_cfg = { 0 };

    VmafContext *vmaf;
    vmaf_init(&vmaf, vmaf_cfg);
    mu_assert("problem during vmaf_init", vmaf);

    VmafCudaState *cu_state;
    VmafCudaConfiguration cuda_cfg = { 0 };
    err = vmaf_cuda_state_init(&cu_state, cuda_cfg);
    mu_assert("problem during vmaf_cuda_state_init", !err);
    err = vmaf_cuda_import_state(vmaf, cu_state);
    mu_assert("problem during vmaf_cuda_import_state", !err);

    VmafCudaPictureConfiguration cuda_pic_cfg = {
        .pic_params = {
            .w = 1920,
            .h = 1080,
            .bpc = 8,
            .pix_fmt = VMAF_PIX_FMT_YUV420P,
        },
        .pic_prealloc_method = VMAF_CUDA_PICTURE_PREALLOCATION_METHOD_HOST_PINNED,
    };

    err = vmaf_cuda_preallocate_pictures(vmaf, cuda_pic_cfg);
    mu_assert("problem during vmaf_cuda_preallocate_pictures", !err);

    VmafModelConfig model_cfg = { 0 };
    VmafModel *model;
    vmaf_model_load(&model, &model_cfg, "vmaf_v0.6.1");
    mu_assert("problem during vmaf_model_load", model);

    err = vmaf_use_features_from_model(vmaf, model);
    mu_assert("problem during vmaf_use_features_from_model", !err);

    for (unsigned i = 0; i < 10; i++) {
        VmafPicture ref, dist;
        err = vmaf_cuda_fetch_preallocated_picture(vmaf, &ref);
        mu_assert("problem during vmaf_cuda_fetch_preallocated_picture", !err);
        err = vmaf_cuda_fetch_preallocated_picture(vmaf, &dist);
        mu_assert("problem during vmaf_cuda_fetch_preallocated_picture", !err);
        err = vmaf_read_pictures(vmaf, &ref, &dist, i);
        mu_assert("problem during vmaf_read_pictures", !err);
    }

    err = vmaf_read_pictures(vmaf, NULL, NULL, 0);
    mu_assert("problem during vmaf_read_pictures", !err);

    err = vmaf_close(vmaf);
    mu_assert("problem during vmaf_read_pictures", !err);

    return NULL;
}

static char *test_cuda_picture_preallocation_method_device()
{
    int err = 0;

    VmafConfiguration vmaf_cfg = { 0 };

    VmafContext *vmaf;
    vmaf_init(&vmaf, vmaf_cfg);
    mu_assert("problem during vmaf_init", vmaf);

    VmafCudaState *cu_state;
    VmafCudaConfiguration cuda_cfg = { 0 };
    err = vmaf_cuda_state_init(&cu_state, cuda_cfg);
    mu_assert("problem during vmaf_cuda_state_init", !err);
    err = vmaf_cuda_import_state(vmaf, cu_state);
    mu_assert("problem during vmaf_cuda_import_state", !err);

    VmafCudaPictureConfiguration cuda_pic_cfg = {
        .pic_params = {
            .w = 1920,
            .h = 1080,
            .bpc = 8,
            .pix_fmt = VMAF_PIX_FMT_YUV420P,
        },
        .pic_prealloc_method = VMAF_CUDA_PICTURE_PREALLOCATION_METHOD_DEVICE,
    };

    err = vmaf_cuda_preallocate_pictures(vmaf, cuda_pic_cfg);
    mu_assert("problem during vmaf_cuda_preallocate_pictures", !err);

    VmafModelConfig model_cfg = { 0 };
    VmafModel *model;
    vmaf_model_load(&model, &model_cfg, "vmaf_v0.6.1");
    mu_assert("problem during vmaf_model_load", model);

    err = vmaf_use_features_from_model(vmaf, model);
    mu_assert("problem during vmaf_use_features_from_model", !err);

    for (unsigned i = 0; i < 10; i++) {
        VmafPicture ref, dist;
        err = vmaf_cuda_fetch_preallocated_picture(vmaf, &ref);
        mu_assert("problem during vmaf_cuda_fetch_preallocated_picture", !err);
        err = vmaf_cuda_fetch_preallocated_picture(vmaf, &dist);
        mu_assert("problem during vmaf_cuda_fetch_preallocated_picture", !err);
        err = vmaf_read_pictures(vmaf, &ref, &dist, i);
        mu_assert("problem during vmaf_read_pictures", !err);
    }

    err = vmaf_read_pictures(vmaf, NULL, NULL, 0);
    mu_assert("problem during vmaf_read_pictures", !err);

    err = vmaf_close(vmaf);
    mu_assert("problem during vmaf_read_pictures", !err);

    return NULL;
}

#define POOL_W 320
#define POOL_H 240
#define POOL_FRAMES 12

/* Frame i of a moving pattern, the distorted one blurred and offset, so that
 * every frame's VMAF differs and stale or half-uploaded data would show. */
static void fill_frame(VmafPicture *pic, unsigned i, int distorted)
{
    for (unsigned p = 0; p < 3; p++) {
        uint8_t *data = pic->data[p];
        for (unsigned y = 0; y < pic->h[p]; y++) {
            for (unsigned x = 0; x < pic->w[p]; x++) {
                unsigned v = (x * 7 + y * 3 + i * 11 + ((x / 8 + y / 8 + i) % 2) * 90) % 256;
                if (distorted)
                    v = (v + ((x + i) % 3) * 9 + (y % 5)) / 2 + 40;
                data[y * pic->stride[p] + x] = (uint8_t)v;
            }
        }
    }
}

/* Whether a picture handed out still holds an earlier frame: a newly
 * allocated picture is zeroed, a reused one keeps what was written. */
static int holds_a_frame(const VmafPicture *pic)
{
    for (unsigned x = 0; x < pic->w[0]; x++) {
        if (((const uint8_t *)pic->data[0])[x])
            return 1;
    }
    return 0;
}

/* VMAF of POOL_FRAMES frames with CUDA, pictures from `method`, or
 * allocated for every frame (vmaf_picture_alloc) when `method` is NONE.
 * `reused`: set when a picture was handed out again, not allocated anew. */
static char *score_frames(enum VmafCudaPicturePreallocationMethod method,
                          double *scores, int *reused)
{
    int err = 0;
    VmafConfiguration vmaf_cfg = { 0 };
    VmafContext *vmaf;
    vmaf_init(&vmaf, vmaf_cfg);
    mu_assert("problem during vmaf_init", vmaf);

    VmafCudaState *cu_state;
    VmafCudaConfiguration cuda_cfg = { 0 };
    err = vmaf_cuda_state_init(&cu_state, cuda_cfg);
    mu_assert("problem during vmaf_cuda_state_init", !err);
    err = vmaf_cuda_import_state(vmaf, cu_state);
    mu_assert("problem during vmaf_cuda_import_state", !err);

    if (method != VMAF_CUDA_PICTURE_PREALLOCATION_METHOD_NONE) {
        VmafCudaPictureConfiguration cuda_pic_cfg = {
            .pic_params = { .w = POOL_W, .h = POOL_H, .bpc = 8,
                            .pix_fmt = VMAF_PIX_FMT_YUV420P },
            .pic_prealloc_method = method,
        };
        err = vmaf_cuda_preallocate_pictures(vmaf, cuda_pic_cfg);
        mu_assert("problem during vmaf_cuda_preallocate_pictures", !err);
    }

    VmafModelConfig model_cfg = { 0 };
    VmafModel *model;
    vmaf_model_load(&model, &model_cfg, "vmaf_v0.6.1");
    mu_assert("problem during vmaf_model_load", model);
    err = vmaf_use_features_from_model(vmaf, model);
    mu_assert("problem during vmaf_use_features_from_model", !err);

    *reused = 0;
    for (unsigned i = 0; i < POOL_FRAMES; i++) {
        VmafPicture ref, dist;
        if (method == VMAF_CUDA_PICTURE_PREALLOCATION_METHOD_NONE) {
            err = vmaf_picture_alloc(&ref, VMAF_PIX_FMT_YUV420P, 8, POOL_W, POOL_H);
            err |= vmaf_picture_alloc(&dist, VMAF_PIX_FMT_YUV420P, 8, POOL_W, POOL_H);
        } else {
            err = vmaf_cuda_fetch_preallocated_picture(vmaf, &ref);
            err |= vmaf_cuda_fetch_preallocated_picture(vmaf, &dist);
        }
        mu_assert("problem getting pictures", !err);
        *reused |= holds_a_frame(&ref) || holds_a_frame(&dist);
        fill_frame(&ref, i, 0);
        fill_frame(&dist, i, 1);
        err = vmaf_read_pictures(vmaf, &ref, &dist, i);
        mu_assert("problem during vmaf_read_pictures", !err);
    }
    err = vmaf_read_pictures(vmaf, NULL, NULL, 0);
    mu_assert("problem flushing", !err);
    for (unsigned i = 0; i < POOL_FRAMES; i++) {
        err = vmaf_score_at_index(vmaf, model, &scores[i], i);
        mu_assert("problem during vmaf_score_at_index", !err);
    }
    vmaf_model_destroy(model);
    err = vmaf_close(vmaf);
    mu_assert("problem during vmaf_close", !err);
    return NULL;
}

static char *test_cuda_host_pictures_reused_and_identical()
{
    double fresh[POOL_FRAMES], host[POOL_FRAMES], pinned[POOL_FRAMES];
    int reused = 0;
    char *msg;

    msg = score_frames(VMAF_CUDA_PICTURE_PREALLOCATION_METHOD_NONE, fresh, &reused);
    if (msg) return msg;
    msg = score_frames(VMAF_CUDA_PICTURE_PREALLOCATION_METHOD_HOST, host, &reused);
    if (msg) return msg;
    mu_assert("HOST pictures were not reused", reused);
    msg = score_frames(VMAF_CUDA_PICTURE_PREALLOCATION_METHOD_HOST_PINNED, pinned, &reused);
    if (msg) return msg;
    mu_assert("HOST_PINNED pictures were not reused", reused);

    for (unsigned i = 0; i < POOL_FRAMES; i++) {
        mu_assert("HOST scores differ from fresh pictures'", host[i] == fresh[i]);
        mu_assert("HOST_PINNED scores differ from fresh pictures'", pinned[i] == fresh[i]);
    }
    // the frames differ, so a stale upload would show
    mu_assert("frames do not differ", fresh[0] != fresh[POOL_FRAMES - 1]);
    return NULL;
}

/* More pictures than the pool holds, taken before any is given back: the
 * fetch allocates the rest instead of waiting, and all are released. */
static char *test_cuda_host_pool_exhausted()
{
    int err = 0;
    VmafConfiguration vmaf_cfg = { 0 };
    VmafContext *vmaf;
    vmaf_init(&vmaf, vmaf_cfg);
    mu_assert("problem during vmaf_init", vmaf);

    VmafCudaState *cu_state;
    VmafCudaConfiguration cuda_cfg = { 0 };
    err = vmaf_cuda_state_init(&cu_state, cuda_cfg);
    mu_assert("problem during vmaf_cuda_state_init", !err);
    err = vmaf_cuda_import_state(vmaf, cu_state);
    mu_assert("problem during vmaf_cuda_import_state", !err);
    VmafCudaPictureConfiguration cuda_pic_cfg = {
        .pic_params = { .w = POOL_W, .h = POOL_H, .bpc = 8,
                        .pix_fmt = VMAF_PIX_FMT_YUV420P },
        .pic_prealloc_method = VMAF_CUDA_PICTURE_PREALLOCATION_METHOD_HOST_PINNED,
    };
    err = vmaf_cuda_preallocate_pictures(vmaf, cuda_pic_cfg);
    mu_assert("problem during vmaf_cuda_preallocate_pictures", !err);

    VmafPicture pics[20];
    for (unsigned i = 0; i < 20; i++) {
        err = vmaf_cuda_fetch_preallocated_picture(vmaf, &pics[i]);
        mu_assert("fetch failed with the pool in use", !err);
    }
    for (unsigned i = 0; i < 20; i++) {
        err = vmaf_picture_unref(&pics[i]);
        mu_assert("problem during vmaf_picture_unref", !err);
    }
    err = vmaf_close(vmaf);
    mu_assert("problem during vmaf_close", !err);
    return NULL;
}

char *run_tests()
{
    mu_run_test(test_cuda_no_init);
    mu_run_test(test_cuda_picture_preallocation_method_none);
    mu_run_test(test_cuda_picture_preallocation_method_host);
    mu_run_test(test_cuda_picture_preallocation_method_host_pinned);
    mu_run_test(test_cuda_picture_preallocation_method_device);
    mu_run_test(test_cuda_host_pictures_reused_and_identical);
    mu_run_test(test_cuda_host_pool_exhausted);
    return NULL;
}
