/**
 *
 *  Copyright 2016-2020 Netflix, Inc.
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

#include <stdint.h>

#include <libvmaf/picture.h>

/* One row of w samples of more than 8 bits as floats (the picture's rows
 * are shared by picture_copy and picture_copy_row, so both give the same
 * values). */
static void copy_row_hbd(float *dst, const uint16_t *data, unsigned w, int offset, float scaler)
{
    for (unsigned j = 0; j < w; j++)
        dst[j] = (float) data[j] / scaler + offset;
}

static void copy_row_8(float *dst, const uint8_t *data, unsigned w, int offset)
{
    for (unsigned j = 0; j < w; j++)
        dst[j] = (float) data[j] + offset;
}

void picture_copy_hbd(float *dst, ptrdiff_t dst_stride,
                      VmafPicture *src, int offset, float scaler, int channel)
{
    float *float_data = dst;
    uint16_t *data = src->data[channel];

    for (unsigned i = 0; i < src->h[channel]; i++) {
        copy_row_hbd(float_data, data, src->w[channel], offset, scaler);
        float_data += dst_stride / sizeof(float);
        data += src->stride[channel] / 2;
    }
    return;
}

/* picture_copy's scaler for samples of bpc bits stored in 16 bits, 0 for
 * samples it reads as 8-bit. */
static float hbd_scaler(unsigned bpc)
{
    return bpc == 10 ? 4.0f : bpc == 12 ? 16.0f : bpc == 16 ? 256.0f : 0.0f;
}

void picture_copy(float *dst, ptrdiff_t dst_stride,
                  VmafPicture *src, int offset, unsigned bpc, int channel)
{
    const float scaler = hbd_scaler(bpc);
    if (scaler) {
        picture_copy_hbd(dst, dst_stride, src, offset, scaler, channel);
        return;
    }

    float *float_data = dst;
    uint8_t *data = src->data[channel];

    for (unsigned i = 0; i < src->h[channel]; i++) {
        copy_row_8(float_data, data, src->w[channel], offset);
        float_data += dst_stride / sizeof(float);
        data += src->stride[channel];
    }

    return;
}

void picture_copy_row(float *dst, const VmafPicture *src, int offset, unsigned bpc,
                      int channel, unsigned row)
{
    const uint8_t *data = (const uint8_t *)src->data[channel] + (size_t)row * src->stride[channel];
    const float scaler = hbd_scaler(bpc);
    if (scaler)
        copy_row_hbd(dst, (const uint16_t *)data, src->w[channel], offset, scaler);
    else
        copy_row_8(dst, data, src->w[channel], offset);
}
