/**
 *
 *  Copyright 2026 libvmaf-fast contributors
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

/* libvmaf-fast: one frame's work shared by helper threads.
 *
 * libvmaf threads by frames: each pool thread runs every extractor on a
 * whole frame pair, so N busy threads hold N pairs and N copies of every
 * extractor's buffers. An extractor whose frame splits into independent
 * chunks (row stripes, block rows) can hand the chunks to a process-wide
 * pool of helper threads instead, and fewer frames need to be in flight to
 * keep the CPU busy.
 *
 * Scores stay bit-identical: chunks are handed out in index order, and a
 * job whose chunks add into running sums is ordered: each chunk hands its
 * addition to vmaf_parallel_in_turn, and the additions run in chunk order,
 * the order a single thread adds them in. With no helpers every chunk runs
 * on the calling thread, one after the other, as the code did before. */

#ifndef __VMAF_PARALLEL_H__
#define __VMAF_PARALLEL_H__

#include <stdbool.h>
#include <stddef.h>

typedef struct VmafParallelJob VmafParallelJob;

/* One chunk of a job: ctx as given to vmaf_parallel_for, the chunk's index,
 * and a scratch buffer of the job's scratch_bytes (64-byte aligned) that
 * belongs to the thread running the chunk until it returns. A chunk never
 * calls vmaf_parallel_for itself. Returns 0, or an error code that
 * vmaf_parallel_for returns (the job's other chunks still run). */
typedef int (*VmafParallelChunk)(void *ctx, unsigned chunk, void *scratch, VmafParallelJob *job);

/* Sets the number of helper threads (0: none, chunks run on the calling
 * threads). Helpers are started or stopped as needed; a running job is
 * finished first. Returns 0, or a negative errno code. */
int vmaf_parallel_set_helpers(unsigned helpers);

/* The number of helper threads set. */
unsigned vmaf_parallel_helpers(void);

/* Runs fn for chunks 0 to n_chunks - 1, handed out in order to the helpers
 * and the calling thread, and returns once every chunk has finished.
 * caller_scratch: scratch_bytes for the chunks the calling thread runs
 * (helpers use their own). ordered: the chunks' in-turn steps run in chunk
 * order (vmaf_parallel_in_turn); a chunk that returns without one has an
 * empty step, so no error path can hold up the chunks after it.
 * Returns 0, or the first error a chunk returned. */
int vmaf_parallel_for(unsigned n_chunks, VmafParallelChunk fn, void *ctx,
                      size_t scratch_bytes, void *caller_scratch, bool ordered);

/* In an ordered job, at most once per chunk: runs fn(arg) after the steps
 * of every chunk before this one, and returns once it has run. It runs on
 * the thread that hands over or runs the step before it (here, if that has
 * run), which runs every step already handed over after its own too, so
 * turns pass without a thread woken for each. fn may read the chunk's
 * scratch and stack, which stay valid until this returns. The wait blocks
 * (no spinning) and runs no other chunk meanwhile. */
void vmaf_parallel_in_turn(VmafParallelJob *job, unsigned chunk, void (*fn)(void *arg),
                           void *arg);

#endif /* __VMAF_PARALLEL_H__ */
