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

/* libvmaf-fast: the helper pool behind parallel.h.
 *
 * One lock guards everything: the list of jobs with chunks left, each job's
 * counters and its turn. A chunk is a stripe of a frame (about a millisecond
 * of work at 4K), so a few lock round trips per chunk cost nothing
 * measurable, and one lock keeps the waits simple to check:
 * - Chunks are claimed in index order, and whoever claims one runs it at
 *   once. A chunk waiting for its turn therefore waits on lower chunks that
 *   are already running; those wait only on lower ones still, so the wait
 *   ends with the lowest unfinished chunk, which waits on nothing. No wait
 *   runs other chunks, so no thread ever holds two.
 * - Whoever hands over the step whose turn it is runs it, and every step
 *   handed over after it, on its own thread, so turns pass without a thread
 *   being woken for each.
 * - The calling thread claims its job's chunks as the helpers do, then waits
 *   for the last running ones holding nothing: frame threads of several
 *   frames, and several contexts, can share the helpers without deadlock. */

#include <errno.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#include "mem.h"
#include "parallel.h"

/* Scratch alignment: a cache line, so two threads' buffers never share one. */
#define SCRATCH_ALIGN 64

/* A chunk's in-turn step, once handed over (fn NULL: the chunk has nothing
 * to do in its turn). */
typedef struct TurnStep {
    void (*fn)(void *arg);
    void *arg;
    bool posted;
} TurnStep;

struct VmafParallelJob {
    VmafParallelChunk fn;
    void *ctx;
    size_t scratch_bytes;
    bool ordered;
    unsigned n_chunks;
    unsigned next;      /* the next chunk to claim */
    unsigned finished;  /* chunks that have returned */
    unsigned turn;      /* in an ordered job: the chunk whose step runs next */
    TurnStep *steps;    /* in an ordered job: each chunk's step */
    int err;            /* the first error a chunk returned */
    pthread_cond_t turn_passed;
    pthread_cond_t all_finished;
    VmafParallelJob *next_job;  /* in the list of jobs with chunks to claim */
};

typedef struct Helper {
    pthread_t thread;
    unsigned index;
    void *scratch;
    size_t scratch_bytes;
} Helper;

static struct {
    pthread_mutex_t lock;
    pthread_cond_t work;    /* a job was posted, or helpers must stop */
    VmafParallelJob *jobs;  /* jobs with chunks to claim, oldest first */
    Helper **helpers;       /* the running helpers */
    unsigned n_helpers;     /* how many run */
    unsigned wanted;        /* how many should: helpers at or above it stop */
} pool = { PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, NULL, NULL, 0, 0 };

/* Held through vmaf_parallel_set_helpers, so two calls never start and stop
 * helpers at once. */
static pthread_mutex_t setting_helpers = PTHREAD_MUTEX_INITIALIZER;

/* Takes the job off the list once its last chunk is claimed (lock held). */
static unsigned claim_chunk(VmafParallelJob *job)
{
    const unsigned chunk = job->next++;
    if (job->next == job->n_chunks) {
        VmafParallelJob **link = &pool.jobs;
        while (*link != job)
            link = &(*link)->next_job;
        *link = job->next_job;
    }
    return chunk;
}

/* Records a chunk's step, and where it is the turn's, runs it and every step
 * handed over after it, in order, on this thread (lock held; released while
 * a step runs). The steps' chunks wait in vmaf_parallel_in_turn until their
 * step has run, so what the steps read stays valid; the thread running them
 * belongs to a chunk that has not finished, so the job outlives them. */
static void post_step(VmafParallelJob *job, unsigned chunk, void (*fn)(void *), void *arg)
{
    job->steps[chunk] = (TurnStep){ fn, arg, true };
    if (job->turn != chunk)
        return;  /* the thread running the step before it runs this one */
    while (job->turn < job->n_chunks && job->steps[job->turn].posted) {
        const TurnStep step = job->steps[job->turn];
        if (step.fn) {
            pthread_mutex_unlock(&pool.lock);
            step.fn(step.arg);
            pthread_mutex_lock(&pool.lock);
        }
        job->turn++;
        pthread_cond_broadcast(&job->turn_passed);
    }
}

/* Counts a chunk that returned `err`, handing over an empty step for it if
 * it handed over none (lock not held on entry or exit). */
static void finish_chunk(VmafParallelJob *job, unsigned chunk, int err)
{
    pthread_mutex_lock(&pool.lock);
    if (job->ordered && !job->steps[chunk].posted)
        post_step(job, chunk, NULL, NULL);
    if (err && !job->err)
        job->err = err;
    if (++job->finished == job->n_chunks)
        pthread_cond_signal(&job->all_finished);
    pthread_mutex_unlock(&pool.lock);
}

/* The job a free helper joins (lock held, a job posted): the one with the
 * fewest chunks running, the oldest of those. All joining the oldest job
 * would pile them up there while other frames' jobs get no one. */
static VmafParallelJob *pick_job(void)
{
    VmafParallelJob *best = pool.jobs;
    for (VmafParallelJob *job = best->next_job; job; job = job->next_job) {
        if (job->next - job->finished < best->next - best->finished)
            best = job;
    }
    return best;
}

/* A helper's scratch, at least `bytes` (lock not held). */
static void *helper_scratch(Helper *helper, size_t bytes)
{
    if (helper->scratch_bytes < bytes) {
        aligned_free(helper->scratch);
        helper->scratch = aligned_malloc(bytes, SCRATCH_ALIGN);
        helper->scratch_bytes = helper->scratch ? bytes : 0;
    }
    return helper->scratch;
}

static void *helper_main(void *data)
{
    Helper *helper = data;
    pthread_mutex_lock(&pool.lock);
    for (;;) {
        while (helper->index < pool.wanted && !pool.jobs)
            pthread_cond_wait(&pool.work, &pool.lock);
        if (helper->index >= pool.wanted)
            break;
        VmafParallelJob *job = pick_job();
        const unsigned chunk = claim_chunk(job);
        pthread_mutex_unlock(&pool.lock);
        void *scratch = helper_scratch(helper, job->scratch_bytes);
        const int err = scratch || !job->scratch_bytes
            ? job->fn(job->ctx, chunk, scratch, job)
            : -ENOMEM;  /* the chunk fails, and still takes its turn */
        finish_chunk(job, chunk, err);
        pthread_mutex_lock(&pool.lock);
    }
    pthread_mutex_unlock(&pool.lock);
    aligned_free(helper->scratch);
    return NULL;
}

int vmaf_parallel_set_helpers(unsigned helpers)
{
    int err = 0;
    pthread_mutex_lock(&setting_helpers);
    pthread_mutex_lock(&pool.lock);
    pool.wanted = helpers;
    if (helpers > pool.n_helpers) {
        Helper **grown = realloc(pool.helpers, helpers * sizeof(*grown));
        if (!grown) {
            err = -ENOMEM;
        } else {
            pool.helpers = grown;
            while (pool.n_helpers < helpers) {
                Helper *helper = calloc(1, sizeof(*helper));
                if (!helper) {
                    err = -ENOMEM;
                    break;
                }
                helper->index = pool.n_helpers;
                if (pthread_create(&helper->thread, NULL, helper_main, helper)) {
                    free(helper);
                    err = -EAGAIN;
                    break;
                }
                pool.helpers[pool.n_helpers++] = helper;
            }
        }
        pool.wanted = pool.n_helpers;
        pthread_mutex_unlock(&pool.lock);
        pthread_mutex_unlock(&setting_helpers);
        return err;
    }
    /* fewer: the helpers above the number end once their chunk is done */
    const unsigned running = pool.n_helpers;
    pool.n_helpers = helpers;
    pthread_cond_broadcast(&pool.work);
    pthread_mutex_unlock(&pool.lock);
    for (unsigned i = helpers; i < running; i++) {
        pthread_join(pool.helpers[i]->thread, NULL);
        free(pool.helpers[i]);
    }
    pthread_mutex_unlock(&setting_helpers);
    return 0;
}

unsigned vmaf_parallel_helpers(void)
{
    pthread_mutex_lock(&pool.lock);
    const unsigned helpers = pool.wanted;
    pthread_mutex_unlock(&pool.lock);
    return helpers;
}

int vmaf_parallel_for(unsigned n_chunks, VmafParallelChunk fn, void *ctx,
                      size_t scratch_bytes, void *caller_scratch, bool ordered)
{
    if (!fn || (scratch_bytes && !caller_scratch))
        return -EINVAL;
    if (!n_chunks)
        return 0;
    /* an ordered job's steps: on the stack up to 8K's 180 stripes */
    TurnStep local_steps[256];
    TurnStep *steps = NULL;
    if (ordered) {
        steps = n_chunks <= 256 ? local_steps : malloc(n_chunks * sizeof(*steps));
        if (!steps)
            return -ENOMEM;
        memset(steps, 0, n_chunks * sizeof(*steps));
    }
    VmafParallelJob job = {
        .fn = fn, .ctx = ctx, .scratch_bytes = scratch_bytes, .ordered = ordered,
        .n_chunks = n_chunks, .steps = steps,
    };
    pthread_cond_init(&job.turn_passed, NULL);
    pthread_cond_init(&job.all_finished, NULL);

    pthread_mutex_lock(&pool.lock);
    /* shared with the helpers, or (no helpers, or one chunk) run here alone */
    const bool shared = pool.wanted && n_chunks > 1;
    if (shared) {
        VmafParallelJob **link = &pool.jobs;
        while (*link)
            link = &(*link)->next_job;
        *link = &job;
        pthread_cond_broadcast(&pool.work);
    }
    /* the calling thread claims chunks too, then waits holding none */
    while (job.next < n_chunks) {
        const unsigned chunk = shared ? claim_chunk(&job) : job.next++;
        pthread_mutex_unlock(&pool.lock);
        finish_chunk(&job, chunk, fn(ctx, chunk, caller_scratch, &job));
        pthread_mutex_lock(&pool.lock);
    }
    while (job.finished < n_chunks)
        pthread_cond_wait(&job.all_finished, &pool.lock);
    pthread_mutex_unlock(&pool.lock);

    pthread_cond_destroy(&job.turn_passed);
    pthread_cond_destroy(&job.all_finished);
    if (steps != local_steps)
        free(steps);
    return job.err;
}

void vmaf_parallel_in_turn(VmafParallelJob *job, unsigned chunk, void (*fn)(void *arg),
                           void *arg)
{
    pthread_mutex_lock(&pool.lock);
    post_step(job, chunk, fn, arg);
    while (job->turn <= chunk)
        pthread_cond_wait(&job->turn_passed, &pool.lock);
    pthread_mutex_unlock(&pool.lock);
}
