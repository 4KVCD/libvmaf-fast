// Derived from libvmaf (https://github.com/Netflix/vmaf): integer_vif.h,
// Copyright 2016-2020 Netflix, Inc., licensed under the BSD+Patent License
// (https://opensource.org/licenses/BSDplusPatent); see LICENSE at the root
// of this repository.
//
// vif_filter1d_table of libvmaf's integer_vif.h: the Gaussian of each scale.
// The vertical pass's results for a pixel, which the horizontal pass filters
// along the rows: the two means (16 bits each) in one word, the three sums of
// squares and products (32 bits each), and the next scale's two samples (16
// bits each) in one word. Eight words a pixel, each value in its own, were
// 60% more memory for the horizontal pass to read 17 times over.
#define VIF_TMP_WORDS 5

static const uint kVifFilter[4][18] = {
    { 489, 935, 1640, 2640, 3896, 5274, 6547, 7455, 7784, 7455, 6547, 5274, 3896, 2640, 1640, 935, 489, 0 },
    { 1244, 3663, 7925, 12590, 14692, 12590, 7925, 3663, 1244, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 3571, 16004, 26386, 16004, 3571, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 10904, 43728, 10904, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
};
