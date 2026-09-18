# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("compute_position")


def reference(extend_prefix_lens, extend_seq_lens, extend_seq_lens_sum):
    bs = extend_seq_lens.shape[0]
    has_prefix = extend_prefix_lens.shape[0] == bs
    cumsum = torch.cumsum(extend_seq_lens, dim=0)
    start = torch.zeros_like(cumsum)
    start[1:] = cumsum[:-1]
    positions = torch.empty(
        extend_seq_lens_sum, dtype=torch.int64, device=extend_seq_lens.device
    )
    for i in range(bs):
        s = int(extend_seq_lens[i])
        p = int(extend_prefix_lens[i]) if has_prefix else 0
        beg = int(start[i])
        positions[beg : beg + s] = torch.arange(
            p, p + s, dtype=torch.int64, device=extend_seq_lens.device
        )
    return positions, start.to(torch.int32)


def make_case(lengths=(0, 1, 511, 512, 513, 1025), has_prefix=True):
    bs = len(lengths)
    prefix = [i % 17 for i in range(bs)]
    vectors = [
        torch.tensor(values, dtype=torch.int32, device="cuda")
        for values in ([prefix], [lengths])
    ]
    total = int(sum(lengths))
    if has_prefix:
        prefix_arg = vectors[0]
    else:
        prefix_arg = torch.empty(0, torch.int32, device="cuda")
    return (prefix_arg, vectors[1], total)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class ComputePositionTest(unittest.TestCase):
    def check(self, args):
        expected = reference(*args)
        snapshots = [x.clone() for x in args[:2]]
        for name, module in MODULES:
            with self.subTest(module=name):
                actual = module.compute_position(*args)
                for got, want in zip(actual, expected):
                    torch.testing.assert_close(
                        got, want, rtol=0, atol=0
                    )
                for value, before in zip(args[:2], snapshots):
                    torch.testing.assert_close(
                        value, before, rtol=0, atol=0
                    )

    def test_basic_and_boundaries(self):
        for has_prefix in (False, True):
            with self.subTest(has_prefix=has_prefix):
                self.check(make_case(has_prefix=has_prefix))
                self.check(make_case(lengths=(0,), has_prefix=has_prefix))
                self.check(make_case(lengths=(1,), has_prefix=has_prefix))
                self.check(
                    make_case(lengths=(8193,), has_prefix=has_prefix)
                )

    def test_large_batch_striped(self):
        # Cross the striped dispatch boundary from both sides and hit
        # the >64-stripe row-widening path.
        for bs in (63, 64, 65, 1023, 1024, 1025, 2049):
            with self.subTest(bs=bs):
                self.check(
                    make_case(
                        lengths=[(i % 7) + 1 for i in range(bs)],
                        has_prefix=True,
                    )
                )
        self.check(
            make_case(lengths=[513] * 65, has_prefix=False)
        )

    def test_all_zero_and_empty(self):
        for lengths in ((), (0, 0, 0), (0, 1, 0)):
            self.check(make_case(lengths=lengths))


RELEASE_REQUIRED_TESTS = [
    "ComputePositionTest.test_basic_and_boundaries",
    "ComputePositionTest.test_large_batch_striped",
    "ComputePositionTest.test_all_zero_and_empty",
]


if __name__ == "__main__":
    unittest.main()
