# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("fused_moe_dispatch_index")


def reference(topk_ids, num_local_experts, m_max):
    flat = topk_ids.reshape(-1)
    masked_m = torch.zeros(
        num_local_experts, dtype=torch.int32, device=flat.device
    )
    src2dst = torch.empty(flat.numel(), dtype=torch.int32, device=flat.device)
    counts = [0] * num_local_experts
    dst = []
    for e in flat.tolist():
        if e < 0:
            dst.append(0)
            continue
        dst.append(e * m_max + counts[e])
        counts[e] += 1
    src2dst.copy_(torch.tensor(dst, dtype=torch.int32, device=flat.device))
    masked_m.copy_(torch.tensor(counts, dtype=torch.int32, device=flat.device))
    return masked_m, src2dst


def make_case(
    tokens=129,
    topk=8,
    experts=16,
    m_max=257,
    pad_ratio=0.0,
):
    ids = torch.randint(
        0, experts, (tokens, topk), dtype=torch.int32, device="cuda"
    )
    if pad_ratio:
        mask = torch.rand(tokens, topk, device="cuda") < pad_ratio
        ids[mask] = -1
    return ids, experts, m_max


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class FusedMoeDispatchIndexTest(unittest.TestCase):
    def check(self, args):
        exp_m, exp_dst = reference(*args)
        snapshot = args[0].clone()
        for name, module in MODULES:
            with self.subTest(module=name):
                masked_m, src2dst = module.fused_moe_dispatch_index(*args)
                torch.testing.assert_close(masked_m, exp_m, rtol=0, atol=0)
                # Bucket-internal order comes from atomics; compare the
                # global multiset instead of element order (valid dst
                # values are unique because buckets are disjoint ranges).
                flat = args[0].reshape(-1)
                self.assertTrue(
                    torch.equal(
                        torch.sort(src2dst).values,
                        torch.sort(exp_dst).values,
                    ),
                    "src2dst multiset mismatch",
                )
                pad = flat < 0
                if pad.any():
                    self.assertTrue(torch.equal(src2dst[pad], exp_dst[pad]))
                torch.testing.assert_close(args[0], snapshot, rtol=0, atol=0)

    def test_basic_and_padding(self):
        for pad in (0.0, 0.25):
            with self.subTest(pad=pad):
                self.check(make_case(pad_ratio=pad))

    def test_shapes_and_grid_edges(self):
        for tokens, topk, experts, m_max in (
            (1, 1, 1, 8),
            (5, 4, 256, 64),
            (64, 16, 1, 2048),
            (8193, 8, 32, 1024),
        ):
            with self.subTest(tokens=tokens, topk=topk, experts=experts):
                self.check(
                    make_case(
                        tokens=tokens,
                        topk=topk,
                        experts=experts,
                        m_max=m_max,
                    )
                )

    def test_empty(self):
        self.check(make_case(tokens=0, topk=8, experts=4, m_max=16))
        self.check(make_case(tokens=4, topk=0, experts=4, m_max=16))

    def check_ownership(self, args):
        self.check(args)
        ids, experts, capacity = args
        flat = ids.reshape(-1)
        for name, module in MODULES:
            counts, dst = module.fused_moe_dispatch_index(*args)
            self.assertEqual(dst.dtype, torch.int32)
            self.assertEqual(counts.shape, (experts,))
            self.assertEqual(dst.shape, (flat.numel(),))
            for e in range(experts):
                selected = dst[flat == e]
                expected = e * capacity + torch.arange(
                    selected.numel(), dtype=torch.int32, device=flat.device
                )
                torch.testing.assert_close(
                    selected.sort().values, expected, rtol=0, atol=0
                )
            self.assertTrue(torch.all(dst[flat < 0] == 0))

    def test_owned_padding_and_extremes(self):
        for n in (1, 255, 256, 257, 1024, 8192, 65544):
            for experts in (1, 17, 64, 256):
                ids = torch.arange(n, device="cuda", dtype=torch.int32)
                ids = (ids % experts).reshape(n, 1)
                ids[::3] = -1
                self.check_ownership((ids, experts, n + 1))
        for n in (1, 257, 65544):
            ids = torch.full((n, 1), -1, dtype=torch.int32, device="cuda")
            self.check_ownership((ids, 17, n + 1))
            ids.fill_(16)
            self.check_ownership((ids, 17, n + 1))

    def test_forced_gridstride_poisoned_output(self):
        for n in (255, 256, 257, 1025, 65544):
            for grid in (1, 2):
                ids = torch.arange(n, device="cuda", dtype=torch.int32) % 17
                ids[::3] = -1
                for name, module in MODULES:
                    # Only the candidate promises an in-kernel pad write.
                    if name not in ("candidate", "generic"):
                        continue
                    backing = torch.full(
                        (n + 16,), -777, dtype=torch.int32, device="cuda"
                    )
                    dst = backing[8 : 8 + n]
                    counts = torch.zeros(17, dtype=torch.int32, device="cuda")
                    module._fused_moe_dispatch_index[(grid,)](
                        ids, dst, counts, n + 1, n, 17, BLOCK=256
                    )
                    self.assertTrue(torch.all(backing[:8] == -777))
                    self.assertTrue(torch.all(backing[-8:] == -777))
                    self.assertTrue(torch.all(dst[ids < 0] == 0))
                    for e in range(17):
                        selected = dst[ids == e]
                        self.assertEqual(int(counts[e]), selected.numel())
                        expected = e * (n + 1) + torch.arange(
                            selected.numel(), device="cuda", dtype=torch.int32
                        )
                        torch.testing.assert_close(
                            selected.sort().values, expected, rtol=0, atol=0
                        )

    def test_noncontiguous_ownership(self):
        ids = torch.arange(257 * 8, device="cuda", dtype=torch.int32)
        ids = (ids % 17).reshape(257, 8).t()
        ids[:, ::3] = -1
        self.assertFalse(ids.is_contiguous())
        self.check_ownership((ids, 17, ids.numel() + 1))


RELEASE_REQUIRED_TESTS = [
    "FusedMoeDispatchIndexTest.test_basic_and_padding",
    "FusedMoeDispatchIndexTest.test_shapes_and_grid_edges",
    "FusedMoeDispatchIndexTest.test_empty",
    "FusedMoeDispatchIndexTest.test_owned_padding_and_extremes",
    "FusedMoeDispatchIndexTest.test_forced_gridstride_poisoned_output",
    "FusedMoeDispatchIndexTest.test_noncontiguous_ownership",
]


if __name__ == "__main__":
    unittest.main()
