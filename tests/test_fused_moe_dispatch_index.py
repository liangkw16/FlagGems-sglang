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

    def test_pair_patterns_and_tails(self):
        pairs = ((3, 3), (3, 5), (-1, 3), (3, -2), (-1, -2), (0, 0))
        for n in (1, 127, 128, 129, 255, 256, 257, 513):
            values = [-1] * n
            for base in range(0, n, 256):
                for lane in range(128):
                    a, b = pairs[lane % len(pairs)]
                    if base + lane < n:
                        values[base + lane] = a
                    if base + lane + 128 < n:
                        values[base + lane + 128] = b
            ids = torch.tensor(
                values, dtype=torch.int32, device="cuda"
            ).reshape(n, 1)
            for capacity in (0, 1, n + 1):
                with self.subTest(n=n, capacity=capacity):
                    self.check_ownership((ids, 7, capacity))

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

    def check_expert_routes(self, ids, experts, capacity):
        flat = ids.reshape(-1)
        valid = flat >= 0
        owners = flat[valid].to(torch.int64)
        expected = torch.bincount(owners, minlength=experts).to(torch.int32)
        snapshot = ids.clone()
        for name, module in MODULES:
            with self.subTest(module=name):
                counts, dst = module.fused_moe_dispatch_index(
                    ids, experts, capacity
                )
                self.assertEqual(counts.dtype, torch.int32)
                self.assertEqual(dst.dtype, torch.int32)
                self.assertEqual(counts.shape, (experts,))
                self.assertEqual(dst.shape, (flat.numel(),))
                torch.testing.assert_close(counts, expected, rtol=0, atol=0)
                # Each route must stay in its owner's interval. Distinct
                # in-range destinations then cover every occupied bucket.
                ranks = dst[valid].to(torch.int64) - owners * capacity
                self.assertTrue(torch.all(ranks >= 0))
                self.assertTrue(torch.all(ranks < expected[owners]))
                self.assertEqual(
                    torch.unique(dst[valid]).numel(), owners.numel()
                )
                self.assertTrue(torch.all(dst[~valid] == 0))
                torch.testing.assert_close(ids, snapshot, rtol=0, atol=0)

    def test_overlapping_capacity_and_int32_edge(self):
        ids = torch.tensor(
            [[0, 1, 0], [1, 0, -1]], dtype=torch.int32, device="cuda"
        )
        for capacity in (0, 1):
            self.check_ownership((ids, 2, capacity))
        ids = torch.tensor([[1, 0, 1, 0]], dtype=torch.int32, device="cuda")
        self.check_ownership((ids, 2, 2147483646))

    def test_expert_grid_boundaries(self):
        for experts in (65535, 65536, 65537):
            with self.subTest(experts=experts):
                ids = torch.tensor(
                    [[experts - 1]], dtype=torch.int32, device="cuda"
                )
                self.check_expert_routes(ids, experts, 1)
        # Exercise prefix accumulation across route blocks and a middle
        # expert as well as both sides of the expert-grid boundary.
        ids = torch.tensor(
            [0, 32768, 65534, 65535, 65536, -1] * 6,
            dtype=torch.int32,
            device="cuda",
        ).reshape(6, 6)
        self.check_expert_routes(ids, 65537, ids.numel() + 1)

    def test_empty_expert_domains(self):
        for experts in (0, 65535, 65536, 65537):
            for shape in ((0, 4), (4, 0), (2, 3)):
                with self.subTest(experts=experts, shape=shape):
                    ids = torch.full(
                        shape, -1, dtype=torch.int32, device="cuda"
                    )
                    self.check_expert_routes(ids, experts, 8)

    def test_forced_expert_grid_stride(self):
        n, experts, padded_experts, blocks = 65, 7, 64, 3
        ids = torch.arange(n, dtype=torch.int32, device="cuda") % experts
        ids[::4] = -1
        block_ids = torch.arange(n, device="cuda") // 32
        keys = block_ids * experts + ids
        expected = (
            torch.bincount(keys[ids >= 0], minlength=blocks * experts)
            .reshape(blocks, experts)
            .to(torch.int32)
        )
        expected_prefix = expected.cumsum(0).to(torch.int32) - expected
        for name, module in MODULES:
            if name == "generic":
                continue
            for grid in (1, 2):
                with self.subTest(module=name, grid=grid):
                    counts = torch.full(
                        (blocks, padded_experts),
                        -777,
                        dtype=torch.int32,
                        device="cuda",
                    )
                    prefix = torch.full_like(counts, -777)
                    masked_m = torch.full(
                        (experts,), -777, dtype=torch.int32, device="cuda"
                    )
                    launch = (
                        {"num_warps": 1, "num_stages": 1}
                        if name == "kunlunxin"
                        else {}
                    )
                    if name == "kunlunxin":
                        module._dispatch_counts[(grid,)](
                            ids,
                            counts,
                            n,
                            experts,
                            padded_experts,
                            blocks,
                            BLOCK=32,
                            EXPERT_TILES=(experts + grid - 1) // grid,
                            **launch,
                        )
                    else:
                        module._dispatch_counts[(grid,)](
                            ids,
                            counts,
                            n,
                            padded_experts,
                            blocks,
                            BLOCK=32,
                            E_TILE=64,
                            **launch,
                        )
                    torch.testing.assert_close(
                        counts[:, :experts], expected, rtol=0, atol=0
                    )
                    module._dispatch_prefix[(grid,)](
                        counts,
                        prefix,
                        masked_m,
                        experts,
                        padded_experts,
                        blocks,
                        EXPERT_TILES=(experts + grid - 1) // grid,
                        **launch,
                    )
                    torch.testing.assert_close(
                        prefix[:, :experts], expected_prefix, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        masked_m,
                        expected.sum(0).to(torch.int32),
                        rtol=0,
                        atol=0,
                    )
                    self.assertTrue(torch.all(prefix[:, experts:] == -777))


RELEASE_REQUIRED_TESTS = [
    "FusedMoeDispatchIndexTest.test_overlapping_capacity_and_int32_edge",
    "FusedMoeDispatchIndexTest.test_basic_and_padding",
    "FusedMoeDispatchIndexTest.test_shapes_and_grid_edges",
    "FusedMoeDispatchIndexTest.test_empty",
    "FusedMoeDispatchIndexTest.test_owned_padding_and_extremes",
    "FusedMoeDispatchIndexTest.test_pair_patterns_and_tails",
    "FusedMoeDispatchIndexTest.test_forced_gridstride_poisoned_output",
    "FusedMoeDispatchIndexTest.test_noncontiguous_ownership",
    "FusedMoeDispatchIndexTest.test_expert_grid_boundaries",
    "FusedMoeDispatchIndexTest.test_empty_expert_domains",
    "FusedMoeDispatchIndexTest.test_forced_expert_grid_stride",
]


if __name__ == "__main__":
    unittest.main()
