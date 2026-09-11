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


RELEASE_REQUIRED_TESTS = [
    "FusedMoeDispatchIndexTest.test_basic_and_padding",
    "FusedMoeDispatchIndexTest.test_shapes_and_grid_edges",
    "FusedMoeDispatchIndexTest.test_empty",
]


if __name__ == "__main__":
    unittest.main()
