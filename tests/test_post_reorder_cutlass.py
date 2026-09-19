# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("post_reorder_cutlass")


def reference(down_output, output, src2dst, topk_ids, topk_weights,
              num_local_experts, topk, num_tokens, hidden_size, scale):
    acc = torch.zeros(num_tokens, hidden_size, dtype=torch.float32, device=output.device)
    for i in range(topk):
        valid = (topk_ids[:, i] != num_local_experts).float()
        rows = down_output[src2dst[:, i].long()].float()
        acc += rows * (topk_weights[:, i].float() * valid)[:, None]
    return (acc * scale).to(output.dtype)


def make_case(tokens=64, topk=8, hidden=1024, experts=32, skip_ratio=0.2):
    down = torch.randn(tokens * topk, hidden, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(tokens, hidden, dtype=torch.bfloat16, device="cuda")
    perm = torch.randperm(tokens * topk, device="cuda").to(torch.int32)
    s2d = perm.reshape(tokens, topk)
    ids = torch.randint(0, experts, (tokens, topk), dtype=torch.int32, device="cuda")
    skip = torch.rand(tokens, topk, device="cuda") < skip_ratio
    ids[skip] = num_local_experts = experts
    w = torch.randn(tokens, topk, dtype=torch.float32, device="cuda").abs()
    return (down, out, s2d, ids, w, experts, topk, tokens, hidden, 2.7)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class PRCtest(unittest.TestCase):
    def check(self, args):
        want = reference(*args)
        for name, module in MODULES:
            with self.subTest(module=name):
                got = module.post_reorder_cutlass(*args)
                torch.testing.assert_close(got.float(), want.float(), rtol=2e-2, atol=2e-2)

    def test_shapes_and_skips(self):
        for tokens, topk, hidden in ((1, 4, 512), (64, 8, 1024), (513, 8, 5120), (7, 3, 1535)):
            with self.subTest(shape=(tokens, topk, hidden)):
                self.check(make_case(tokens, topk, hidden))
        args = list(make_case(128, 8, 1024))
        args[3][:, :] = args[5]  # all slots off-rank
        self.check(tuple(args))


RELEASE_REQUIRED_TESTS = [
    "PRCtest.test_shapes_and_skips",
]


if __name__ == "__main__":
    unittest.main()
