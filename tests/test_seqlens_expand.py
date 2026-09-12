# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("seqlens_expand")


def reference(extend_seq_lens, seq_lens, total_len, max_q_len):
    device = extend_seq_lens.device
    n = extend_seq_lens.shape[0]
    offsets = torch.zeros(n + 1, dtype=torch.int32, device=device)
    torch.cumsum(extend_seq_lens, dim=0, out=offsets[1:])
    out = torch.empty(total_len, dtype=torch.int32, device=device)
    for i in range(n):
        qo = int(extend_seq_lens[i])
        kv = int(seq_lens[i])
        beg = int(offsets[i])
        out[beg : beg + qo] = torch.clamp(
            kv - qo + 1 + torch.arange(qo, dtype=torch.int32, device=device),
            min=0,
        )
    return out


def make_case(qos=(0, 1, 5, 128, 513), kvs=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    if kvs is None:
        kvs = [q + int(torch.randint(0, 64, (1,), generator=g)) for q in qos]
    total = sum(qos)
    extend = torch.tensor(qos, dtype=torch.int32, device="cuda")
    seq = torch.tensor(kvs, dtype=torch.int32, device="cuda")
    return extend, seq, total, max(qos) if qos else 1


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class SeqlensExpandTest(unittest.TestCase):
    def check(self, args):
        expected = reference(*args)
        snapshots = [args[0].clone(), args[1].clone()]
        for name, module in MODULES:
            with self.subTest(module=name):
                actual = module.seqlens_expand(*args)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(
                    args[0], snapshots[0], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    args[1], snapshots[1], rtol=0, atol=0
                )

    def test_basic_and_negative_starts(self):
        self.check(make_case(qos=(0, 1, 5, 128, 513)))
        # kv < qo forces clamped (negative) starts mid-request
        self.check(make_case(qos=(4, 7, 64), kvs=(1, 7, 3)))
        self.check(make_case(qos=(16, 16), kvs=(15, 0)))

    def test_tail_and_single(self):
        self.check(make_case(qos=(1,), kvs=(0,)))
        self.check(make_case(qos=(1025, 2048)))
        self.check(make_case(qos=(3, 0, 0, 9), kvs=(2, 100, 0, 1)))

    def test_empty(self):
        self.check(make_case(qos=()))
        self.check(
            (
                torch.zeros(0, dtype=torch.int32, device="cuda"),
                torch.zeros(0, dtype=torch.int32, device="cuda"),
                0,
                1,
            )
        )


RELEASE_REQUIRED_TESTS = [
    "SeqlensExpandTest.test_basic_and_negative_starts",
    "SeqlensExpandTest.test_tail_and_single",
    "SeqlensExpandTest.test_empty",
]


if __name__ == "__main__":
    unittest.main()
