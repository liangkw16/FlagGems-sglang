# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("unpad_draft_extend_output")


def reference(raw_out, cu, lens, total):
    bs = lens.shape[0]
    out = torch.empty((total, raw_out.shape[2], raw_out.shape[3]), dtype=raw_out.dtype, device=raw_out.device)
    for b in range(bs):
        s = int(lens[b])
        beg = int(cu[b])
        out[beg : beg + s] = raw_out[b, :s]
    return out


def make_case(lens=(0, 3, 1, 5), heads=8, dim=128, tpb=None):
    bs = len(lens)
    if tpb is None:
        tpb = max(lens) + 2
    cum = [0]
    for n in lens:
        cum.append(cum[-1] + n)
    raw = torch.randn(bs, tpb, heads, dim, dtype=torch.bfloat16, device="cuda")
    return (
        raw,
        torch.tensor(cum, dtype=torch.int32, device="cuda"),
        torch.tensor(lens, dtype=torch.int32, device="cuda"),
        cum[-1],
    )


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class UnpadTest(unittest.TestCase):
    def check(self, args):
        want = reference(*args)
        for name, module in MODULES:
            with self.subTest(module=name):
                got = module.unpad_draft_extend_output(*args)
                torch.testing.assert_close(got.view(torch.uint8), want.contiguous().view(torch.uint8), rtol=0, atol=0)

    def test_u32_fallback_conditions(self):
        # H*D even but dim odd must take the element path (the u32 view
        # halves the last dim, not the span); an odd storage offset on a
        # contiguous slice must also fall back.
        self.check(make_case(lens=(1, 3), heads=2, dim=3, tpb=4))
        self.check(make_case(lens=(2,), heads=6, dim=5, tpb=2))
        args = make_case(lens=(1, 2), heads=2, dim=4, tpb=3)
        base = torch.randn(
            3, args[0].shape[1], args[0].shape[2], args[0].shape[3],
            dtype=args[0].dtype, device="cuda",
        )
        sliced = base[1:]
        sliced.copy_(args[0])
        self.check((sliced, *args[1:]))

    def test_strided_len_tensors(self):
        # lens/cum sliced views must keep semantics (kernel takes strides).
        lens = (0, 3, 1, 5, 2)
        heads, dim, tpb = 8, 128, 8
        cum = [0]
        for n in lens:
            cum.append(cum[-1] + n)
        raw = torch.randn(5, tpb, heads, dim, dtype=torch.bfloat16, device="cuda")
        # interleave: real values at even indices, filler at odd, so the
        # ::2 views carry exactly the real lens/cum entries
        pad = 5
        dense_lens = [0] * (len(lens) + pad)
        dense_cum = [0] * (len(cum) + pad)
        for i, v in enumerate(lens):
            dense_lens[2 * i] = v
        for i, v in enumerate(cum):
            dense_cum[2 * i] = v
        self.check(
            (
                raw,
                torch.tensor(dense_cum, dtype=torch.int32, device="cuda")[::2],
                torch.tensor(dense_lens, dtype=torch.int32, device="cuda")[::2],
                cum[-1],
            )
        )

    def test_ragged_and_boundaries(self):
        self.check(make_case())
        self.check(make_case(lens=(0,), tpb=1))
        self.check(make_case(lens=(tpb_len := 64,), tpb=64, heads=2, dim=64))
        self.check(make_case(lens=(7,) * 40, heads=4, dim=96, tpb=7))

    def test_persistent_rotation(self):
        # e21 ascend semantics: grid capped at 64 programs rotating over
        # segments (p, p+num_programs, ...). bs > 64 forces some programs
        # to serve several segments; ragged lens with zeros exercise the
        # skipped segment and sub-BLOCK tails; the wide-span case crosses
        # multiple BLOCK=16384 tiles with a non-divisible remainder.
        ragged = tuple((i * 7) % 9 for i in range(100))  # 0..8, zeros at i%9==0
        self.check(make_case(lens=ragged, heads=8, dim=128, tpb=9))
        lens = [0, 33] + [64] * 30 + [17]
        self.check(make_case(lens=tuple(lens), heads=16, dim=128, tpb=64))

    def test_unmasked_main_masked_tail_split(self):
        # e21r ascend semantics: whole BLOCK=16384 tiles stream unmasked
        # and only the remainder drains through the BLOCK_TAIL=2048 fp32
        # masked loop. With span=1024 (heads=8, dim=128) the lens below
        # walk every split shape: 41 -> 41984 = 2 full blocks + 9216
        # remainder (4 full tail tiles + one partial); 48 -> 49152 = 3
        # full blocks exactly (zero-length tail must write nothing);
        # 35 -> 3072 remainder (one full + one partial tail tile); 34 ->
        # 2048 remainder (exactly one tail tile, all-true mask); 1 ->
        # 1024 < BLOCK_TAIL (main loop runs zero times); 0 -> both loops
        # skip the segment. An odd span keeps the element path honest
        # through the same split.
        self.check(make_case(lens=(41, 48, 35, 34, 1, 0), heads=8, dim=128, tpb=64))
        self.check(make_case(lens=(41,), heads=2, dim=3, tpb=64))

    def test_masked_tail_no_other_prefill(self):
        # e22 ascend semantics: the masked load carries NO other=0
        # prefill (chip-rulesets.md:25 - the prefill serialises MTE2 on
        # Ascend), so masked-out lanes hold undef values that must never
        # reach memory: the store carries the same mask m. With span=1024
        # (heads=8, dim=128) one BLOCK=16384 tile is 16 tokens, so every
        # lens % 16 != 0 forces a partial final tile whose undef tail
        # sits immediately before the NEXT segment's rows - a dropped or
        # loosened store mask would overwrite the following segment's
        # correct prefix and fail the byte-exact check. 64 is the
        # exact-multiple control (65536 = 4 tiles exactly, zero-length
        # tail must write nothing extra); 1 is the single-row all-tail
        # minimum; 63 leaves a near-full 15360-element tail; tpb=64 with
        # small lens also launches idle tile programs whose loop never
        # runs. The odd-span case keeps the element path honest through
        # the same undef-tail exposure.
        self.check(make_case(lens=(33, 17, 1, 64, 40, 63, 15), heads=8, dim=128, tpb=64))
        self.check(make_case(lens=(41,), heads=2, dim=3, tpb=64))


RELEASE_REQUIRED_TESTS = [
    "UnpadTest.test_u32_fallback_conditions",
    "UnpadTest.test_ragged_and_boundaries",
    "UnpadTest.test_persistent_rotation",
    "UnpadTest.test_unmasked_main_masked_tail_split",
    "UnpadTest.test_masked_tail_no_other_prefill",
]


if __name__ == "__main__":
    unittest.main()
