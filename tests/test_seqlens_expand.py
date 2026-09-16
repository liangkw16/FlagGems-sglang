# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest import mock

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


def output_search_reference(args):
    # Literal task arithmetic stays int32, including wrap before clamp.
    # output_size avoids device-to-host length discovery in repeat_interleave.
    extend, seq, total, _ = args
    n = extend.numel()
    prefix = torch.cumsum(extend, dim=0, dtype=torch.int32) - extend
    req = torch.repeat_interleave(
        torch.arange(n, device=extend.device), extend, output_size=total
    )
    pos = torch.arange(total, dtype=torch.int32, device=extend.device)
    base = seq - extend + 1
    return (base[req] + (pos - prefix[req])).clamp(min=0)


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

    def test_strided_inputs(self):
        # E4 regression: the fused kernel's prefix accumulation used to
        # read extend without the element stride while the current-row
        # load used it, so strided views produced wrong bases. Both the
        # fused (n <= 1024) and scan (n > 1024) paths must honor es/ss.
        for qos in (
            (0, 1, 5, 128, 513),
            tuple((i % 37) + 1 for i in range(1500)),
        ):
            extend, seq, total, mq = make_case(qos=qos, seed=7)
            # dim=1 interleaves [value, filler] so [::2] is a stride-2
            # view carrying exactly the original values.
            ext_base = torch.stack(
                [extend, torch.full_like(extend, -1)], dim=1
            ).flatten()[::2]
            seq_base = torch.stack(
                [seq, torch.full_like(seq, -1)], dim=1
            ).flatten()[::2]
            assert ext_base.stride(0) == 2 and seq_base.stride(0) == 2
            assert torch.equal(ext_base, extend) and torch.equal(seq_base, seq)
            self.check((ext_base, seq_base, total, mq))

    def test_large_batch_two_path(self):
        # E4: n > 1024 takes the scan + expand-prefixed path; a mixed
        # qos/kvs profile keeps clamps, zero-length rows and the tail in
        # play on both kernels.
        g = torch.Generator().manual_seed(11)
        qos = torch.randint(0, 700, (2050,), generator=g).tolist()
        kvs = (
            torch.tensor(qos) + torch.randint(0, 64, (2050,), generator=g)
        ).tolist()
        self.check(make_case(qos=qos, kvs=kvs, seed=11))

    def test_prefix_shape_boundaries(self):
        # Powers of two change the fused reduction width; n > 1024
        # controls retain the separate scan. Include empty requests,
        # clamped starts, and a request spanning two output tiles.
        for n in (1, 3, 31, 32, 33, 513, 1023, 1024, 1025, 2050):
            qos = [(i * 7) % 19 for i in range(n)]
            qos[-1] = 1025
            kvs = [max(0, q + i % 7 - 3) for i, q in enumerate(qos)]
            with self.subTest(n=n):
                self.check(make_case(qos=qos, kvs=kvs))

    def check_output_search(self, args):
        expected = output_search_reference(args)
        snapshots = [args[0].clone(), args[1].clone()]
        for name, module in MODULES:
            with self.subTest(module=name):
                actual = module.seqlens_expand(*args)
                self.assertEqual(actual.dtype, torch.int32)
                self.assertEqual(actual.shape, (args[2],))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for value, snapshot in zip(args[:2], snapshots):
                    torch.testing.assert_close(value, snapshot, rtol=0, atol=0)

    def test_output_search_large_patterns(self):
        for n in (1025, 2048, 4096):
            mixed = [i % 5 if i % 7 == 0 else 0 for i in range(n)]
            mixed[0], mixed[-1] = 0, 257
            skew = [0] * n
            skew[1], skew[n // 2], skew[-2] = 1, 1025, 4097
            single = [0] * n
            single[n // 2] = 1
            dense = [1] * n
            dense[-1] = 2
            for profile, qos in (
                ("zero_runs", mixed),
                ("uneven", skew),
                ("dense_tail", dense),
                ("all_zero", [0] * n),
                ("single", single),
            ):
                kvs = [max(0, q + i % 7 - 3) for i, q in enumerate(qos)]
                with self.subTest(n=n, profile=profile):
                    self.check_output_search(make_case(qos=qos, kvs=kvs))

        # New output tile B-1/B/B+1, plus one output and long zero runs.
        for total in (1, 255, 256, 257):
            qos = [0] * 1025
            qos[-1] = total
            self.check_output_search(make_case(qos=qos, kvs=[0] * 1025))
        # Force the new output kernel through three iterations and a tail.
        qos = [0] * 1025
        qos[1], qos[-2] = 256, 257
        args = make_case(qos=qos, kvs=[0] * 1025)
        ext, seq, total, _ = args
        expected = output_search_reference(args)
        for name, module in MODULES:
            with self.subTest(module=name, manual_output_grid=1):
                prefix = torch.empty(1025, dtype=torch.int32, device="cuda")
                out = torch.full_like(expected, -123)
                module._seqlens_prefix[(1,)](ext, prefix, 1025, 1, BLOCK=1024)
                module._seqlens_expand_grouped[(1,)](
                    ext,
                    seq,
                    prefix,
                    out,
                    1,
                    1,
                    1025,
                    (1025 + 3) // 4,
                    GROUP=4,
                    BLOCK=32,
                )
                torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_output_search_strides_int32_wrap(self):
        lo, hi = -(2**31), 2**31 - 1
        for n in (1025, 2048, 4096):
            qos = [0] * n
            qos[0], qos[1], qos[n // 2], qos[-2] = 2, 3, 257, 259
            kvs = [lo, lo + 1] + [hi if i % 2 else lo for i in range(n - 2)]
            extend, seq, total, qmax = make_case(qos=qos, kvs=kvs)
            ext_storage = torch.full(
                (2 * n + 3,), -999, dtype=torch.int32, device="cuda"
            )
            seq_storage = torch.full(
                (3 * n + 5,), -777, dtype=torch.int32, device="cuda"
            )
            ext_view = ext_storage[1 : 1 + 2 * n : 2]
            seq_view = seq_storage[2 : 2 + 3 * n : 3]
            ext_view.copy_(extend)
            seq_view.copy_(seq)
            self.assertEqual(ext_view.stride(0), 2)
            self.assertEqual(seq_view.stride(0), 3)
            self.assertEqual(ext_view.storage_offset(), 1)
            self.assertEqual(seq_view.storage_offset(), 2)
            args = (ext_view, seq_view, total, qmax + 4096)
            expected = output_search_reference(args)
            torch.testing.assert_close(
                expected[:5],
                torch.tensor(
                    [hi, 0, hi, 0, 0], dtype=torch.int32, device="cuda"
                ),
                rtol=0,
                atol=0,
            )
            with self.subTest(n=n):
                self.check_output_search(args)
                one = torch.tensor([1], dtype=torch.int32, device="cuda")
                three = torch.tensor([3], dtype=torch.int32, device="cuda")
                self.check_output_search(
                    (one.expand(n), three.expand(n), n, 1)
                )

    def test_output_search_request_grid_boundary(self):
        for n in (65535, 65536, 65537):
            qos = [0] * n
            qos[0], qos[n // 2], qos[-1] = 1, 256, 257
            kvs = [0] * n
            with self.subTest(n=n):
                self.check_output_search(make_case(qos=qos, kvs=kvs))

    def test_fallback_row_tile_grid_stride(self):
        qos = (
            0,
            1,
            1023,
            1024,
            1025,
            2047,
            2048,
            2049,
            3073,
            4096,
            4097,
            0,
            1,
            2,
            3,
            5121,
            9,
        )
        args = make_case(qos=qos, kvs=[max(0, q - 7) for q in qos])
        expected = output_search_reference(args)
        extend, seq, total, qmax = args
        for name, module in MODULES:
            with self.subTest(module=name):
                prefix = torch.empty(
                    len(qos), dtype=torch.int32, device="cuda"
                )
                out = torch.full_like(expected, -123)
                module._seqlens_prefix[(1,)](
                    extend, prefix, len(qos), extend.stride(0), BLOCK=1024
                )
                call = [
                    extend,
                    seq,
                    prefix,
                    out,
                    extend.stride(0),
                    seq.stride(0),
                    qmax,
                ]
                if hasattr(module, "_seqlens_expand_p_safe"):
                    # Force both row and tile iterations on the new fallback.
                    call.append(len(qos))
                    grid = (3, 2)
                else:
                    # E4 baseline lacks row-stride; same numeric input uses
                    # its supported request grid, still with tile stepping.
                    grid = (len(qos), 2)
                kernel = getattr(
                    module, "_seqlens_expand_p_safe", module._seqlens_expand_p
                )
                kernel[grid](*call, BLOCK=1024)
                torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_wide_prefix_and_wrapped_values(self):
        imax = 2**31 - 1
        extend = torch.tensor([imax, 1, 1], dtype=torch.int32, device="cuda")
        expected = torch.tensor(
            [0, imax, imax + 1], dtype=torch.int64, device="cuda"
        )
        for name, module in MODULES:
            with self.subTest(module=name):
                # Scalar value 1 is specialized by Triton: it must also
                # work in every newly added wide-index conversion.
                one = torch.ones(1, dtype=torch.int32, device="cuda")
                one_prefix = torch.empty(1, dtype=torch.int64, device="cuda")
                one_out = torch.empty_like(one)
                module._seqlens_prefix_wide[(1,)](
                    one, one_prefix, 1, 1, BLOCK=1024
                )
                module._seqlens_expand_p_safe[(1, 1)](
                    one, one, one_prefix, one_out, 1, 1, 1, 1, BLOCK=1024
                )
                torch.testing.assert_close(one_out, one, rtol=0, atol=0)
                prefix = torch.empty(3, dtype=torch.int64, device="cuda")
                module._seqlens_prefix_wide[(1,)](
                    extend, prefix, 3, 1, BLOCK=1024
                )
                torch.testing.assert_close(prefix, expected, rtol=0, atol=0)
                # A tiny allocation also tests carry across the 1024-row
                # chunk, including the tail after a wrapped int32 sum.
                carry_input = torch.tensor(
                    [imax] + [0] * 1022 + [1, 1],
                    dtype=torch.int32,
                    device="cuda",
                )
                carry_prefix = torch.empty(
                    1025, dtype=torch.int64, device="cuda"
                )
                module._seqlens_prefix_wide[(1,)](
                    carry_input, carry_prefix, 1025, 1, BLOCK=1024
                )
                carry_expected = (
                    torch.cumsum(carry_input, dim=0, dtype=torch.int64)
                    - carry_input
                )
                torch.testing.assert_close(
                    carry_prefix, carry_expected, rtol=0, atol=0
                )
                self.assertEqual(int(carry_prefix[-1]), imax + 1)
                # Small output fixture exercises i64 physical prefix with
                # int32 wrap in values, without allocating >8GB output.
                args = make_case(qos=(2, 3), kvs=(-(2**31), -(2**31) + 1))
                ext, seq, total, qmax = args
                prefix = torch.empty(2, dtype=torch.int64, device="cuda")
                out = torch.empty(total, dtype=torch.int32, device="cuda")
                module._seqlens_prefix_wide[(1,)](
                    ext, prefix, 2, 1, BLOCK=1024
                )
                module._seqlens_expand_p_safe[(1, 1)](
                    ext, seq, prefix, out, 1, 1, qmax, 2, BLOCK=1024
                )
                torch.testing.assert_close(
                    out, output_search_reference(args), rtol=0, atol=0
                )

    def test_output_search_metadata_bounds(self):
        # Capture dispatch without allocating large output/storage buffers.
        class TensorMeta:
            ndim = 1
            dtype = torch.int32
            device = "cuda"

            def __init__(self, n, stride):
                self.n, self.step = n, stride

            def numel(self):
                return self.n

            def stride(self, axis):
                assert axis == 0
                return self.step

        class Capture:
            def __init__(self):
                self.calls = []

            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    self.calls.append((grid, args, kwargs))

                return launch

        imax = 2**31 - 1
        cases = (
            (1024, 257, 257, 1, 1),
            (1024, 257, 63 * 1024, 1, 1),
            (1024, 257, 64 * 1024, 1, 1),
            (1024, 257, 65 * 1024, 1, 1),
            (257, 257, 255 * 1024, 1, 1),
            (258, 258, 255 * 1024, 1, 1),
            (1025, 0, 1, 1, 1),
            (1025, 17, 17, 1, 1),
            (1025, imax + 1, 17, 1, 1),
            (65537, 257, 17, 1, 1),
            (1025, 17, 17, imax, 1),
            (1025, 65535 * 256, 1024, 1, 1),
            (1025, 65535 * 256 + 1, 1024, 1, 1),
            (1025, imax - 1, 1024, 1, 1),
            (1025, imax, 1024, 1, 1),
            (1025, imax + 1, 1024, 1, 1),
            (2, imax + 1, imax, 1, 1),
            (65535, imax + 1, 261121, 1, 1),
            (65536, imax + 1, 261121, 1, 1),
            (65537, imax + 1, 261121, 1, 1),
            ((imax // 1024) * 1024, 1, 1, 1, 1),
            ((imax // 1024) * 1024 + 1, 1, 1, 1, 1),
            (imax, 1, 1, 1, 1),
            (imax + 1, 1, 1, 1, 1),
            (2, 1, 1, imax, 1),
            (3, 1, 1, imax, 1),
            (3, 1, 1, 1, imax),
            (2, 1, imax - 1023, 1, 1),
            (2, 1, imax - 1022, 1, 1),
        )
        for name, module in MODULES:
            for n, total, qmax, es, ss in cases:
                with self.subTest(
                    module=name, n=n, total=total, qmax=qmax, es=es, ss=ss
                ):
                    names = (
                        "_seqlens_expand",
                        "_seqlens_prefix",
                        "_seqlens_prefix_wide",
                        "_seqlens_expand_p",
                        "_seqlens_expand_p_safe",
                        "_seqlens_expand_grouped",
                    )
                    captures = {key: Capture() for key in names}
                    fused, prefix, wide, old, safe, grouped = captures.values()
                    output = object()
                    allocation = mock.Mock(return_value=output)
                    with (
                        mock.patch.multiple(module, **captures),
                        mock.patch.object(module.torch, "empty", allocation),
                    ):
                        actual = module.seqlens_expand(
                            TensorMeta(n, es), TensorMeta(n, ss), total, qmax
                        )
                    self.assertIs(actual, output)
                    if total == 0:
                        self.assertFalse(
                            any(c.calls for c in captures.values())
                        )
                        continue
                    wide_domain = (
                        total > imax
                        or n > (imax // 1024) * 1024
                        or (n - 1) * max(es, ss) > imax
                        or qmax > imax - 1023
                    )
                    tiles = min((max(1, qmax) + 1023) // 1024, 255)
                    safe_domain = wide_domain or n * tiles > 65535
                    safe_x = min(n, 65535)
                    safe_grid = (safe_x, min(tiles, 65535 // safe_x))
                    if n <= 1024 and not safe_domain:
                        self.assertEqual(fused.calls[0][0], (n, tiles))
                        self.assertEqual(
                            sum(len(c.calls) for c in captures.values()), 1
                        )
                    else:
                        self.assertFalse(fused.calls)
                        self.assertEqual(
                            allocation.call_args_list[1].kwargs["dtype"],
                            torch.int64 if wide_domain else torch.int32,
                        )
                        scanner = wide if wide_domain else prefix
                        self.assertEqual(scanner.calls[0][0], (1,))
                        self.assertFalse(
                            (prefix if wide_domain else wide).calls
                        )
                        if n > 1024 and qmax <= 32:
                            self.assertEqual(
                                grouped.calls[0][0],
                                (min((n + 3) // 4, 65535),),
                            )
                            self.assertEqual(grouped.calls[0][2]["GROUP"], 4)
                            self.assertEqual(grouped.calls[0][2]["BLOCK"], 32)
                            self.assertFalse(old.calls or safe.calls)
                        elif safe_domain:
                            self.assertEqual(safe.calls[0][0], safe_grid)
                            self.assertEqual(safe.calls[0][1][-1], n)
                            self.assertFalse(old.calls or grouped.calls)
                        else:
                            self.assertEqual(old.calls[0][0], (n, tiles))
                            self.assertFalse(safe.calls or grouped.calls)
                        self.assertEqual(
                            sum(len(c.calls) for c in captures.values()), 2
                        )
                    for capture in captures.values():
                        for grid, _, _ in capture.calls:
                            product = 1
                            for axis in grid:
                                product *= axis
                            self.assertLessEqual(product, 65535)

    def test_grouped_short_hint_and_strides(self):
        # New route: lengths, prefix duplicates, int32 wrap, independent strides.
        for n in (1025, 4096):
            for qmax in (1, 8, 17, 31, 32):
                qos = [0 if i % 5 == 0 else i % (qmax + 1) for i in range(n)]
                qos[-1] = qmax
                kvs = [-(2**31) if i % 3 == 0 else 2**31 - 1 for i in range(n)]
                ext, seq, total, _ = make_case(qos=qos, kvs=kvs)
                ext_storage = torch.full(
                    (2 * n + 1,), -111, dtype=torch.int32, device="cuda"
                )
                seq_storage = torch.full(
                    (3 * n + 2,), -222, dtype=torch.int32, device="cuda"
                )
                es, ss = ext_storage[1::2], seq_storage[2::3]
                es.copy_(ext)
                ss.copy_(seq)
                self.check_output_search((es, ss, total, qmax))

    def test_grouped_actual_lengths_ignore_short_hint(self):
        # Underreported hint must not omit long rows in the new grouped route.
        for n in (1025, 4096):
            for qlong in (33, 257, 8193):
                qos = [0] * n
                qos[1], qos[-1] = 1, qlong
                ext, seq, total, _ = make_case(qos=qos, kvs=[0] * n)
                for hint in (0, 1, 17, 32):
                    self.check_output_search((ext, seq, total, hint))
                    # Nonzero expected values also expose omitted writes.
                    self.check_output_search((ext, ext + 17, total, hint))

    def test_grouped_wide_prefix_small_allocation(self):
        # Execute the wide scratch route without pretending to allocate >8GB.
        qos = [0, 1, 2, 0, 33]
        args = make_case(qos=qos, kvs=[-(2**31)] * len(qos))
        ext, seq, total, _ = args
        for _, module in MODULES:
            prefix = torch.empty(len(qos), dtype=torch.int64, device="cuda")
            out = torch.full((total,), -777, dtype=torch.int32, device="cuda")
            module._seqlens_prefix_wide[(1,)](
                ext, prefix, len(qos), 1, BLOCK=1024
            )
            module._seqlens_expand_grouped[(1,)](
                ext,
                seq,
                prefix,
                out,
                1,
                1,
                len(qos),
                (len(qos) + 3) // 4,
                GROUP=4,
                BLOCK=32,
            )
            torch.testing.assert_close(
                out, output_search_reference(args), rtol=0, atol=0
            )


RELEASE_REQUIRED_TESTS = [
    "SeqlensExpandTest.test_grouped_short_hint_and_strides",
    "SeqlensExpandTest.test_grouped_actual_lengths_ignore_short_hint",
    "SeqlensExpandTest.test_grouped_wide_prefix_small_allocation",
    "SeqlensExpandTest.test_basic_and_negative_starts",
    "SeqlensExpandTest.test_tail_and_single",
    "SeqlensExpandTest.test_empty",
    "SeqlensExpandTest.test_strided_inputs",
    "SeqlensExpandTest.test_large_batch_two_path",
    "SeqlensExpandTest.test_prefix_shape_boundaries",
    "SeqlensExpandTest.test_output_search_large_patterns",
    "SeqlensExpandTest.test_output_search_strides_int32_wrap",
    "SeqlensExpandTest.test_output_search_request_grid_boundary",
    "SeqlensExpandTest.test_fallback_row_tile_grid_stride",
    "SeqlensExpandTest.test_wide_prefix_and_wrapped_values",
    "SeqlensExpandTest.test_output_search_metadata_bounds",
]


if __name__ == "__main__":
    unittest.main()
