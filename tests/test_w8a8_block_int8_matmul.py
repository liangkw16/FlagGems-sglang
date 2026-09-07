# Copyright 2026 FlagOS Contributors
# (same license)
import importlib.util
import unittest
from pathlib import Path

import torch

MODULE_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "flaggems_sglang"
    / "ops"
    / "w8a8_block_int8_matmul.py"
)
SPEC = importlib.util.spec_from_file_location("w8a8_module", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def reference(A, B, As, Bs, block_size, output_dtype):
    A = A.to(torch.float32)
    B = B.to(torch.float32)
    block_n, block_k = block_size
    M, K = A.shape
    N, _ = B.shape
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k
    C = torch.zeros(M, N, dtype=torch.float32, device=A.device)
    for i in range(k_tiles):
        k_lo, k_hi = i * block_k, min((i + 1) * block_k, K)
        a_tile = A[:, k_lo:k_hi]
        a_s = As[:, i : i + 1]
        for j in range(n_tiles):
            n_lo, n_hi = j * block_n, min((j + 1) * block_n, N)
            b_tile = B[n_lo:n_hi, k_lo:k_hi]
            s = a_s * Bs[j, i]
            C[:, n_lo:n_hi] += torch.matmul(a_tile, b_tile.t()) * s
    return C.to(output_dtype)


def make_case(M, N, K, block_n=128, block_k=128, dtype=torch.float16, seed=0):
    # Platform contract: int8 operands, fp32 scales; dtype param is output dtype.
    g = torch.Generator().manual_seed(seed)
    A_int = torch.randint(
        -128, 128, (M, K), dtype=torch.int8, generator=g
    ).cuda()
    B_int = torch.randint(
        -128, 128, (N, K), dtype=torch.int8, generator=g
    ).cuda()
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k
    As = torch.randn(M, k_tiles, generator=g).cuda().abs() * 0.01
    Bs = torch.randn(n_tiles, k_tiles, generator=g).cuda().abs() * 0.01
    return A_int, B_int, As, Bs, [block_n, block_k], dtype


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class W8A8Test(unittest.TestCase):
    def _check(self, A, B, As, Bs, bs, dtype):
        actual = MODULE.w8a8_block_int8_matmul(A, B, As, Bs, bs, dtype)
        expected = reference(A, B, As, Bs, bs, dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=1e-2, rtol=1e-2
        )

    def test_basic(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                args = make_case(64, 128, 256, dtype=dtype)
                self._check(*args)

    def test_block_sizes(self):
        for bn, bk in ((32, 32), (64, 64), (128, 128), (128, 64)):
            with self.subTest(bn=bn, bk=bk):
                args = make_case(32, 128, 256, block_n=bn, block_k=bk)
                self._check(*args)

    def test_shapes(self):
        for M, N, K in ((16, 64, 128), (128, 256, 512), (1, 32, 64)):
            with self.subTest(M=M, N=N, K=K):
                args = make_case(M, N, K)
                self._check(*args)

    def test_empty(self):
        out = MODULE.w8a8_block_int8_matmul(
            torch.zeros(0, 64, dtype=torch.int8, device="cuda"),
            torch.zeros(32, 64, dtype=torch.int8, device="cuda"),
            torch.zeros(0, 1, device="cuda"),
            torch.zeros(1, 1, device="cuda"),
            [32, 64],
            torch.float16,
        )
        self.assertEqual(out.shape, (0, 32))


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class W8A8VariantsTest(unittest.TestCase):
    """Core matrix across every backend variant (generic + vendors)."""

    from tests._op_variants import load_operator_modules

    MODULES = load_operator_modules("w8a8_block_int8_matmul")

    def test_int8_group_and_tile_tails(self):
        for m, n, k in ((63, 127, 127), (64, 128, 128), (65, 129, 129)):
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                args = make_case(
                    m, n, k, block_n=128, block_k=128, dtype=dtype, seed=58
                )
                self.assertEqual(args[0].dtype, torch.int8)
                self.assertEqual(args[1].dtype, torch.int8)
                expected = reference(*args)
                for name, module in self.MODULES:
                    with self.subTest(
                        module=name, shape=(m, n, k), dtype=dtype
                    ):
                        actual = module.w8a8_block_int8_matmul(*args)
                        torch.testing.assert_close(
                            actual, expected, atol=0.5, rtol=1e-2
                        )

    def test_variants_match_reference(self):
        for M, N, K, bn, bk, dtype in (
            (16, 128, 256, 128, 64, torch.bfloat16),
            (64, 512, 1024, 128, 128, torch.bfloat16),
            (128, 1024, 2048, 128, 64, torch.float16),
            (32, 128, 256, 64, 64, torch.float16),
        ):
            args = make_case(
                M, N, K, block_n=bn, block_k=bk, dtype=dtype, seed=M
            )
            ref = reference(*args)
            for name, module in self.MODULES:
                with self.subTest(module=name, M=M, N=N, K=K):
                    out = module.w8a8_block_int8_matmul(*args)
                    torch.testing.assert_close(
                        out.float(), ref.float(), atol=1e-2, rtol=1e-2
                    )


RELEASE_REQUIRED_TESTS = [
    "W8A8Test.test_basic",
    "W8A8Test.test_block_sizes",
    "W8A8Test.test_shapes",
    "W8A8Test.test_empty",
    "W8A8VariantsTest.test_int8_group_and_tile_tails",
    "W8A8VariantsTest.test_variants_match_reference",
]


if __name__ == "__main__":
    unittest.main()
