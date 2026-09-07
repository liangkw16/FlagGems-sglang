# Copyright 2026 FlagOS Contributors
# (same license)
import importlib.util
import unittest
from pathlib import Path

import torch

MODULE_PATH = (
    Path(__file__).parents[1] / "src" / "flaggems_sglang" / "ops" / "l2norm.py"
)
SPEC = importlib.util.spec_from_file_location("l2norm_module", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

TOL = {
    torch.float32: (1e-4, 1e-4),
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (1.5e-2, 1.5e-2),
}

# Keep the contract matrix visible even if a future load_tests hook filters it.
RELEASE_REQUIRED_TESTS = [
    f"L2NormTest.{name}"
    for name in (
        "test_dtypes",
        "test_short_row_tiles",
        "test_shapes",
        "test_multi_dim",
        "test_transposed_leading_dims",
        "test_non_contiguous",
        "test_empty",
        "test_input_not_modified",
    )
]


def reference(x, eps=1e-6):
    xf = x.float()
    rstd = (xf.pow(2).sum(dim=-1, keepdim=True) + eps).rsqrt()
    return (xf * rstd).to(x.dtype)


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class L2NormTest(unittest.TestCase):
    def _check(self, x, eps=1e-6):
        actual = MODULE.l2norm(x, eps)
        expected = reference(x, eps)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        atol, rtol = TOL[x.dtype]
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

    def test_short_row_tiles(self):
        torch.manual_seed(56)
        for rows in (7, 8, 9):
            for dim in (63, 64, 65, 127, 128, 129):
                for dtype in TOL:
                    with self.subTest(rows=rows, dim=dim, dtype=dtype):
                        x = torch.randn(rows, dim, device="cuda", dtype=dtype)
                        x[0] = 0
                        self._check(x)

    def test_dtypes(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self._check(
                    torch.randn(32, 512, device="cuda", dtype=dtype) * 3
                )

    def test_shapes(self):
        for shape in ((1, 16), (100, 1023), (5, 1024), (3, 4096), (8, 8192)):
            with self.subTest(shape=shape):
                self._check(torch.randn(*shape, device="cuda"))

    def test_multi_dim(self):
        for dtype in TOL:
            for shape in ((128,), (2, 3, 128), (4, 5, 6, 256)):
                with self.subTest(shape=shape, dtype=dtype):
                    self._check(
                        torch.randn(*shape, device="cuda", dtype=dtype)
                    )

    def test_transposed_leading_dims(self):
        x = torch.randn(2, 3, 65, device="cuda").transpose(0, 1)
        self.assertFalse(x.is_contiguous())
        self._check(x)

    def test_non_contiguous(self):
        base = torch.randn(32, 1024, device="cuda")
        x = base[:, ::2]
        self.assertFalse(x.is_contiguous())
        self._check(x)

    def test_empty(self):
        out = MODULE.l2norm(torch.randn(0, 128, device="cuda"))
        self.assertEqual(out.shape, (0, 128))

    def test_input_not_modified(self):
        x = torch.randn(16, 256, device="cuda", dtype=torch.float16)
        snap = x.clone()
        self._check(x)
        torch.testing.assert_close(x, snap)


if __name__ == "__main__":
    unittest.main()
