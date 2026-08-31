# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import unittest
from pathlib import Path

import torch

MODULE_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "flaggems_sglang"
    / "ops"
    / "per_token_group_quant_int8.py"
)
SPEC = importlib.util.spec_from_file_location(
    "per_token_group_quant_int8_module", MODULE_PATH
)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def _load_vendor(name):
    path = (
        Path(__file__).parents[1]
        / "src"
        / "flaggems_sglang"
        / "runtime"
        / "backend"
        / f"_{name}"
        / "ops"
        / "per_token_group_quant_int8.py"
    )
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(
        f"per_token_group_quant_int8_{name}", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VENDOR_MODULES = {
    name: module
    for name, module in (
        (vendor, _load_vendor(vendor))
        for vendor in ("ascend", "enflame", "kunlunxin", "metax")
    )
    if module is not None
}

_EPS = 1e-10


def reference(x, group_size, dtype=torch.int8):
    iinfo = torch.iinfo(dtype)
    int8_min, int8_max = iinfo.min, iinfo.max
    x_ = x.reshape(x.numel() // group_size, group_size)
    amax = (
        x_.abs().max(dim=-1, keepdim=True)[0].clamp(min=_EPS).to(torch.float32)
    )
    x_s = amax / int8_max
    x_q = (x_ / x_s).clamp(min=int8_min, max=int8_max).to(dtype)
    x_q = x_q.reshape(x.shape)
    x_s = x_s.reshape(x.shape[:-1] + (x.shape[-1] // group_size,))
    return x_q, x_s


class TestPerTokenGroupQuantInt8(unittest.TestCase):
    def _check(self, x, group_size, module=MOD):
        x_q, x_s = module.per_token_group_quant_int8(x, group_size)
        # platform semantics: the reference runs on the same device; CPU
        # torch rounds fp division differently at boundary ulps than
        # device torch, so device-side comparison is the contract
        ref_q, ref_s = reference(x, group_size)
        self.assertEqual(x_q.dtype, torch.int8)
        self.assertEqual(x_s.dtype, torch.float32)
        self.assertEqual(x_q.shape, x.shape)
        self.assertEqual(
            x_s.shape, x.shape[:-1] + (x.shape[-1] // group_size,)
        )
        self.assertTrue(torch.equal(x_q, ref_q))
        torch.testing.assert_close(x_s, ref_s, rtol=1e-6, atol=1e-8)

    def test_dtype_shape_matrix(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for rows, k, gs in [
                (256, 512, 64),
                (1024, 2560, 128),
                (8192, 512, 64),
                (129, 1025, 32),
                (1, 128, 128),
                (64, 96, 96),
                (65536, 256, 64),
            ]:
                if k % gs != 0:
                    continue
                torch.manual_seed(rows + k)
                x = torch.randn(rows, k, dtype=dtype, device="cuda") * 3
                with self.subTest(dtype=dtype, rows=rows, k=k, gs=gs):
                    self._check(x, gs)

    def test_trunc_semantics(self):
        # values must truncate toward zero, not round
        x = torch.tensor(
            [[0.9, -0.9, 1.6, -1.6, 127.6, -200.0, 0.0, 255.0]],
            dtype=torch.float32,
            device="cuda",
        )
        x_q, x_s = MOD.per_token_group_quant_int8(x, 8)
        ref_q, ref_s = reference(x, 8)
        torch.testing.assert_close(x_q, ref_q, rtol=0, atol=0)

    def test_zero_group_epsilon(self):
        x = torch.zeros(4, 128, dtype=torch.float32, device="cuda")
        x_q, x_s = MOD.per_token_group_quant_int8(x, 128)
        ref_q, ref_s = reference(x, 128)
        torch.testing.assert_close(x_q, ref_q, rtol=0, atol=0)
        torch.testing.assert_close(x_s, ref_s, rtol=1e-6, atol=1e-12)

    def test_extreme_values(self):
        x = torch.zeros(2, 64, dtype=torch.float16, device="cuda")
        x[0, :32] = -65504.0
        x[1, 32:] = 65504.0
        with self.subTest("finite fp16 extremes"):
            self._check(x, 32)

    def test_multi_leading_dims_and_noncontig(self):
        torch.manual_seed(7)
        base = torch.randn(2, 3, 5, 256, dtype=torch.float16, device="cuda")
        with self.subTest("multi-dim contiguous"):
            self._check(base, 64)
        sliced = base[:, :, :, ::2]
        with self.subTest("non-contiguous input handled"):
            x_q, x_s = MOD.per_token_group_quant_int8(sliced, 64)
            ref_q, ref_s = reference(sliced.contiguous(), 64)
            torch.testing.assert_close(x_q, ref_q, rtol=0, atol=0)

    def test_input_invariance(self):
        x = torch.randn(128, 256, dtype=torch.float16, device="cuda")
        x_clone = x.clone()
        MOD.per_token_group_quant_int8(x, 128)
        self.assertTrue(torch.equal(x, x_clone))

    def test_empty(self):
        x = torch.empty(0, 128, dtype=torch.float16, device="cuda")
        x_q, x_s = MOD.per_token_group_quant_int8(x, 128)
        self.assertEqual(x_q.shape, (0, 128))
        self.assertEqual(x_s.shape, (0, 1))

    def test_vendor_matrix_matches_generic_semantics(self):
        # every shipped kernel variant (e7 kunlunxin tile included) must
        # reproduce the reference bit-for-bit on the same coverage
        for name, module in VENDOR_MODULES.items():
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                for rows, k, gs in [
                    (256, 512, 64),
                    (129, 1025, 32),
                    (64, 96, 96),
                ]:
                    if k % gs != 0:
                        continue
                    torch.manual_seed(rows + k)
                    x = torch.randn(rows, k, dtype=dtype, device="cuda") * 3
                    with self.subTest(name=name, dtype=dtype, gs=gs):
                        self._check(x, gs, module=module)


if __name__ == "__main__":
    unittest.main()
