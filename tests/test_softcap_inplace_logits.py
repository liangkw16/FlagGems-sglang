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

ROOT = Path(__file__).parents[1]
MODULE_PATHS = {
    "generic": ROOT
    / "src"
    / "flaggems_sglang"
    / "ops"
    / "softcap_inplace_logits.py",
    **{
        vendor: ROOT
        / "src"
        / "flaggems_sglang"
        / "runtime"
        / "backend"
        / f"_{vendor}"
        / "ops"
        / "softcap_inplace_logits.py"
        for vendor in ("ascend", "enflame", "kunlunxin", "metax")
    },
}


def _load(name, module_path):
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULES = {
    name: _load(f"softcap_inplace_logits_{name}", module_path)
    for name, module_path in MODULE_PATHS.items()
}


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class SoftcapInplaceLogitsTest(unittest.TestCase):
    def _check(self, module, source, cap=30.0):
        expected = torch.tanh(source / cap) * cap
        pointer = source.data_ptr()

        actual = module.softcap_inplace_logits(source, cap)

        self.assertEqual(actual.data_ptr(), pointer)
        self.assertEqual(
            (actual.shape, actual.dtype), (source.shape, source.dtype)
        )
        tolerance = {
            torch.float16: 1e-2,
            torch.bfloat16: 1.5e-2,
            torch.float32: 1e-4,
        }[source.dtype]
        torch.testing.assert_close(
            actual,
            expected,
            atol=tolerance,
            rtol=tolerance,
            equal_nan=True,
        )

    def test_dtypes_shapes_and_vendor_paths(self):
        for name, module in MODULES.items():
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                for shape in ((1,), (3, 257), (2, 3, 4097)):
                    with self.subTest(name=name, dtype=dtype, shape=shape):
                        source = (
                            torch.linspace(
                                -60.0,
                                60.0,
                                torch.tensor(shape).prod().item(),
                                device="cuda",
                                dtype=torch.float32,
                            )
                            .reshape(shape)
                            .to(dtype)
                        )
                        self._check(module, source)

    def test_tile_boundaries_and_grid_stride(self):
        cases = {
            "generic": (1023, 1024, 1025),
            "ascend": (511, 512, 513, 48 * 512 + 17),
            "enflame": (32767, 32768, 32769, 12 * 32768 + 17),
            "kunlunxin": (4095, 4096, 4097),
            "metax": (2047, 2048, 2049),
        }
        for name, lengths in cases.items():
            for length in lengths:
                with self.subTest(name=name, length=length):
                    source = torch.linspace(-60.0, 60.0, length, device="cuda")
                    self._check(MODULES[name], source)

    def test_noncontiguous_rows_preserve_padding(self):
        for name, module in MODULES.items():
            with self.subTest(name=name):
                storage = torch.randn(5, 2051, device="cuda")
                source = storage[:, :2049]
                self.assertFalse(source.is_contiguous())
                padding = storage[:, 2049:].clone()
                self._check(module, source)
                torch.testing.assert_close(
                    storage[:, 2049:], padding, atol=0.0, rtol=0.0
                )

    def test_empty_and_special_values(self):
        for name, module in MODULES.items():
            with self.subTest(name=name, case="empty"):
                source = torch.empty(0, device="cuda", dtype=torch.float16)
                self._check(module, source)
            with self.subTest(name=name, case="special"):
                source = torch.tensor(
                    [
                        float("-inf"),
                        -1.0,
                        -0.0,
                        0.0,
                        1.0,
                        float("inf"),
                        float("nan"),
                    ],
                    device="cuda",
                )
                self._check(module, source)

    def test_cap_boundaries(self):
        caps = (0.0, float.fromhex("0x1p-128"), float("inf"), float("nan"))
        for name, module in MODULES.items():
            for cap in caps:
                with self.subTest(name=name, cap=cap):
                    source = torch.tensor(
                        [-1.0, -0.0, 0.0, 1.0], device="cuda"
                    )
                    self._check(module, source, cap)


if __name__ == "__main__":
    unittest.main()
