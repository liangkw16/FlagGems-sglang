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
    / "draft_topk1.py"
)
SPEC = importlib.util.spec_from_file_location(
    "draft_topk1_module", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def reference(
    next_token_logits, positions, draft_tokens=None, draft_token_column=0
):
    bs = next_token_logits.shape[0]
    topk_index = next_token_logits.argmax(dim=-1, keepdim=True).to(torch.int64)
    topk_p = torch.ones(
        bs, 1, dtype=torch.float32, device=next_token_logits.device
    )
    out_positions = positions + 1
    out_draft_tokens = None
    if draft_tokens is not None:
        out_draft_tokens = draft_tokens.clone()
        out_draft_tokens[:, draft_token_column] = topk_index.squeeze(-1)
    return topk_p, topk_index, out_positions, out_draft_tokens


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class DraftTopk1Test(unittest.TestCase):
    def _check(self, logits, positions, draft=None, column=0):
        actual = MODULE.draft_topk1(logits, positions, draft, column)
        expected = reference(logits, positions, draft, column)
        torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
        torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)
        torch.testing.assert_close(actual[2], expected[2], atol=0, rtol=0)
        if draft is None:
            self.assertIsNone(actual[3])
            self.assertIsNone(expected[3])
        else:
            torch.testing.assert_close(actual[3], expected[3], atol=0, rtol=0)
        return actual

    def test_dtypes_and_shapes_match_reference(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for bs, vocab in (
                (1, 1),
                (1, 17),
                (8, 1023),
                (8, 1024),
                (5, 1025),
                (3, 50000),
            ):
                with self.subTest(dtype=dtype, bs=bs, vocab=vocab):
                    logits = torch.randn(bs, vocab, device="cuda").to(dtype)
                    positions = torch.arange(
                        bs, device="cuda", dtype=torch.int64
                    )
                    draft = torch.randint(
                        0, 100, (bs, 4), device="cuda", dtype=torch.int32
                    )
                    self._check(logits, positions, draft, 2)

    def test_tie_breaks_pick_first_index(self):
        logits = torch.full((1, 2048), -1.0, device="cuda")
        logits[0, 900] = 5.0
        logits[0, 3] = 5.0
        positions = torch.zeros(1, device="cuda", dtype=torch.int64)
        out = self._check(logits, positions)
        self.assertEqual(out[1].item(), 3)

        logits = torch.zeros(4, 300, device="cuda")
        positions = torch.arange(4, device="cuda", dtype=torch.int64)
        out = self._check(logits, positions)
        torch.testing.assert_close(
            out[1], torch.zeros((4, 1), device="cuda", dtype=torch.int64)
        )

    def test_fp16_duplicate_values_tie(self):
        values = torch.randn(1024, device="cuda").to(torch.float16)
        logits = values.unsqueeze(0).repeat(2, 1).contiguous()
        logits[0, 777] = logits[0, 5]
        logits[1, 999] = logits[1, 0]
        positions = torch.tensor([10, 20], device="cuda", dtype=torch.int64)
        out = self._check(logits, positions)
        first = logits[0].argmax().item()
        second = logits[1].argmax().item()
        self.assertEqual(out[1][0].item(), first)
        self.assertEqual(out[1][1].item(), second)

    def test_large_vocab_realistic(self):
        vocab = 128256
        logits = torch.randn(16, vocab, device="cuda").to(torch.float16)
        positions = torch.arange(16, device="cuda", dtype=torch.int64) * 7
        draft = torch.randint(
            0, vocab, (16, 8), device="cuda", dtype=torch.int64
        )
        self._check(logits, positions, draft, 0)
        self._check(logits, positions, draft, 7)

    def test_draft_none_passthrough(self):
        logits = torch.randn(6, 128, device="cuda")
        positions = torch.arange(6, device="cuda", dtype=torch.int64)
        self._check(logits, positions, None)

    def test_draft_dtypes_and_columns(self):
        for dtype in (torch.int32, torch.int64):
            with self.subTest(dtype=dtype):
                logits = torch.randn(7, 300, device="cuda")
                positions = torch.arange(7, device="cuda", dtype=torch.int64)
                draft = torch.randint(
                    0, 50, (7, 5), device="cuda", dtype=dtype
                )
                for column in (0, 1, 4):
                    with self.subTest(column=column):
                        self._check(logits, positions, draft, column)

    def test_non_contiguous_inputs(self):
        base = torch.randn(4, 512, device="cuda")
        logits = base[:, ::2]
        positions_base = torch.arange(8, device="cuda", dtype=torch.int64)
        positions = positions_base[::2]
        self.assertEqual(positions.shape[0], logits.shape[0])
        draft_base = torch.randint(
            0, 99, (4, 16), device="cuda", dtype=torch.int32
        )
        draft = draft_base[:, ::4]
        self.assertFalse(logits.is_contiguous())
        self.assertFalse(positions.is_contiguous())
        self.assertFalse(draft.is_contiguous())
        self._check(logits, positions, draft, 1)

    def test_inputs_not_modified(self):
        logits = torch.randn(5, 256, device="cuda", dtype=torch.float16)
        positions = torch.arange(5, device="cuda", dtype=torch.int64)
        draft = torch.randint(0, 10, (5, 3), device="cuda", dtype=torch.int64)
        snapshots = (logits.clone(), positions.clone(), draft.clone())
        self._check(logits, positions, draft, 1)
        torch.testing.assert_close(logits, snapshots[0])
        torch.testing.assert_close(positions, snapshots[1])
        torch.testing.assert_close(draft, snapshots[2])

    def test_empty_batch(self):
        logits = torch.randn(0, 128, device="cuda")
        positions = torch.zeros(0, device="cuda", dtype=torch.int64)
        draft = torch.zeros(0, 4, device="cuda", dtype=torch.int32)
        out = MODULE.draft_topk1(logits, positions, draft, 0)
        self.assertEqual(out[0].shape, (0, 1))
        self.assertEqual(out[1].shape, (0, 1))
        self.assertEqual(out[2].shape, (0,))
        self.assertEqual(out[3].shape, (0, 4))

    def test_row_grid_fold_path(self):
        rows = 70000
        logits = torch.randn(rows, 16, device="cuda")
        positions = torch.arange(rows, device="cuda", dtype=torch.int64)
        self.assertGreater(rows, 65535)
        self._check(logits, positions)


if __name__ == "__main__":
    unittest.main()
