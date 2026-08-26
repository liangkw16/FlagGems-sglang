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
    / "fused_recurrent_gdn.py"
)
SPEC = importlib.util.spec_from_file_location(
    "fused_recurrent_gdn_module", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def reference(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    output_final_state,
    use_qk_l2norm_in_kernel=False,
):
    batch, sequence_length, query_heads, _ = q.shape
    value_heads = v.shape[2]
    ratio = value_heads // query_heads
    beta_is_vector = beta.dim() == v.dim()

    if initial_state is None:
        state = q.new_zeros(
            batch,
            value_heads,
            v.shape[-1],
            q.shape[-1],
            dtype=torch.float32,
        )
    else:
        state = initial_state.float().clone()
    output = q.new_zeros(
        batch,
        sequence_length,
        value_heads,
        v.shape[-1],
        dtype=torch.float32,
    )

    for timestep in range(sequence_length):
        query = q[:, timestep].float()
        key = k[:, timestep].float()
        value = v[:, timestep].float()
        gate = g[:, timestep].float()
        if use_qk_l2norm_in_kernel:
            query = (
                query / (query.square().sum(-1, keepdim=True) + 1e-6).sqrt()
            )
            key = key / (key.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        query = query * scale
        if ratio > 1:
            query = query.repeat_interleave(ratio, dim=1)
            key = key.repeat_interleave(ratio, dim=1)

        state = state * gate.exp()[:, :, None, None]
        prediction = torch.einsum("bhvk,bhk->bhv", state, key)
        correction = value - prediction
        timestep_beta = beta[:, timestep].float()
        if not beta_is_vector:
            timestep_beta = timestep_beta.unsqueeze(-1)
        correction = correction * timestep_beta
        state = state + correction.unsqueeze(-1) * key.unsqueeze(-2)
        output[:, timestep] = torch.einsum("bhvk,bhk->bhv", state, query)

    return output.to(v.dtype), state if output_final_state else None


def make_case(
    dtype,
    *,
    batch=2,
    sequence_length=5,
    query_heads=2,
    value_heads=4,
    key_dim=13,
    value_dim=19,
    beta_is_vector=False,
    use_initial_state=False,
):
    generator = torch.Generator(device="cuda").manual_seed(
        20260824 + sequence_length + key_dim + value_dim
    )
    q_storage = torch.randn(
        (batch, sequence_length, query_heads, key_dim * 2),
        generator=generator,
        device="cuda",
        dtype=dtype,
    )
    k_storage = torch.randn(
        q_storage.shape,
        generator=generator,
        device=q_storage.device,
        dtype=q_storage.dtype,
    )
    v_storage = torch.randn(
        (batch, sequence_length, value_heads, value_dim * 2),
        generator=generator,
        device="cuda",
        dtype=dtype,
    )
    g_storage = torch.empty(
        (batch, sequence_length, value_heads * 2),
        device="cuda",
        dtype=dtype,
    )
    q = q_storage[..., ::2]
    k = k_storage[..., 1::2]
    v = v_storage[..., ::2]
    g = g_storage[..., 1::2]
    g.copy_(
        -torch.rand(
            g.shape,
            generator=generator,
            device=g.device,
            dtype=g.dtype,
        )
        * 0.2
    )

    if beta_is_vector:
        beta_storage = torch.empty(
            (batch, sequence_length, value_heads, value_dim * 2),
            device="cuda",
            dtype=dtype,
        )
    else:
        beta_storage = torch.empty(
            (batch, sequence_length, value_heads * 2),
            device="cuda",
            dtype=dtype,
        )
    beta = beta_storage[..., ::2]
    beta.copy_(
        torch.rand(
            beta.shape,
            generator=generator,
            device=beta.device,
            dtype=beta.dtype,
        )
        * 0.6
        + 0.2
    )

    initial_state = None
    if use_initial_state:
        state_storage = torch.randn(
            (batch, value_heads, value_dim, key_dim * 2),
            generator=generator,
            device="cuda",
            dtype=dtype,
        )
        initial_state = state_storage[..., 1::2]
    return q, k, v, g, beta, initial_state


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class FusedRecurrentGdnTest(unittest.TestCase):
    def assert_matches(
        self,
        case,
        *,
        output_final_state,
        use_qk_l2norm_in_kernel,
        atol=1e-2,
        rtol=1e-2,
        equal_nan=False,
    ):
        q, k, v, g, beta, initial_state = case
        scale = 0.37
        tensors = tuple(
            tensor
            for tensor in (q, k, v, g, beta, initial_state)
            if tensor is not None
        )
        snapshots = tuple(tensor.clone() for tensor in tensors)

        actual_output, actual_state = MODULE.fused_recurrent_gdn(
            q,
            k,
            v,
            g,
            beta,
            scale,
            initial_state,
            output_final_state,
            use_qk_l2norm_in_kernel,
        )
        expected_output, expected_state = reference(
            q,
            k,
            v,
            g,
            beta,
            scale,
            initial_state,
            output_final_state,
            use_qk_l2norm_in_kernel,
        )

        self.assertEqual(
            (actual_output.shape, actual_output.dtype),
            (
                (q.shape[0], q.shape[1], v.shape[2], v.shape[-1]),
                v.dtype,
            ),
        )
        torch.testing.assert_close(
            actual_output,
            expected_output,
            atol=atol,
            rtol=rtol,
            equal_nan=equal_nan,
        )
        if output_final_state:
            self.assertEqual(actual_state.dtype, torch.float32)
            torch.testing.assert_close(
                actual_state,
                expected_state,
                atol=atol,
                rtol=rtol,
                equal_nan=equal_nan,
            )
        else:
            self.assertIsNone(actual_state)
        for tensor, snapshot in zip(tensors, snapshots):
            torch.testing.assert_close(tensor, snapshot, atol=0.0, rtol=0.0)

    def test_gqa_beta_forms_l2_initial_state_strides_and_dtypes(self):
        cases = (
            (make_case(torch.float16), True, False),
            (
                make_case(
                    torch.bfloat16,
                    batch=1,
                    sequence_length=4,
                    key_dim=17,
                    value_dim=11,
                    beta_is_vector=True,
                    use_initial_state=True,
                ),
                True,
                True,
            ),
            (
                make_case(
                    torch.float32,
                    batch=1,
                    sequence_length=3,
                    query_heads=3,
                    value_heads=3,
                    key_dim=7,
                    value_dim=5,
                    use_initial_state=True,
                ),
                False,
                False,
            ),
        )
        for case, output_final_state, use_l2norm in cases:
            with self.subTest(
                dtype=case[0].dtype,
                output_final_state=output_final_state,
                use_l2norm=use_l2norm,
            ):
                self.assert_matches(
                    case,
                    output_final_state=output_final_state,
                    use_qk_l2norm_in_kernel=use_l2norm,
                )

    def test_empty_sequence_preserves_initial_state(self):
        case = make_case(
            torch.float16,
            batch=2,
            sequence_length=0,
            key_dim=9,
            value_dim=7,
            use_initial_state=True,
        )

        self.assert_matches(
            case,
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
        )

    def test_key_dimension_warp_boundary(self):
        for dtype, key_dim in (
            (torch.float16, 128),
            (torch.float32, 128),
            (torch.float16, 129),
        ):
            with self.subTest(dtype=dtype, key_dim=key_dim):
                self.assert_matches(
                    make_case(
                        dtype,
                        batch=1,
                        sequence_length=2,
                        query_heads=1,
                        value_heads=1,
                        key_dim=key_dim,
                        value_dim=7,
                    ),
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=False,
                )

    def test_k64_bfloat16_specialized_contract(self):
        self.assert_matches(
            make_case(
                torch.bfloat16,
                batch=2,
                sequence_length=3,
                query_heads=2,
                value_heads=4,
                key_dim=64,
                value_dim=17,
                beta_is_vector=True,
                use_initial_state=True,
            ),
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
        )

    def test_long_bfloat16_recurrence(self):
        for batch, sequence_length in ((32, 32), (8, 128)):
            with self.subTest(batch=batch, sequence_length=sequence_length):
                self.assert_matches(
                    make_case(
                        torch.bfloat16,
                        batch=batch,
                        sequence_length=sequence_length,
                        query_heads=4,
                        value_heads=8,
                        key_dim=64,
                        value_dim=64,
                    ),
                    output_final_state=False,
                    use_qk_l2norm_in_kernel=False,
                    atol=1.5e-2,
                    rtol=1.5e-2,
                    equal_nan=True,
                )


if __name__ == "__main__":
    unittest.main()
