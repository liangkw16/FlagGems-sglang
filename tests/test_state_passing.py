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

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("state_passing")


def reference(states, dA_cumsum, initial_states=None):
    batch, nchunks, nheads, dim = states.shape
    if initial_states is None:
        current = states.new_zeros(
            batch,
            nheads,
            dim,
            dtype=torch.float32,
        )
    else:
        current = initial_states.float().clone()

    out = torch.empty(
        batch,
        nchunks,
        nheads,
        dim,
        device=states.device,
        dtype=states.dtype,
    )
    states_f = states.float()
    dA_last = dA_cumsum[..., -1].float().permute(0, 2, 1)
    for chunk in range(nchunks):
        out[:, chunk] = current.to(states.dtype)
        decay = torch.exp(dA_last[:, chunk]).unsqueeze(-1)
        current = current * decay + states_f[:, chunk]
    return out, current


def make_case(
    dtype,
    *,
    batch=2,
    nchunks=5,
    nheads=3,
    dim=17,
    length=7,
    use_initial_states=False,
):
    generator = torch.Generator(device="cuda").manual_seed(
        20260830 + nchunks + dim + length
    )
    states_storage = torch.randn(
        (batch, nchunks, nheads, dim * 2),
        generator=generator,
        device="cuda",
        dtype=dtype,
    )
    states = states_storage[..., ::2]

    dA_storage = torch.empty(
        (batch, nheads, nchunks, length * 2),
        device="cuda",
        dtype=torch.float32,
    )
    dA_cumsum = dA_storage[..., 1::2]
    dA_cumsum.copy_(
        -torch.rand(
            dA_cumsum.shape,
            generator=generator,
            device=dA_cumsum.device,
            dtype=dA_cumsum.dtype,
        )
        * 0.2
    )

    initial_states = None
    if use_initial_states:
        initial_storage = torch.randn(
            (batch, nheads, dim * 2),
            generator=generator,
            device="cuda",
            dtype=dtype,
        )
        initial_states = initial_storage[..., 1::2]
    return states, dA_cumsum, initial_states


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class StatePassingTest(unittest.TestCase):
    def assert_matches(self, case, module):
        states, dA_cumsum, initial_states = case
        tensors = tuple(
            tensor
            for tensor in (states, dA_cumsum, initial_states)
            if tensor is not None
        )
        snapshots = tuple(tensor.clone() for tensor in tensors)

        actual_out, actual_final = module.state_passing(
            states,
            dA_cumsum,
            initial_states,
        )
        expected_out, expected_final = reference(
            states,
            dA_cumsum,
            initial_states,
        )

        self.assertEqual(actual_out.shape, states.shape)
        self.assertEqual(actual_out.dtype, states.dtype)
        self.assertEqual(
            actual_final.shape,
            (states.shape[0], states.shape[2], states.shape[3]),
        )
        self.assertEqual(actual_final.dtype, torch.float32)
        tolerance = {
            torch.float32: 1e-4,
            torch.float16: 1e-2,
            torch.bfloat16: 1.5e-2,
        }[states.dtype]
        torch.testing.assert_close(
            actual_out,
            expected_out,
            atol=tolerance,
            rtol=tolerance,
        )
        torch.testing.assert_close(
            actual_final,
            expected_final,
            atol=tolerance,
            rtol=tolerance,
        )
        for tensor, snapshot in zip(tensors, snapshots):
            torch.testing.assert_close(tensor, snapshot, atol=0.0, rtol=0.0)

    def test_dtypes_initial_states_strides_and_dim_tail(self):
        cases = (
            make_case(torch.float16),
            make_case(
                torch.bfloat16,
                batch=1,
                nchunks=4,
                nheads=2,
                dim=257,
                length=5,
                use_initial_states=True,
            ),
            make_case(
                torch.float32,
                batch=2,
                nchunks=1,
                nheads=1,
                dim=33,
                length=3,
                use_initial_states=True,
            ),
        )
        for name, module in MODULES:
            for case in cases:
                with self.subTest(
                    module=name,
                    dtype=case[0].dtype,
                    shape=tuple(case[0].shape),
                    initial_states=case[2] is not None,
                ):
                    self.assert_matches(case, module)

    def test_pre_update_snapshot_and_last_dA_lane(self):
        states = torch.tensor(
            [[[[1.0, -2.0]], [[3.0, 4.0]]]],
            device="cuda",
        )
        initial_states = torch.tensor(
            [[[0.25, -0.5]]],
            device="cuda",
        )
        dA_a = torch.tensor(
            [[[[100.0, -100.0, -0.1], [50.0, 25.0, -0.2]]]],
            device="cuda",
        )
        dA_b = dA_a.clone()
        dA_b[..., :-1] = -dA_b[..., :-1]

        expected_out, expected_final = reference(
            states,
            dA_a,
            initial_states,
        )
        for name, module in MODULES:
            with self.subTest(module=name):
                out_a, final_a = module.state_passing(
                    states,
                    dA_a,
                    initial_states,
                )
                out_b, final_b = module.state_passing(
                    states,
                    dA_b,
                    initial_states,
                )
                torch.testing.assert_close(out_a[:, 0], initial_states)
                torch.testing.assert_close(out_a, out_b)
                torch.testing.assert_close(final_a, final_b)
                torch.testing.assert_close(
                    out_a, expected_out, atol=1e-4, rtol=1e-4
                )
                torch.testing.assert_close(
                    final_a,
                    expected_final,
                    atol=1e-4,
                    rtol=1e-4,
                )

    def test_empty_chunks(self):
        for name, module in MODULES:
            for use_initial_states in (False, True):
                with self.subTest(
                    module=name, initial_states=use_initial_states
                ):
                    case = make_case(
                        torch.float16,
                        nchunks=0,
                        use_initial_states=use_initial_states,
                    )
                    self.assert_matches(case, module)
                    if case[2] is not None:
                        _, final_states = module.state_passing(*case)
                        self.assertNotEqual(
                            final_states.data_ptr(),
                            case[2].data_ptr(),
                        )

    def test_zero_dimensions(self):
        cases = (
            make_case(torch.float16, batch=0),
            make_case(torch.float16, nheads=0),
            make_case(torch.float16, dim=0, use_initial_states=True),
        )
        for name, module in MODULES:
            for case in cases:
                with self.subTest(module=name, shape=tuple(case[0].shape)):
                    self.assert_matches(case, module)

    def test_capped_grid_covers_every_tile(self):
        states = torch.full(
            (256, 1, 256, 129),
            4.0,
            dtype=torch.float16,
            device="cuda",
        )
        dA_cumsum = torch.zeros(
            (256, 256, 1, 1),
            device="cuda",
        )
        initial_states = torch.full(
            (256, 256, 129),
            3.0,
            device="cuda",
        )
        for name, module in MODULES:
            with self.subTest(module=name):
                out, final_states = module.state_passing(
                    states,
                    dA_cumsum,
                    initial_states,
                )
                torch.testing.assert_close(
                    out[:, 0], torch.full_like(out[:, 0], 3.0)
                )
                torch.testing.assert_close(
                    final_states,
                    torch.full_like(final_states, 7.0),
                )


if __name__ == "__main__":
    unittest.main()
