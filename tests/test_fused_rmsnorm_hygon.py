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

import ast
import unittest
from pathlib import Path

SOURCE_ROOT = Path(__file__).parents[1] / "src" / "flaggems_sglang"
GENERIC_PATH = SOURCE_ROOT / "ops" / "fused_rmsnorm.py"
HYGON_PATH = (
    SOURCE_ROOT / "runtime" / "backend" / "_hygon" / "ops" / "fused_rmsnorm.py"
)


def _function(tree, name):
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _launch_call(function):
    return next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Subscript)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "_fused_rmsnorm_kernel"
    )


class HygonFusedRmsnormSourceTest(unittest.TestCase):
    def setUp(self):
        self.generic_tree = ast.parse(GENERIC_PATH.read_text())
        self.hygon_tree = ast.parse(HYGON_PATH.read_text())

    def test_only_launch_width_is_tuned(self):
        generic_kernel = _function(self.generic_tree, "_fused_rmsnorm_kernel")
        hygon_kernel = _function(self.hygon_tree, "_fused_rmsnorm_kernel")
        self.assertEqual(
            ast.dump(generic_kernel.args), ast.dump(hygon_kernel.args)
        )
        self.assertEqual(
            ast.dump(ast.Module(body=generic_kernel.body, type_ignores=[])),
            ast.dump(ast.Module(body=hygon_kernel.body, type_ignores=[])),
        )

        generic_wrapper = _function(self.generic_tree, "fused_rmsnorm")
        hygon_wrapper = _function(self.hygon_tree, "fused_rmsnorm")
        generic_launch = _launch_call(generic_wrapper)
        generic_launch.keywords = [
            keyword
            for keyword in generic_launch.keywords
            if keyword.arg not in {"num_warps", "num_stages"}
        ]
        self.assertEqual(
            ast.dump(ast.Module(body=generic_wrapper.body, type_ignores=[])),
            ast.dump(ast.Module(body=hygon_wrapper.body, type_ignores=[])),
        )

    def test_autotune_uses_official_hygon_warp_widths(self):
        kernel = _function(self.hygon_tree, "_fused_rmsnorm_kernel")
        autotune = next(
            decorator
            for decorator in kernel.decorator_list
            if isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr == "autotune"
        )
        keywords = {
            keyword.arg: keyword.value for keyword in autotune.keywords
        }
        self.assertEqual(ast.literal_eval(keywords["key"]), ["hidden_size"])

        configs = keywords["configs"]
        self.assertIsInstance(configs, ast.List)
        warp_widths = []
        for config in configs.elts:
            values = {
                keyword.arg: ast.literal_eval(keyword.value)
                for keyword in config.keywords
            }
            warp_widths.append(values["num_warps"])
            self.assertEqual(values["num_stages"], 1)
        self.assertEqual(warp_widths, [4, 8, 16])


if __name__ == "__main__":
    unittest.main()
