---
name: kernelgen-flagos
description: >
  Unified GPU kernel operator generation and optimization skill. Automatically detects the target
  repository type (FlagGems, vLLM, or general Python/Triton) and dispatches to the appropriate
  specialized sub-skill. Includes operator generation, MCP-based iterative optimization, and
  feedback submission sub-skills. Use this skill when the user wants to generate or optimize a
  GPU kernel operator, create a Triton kernel, or says things like "generate an operator",
  "create a kernel for X", "optimize triton kernel", or "/kernelgen-flagos".
argument-hint: "<operator_name> [--func-type <type>]"
user-invokable: true
compatibility: "Python 3.8+, PyTorch with CUDA, Triton"
metadata:
  version: "1.0.0"
  author: flagos-ai
  category: gpu-kernel-generation
  tags: [kernelgen, triton, gpu, mcp, operator-generation, operator-optimization, flaggems, vllm, feedback]
allowed-tools:
  - Bash
  - Bash(gh:*)
  - Bash(python:*)
  - Bash(python3:*)
  - Bash(command:*)
  - Read
  - Write
  - Edit
  - Glob
  - Grep
  - AskUserQuestion
---

<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->


# kernelgen-flagos — Unified GPU Operator Generation Skill

This is a **unified entry point** that bundles generation and optimization sub-skills into one:

| Sub-skill file | Purpose |
|---|---|
| **Generation** | |
| `kernelgen-generate.md` | Generate GPU kernels for **any** Python/Triton repository |
| `kernelgen-generate-for-flaggems.md` | Specialized generation for **FlagGems** repositories |
| `kernelgen-generate-for-vllm.md` | Specialized generation for **vLLM** repositories |
| **Optimization** | |
| `kernelgen-optimize.md` | Optimize existing Triton kernels via MCP iterative optimization (general purpose) |
| `kernelgen-optimize-for-flaggems.md` | Optimize Triton operators and integrate into **FlagGems** (3 modes: built-in/external/experimental) |
| `kernelgen-optimize-for-vllm.md` | Optimize Triton operators and integrate into **vLLM** (with CustomOp registration) |
| **Platform Specialization** | |
| `kernelgen-specialize.md` | Specialize Triton operators to target platforms (e.g., GPU → Ascend NPU) via MCP `specialize_kernel` |
| `kernelgen-specialize-for-flaggems.md` | Platform specialization + **FlagGems** integration (4 modes: vendor-ops/vendor-fused/override-builtin/experimental) |
| **MCP Configuration** | |
| `kernelgen-mcp-setup.md` | Check and auto-configure the `kernelgen-server` MCP service (URL built-in, user only provides Token) |
| **Feedback** | |
| `kernelgen-submit-feedback.md` | Submit bug reports and feedback via GitHub or email |

All sub-skill files are located in the **same directory** as this `SKILL.md` file.

---

## Routing Protocol — Follow This BEFORE Doing Anything Else

### Phase 0: MCP Transport Check

The kernelgen MCP service lives at `https://kernelgen.flagos.io/sse`
(Streamable HTTP, **no trailing slash** — with a trailing slash it 307-redirects
to the SPA and fails). Two transports are supported; pick whichever works:

1. **Native MCP tools**: if `mcp__kernelgen-server__*` tools
   (generate_kernel / optimize_kernel / specialize_kernel / autotune_kernel)
   appear in the current tool list, use them directly → proceed to Phase 1.
2. **Bundled script client** (the normal case in ZCode — it does NOT auto-mount
   project `.mcp.json`): call via Bash and verify connectivity first:

   ```bash
   python3 <this skill's directory>/scripts/kernelgen_mcp.py list
   # expect: "4 tools at https://kernelgen.flagos.io/sse"
   ```

   Token resolution is automatic: env `KERNELGEN_MCP_TOKEN`, else the
   `kernelgen-server` entry in the repo root `.mcp.json`. If the token is
   missing or `list` fails, follow `kernelgen-mcp-setup.md` in this directory.

**Transport substitution rule (applies to ALL sub-skills)**: wherever a
sub-skill instructs to call `mcp__kernelgen-mcp__<tool>` or
`mcp__kernelgen-server__<tool>` and that native tool is not mounted, call
instead:

```bash
python3 <this skill's directory>/scripts/kernelgen_mcp.py call <tool> '<json-args>'
```

The request arguments and the JSON-RPC result are identical to the native tool.
For `autotune_kernel`, poll by re-calling with
`{"job_id": ..., "last_seen_attempt": N, "last_seen_version": M, "device": ...}`
until status is `completed`/`failed` (~2 min on real chips; keep at most 2
concurrent jobs server-wide).

### Phase 1: Detect Repository Type

Use the Glob tool to check for project identity files in the current working directory:

```
Glob: pyproject.toml
Glob: setup.py
Glob: setup.cfg
```

Then use the Read tool to read whichever file exists. Determine the **project name** from
the file contents (e.g., `name = "flag_gems"` in pyproject.toml, or `name='vllm'` in setup.py).

Also use the Glob tool to check for characteristic directory structures:

**FlagGems indicators** (match ANY):
- `src/flag_gems/` directory exists
- Project name is `flag_gems` or `flag-gems` or `FlagGems`
- `import flag_gems` appears in test files

**vLLM indicators** (match ANY):
- `vllm/` directory exists at the repo root (with `vllm/__init__.py`)
- Project name is `vllm`
- `csrc/` directory exists alongside `vllm/`

### Phase 2: Dispatch to Sub-skill

Based on the detection result, use the **Read tool** to read the appropriate sub-skill file
from this skill's directory, then **follow the instructions in that file exactly**.

**To locate the sub-skill files**: They are in the same directory as this SKILL.md. Use the
Glob tool to find the path:

```
Glob: **/skills/kernelgen-flagos/kernelgen-generate.md
```

Then use the Read tool to read the matched path.

#### Decision Table

**Generation requests** (user wants to create/generate a new operator):

| Detection Result | Action |
|---|---|
| FlagGems repository detected | Read `kernelgen-generate-for-flaggems.md` and follow it |
| vLLM repository detected | Read `kernelgen-generate-for-vllm.md` and follow it |
| Neither detected (or unknown) | Read `kernelgen-generate.md` and follow it |

**Optimization requests** (user wants to optimize an existing operator, mentions "optimize", "speedup", "improve performance"):

| Detection Result | Action |
|---|---|
| FlagGems repository detected | Read `kernelgen-optimize-for-flaggems.md` and follow it |
| vLLM repository detected | Read `kernelgen-optimize-for-vllm.md` and follow it |
| Neither detected (or unknown) | Read `kernelgen-optimize.md` and follow it |

**Specialization requests** (user wants to migrate/specialize an operator to a different platform, mentions "specialize", "migrate to Ascend/NPU", "platform migration"):

| Detection Result | Action |
|---|---|
| FlagGems repository detected | Read `kernelgen-specialize-for-flaggems.md` and follow it |
| Neither detected (or unknown) | Read `kernelgen-specialize.md` and follow it |

**Feedback requests**:

| Detection Result | Action |
|---|---|
| User reports a bug or requests feedback submission | Read `kernelgen-submit-feedback.md` and follow it |

**Important rules:**
1. **Always detect first, dispatch second.** Never skip detection.
2. **Read the entire sub-skill file** before starting execution — do not partially read it.
3. **Follow the sub-skill instructions exactly** as if they were the main SKILL.md. All steps,
   rules, and protocols in the sub-skill apply fully.
4. **Do not mix sub-skills.** Once you dispatch to a sub-skill, follow it to completion.
5. If the user explicitly requests a specific sub-skill (e.g., "use the FlagGems version"),
   honor that request regardless of auto-detection results.
6. **CRITICAL — MCP is mandatory**: ALL operator code generation MUST go through the
   kernelgen MCP service — either the native `mcp__kernelgen-server__*` tools or the
   bundled `scripts/kernelgen_mcp.py` client (see Phase 0). NEVER generate Triton
   kernels, PyTorch wrappers, or operator implementations yourself. If the MCP service
   is not reachable via either transport after retries, STOP and report the issue — do
   NOT fall back to writing code manually.

### Phase 3: Feedback Handling

At **any point** during the workflow, if the user reports a bug, says something is broken,
or asks to submit feedback about the skill:

1. Use the Read tool to read `kernelgen-submit-feedback.md` from this skill's directory.
2. Follow the feedback submission workflow described in that file.
3. After feedback is submitted, ask the user if they want to continue with the operator
   generation workflow or stop.

---

## Quick Reference for Users

```bash
# === Generation ===
# Generate a kernel operator (auto-detects repo type)
/kernelgen-flagos relu

# Generate with explicit function type
/kernelgen-flagos rms_norm --func-type normalization

# === Optimization ===
# Optimize an existing Triton kernel (auto-detects repo type)
# Just say "optimize the relu kernel" or "improve kernel performance"
# The skill will automatically dispatch to the right optimization sub-skill

# The skill will automatically:
# - Detect if you're in a FlagGems repo → use FlagGems-specific workflow
# - Detect if you're in a vLLM repo → use vLLM-specific workflow
# - Otherwise → use the general-purpose workflow
```

If you encounter any issues during generation, just say "submit feedback" or "report a bug"
and the skill will guide you through the feedback submission process.
