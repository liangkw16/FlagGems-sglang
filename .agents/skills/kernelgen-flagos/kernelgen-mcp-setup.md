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

# KernelGen MCP Configuration Check & Auto-Setup

This file ensures the kernelgen MCP service is reachable before any sub-skill
(generate / optimize / specialize) executes.

**Transport reality per agent host** (verified 2026-09-01):

| Host | Project `.mcp.json` auto-mounted? | Working transport |
|---|---|---|
| Claude Code | yes (after restart) | native `mcp__kernelgen-server__*` tools |
| ZCode | **no** (never reads it) | `scripts/kernelgen_mcp.py` via Bash |

The bundled script client works on every host, so it is the default transport
in ZCode and the fallback everywhere else.

---

## Step 1: Verify Connectivity

Run from any directory inside the repository:

```bash
python3 <this skill's directory>/scripts/kernelgen_mcp.py list
```

- Prints `4 tools at https://kernelgen.flagos.io/sse` → **configured**, return
  and continue the workflow.
- Exits with a token error or HTTP error → continue to Step 2.

## Step 2: Guide the User to Obtain a Token

Output the following message:

```
The KernelGen MCP toolset is not yet configured.

1. Visit https://kernelgen.flagos.io/mcp to register and obtain your KernelGen Token
   (trial application required; state the purpose, e.g. "Kernel Challenge")
2. Paste the KernelGen Token here, and I will complete the configuration automatically
```

Wait for the user to provide the token.

## Step 3: Write the Configuration

Write (or merge into) `.mcp.json` at the **repository root** — do not delete
other MCP entries if the file exists:

```json
{
  "mcpServers": {
    "kernelgen-server": {
      "type": "http",
      "url": "https://kernelgen.flagos.io/sse",
      "headers": {
        "Authorization": "Bearer <USER_TOKEN>"
      }
    }
  }
}
```

**Endpoint notes**:
- URL must be `https://kernelgen.flagos.io/sse` with **no trailing slash**.
  The `/sse/` form 307-redirects to the SPA and every request fails.
- Keep `.mcp.json` out of git (`.git/info/exclude` or `.gitignore`) — it holds
  a bearer token.
- Token lifetime ≈ through 2026-12; if calls suddenly return 401, re-register.

## Step 4: Re-verify and Continue

Re-run the Step 1 command; on success the workflow continues immediately —
**no restart is needed** for the script transport (a restart is only required
for Claude Code to mount native MCP tools).

---

## Troubleshooting

- `HTTP 401` → token wrong/expired, or `Authorization` header not in
  `headers` of the `kernelgen-server` entry.
- `no JSON-RPC reply with id 1` → wrong URL form (trailing slash) or SPA
  redirect; print the endpoint exactly as in Step 3.
- Timeouts on `call` → server occupies real chips; default timeout is 900 s,
  raise with `--timeout`, and keep at most 2 concurrent jobs server-wide.
