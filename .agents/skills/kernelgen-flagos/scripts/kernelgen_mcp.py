#!/usr/bin/env python3
"""kernelgen MCP client over Streamable HTTP JSON-RPC (stdlib only).

ZCode does not auto-mount project `.mcp.json`, so the kernelgen MCP tools
(`mcp__kernelgen-server__*`) are normally unavailable as native tools. This
script is the drop-in transport: same server, same tools, same JSON-RPC
results, callable via Bash from anywhere.

Endpoint: https://kernelgen.flagos.io/sse  (Streamable HTTP, NO trailing
slash — the sse-with-slash form 307-redirects to the SPA and fails).

Usage:
  kernelgen_mcp.py list
  kernelgen_mcp.py ping
  kernelgen_mcp.py call <tool> '<json-args>'
  kernelgen_mcp.py call <tool> --file req.json
  kernelgen_mcp.py call <tool> --raw '<json-args>'

Token resolution (first hit wins):
  1. --token / env KERNELGEN_MCP_TOKEN (raw token or full "Bearer ..." value)
  2. .mcp.json next to the repo root or in any parent of the cwd
     (mcpServers key containing "kernelgen", headers.Authorization)

autotune_kernel polling: re-call the same tool with
  {"job_id": ..., "last_seen_attempt": N, "last_seen_version": M, "device": ...}
until status becomes completed/failed (~2 min on real chips; keep at most 2
concurrent jobs server-wide).

Exit codes: 0 ok · 1 transport/protocol error · 2 tool reported isError.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_URL = "https://kernelgen.flagos.io/sse"


def find_mcp_json():
    """Return path of the nearest .mcp.json: cwd upward chain, then repo root."""
    candidates = [Path.cwd(), *Path.cwd().parents]
    candidates.append(Path(__file__).resolve().parents[4])  # scripts/ → repo root
    seen = set()
    for base in candidates:
        if base in seen:
            continue
        seen.add(base)
        p = base / ".mcp.json"
        if p.is_file():
            return p
    return None


def resolve_token(cli_token):
    if cli_token:
        return cli_token
    env = os.environ.get("KERNELGEN_MCP_TOKEN")
    if env:
        return env
    p = find_mcp_json()
    if p:
        try:
            servers = json.loads(p.read_text()).get("mcpServers", {})
            for name, cfg in servers.items():
                if "kernelgen" in name.lower():
                    auth = (cfg.get("headers") or {}).get("Authorization")
                    if auth:
                        return auth
        except (json.JSONDecodeError, OSError):
            pass
    sys.exit(
        "error: no token — pass --token, set KERNELGEN_MCP_TOKEN, or put a "
        "kernelgen-server entry (headers.Authorization) in .mcp.json"
    )


def post(url, authorization, message, timeout):
    req = urllib.request.Request(
        url,
        data=json.dumps(message).encode(),
        headers={
            "Authorization": authorization,
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ctype = resp.headers.get("Content-Type", "")
            body = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:500]
        sys.exit(f"error: HTTP {e.code} from {url}: {detail}")
    except urllib.error.URLError as e:
        sys.exit(f"error: cannot reach {url}: {e.reason}")

    if "event-stream" in ctype:
        for line in body.splitlines():
            if line.startswith("data:"):
                data = line[5:].strip()
                if not data:
                    continue
                try:
                    msg = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if isinstance(msg, dict) and msg.get("id") == message["id"]:
                    return msg
        sys.exit(f"error: no JSON-RPC reply with id {message['id']} in SSE stream")
    try:
        msg = json.loads(body)
    except json.JSONDecodeError:
        sys.exit(f"error: non-JSON response ({ctype}): {body[:500]}")
    if isinstance(msg, list):  # batch: pick our id
        for m in msg:
            if isinstance(m, dict) and m.get("id") == message["id"]:
                return m
        sys.exit(f"error: reply id mismatch in batch response")
    return msg


def rpc(method, params, url, authorization, timeout):
    msg = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    reply = post(url, authorization, msg, timeout)
    if "error" in reply and reply["error"] is not None:
        sys.exit(f"error: JSON-RPC error: {json.dumps(reply['error'], ensure_ascii=False)}")
    return reply.get("result")


def tool_text(result):
    parts = []
    for c in result.get("content", []):
        if c.get("type") == "text":
            parts.append(c["text"])
        else:
            parts.append(json.dumps(c, ensure_ascii=False))
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--url", default=os.environ.get("KERNELGEN_MCP_URL", DEFAULT_URL))
    ap.add_argument("--token", help="Bearer token (else KERNELGEN_MCP_TOKEN / .mcp.json)")
    ap.add_argument("--timeout", type=int, default=900, help="seconds (default 900)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list available MCP tools")
    sub.add_parser("ping", help="verify endpoint + token via tools/list")
    call_p = sub.add_parser("call", help="call one MCP tool")
    call_p.add_argument("tool")
    call_p.add_argument("args", nargs="?", default="{}", help="JSON arguments")
    call_p.add_argument("--file", help="read JSON arguments from file")
    call_p.add_argument("--raw", action="store_true", help="print full JSON-RPC result")
    args = ap.parse_args()

    authorization = resolve_token(args.token)
    if not authorization.lower().startswith("bearer "):
        authorization = f"Bearer {authorization}"
    url = args.url.rstrip("/")

    if args.cmd in ("list", "ping"):
        result = rpc("tools/list", {}, url, authorization, args.timeout)
        tools = result.get("tools", [])
        print(f"{len(tools)} tools at {url}:")
        for t in tools:
            desc = " ".join((t.get("description") or "").split())[:100]
            print(f"  {t['name']}: {desc}")
        return

    raw_args = Path(args.file).read_text() if args.file else args.args
    try:
        arguments = json.loads(raw_args)
    except json.JSONDecodeError as e:
        sys.exit(f"error: arguments are not valid JSON: {e}")
    result = rpc("tools/call", {"name": args.tool, "arguments": arguments},
                 url, authorization, args.timeout)
    if args.raw:
        print(json.dumps(result, ensure_ascii=False, indent=1))
    else:
        text = tool_text(result)
        try:
            print(json.dumps(json.loads(text), ensure_ascii=False, indent=1))
        except (json.JSONDecodeError, TypeError):
            print(text)
    if result.get("isError"):
        sys.exit(2)


if __name__ == "__main__":
    main()
