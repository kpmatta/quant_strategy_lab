from __future__ import annotations

import json
import os
from typing import Any, Dict, List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from mcp_quant.mcp_client import MCPClientError, call_mcp_tool


class LLMConfigError(RuntimeError):
    pass


class LLMResponseError(RuntimeError):
    pass


SYSTEM_PROMPT = (
    "You are an MCP tool-calling agent for quantitative strategy backtests. "
    "You can call these tools:\n"
    "- list_strategies: {} arguments\n"
    "- get_strategy_schema: {\"name\": string}\n"
    "- sample_price_series: {\"length\": int, \"start\": float, \"drift\": float, \"vol\": float, \"seed\": int}\n"
    "- run_backtest: {\"prices\": [float], \"strategy\": string, \"params\": object, "
    "\"start_cash\": float, \"fee_bps\": float}\n\n"
    "Respond with JSON only. Choose exactly one of:\n"
    "1) {\"tool\": \"<tool_name>\", \"arguments\": { ... }}\n"
    "2) {\"final\": \"<concise answer>\"}\n\n"
    "If you need prices and they are not provided by the user, call sample_price_series first."
)


def _llm_config() -> tuple[str, str, str | None]:
    base = os.getenv("LLM_API_BASE", "https://api.openai.com").rstrip("/")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key and base.startswith("https://api.openai.com"):
        raise LLMConfigError("Set LLM_API_KEY or OPENAI_API_KEY for OpenAI API access")
    return base, model, key


def _extract_json(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        raise LLMResponseError("LLM response did not include JSON")
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMResponseError(f"Invalid JSON from LLM: {exc}") from exc


def _call_llm(messages: List[Dict[str, str]], temperature: float) -> str:
    base, model, key = _llm_config()
    url = f"{base}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
    try:
        with urlopen(request, timeout=30) as response:
            data = json.load(response)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", "ignore") if exc.fp else ""
        raise LLMResponseError(f"LLM HTTP error {exc.code}: {body}") from exc
    except URLError as exc:
        raise LLMResponseError(f"LLM connection error: {exc.reason}") from exc
    choices = data.get("choices") or []
    if not choices:
        raise LLMResponseError("LLM response missing choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not content:
        raise LLMResponseError("LLM response missing content")
    return content


async def run_llm_agent(prompt: str, max_steps: int = 3, temperature: float = 0.2) -> Dict[str, Any]:
    max_steps = max(1, min(int(max_steps), 6))
    temperature = max(0.0, min(float(temperature), 1.5))
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    steps: List[Dict[str, Any]] = []

    for _ in range(max_steps):
        content = _call_llm(messages, temperature)
        parsed = _extract_json(content)
        if "final" in parsed:
            return {
                "final": parsed["final"],
                "steps": steps,
            }
        tool_name = parsed.get("tool")
        if not tool_name:
            raise LLMResponseError("LLM response missing tool or final")
        arguments = parsed.get("arguments") or {}
        print(f"LLM called tool {tool_name} with args: {json.dumps(arguments)}")
        try:
            result = await call_mcp_tool(tool_name, arguments)
        except MCPClientError as exc:
            raise LLMResponseError(f"MCP tool error: {exc}") from exc
        steps.append({"tool": tool_name, "arguments": arguments, "result": result})
        messages.append({"role": "assistant", "content": content})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Tool result for {tool_name}: {json.dumps(result)}. "
                    "Provide the next action as JSON."
                ),
            }
        )

    return {
        "final": "Max tool steps reached without a final response.",
        "steps": steps,
    }
