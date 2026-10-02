"""2.2.3 带工具调用的多轮交互：Agent 的核心循环。"""

import argparse
import ast
import json
import math
import operator
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from openai import APIConnectionError, APIStatusError, OpenAI


SYSTEM_PROMPT = """你是一个可以调用工具的中文助手。
遇到数学计算时必须调用 calculate 工具，询问当前时间时必须调用 get_current_time 工具。
拿到工具结果后，再用自然、简洁的中文回答用户。"""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "安全地计算一个数学表达式，支持加减乘除、整除、取余和乘方。",
            "parameters": {  # parameters = 参数整体
                "type": "object",
                "properties": {  # properties = 参数里面有哪些具体字段
                    "expression": {
                        "type": "string",
                        "description": "需要计算的数学表达式，例如 (23 + 7) * 4",
                    }
                },
                "required": ["expression"],  # expression 必须字段
                "additionalProperties": False,  # 对象里不能出现 properties 没有声明过的额外字段。（强迫模型调用tool生成参数字段的限制）
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取运行此程序的计算机当前本地日期、时间和 UTC 偏移量。",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
]


class SafeCalculator(ast.NodeVisitor):
    """只计算允许的数字和运算符，不使用 eval。"""

    binary_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    unary_operators = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def visit_Expression(self, node: ast.Expression) -> int | float:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> int | float:
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("表达式中只能包含数字")
        return node.value

    def visit_BinOp(self, node: ast.BinOp) -> int | float:
        operation = self.binary_operators.get(type(node.op))
        if operation is None:
            raise ValueError("表达式包含不支持的二元运算符")
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("乘方指数不能超过 100")
        return operation(left, right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> int | float:
        operation = self.unary_operators.get(type(node.op))
        if operation is None:
            raise ValueError("表达式包含不支持的一元运算符")
        return operation(self.visit(node.operand))

    def generic_visit(self, node: ast.AST) -> Any:
        raise ValueError(f"表达式包含不支持的语法：{type(node).__name__}")


def calculate(expression: str) -> dict[str, int | float | str]:
    """计算模型传入的数学表达式。"""
    if len(expression) > 200:
        raise ValueError("表达式长度不能超过 200 个字符")
    tree = ast.parse(expression, mode="eval")
    result = SafeCalculator().visit(tree)
    if isinstance(result, float) and not math.isfinite(result):
        raise ValueError("计算结果不是有限数值")
    return {"expression": expression, "result": result}


def get_current_time() -> dict[str, str]:
    """返回计算机当前本地时间。"""
    now = datetime.now().astimezone()
    return {
        "datetime": now.isoformat(timespec="seconds"),
        "timezone": now.tzname() or "local",
        "utc_offset": now.strftime("%z"),
    }


TOOL_HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {
    "calculate": calculate,
    "get_current_time": get_current_time,
}


def parse_tool_arguments(raw_arguments: str | dict[str, Any] | None) -> dict[str, Any]:
    if raw_arguments is None:
        return {}
    if isinstance(raw_arguments, dict):
        return raw_arguments
    arguments = json.loads(raw_arguments)
    if not isinstance(arguments, dict):
        raise ValueError("工具参数必须是 JSON 对象")
    return arguments


def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """执行白名单中的工具，并把成功或错误统一转成 JSON 数据。"""
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return {"ok": False, "error": f"未知工具：{name}"}
    try:
        return {"ok": True, "data": handler(**arguments)}
    except (ArithmeticError, SyntaxError, TypeError, ValueError) as exc:
        return {"ok": False, "error": str(exc)}


def run_agent_turn(
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    max_tool_rounds: int,
) -> str:
    """运行一次用户回合，直到模型给出最终文本或超过工具调用上限。"""
    executed_rounds = 0

    while True:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            temperature=0.7,
        )
        assistant_message = response.choices[0].message
        tool_calls = assistant_message.tool_calls or []

        normalized_calls = []
        for index, tool_call in enumerate(tool_calls):
            call_data = tool_call.model_dump(exclude_none=True)  # 转换为dict，并去除none值
            call_data["id"] = tool_call.id or f"call_{executed_rounds}_{index}"
            normalized_calls.append(call_data)

        assistant_entry: dict[str, Any] = {
            "role": "assistant",
            "content": assistant_message.content,
        }
        if normalized_calls:
            assistant_entry["tool_calls"] = normalized_calls
        messages.append(assistant_entry)  # 对于role为assistant和tool的消息需要先处理一下再放入messages。

        if not tool_calls:
            return assistant_message.content or ""

        if executed_rounds >= max_tool_rounds:
            raise RuntimeError(f"工具调用超过上限（{max_tool_rounds} 轮）")
        executed_rounds += 1

        for tool_call, call_data in zip(tool_calls, normalized_calls):  # tool_calls 是模型返回的原始工具调用对象，normalized_calls 是程序整理后的标准字典。
            tool_name = tool_call.function.name
            try:
                arguments = parse_tool_arguments(tool_call.function.arguments)
                tool_result = execute_tool(tool_name, arguments)
            except (json.JSONDecodeError, ValueError) as exc:
                arguments = {}
                tool_result = {"ok": False, "error": f"工具参数解析失败：{exc}"}

            print(
                f"[工具调用] {tool_name}"
                f"({json.dumps(arguments, ensure_ascii=False)})"
            )
            print(f"[工具结果] {json.dumps(tool_result, ensure_ascii=False)}")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_data["id"],
                    "content": json.dumps(tool_result, ensure_ascii=False),
                }
            )


def save_history(path: Path, model: str, messages: list[dict[str, Any]]) -> None:
    """将完整多轮上下文保存成一个格式化 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(
            {"model": model, "messages": messages},
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        file.write("\n")
    temporary_path.replace(path)


def ask_once(
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    prompt: str,
    max_tool_rounds: int,
    history_output: Path,
) -> None:
    messages.append({"role": "user", "content": prompt})
    answer = run_agent_turn(client, model, messages, max_tool_rounds)
    print(f"助手：{answer}")
    save_history(history_output, model, messages)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="带工具调用的多轮 Agent")
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLAMA_BASE_URL", "http://127.0.0.1:11434/v1"),
        help="OpenAI 兼容接口地址",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LLAMA_MODEL", "qwen3:0.6b"),
        help="服务端模型名",
    )
    parser.add_argument(
        "--prompt",
        help="执行单个用户回合；省略时进入多轮交互模式",
    )
    parser.add_argument(
        "--history-output",
        type=Path,
        default=Path(__file__).with_name("agent_history.json"),
        help="多轮上下文 JSON 保存路径",
    )
    parser.add_argument(
        "--max-tool-rounds",
        type=int,
        default=5,
        help="单个用户回合最多允许的工具调用轮数",
    )
    args = parser.parse_args()
    if args.max_tool_rounds < 1:
        parser.error("--max-tool-rounds 必须大于等于 1")
    return args


def main() -> None:
    args = parse_args()
    client = OpenAI(
        base_url=args.base_url,
        api_key=os.getenv("LLAMA_API_KEY", "local-not-used"),
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    try:
        if args.prompt:
            print(f"用户：{args.prompt}")
            ask_once(
                client,
                args.model,
                messages,
                args.prompt,
                args.max_tool_rounds,
                args.history_output,
            )
            return

        print("进入多轮 Agent 对话。输入 exit 或 quit 结束。")
        while True:
            try:
                prompt = input("用户：").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n对话结束。")
                break
            if prompt.lower() in {"exit", "quit"}:
                print("对话结束。")
                break
            if not prompt:
                continue
            ask_once(
                client,
                args.model,
                messages,
                prompt,
                args.max_tool_rounds,
                args.history_output,
            )
    except APIConnectionError:
        raise SystemExit(
            f"无法连接本地模型服务：{args.base_url}\n请确认 Ollama 正在运行。"
        ) from None
    except APIStatusError as exc:
        raise SystemExit(
            f"模型服务返回 HTTP {exc.status_code}：{exc.message}"
        ) from None
    except RuntimeError as exc:
        raise SystemExit(f"Agent 运行失败：{exc}") from None


if __name__ == "__main__":
    main()
