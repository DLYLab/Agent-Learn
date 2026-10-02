"""2.2.2 单轮对话：调用本地 OpenAI 兼容的 Llama 服务。"""

import argparse
import json
import os
from pathlib import Path

from openai import APIConnectionError, APIStatusError, OpenAI


def chat_once(base_url: str, model: str, prompt: str, output: Path) -> str:
    """发送一条用户消息，并返回模型的一条回复。"""
    client = OpenAI(
        base_url=base_url,
        # 本地 llama.cpp 服务通常不会校验 API Key，但 SDK 要求提供该字段。
        api_key=os.getenv("LLAMA_API_KEY", "local-not-used"),
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    response_data = response.model_dump()
    output.parent.mkdir(parents=True, exist_ok=True)
    # 先写临时文件再替换，避免写入中断时破坏原有记录。
    temporary_output = output.with_name(f".{output.name}.tmp")
    with temporary_output.open("w", encoding="utf-8") as file:
        json.dump(response_data, file, ensure_ascii=False, indent=2, default=str)
        file.write("\n")
    temporary_output.replace(output)

    return response.choices[0].message.content or ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="向本地 Qwen3-0.6B 发起一次单轮对话")
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLAMA_BASE_URL", "http://127.0.0.1:11434/v1"),
        help="OpenAI 兼容接口地址（默认读取 LLAMA_BASE_URL）",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LLAMA_MODEL", "qwen3:0.6b"),
        help="服务端模型名（默认读取 LLAMA_MODEL）",
    )
    parser.add_argument(
        "--prompt",
        default="你有多大参数",
        help="本轮发送给模型的消息",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("response.json"),
        help="完整响应的 JSON 输出路径（默认保存到脚本所在目录）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"用户：{args.prompt}")

    try:
        answer = chat_once(args.base_url, args.model, args.prompt, args.output)
    except APIConnectionError:
        raise SystemExit(
            f"无法连接本地模型服务：{args.base_url}\n"
            "请先启动 llama.cpp server，或通过 --base-url 指定实际地址。"
        ) from None
    except APIStatusError as exc:
        raise SystemExit(
            f"模型服务返回 HTTP {exc.status_code}：{exc.message}\n"
            "请检查 --model 与服务端加载的模型名是否一致。"
        ) from None

    print(f"助手：{answer}")


if __name__ == "__main__":
    main()
