# 2.2.2 单轮对话：最简单的 API 调用

这个示例向本地部署的 Qwen3-0.6B 发送一条用户消息，并打印一条模型回复。每次运行都是独立的单轮对话，不保存历史消息。

## 运行前提

- Conda 环境：`agent`
- 本地模型服务提供 OpenAI 兼容的 `/v1/chat/completions` 接口
- 默认接口：`http://127.0.0.1:11434/v1`（本机 Ollama）
- 默认模型名：`qwen3:0.6b`

本机已安装 Ollama，并在 `127.0.0.1:11434` 提供服务。可用下面的命令确认服务和模型：

```powershell
ollama list
```

如果改用 llama.cpp，可先用你本机的模型文件启动服务，例如：

```powershell
llama-server.exe -m "你的模型文件.gguf" --host 127.0.0.1 --port 8080
```

`llama-server.exe` 的名称、位置及参数可能因你的安装方式而不同；已经启动服务时无需重复启动。

## 运行

在项目根目录执行：

```powershell
conda run -n agent python ".\对话测试\single_turn_chat.py"
```

发送自定义问题：

```powershell
conda run -n agent python ".\对话测试\single_turn_chat.py" --prompt "北京有哪些著名景点？"
```

如果你的接口地址或模型名不同：

```powershell
conda run -n agent python ".\对话测试\single_turn_chat.py" `
  --base-url "http://127.0.0.1:11434/v1" `
  --model "你的服务端模型名" `
  --prompt "你好"
```

也可以使用环境变量，避免每次传参：

```powershell
$env:LLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
$env:LLAMA_MODEL = "qwen3:0.6b"
conda run -n agent python ".\对话测试\single_turn_chat.py"
```

如果服务启用了鉴权，再设置 `$env:LLAMA_API_KEY`；本地 llama.cpp 默认通常不需要真实密钥。

## 核心代码

单轮对话的关键是 `messages` 中只有当前这一条用户消息：

```python
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": prompt}],
)
```

如果下一次调用时把上一轮消息也放进 `messages`，就变成了多轮对话。
