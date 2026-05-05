"""教学用：模拟 MCP 风格的 JSON-RPC 往返（非官方 MCP SDK）。

运行（在仓库根目录）：`python example/mcp-jsonrpc-concept-demo.py`
对应文档：`docs/9-mcp-model-context-protocol.md`
"""
import json


def main() -> None:
    print("--- MCP 概念演示：Host / Client / Server 消息流 ---")
    print("✅ Host 已获取用户对「本地 demo-server」的连接授权")
    print("🧠 Client 发送 initialize（能力协商）")
    init_req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "demo-host", "version": "0.1"},
        },
    }
    print(json.dumps(init_req, ensure_ascii=False))
    print("👀 Server 返回 initialize 结果（声明可提供 tools）")
    init_res = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "demo-server", "version": "0.1"},
        },
    }
    print(json.dumps(init_res, ensure_ascii=False))
    print("🔍 Client 列出工具：tools/list")
    print('{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}')
    print("👀 Server：当前注册 1 个 tool → `read_file`")
    print(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {
                    "tools": [
                        {
                            "name": "read_file",
                            "description": "Read a UTF-8 text file by path",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string"}
                                },
                                "required": ["path"],
                            },
                        }
                    ]
                },
            },
            ensure_ascii=False,
        )
    )
    print("🎬 Host 侧 LLM 决定调用工具（模型输出 tool call）")
    print("🔍 Client 执行 tools/call：read_file")
    call_req = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {"name": "read_file", "arguments": {"path": "./README.md"}},
    }
    print(json.dumps(call_req, ensure_ascii=False))
    print("👀 Server 返回文本内容（截断示意）")
    print(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "result": {
                    "content": [
                        {"type": "text", "text": "# multi-agent\n...(truncated)..."}
                    ],
                    "isError": False,
                },
            },
            ensure_ascii=False,
        )
    )
    print("🎉 Host 将 Observation 写回对话，供下一轮 LLM 生成最终答案")


if __name__ == "__main__":
    main()
