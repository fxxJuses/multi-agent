# 第十一章 claw-code 的 MCP Client（二）：stdio Manager 的生命周期

上一章我们看到，`claw-code` 会把 `mcpServers` 配置解析成 `McpClientBootstrap`。但 bootstrap 只是“连接说明书”，还没有真的启动 MCP Server。本章进入核心执行层：`McpServerManager` 如何管理本地 stdio MCP Server 的生命周期。

如果说上一章像“把插座线接好”，那么本章就像“真正通电”：启动进程、发送 `initialize`、分页 `tools/list`、维护工具路由、执行 `tools/call`，并在失败时重置 server。

> 源码路径说明：本章重点分析 `/Users/fan/workspace/Agent/claw-code/rust/crates/runtime/src/mcp_stdio.rs` 与 `/Users/fan/workspace/Agent/claw-code/rust/crates/runtime/src/mcp_lifecycle_hardened.rs`。

---

## 11.1 stdio Manager 的工作原理

MCP Client 最朴素的实现方式是：Host 启动一个本地 MCP Server 子进程，然后通过 stdin/stdout 发送 JSON-RPC 消息。`claw-code` 的 `McpServerManager` 就是这条路径的总控模块。

它内部维护四类状态：

- **servers**：每个可管理的 stdio server；
- **unsupported_servers**：配置里出现但当前 manager 不支持的远端 server；
- **tool_index**：模型可见的 qualified tool name 到真实 server/tool 的路由表；
- **next_request_id**：JSON-RPC 请求 id 递增计数器。

我们可以把 manager 的主循环形式化成：

$$\text{call}(qname, args) = \text{send}_{server(qname)}(\text{tools/call}, raw\_tool(qname), args)$$

其中 $qname$ 是 `mcp__server__tool` 这种模型可见名称，`server(qname)` 和 `raw_tool(qname)` 来自 `tool_index`。

本章我们要回答的问题是：**为什么 `claw-code` 当前不能直接连远端 MCP Server？为什么工具发现失败不会让整个 CLI 彻底崩掉？为什么工具调用前还要反复 `ensure_server_ready`？**

---

## 11.2 Manager 注册：只有 stdio 会被真正管理

（1）`from_servers`：把可运行与不可运行的 server 分开

`McpServerManager::from_servers` 是一个非常关键的分水岭。配置层支持很多 transport，但 manager 这里只接收 `stdio`：

```rust
pub fn from_servers(servers: &BTreeMap<String, ScopedMcpServerConfig>) -> Self {
    let mut managed_servers = BTreeMap::new();
    let mut unsupported_servers = Vec::new();

    for (server_name, server_config) in servers {
        if server_config.transport() == McpTransport::Stdio {
            let bootstrap = McpClientBootstrap::from_scoped_config(server_name, server_config);
            managed_servers.insert(server_name.clone(), ManagedMcpServer::new(bootstrap));
        } else {
            unsupported_servers.push(UnsupportedMcpServer {
                server_name: server_name.clone(),
                transport: server_config.transport(),
                reason: format!(
                    "transport {:?} is not supported by McpServerManager",
                    server_config.transport()
                ),
            });
        }
    }

    Self {
        servers: managed_servers,
        unsupported_servers,
        tool_index: BTreeMap::new(),
        next_request_id: 1,
    }
}
```

这段代码的设计要点：

- **`McpTransport::Stdio` 白名单**：只有本地 stdio server 会进入 `managed_servers`。
- **`unsupported_servers`**：远端配置不会被静默丢弃，而是进入 degraded/pending 报告。
- **`tool_index` 延后构建**：只有 `tools/list` 成功后，工具路由才会被写入。

这也解释了前面讨论过的问题：`claw-code` 的 MCP Client 当前**不能直接对接各类云平台远端 MCP**。配置能表达远端，但 `McpServerManager` 没有 HTTP/SSE/WebSocket transport 实现。

（2）`ManagedMcpServer`：每个 server 的运行态

```rust
struct ManagedMcpServer {
    bootstrap: McpClientBootstrap,
    process: Option<McpStdioProcess>,
    initialized: bool,
}

impl ManagedMcpServer {
    fn new(bootstrap: McpClientBootstrap) -> Self {
        Self {
            bootstrap,
            process: None,
            initialized: false,
        }
    }
}
```

这个结构的设计要点：

- **`bootstrap`**：保存启动命令和工具命名前缀。
- **`process`**：只有真正需要连接时才启动子进程，避免配置加载阶段产生副作用。
- **`initialized`**：区分“进程已启动”和“MCP 握手已完成”。

---

## 11.3 wire format：为什么这里不是一行一个 JSON

（1）Content-Length framing

我们在 `multi-agent` 里写的教学 client 用的是“每行一个 JSON”的简化版本；`claw-code` 更接近 LSP/MCP 的 stdio framing：每条消息前面带 `Content-Length`。

```rust
fn encode_frame(payload: &[u8]) -> Vec<u8> {
    let header = format!("Content-Length: {}\r\n\r\n", payload.len());
    let mut framed = header.into_bytes();
    framed.extend_from_slice(payload);
    framed
}
```

这段代码的设计要点：

- **`Content-Length`**：告诉接收方接下来要读多少字节，避免 JSON 内部换行破坏消息边界。
- **`\r\n\r\n`**：header 与 body 的分隔符，沿用 LSP 风格。
- **字节长度**：长度按 UTF-8 bytes 计算，而不是字符数。

（2）读取 frame：先读 header，再读 body

```rust
pub async fn read_frame(&mut self) -> io::Result<Vec<u8>> {
    let mut content_length = None;
    loop {
        let mut line = String::new();
        let bytes_read = self.stdout.read_line(&mut line).await?;
        if bytes_read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "MCP stdio stream closed while reading headers",
            ));
        }
        if line == "\r\n" {
            break;
        }
        let header = line.trim_end_matches(['\r', '\n']);
        if let Some((name, value)) = header.split_once(':') {
            if name.trim().eq_ignore_ascii_case("Content-Length") {
                content_length = Some(value.trim().parse::<usize>()?);
            }
        }
    }

    let content_length = content_length.ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "missing Content-Length header")
    })?;
    let mut payload = vec![0_u8; content_length];
    self.stdout.read_exact(&mut payload).await?;
    Ok(payload)
}
```

这个读取器的设计要点：

- **大小写兼容**：`Content-Length` header 用 `eq_ignore_ascii_case` 识别。
- **EOF 明确报错**：server 退出或 stdout 关闭会被归类为传输错误。
- **严格长度读取**：只读指定长度，避免多条消息粘包。

---

## 11.4 连接生命周期：`ensure_server_ready`

（1）先检查进程，再决定是否 spawn

每次发现工具、列资源、调用工具之前，manager 都会调用 `ensure_server_ready`。这看起来有点重复，但它让调用路径具备自修复能力：server 进程死了，就重置并重新启动。

```rust
async fn ensure_server_ready(
    &mut self,
    server_name: &str,
) -> Result<(), McpServerManagerError> {
    if self.server_process_exited(server_name)? {
        self.reset_server(server_name).await?;
    }

    let needs_spawn = self
        .servers
        .get(server_name)
        .map(|server| server.process.is_none())
        .ok_or_else(|| McpServerManagerError::UnknownServer {
            server_name: server_name.to_string(),
        })?;

    if needs_spawn {
        let server = self.server_mut(server_name)?;
        server.process = Some(spawn_mcp_stdio_process(&server.bootstrap)?);
        server.initialized = false;
    }

    // 后续继续判断是否需要 initialize
    Ok(())
}
```

这段代码的设计要点：

- **懒启动**：只有真正要用 server 时才启动进程。
- **进程死亡检测**：如果子进程已经退出，先 `reset_server`，避免拿旧句柄继续读写。
- **初始化状态清零**：重新 spawn 后必须重新 `initialize`。

（2）发送 initialize

真正握手时，manager 会发送 `initialize`，等待 result，并把 `initialized` 标记为 `true`：

```rust
let request_id = self.take_request_id();
let response = {
    let server = self.server_mut(server_name)?;
    let process = server.process.as_mut().ok_or_else(|| {
        McpServerManagerError::InvalidResponse {
            server_name: server_name.to_string(),
            method: "initialize",
            details: "server process missing before initialize".to_string(),
        }
    })?;
    Self::run_process_request(
        server_name,
        "initialize",
        MCP_INITIALIZE_TIMEOUT_MS,
        process.initialize(request_id, default_initialize_params()),
    )
    .await
};

let server = self.server_mut(server_name)?;
server.initialized = true;
```

这段代码的设计要点：

- **`request_id`**：每个 JSON-RPC 请求都有唯一 id，用于响应匹配。
- **初始化超时**：生产环境 `initialize` 默认 10 秒，防止 server 挂死。
- **结果校验**：如果没有 result payload，会被当成 invalid response 并 reset。

（3）默认初始化参数

```rust
fn default_initialize_params() -> McpInitializeParams {
    McpInitializeParams {
        protocol_version: "2025-03-26".to_string(),
        capabilities: JsonValue::Object(serde_json::Map::new()),
        client_info: McpInitializeClientInfo {
            name: "runtime".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        },
    }
}
```

这个参数块的设计要点：

- **`protocol_version`**：声明客户端希望使用的 MCP 版本。
- **`capabilities`**：当前传空对象，表示最小 client 能力。
- **`client_info`**：给 server 记录调用来源。

---

## 11.5 工具发现：`tools/list` 如何建立路由表

（1）分页发现工具

`discover_tools_for_server_once` 会循环调用 `tools/list`，直到没有 `next_cursor`：

```rust
let mut discovered_tools = Vec::new();
let mut cursor = None;
loop {
    let request_id = self.take_request_id();
    let response = {
        let server = self.server_mut(server_name)?;
        let process = server.process.as_mut().ok_or_else(|| {
            McpServerManagerError::InvalidResponse {
                server_name: server_name.to_string(),
                method: "tools/list",
                details: "server process missing after initialization".to_string(),
            }
        })?;
        Self::run_process_request(
            server_name,
            "tools/list",
            MCP_LIST_TOOLS_TIMEOUT_MS,
            process.list_tools(request_id, Some(McpListToolsParams { cursor: cursor.clone() })),
        )
        .await?
    };

    let result = response.result.ok_or_else(|| McpServerManagerError::InvalidResponse {
        server_name: server_name.to_string(),
        method: "tools/list",
        details: "missing result payload".to_string(),
    })?;

    for tool in result.tools {
        let qualified_name = mcp_tool_name(server_name, &tool.name);
        discovered_tools.push(ManagedMcpTool {
            server_name: server_name.to_string(),
            qualified_name,
            raw_name: tool.name.clone(),
            tool,
        });
    }

    match result.next_cursor {
        Some(next_cursor) => cursor = Some(next_cursor),
        None => break,
    }
}
```

这段代码的设计要点：

- **分页协议**：`cursor` / `next_cursor` 支持大量工具列表。
- **qualified name**：发现阶段就把 raw tool name 转成模型可见名称。
- **保留 raw name**：真正 `tools/call` 时仍然要把 server 原始工具名发回去。

（2）写入 `tool_index`

发现工具成功后，manager 会建立 `qualified_name -> ToolRoute` 的路由表：

```rust
for tool in server_tools {
    self.tool_index.insert(
        tool.qualified_name.clone(),
        ToolRoute {
            server_name: tool.server_name.clone(),
            raw_name: tool.raw_name.clone(),
        },
    );
    discovered_tools.push(tool);
}
```

这个路由表的设计要点：

- **模型用 qualified name**：例如 `mcp__playwright__browser_screenshot`。
- **server 用 raw name**：例如 `browser_screenshot`。
- **路由解耦**：模型侧不需要知道工具来自哪个 server，manager 根据索引反查。

---

## 11.6 工具调用：从 qualified name 回到 `tools/call`

当模型选择了一个 MCP 工具时，CLI 会把工具名传给 manager。manager 首先用 `tool_index` 查路由，再向对应 server 发 `tools/call`。

```rust
pub async fn call_tool(
    &mut self,
    qualified_tool_name: &str,
    arguments: Option<JsonValue>,
) -> Result<JsonRpcResponse<McpToolCallResult>, McpServerManagerError> {
    let route = self
        .tool_index
        .get(qualified_tool_name)
        .cloned()
        .ok_or_else(|| McpServerManagerError::UnknownTool {
            qualified_name: qualified_tool_name.to_string(),
        })?;

    let timeout_ms = self.tool_call_timeout_ms(&route.server_name)?;

    self.ensure_server_ready(&route.server_name).await?;
    let request_id = self.take_request_id();
    let response = {
        let server = self.server_mut(&route.server_name)?;
        let process = server.process.as_mut().ok_or_else(|| {
            McpServerManagerError::InvalidResponse {
                server_name: route.server_name.clone(),
                method: "tools/call",
                details: "server process missing after initialization".to_string(),
            }
        })?;
        Self::run_process_request(
            &route.server_name,
            "tools/call",
            timeout_ms,
            process.call_tool(
                request_id,
                McpToolCallParams {
                    name: route.raw_name,
                    arguments,
                    meta: None,
                },
            ),
        )
        .await
    };

    if let Err(error) = &response {
        if Self::should_reset_server(error) {
            self.reset_server(&route.server_name).await?;
        }
    }

    response
}
```

这段代码的设计要点：

- **`UnknownTool`**：如果工具没在发现阶段建立路由，不能调用。
- **per-server timeout**：工具调用超时可以由 server 配置覆盖。
- **调用前保活**：即使发现工具后 server 退出，调用前也会重新确保 ready。
- **失败后 reset**：传输错误或超时会触发 reset，避免下次继续用坏进程。

---

## 11.7 错误分类与 degraded startup

（1）错误按生命周期阶段归类

`McpServerManagerError` 不只是字符串，它会映射到 MCP 生命周期阶段：

```rust
fn lifecycle_phase_for_method(method: &str) -> McpLifecyclePhase {
    match method {
        "initialize" => McpLifecyclePhase::InitializeHandshake,
        "tools/list" => McpLifecyclePhase::ToolDiscovery,
        "resources/list" => McpLifecyclePhase::ResourceDiscovery,
        "resources/read" | "tools/call" => McpLifecyclePhase::Invocation,
        _ => McpLifecyclePhase::ErrorSurfacing,
    }
}
```

这个分类的设计要点：

- **可观测性**：错误不再只是“失败了”，而是“在哪个阶段失败”。
- **恢复策略**：`Transport` / `Timeout` 这类错误可以被标为 recoverable。
- **用户提示**：degraded report 可以告诉用户哪些 server pending，哪些工具仍可用。

（2）best effort discovery

启动时 `discover_tools_best_effort` 不会因为一个 server 坏掉就让全部 MCP 不可用：

```rust
pub async fn discover_tools_best_effort(&mut self) -> McpToolDiscoveryReport {
    let server_names = self.server_names();
    let mut discovered_tools = Vec::new();
    let mut working_servers = Vec::new();
    let mut failed_servers = Vec::new();

    for server_name in server_names {
        match self.discover_tools_for_server(&server_name).await {
            Ok(server_tools) => {
                working_servers.push(server_name.clone());
                self.clear_routes_for_server(&server_name);
                // 成功工具写入 tool_index
            }
            Err(error) => {
                self.clear_routes_for_server(&server_name);
                failed_servers.push(error.discovery_failure(&server_name));
            }
        }
    }

    McpToolDiscoveryReport {
        tools: discovered_tools,
        failed_servers,
        unsupported_servers: self.unsupported_servers.clone(),
        degraded_startup,
    }
}
```

这段代码的设计要点：

- **部分成功**：一个 MCP Server 坏了，不影响另一个 MCP Server 的工具注册。
- **清理旧路由**：失败 server 的旧工具路由会被清掉，避免模型调用不存在的工具。
- **degraded report**：远端 unsupported server 和启动失败 server 都会被报告出来。

---

## 11.8 运行实例与分析

下面是按 `McpServerManager` 路径整理的一次运行记录：

```text
📚 读取 mcpServers：playwright(stdio), notion(http)
✅ playwright 进入 managed_servers
⚠️ notion 进入 unsupported_servers：transport Http is not supported by McpServerManager
🧩 ensure_server_ready(playwright)
✅ spawn stdio process：npx @playwright/mcp
🧠 发送 initialize，protocolVersion=2025-03-26
✅ initialize 成功，server.initialized=true
🔍 调用 tools/list cursor=null
🎯 发现工具：browser_navigate
🎯 发现工具：browser_screenshot
✅ 建立路由：mcp__playwright__browser_screenshot -> playwright/browser_screenshot
🎬 模型调用 mcp__playwright__browser_screenshot
🔍 manager 路由到 tools/call name=browser_screenshot
👀 Server 返回 content / structuredContent
🎉 CLI 将结果序列化回 Tool Observation
```

从上面的输出可以看到，`claw-code` 的 stdio manager 清晰展示了三个特征：

1. **当前运行能力是 stdio-first**：远端配置进入 degraded，而不是被真正连接；
2. **发现阶段建立路由**：模型看到的工具名和 server 内部 raw name 被解耦；
3. **调用阶段持续保活**：每次调用前都确保 server ready，失败后 reset。

---

## 11.9 特点、局限性与调试技巧

（1）主要特点

1. **真实协议 framing**：采用 `Content-Length` frame，而不是教学版 newline JSON。
2. **生命周期清晰**：spawn、initialize、discover、invoke、shutdown 都有明确函数边界。
3. **degraded 可用**：部分 server 失败时仍保留已发现工具，提升启动韧性。

（2）固有局限性

1. **只支持 stdio 运行态**：HTTP/SSE/WS 配置会进入 unsupported，而不是远端连接。
2. **同步路由依赖发现结果**：如果启动时 `tools/list` 失败，相关工具不会进入 `tool_index`。
3. **Server stdout 必须干净**：stdio 协议 stdout 只能写 framed JSON-RPC，日志应写 stderr。

（3）调试技巧

- **先看 unsupported_servers**：如果远端 MCP 没出现工具，确认它是否被 manager 标为 unsupported。
- **检查 stdout 污染**：本地 MCP Server 如果把日志写到 stdout，会破坏 `Content-Length` framing。
- **按生命周期定位错误**：`initialize` 失败看启动/协议版本；`tools/list` 失败看工具 schema；`tools/call` 失败看参数和执行环境。
- **确认 tool_index**：模型调用失败为 `UnknownTool` 时，通常说明该工具没有成功发现。

下一章我们会继续往上走：这些 `ManagedMcpTool` 如何变成 LLM 可见的 `RuntimeToolDefinition`，又如何通过 CLI 的 `ToolExecutor` 真正被模型调用。

---

## 参考源码

[1] `/Users/fan/workspace/Agent/claw-code/rust/crates/runtime/src/mcp_stdio.rs`

[2] `/Users/fan/workspace/Agent/claw-code/rust/crates/runtime/src/mcp_lifecycle_hardened.rs`
