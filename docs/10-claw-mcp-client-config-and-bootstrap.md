# 第十章 claw-code 的 MCP Client（一）：从配置到 Bootstrap

在第九章里，我们从协议层理解了 MCP：Host 通过 Client 连接 Server，先 `initialize`，再 `tools/list`，最后 `tools/call`。然而一旦把这个协议放进真实 CLI 工程里，问题会立刻变得更具体：**用户写在配置文件里的 `mcpServers`，到底怎样变成一个可以被 Agent 调用的工具？**

本章我们借 `claw-code` 的 Rust 实现，把 MCP Client 的第一段链路拆开：**配置加载 → 作用域合并 → transport 建模 → bootstrap → 工具命名**。如果说第九章的 MCP 像“国标插座”，那么本章要看的就是“电工怎么把墙里的线接到插座背面”。

> 源码路径说明：本章分析的是 `/Users/fan/workspace/Agent/claw-code` 中的实现，重点文件包括 `rust/crates/runtime/src/config.rs`、`rust/crates/runtime/src/mcp_client.rs`、`rust/crates/runtime/src/mcp.rs`。

---

## 10.1 配置进入 MCP Client 的工作原理

在一个真实 Agent CLI 中，MCP 不是从 `MCPStdioClient::new(...)` 这种硬编码入口开始的，而是从配置开始的。用户可能在全局配置、项目配置、本地配置里写不同的 `mcpServers`，然后运行时要合并出一份“最终可用”的 MCP server 列表。

`claw-code` 的做法可以概括成四步：

- **ConfigLoader**：发现并读取多层设置文件；
- **merge_mcp_servers**：从每个设置文件里抽取 `mcpServers` 并按作用域覆盖；
- **McpServerConfig**：把 JSON 配置解析成结构化 enum；
- **McpClientBootstrap**：把某个 server 的配置变成“连接前的启动说明书”。

用一个形式化表达就是：我们把原始配置集合 $\mathcal{C}$ 变成一组 bootstrap $\mathcal{B}$：

$$\mathcal{B} = \{\text{bootstrap}(name, cfg) \mid (name, cfg) \in \text{merge}(\mathcal{C})\}$$

这里最重要的直觉是：**bootstrap 还不是连接**。它只是告诉后续的 manager：“这个 server 叫什么、工具名前缀是什么、要用哪种 transport、签名是什么”。真正启动进程、握手、发现工具，要到下一章的 `McpServerManager` 才发生。

这种拆分特别适合以下场景：

- 多层配置会覆盖：例如 user / project / local 三层设置共同决定最终 server 列表；
- 支持多种 transport 形状：即使当前只实现 `stdio`，配置层也可以先表达 `http`、`sse`、`ws`；
- 后续需要可观测性：bootstrap 中的 signature、normalized name 可以帮助判断“配置是否变化”。

本章我们要解决的问题是：**读者看到一个 `mcpServers` 配置时，能顺着源码回答它会被解析成什么、命名成什么、是否真的可执行。**

---

## 10.2 配置模型：`McpServerConfig` 如何描述不同 Server

（1）transport enum：先把“连接方式”抽象出来

`config.rs` 里先定义了 MCP server 支持的 transport 家族：

```rust
pub enum McpTransport {
    Stdio,
    Sse,
    Http,
    Ws,
    Sdk,
    ManagedProxy,
}

pub enum McpServerConfig {
    Stdio(McpStdioServerConfig),
    Sse(McpRemoteServerConfig),
    Http(McpRemoteServerConfig),
    Ws(McpWebSocketServerConfig),
    Sdk(McpSdkServerConfig),
    ManagedProxy(McpManagedProxyServerConfig),
}
```

这段代码的设计要点：

- **`McpTransport`**：给上层一个轻量标签，便于判断某个 server 属于哪类连接方式。
- **`McpServerConfig`**：用 Rust enum 承载不同 transport 的专属字段，避免所有字段揉成一个大 struct。
- **配置层先行**：这里能表达 `Http` / `Sse` / `Ws`，并不等于运行时 manager 已经实现了这些远端传输。

（2）stdio 配置：本地进程最小闭环

`stdio` server 对应的是本地可执行命令：

```rust
pub struct McpStdioServerConfig {
    pub command: String,
    pub args: Vec<String>,
    pub env: BTreeMap<String, String>,
    pub tool_call_timeout_ms: Option<u64>,
}
```

这个配置的设计要点：

- **`command` / `args`**：决定本地要启动哪个 MCP Server 进程。
- **`env`**：把 token、路径、实验开关等变量注入 server 子进程。
- **`tool_call_timeout_ms`**：把工具调用超时做成 per-server 配置，而不是写死成全局值。

（3）remote 配置：先能解析，但不代表已能执行

远端 HTTP/SSE server 使用统一的远端配置：

```rust
pub struct McpRemoteServerConfig {
    pub url: String,
    pub headers: BTreeMap<String, String>,
    pub headers_helper: Option<String>,
    pub oauth: Option<McpOAuthConfig>,
}
```

这个配置的设计要点：

- **`url`**：远端 MCP endpoint。
- **`headers` / `headers_helper`**：支持静态 header，也支持通过 helper 动态生成 header。
- **`oauth`**：配置层已经为远端授权预留了结构，但 `McpServerManager` 当前没有真正实现远端连接。

这里要特别强调一个容易误读的点：**`claw-code` 的配置层“认识”远端 MCP，但当前运行时 manager 只真正支持 `stdio`。**远端配置会被加载、合并、记录为 pending/degraded，但不会被 `McpServerManager` 连接成功。

---

## 10.3 解析规则：`mcpServers` 如何变成结构化配置

（1）从设置文件里合并 `mcpServers`

`merge_mcp_servers` 的核心逻辑很直接：如果当前配置文件没有 `mcpServers`，跳过；如果有，就逐个 server 解析并写入目标 map。

```rust
fn merge_mcp_servers(
    target: &mut BTreeMap<String, ScopedMcpServerConfig>,
    source: ConfigSource,
    root: &BTreeMap<String, JsonValue>,
    path: &Path,
) -> Result<(), ConfigError> {
    let Some(mcp_servers) = root.get("mcpServers") else {
        return Ok(());
    };
    let servers = expect_object(mcp_servers, &format!("{}: mcpServers", path.display()))?;
    for (name, value) in servers {
        let parsed = parse_mcp_server_config(
            name,
            value,
            &format!("{}: mcpServers.{name}", path.display()),
        )?;
        target.insert(
            name.clone(),
            ScopedMcpServerConfig {
                scope: source,
                config: parsed,
            },
        );
    }
    Ok(())
}
```

这段代码的设计要点：

- **`ScopedMcpServerConfig`**：保留 server 来自哪个作用域，便于后续调试与覆盖判断。
- **`target.insert`**：同名 server 后写入的配置会覆盖前面的配置，符合“更近作用域优先”的直觉。
- **`context` 字符串**：错误信息会带上 `mcpServers.{name}`，让配置错误能定位到具体 server。

（2）类型推断：有 `url` 默认就是 HTTP

配置解析函数通过 `type` 字段决定 server 类型；如果没有显式 `type`，则根据是否存在 `url` 推断：

```rust
fn parse_mcp_server_config(
    server_name: &str,
    value: &JsonValue,
    context: &str,
) -> Result<McpServerConfig, ConfigError> {
    let object = expect_object(value, context)?;
    let server_type =
        optional_string(object, "type", context)?.unwrap_or_else(|| infer_mcp_server_type(object));
    match server_type {
        "stdio" => Ok(McpServerConfig::Stdio(McpStdioServerConfig {
            command: expect_string(object, "command", context)?.to_string(),
            args: optional_string_array(object, "args", context)?.unwrap_or_default(),
            env: optional_string_map(object, "env", context)?.unwrap_or_default(),
            tool_call_timeout_ms: optional_u64(object, "toolCallTimeoutMs", context)?,
        })),
        "sse" => Ok(McpServerConfig::Sse(parse_mcp_remote_server_config(object, context)?)),
        "http" => Ok(McpServerConfig::Http(parse_mcp_remote_server_config(object, context)?)),
        other => Err(ConfigError::Parse(format!(
            "{context}: unsupported MCP server type for {server_name}: {other}"
        ))),
    }
}

fn infer_mcp_server_type(object: &BTreeMap<String, JsonValue>) -> &'static str {
    if object.contains_key("url") {
        "http"
    } else {
        "stdio"
    }
}
```

这段代码的设计要点：

- **显式优先**：配置里写了 `type` 就按 `type` 走。
- **URL 推断**：只写 `url` 的配置会被当成 `http` MCP server。
- **本地默认**：没有 `url` 时默认 `stdio`，符合多数本地 MCP server 的配置习惯。

---

## 10.4 Bootstrap：配置到连接目标的最后一步

配置解析后，`mcp_client.rs` 负责把它变成 `McpClientBootstrap`。这一步像是在真正打电话之前，把联系人姓名、号码、备注和拨号方式写到通讯录里。

（1）bootstrap 保存连接前需要的元信息

```rust
pub struct McpClientBootstrap {
    pub server_name: String,
    pub normalized_name: String,
    pub tool_prefix: String,
    pub signature: Option<String>,
    pub transport: McpClientTransport,
}

impl McpClientBootstrap {
    pub fn from_scoped_config(server_name: &str, config: &ScopedMcpServerConfig) -> Self {
        Self {
            server_name: server_name.to_string(),
            normalized_name: normalize_name_for_mcp(server_name),
            tool_prefix: mcp_tool_prefix(server_name),
            signature: mcp_server_signature(&config.config),
            transport: McpClientTransport::from_config(&config.config),
        }
    }
}
```

这个 bootstrap 的设计要点：

- **`server_name`**：保留用户配置里的原始名字，用于错误信息与 UI 展示。
- **`normalized_name` / `tool_prefix`**：为模型工具名做稳定、可预测的命名。
- **`signature`**：对 server 的连接目标做摘要，例如 `stdio:[uvx|mcp-server]`，便于判断配置变化。
- **`transport`**：把配置 enum 转成 client 启动时实际需要的 transport enum。

（2）transport 映射：远端形状存在，但运行时不一定支持

```rust
impl McpClientTransport {
    pub fn from_config(config: &McpServerConfig) -> Self {
        match config {
            McpServerConfig::Stdio(config) => Self::Stdio(McpStdioTransport {
                command: config.command.clone(),
                args: config.args.clone(),
                env: config.env.clone(),
                tool_call_timeout_ms: config.tool_call_timeout_ms,
            }),
            McpServerConfig::Sse(config) => Self::Sse(McpRemoteTransport {
                url: config.url.clone(),
                headers: config.headers.clone(),
                headers_helper: config.headers_helper.clone(),
                auth: McpClientAuth::from_oauth(config.oauth.clone()),
            }),
            McpServerConfig::Http(config) => Self::Http(McpRemoteTransport {
                url: config.url.clone(),
                headers: config.headers.clone(),
                headers_helper: config.headers_helper.clone(),
                auth: McpClientAuth::from_oauth(config.oauth.clone()),
            }),
            McpServerConfig::Ws(config) => Self::WebSocket(McpRemoteTransport {
                url: config.url.clone(),
                headers: config.headers.clone(),
                headers_helper: config.headers_helper.clone(),
                auth: McpClientAuth::None,
            }),
            McpServerConfig::Sdk(config) => Self::Sdk(McpSdkTransport {
                name: config.name.clone(),
            }),
            McpServerConfig::ManagedProxy(config) => Self::ManagedProxy(McpManagedProxyTransport {
                url: config.url.clone(),
                id: config.id.clone(),
            }),
        }
    }
}
```

这段代码的设计要点：

- **字段复制**：bootstrap 是运行时使用的独立值，避免后续再依赖原始 JSON。
- **OAuth 显式建模**：`McpClientAuth::OAuth` 能表达远端鉴权需求。
- **实现分层**：`from_config` 只负责“映射形状”，不负责“真的连上”。

---

## 10.5 工具名规范：为什么模型看到的是 `mcp__server__tool`

MCP Server 返回的工具名通常只是 `read_file`、`browser_screenshot`、`query`。如果多个 server 都有同名工具，Host 侧必须给它们加命名空间。

`claw-code` 在 `mcp.rs` 里定义了统一规则：

```rust
pub fn normalize_name_for_mcp(name: &str) -> String {
    name.chars()
        .map(|ch| match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '_' | '-' => ch,
            _ => '_',
        })
        .collect::<String>()
}

pub fn mcp_tool_prefix(server_name: &str) -> String {
    format!("mcp__{}__", normalize_name_for_mcp(server_name))
}

pub fn mcp_tool_name(server_name: &str, tool_name: &str) -> String {
    format!(
        "{}{}",
        mcp_tool_prefix(server_name),
        normalize_name_for_mcp(tool_name)
    )
}
```

这段代码的设计要点：

- **命名空间隔离**：`mcp__github__search` 和 `mcp__notion__search` 不会冲突。
- **模型可读**：双下划线分隔让模型能看出 “MCP / server / tool” 三段结构。
- **字符归一化**：空格、中文标点等不适合当工具名的字符会被替换成 `_`。

可以把这个命名规则写成：

$$\text{qualified\_tool} = \text{mcp\\_\\_} + \text{normalize}(server) + \text{\\_\\_} + \text{normalize}(tool)$$

例如：

```text
server = "playwright local"
tool   = "browser.screenshot"

qualified_tool = "mcp__playwright_local__browser_screenshot"
```

这就是后面 CLI 传给 LLM 的工具名，也是 `McpServerManager::call_tool` 路由回具体 server 的 key。

---

## 10.6 运行实例与分析

下面是一段按 `claw-code` 实现路径整理的运行记录，用来展示“配置 → bootstrap → 工具名”的转换过程：

```text
📚 加载 settings.json
✅ 发现 mcpServers.playwright
🧩 解析 server 类型：未显式声明 type，且没有 url → stdio
✅ 生成 McpServerConfig::Stdio(command="npx", args=["@playwright/mcp"])
🧩 生成 McpClientBootstrap
✅ server_name = "playwright"
✅ tool_prefix = "mcp__playwright__"
🎯 假设 Server 返回 tool.name = "browser_screenshot"
✅ qualified tool = "mcp__playwright__browser_screenshot"
```

从上面的输出可以看到，`claw-code` 在真正连接 MCP Server 之前，已经做完了三件关键工作：

1. **配置结构化**：把 JSON 配置变成类型安全的 Rust enum；
2. **连接目标标准化**：用 bootstrap 保存启动命令、transport、签名；
3. **工具名命名空间化**：提前解决多 server 同名工具冲突。

---

## 10.7 特点、局限性与调试技巧

（1）主要特点

1. **配置层支持多 transport**：`stdio`、`http`、`sse`、`ws`、`sdk`、`managed proxy` 都有配置形状。
2. **bootstrap 与连接解耦**：`McpClientBootstrap` 只描述连接目标，不启动进程。
3. **工具名稳定可预测**：`mcp__server__tool` 让 LLM 工具列表避免命名冲突。

（2）固有局限性

1. **配置支持不等于运行支持**：远端 transport 可被解析，但当前 manager 只真正执行 `stdio`。
2. **同名 server 覆盖需要小心**：多层配置里同名 server 会被后写入的作用域覆盖。
3. **signature 不是安全校验**：它更像配置指纹，不是防篡改签名。

（3）调试技巧

- **先看 `mcpServers` 是否被解析**：配置错误通常会带上 `mcpServers.{name}` 的路径。
- **确认 URL-only 推断**：只写 `url` 会走 `http`，而当前运行时会把它标为 unsupported。
- **检查工具名归一化**：如果模型看到的工具名和预期不同，先看 `normalize_name_for_mcp`。

在掌握了配置与 bootstrap 之后，下一章我们将进入最核心的部分：`McpServerManager` 如何启动 stdio server、完成握手、发现工具并执行 `tools/call`。

---

## 参考源码

[1] `/Users/fan/workspace/Agent/claw-code/rust/crates/runtime/src/config.rs`

[2] `/Users/fan/workspace/Agent/claw-code/rust/crates/runtime/src/mcp_client.rs`

[3] `/Users/fan/workspace/Agent/claw-code/rust/crates/runtime/src/mcp.rs`
