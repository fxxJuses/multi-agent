# 第十二章 claw-code 的 MCP Client（三）：从 MCP Tool 到 Agent Tool

前两章我们已经走完了 MCP Client 的底层链路：配置被解析成 bootstrap，`McpServerManager` 启动 stdio server、完成 `initialize`、执行 `tools/list`，并建立 `mcp__server__tool` 到真实 `tools/call` 的路由。

但对 Agent 来说，这还不够。模型最终看到的不是 `McpServerManager`，而是一组 **LLM tool schema**。因此本章继续向上追问：**`claw-code` 如何把发现到的 MCP tool 注册进 CLI 的工具系统？权限如何判定？模型调用时又如何回到 `McpServerManager::call_tool`？**

如果说第十一章解决的是“怎么和 MCP Server 通信”，那么本章解决的是“怎么让模型安全地使用这些通信能力”。

> 源码路径说明：本章重点分析 `/Users/fan/workspace/Agent/claw-code/rust/crates/rusty-claude-cli/src/main.rs`、`/Users/fan/workspace/Agent/claw-code/rust/crates/tools/src/lib.rs`、`/Users/fan/workspace/Agent/claw-code/rust/crates/runtime/src/mcp_tool_bridge.rs`。

---

## 12.1 Agent 集成的工作原理

MCP tool 要进入 Agent 主循环，至少要完成三次“翻译”：

- **协议工具 → RuntimeToolDefinition**：把 MCP 的 `name / description / inputSchema / annotations` 变成 CLI 工具定义；
- **RuntimeToolDefinition → GlobalToolRegistry**：把 MCP 工具和内置工具、插件工具合并到一个工具表；
- **LLM tool call → MCP tools/call**：模型调用 `mcp__server__tool` 后，CLI executor 再路由回 `McpServerManager`。

我们可以把这条链路写成：

$$\text{LLMTools} = \text{BuiltinTools} \cup \text{PluginTools} \cup \text{McpTools}$$

其中：

$$\text{McpTools} = \{\text{RuntimeToolDefinition}(t) \mid t \in \text{tools/list}(servers)\}$$

直觉上，MCP 并没有让 Agent 主循环“自动懂外部世界”。它只是把外部 Server 暴露的工具 schema，翻译成模型已经能理解的工具列表，再由 CLI 的执行器把调用转发出去。

---

## 12.2 `RuntimeMcpState`：CLI 侧 MCP 状态容器

（1）启动时做 best-effort discovery

`RuntimeMcpState::new` 是 CLI 侧把 manager 接进来的入口。它创建 `McpServerManager`，启动 Tokio runtime，然后调用 `discover_tools_best_effort`：

```rust
impl RuntimeMcpState {
    fn new(
        runtime_config: &runtime::RuntimeConfig,
    ) -> Result<Option<(Self, runtime::McpToolDiscoveryReport)>, Box<dyn std::error::Error>> {
        let mut manager = McpServerManager::from_runtime_config(runtime_config);
        if manager.server_names().is_empty() && manager.unsupported_servers().is_empty() {
            return Ok(None);
        }

        let runtime = tokio::runtime::Runtime::new()?;
        let discovery = runtime.block_on(manager.discover_tools_best_effort());
        let pending_servers = discovery
            .failed_servers
            .iter()
            .map(|failure| failure.server_name.clone())
            .chain(
                discovery
                    .unsupported_servers
                    .iter()
                    .map(|server| server.server_name.clone()),
            )
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();

        Ok(Some((Self { runtime, manager, pending_servers, degraded_report }, discovery)))
    }
}
```

这段代码的设计要点：

- **没有 MCP 就返回 `None`**：避免无配置时还创建空 runtime。
- **best effort**：启动时尽可能发现能用的工具，同时记录失败 server。
- **`pending_servers`**：把失败和 unsupported 的 server 暴露给工具搜索/用户反馈。

（2）调用工具时返回 JSON 字符串

`RuntimeMcpState::call_tool` 把 async manager 包在同步 CLI 调用里，并把结果序列化成 pretty JSON：

```rust
fn call_tool(
    &mut self,
    qualified_tool_name: &str,
    arguments: Option<serde_json::Value>,
) -> Result<String, ToolError> {
    let response = self
        .runtime
        .block_on(self.manager.call_tool(qualified_tool_name, arguments))
        .map_err(|error| ToolError::new(error.to_string()))?;
    if let Some(error) = response.error {
        return Err(ToolError::new(format!(
            "MCP tool `{qualified_tool_name}` returned JSON-RPC error: {} ({})",
            error.message, error.code
        )));
    }

    let result = response.result.ok_or_else(|| {
        ToolError::new(format!(
            "MCP tool `{qualified_tool_name}` returned no result payload"
        ))
    })?;
    serde_json::to_string_pretty(&result).map_err(|error| ToolError::new(error.to_string()))
}
```

这段代码的设计要点：

- **同步外壳**：CLI tool executor 是同步接口，内部用 `runtime.block_on` 调 async manager。
- **JSON-RPC 错误显式化**：server 返回 `error` 时转成 `ToolError`。
- **保留完整结果**：没有只抽 text，而是把 MCP result 整体 pretty print 给 Agent。

---

## 12.3 MCP tool 如何变成 LLM tool schema

（1）发现工具后生成 runtime tools

`build_runtime_mcp_state` 会把 discovery 里成功发现的 `ManagedMcpTool` 转成 `RuntimeToolDefinition`：

```rust
fn build_runtime_mcp_state(
    runtime_config: &runtime::RuntimeConfig,
) -> Result<RuntimePluginStateBuildOutput, Box<dyn std::error::Error>> {
    let Some((mcp_state, discovery)) = RuntimeMcpState::new(runtime_config)? else {
        return Ok((None, Vec::new()));
    };

    let mut runtime_tools = discovery
        .tools
        .iter()
        .map(mcp_runtime_tool_definition)
        .collect::<Vec<_>>();
    if !mcp_state.server_names().is_empty() {
        runtime_tools.extend(mcp_wrapper_tool_definitions());
    }

    Ok((Some(Arc::new(Mutex::new(mcp_state))), runtime_tools))
}
```

这个函数的设计要点：

- **成功发现才注册**：只有 `discovery.tools` 里的工具会变成模型可见工具。
- **wrapper tools**：只要存在 MCP server，就额外注册资源列表/读取等包装工具。
- **共享状态**：`RuntimeMcpState` 被包进 `Arc<Mutex<_>>`，让 executor 可以跨调用复用同一个 manager。

（2）单个 MCP tool 到 `RuntimeToolDefinition`

```rust
fn mcp_runtime_tool_definition(tool: &runtime::ManagedMcpTool) -> RuntimeToolDefinition {
    RuntimeToolDefinition {
        name: tool.qualified_name.clone(),
        description: Some(
            tool.tool
                .description
                .clone()
                .unwrap_or_else(|| format!("Invoke MCP tool `{}`.", tool.qualified_name)),
        ),
        input_schema: tool
            .tool
            .input_schema
            .clone()
            .unwrap_or_else(|| json!({ "type": "object", "additionalProperties": true })),
        required_permission: permission_mode_for_mcp_tool(&tool.tool),
    }
}
```

这段代码的设计要点：

- **`name` 使用 qualified name**：模型看到的是 `mcp__server__tool`，不是 raw tool name。
- **description fallback**：MCP Server 没写描述时，CLI 仍生成一个可用描述。
- **schema fallback**：没有 `inputSchema` 时允许任意 object，保证工具仍能注册。
- **权限由 annotations 推断**：这一步决定工具属于只读、工作区写入，还是高风险。

（3）wrapper tools：资源访问与通用调用入口

除了每个 MCP 原生工具，`claw-code` 还注册了三个 wrapper：

```rust
fn mcp_wrapper_tool_definitions() -> Vec<RuntimeToolDefinition> {
    vec![
        RuntimeToolDefinition {
            name: "MCPTool".to_string(),
            description: Some(
                "Call a configured MCP tool by its qualified name and JSON arguments.".to_string(),
            ),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "qualifiedName": { "type": "string" },
                    "arguments": {}
                },
                "required": ["qualifiedName"],
                "additionalProperties": false
            }),
            required_permission: PermissionMode::DangerFullAccess,
        },
        // ListMcpResourcesTool / ReadMcpResourceTool 省略
    ]
}
```

这些 wrapper 的设计要点：

- **`MCPTool`**：提供“按 qualifiedName 调任意 MCP 工具”的通用入口，因此权限是 `DangerFullAccess`。
- **`ListMcpResourcesTool`**：列出一个或全部 server 的 MCP resources，属于只读。
- **`ReadMcpResourceTool`**：按 `server + uri` 读取资源，也属于只读。

---

## 12.4 权限：MCP annotations 如何影响安全等级

MCP tool 可以带 `annotations`，例如 `readOnlyHint`、`destructiveHint`、`openWorldHint`。`claw-code` 用这些 hint 推断权限模式：

```rust
fn permission_mode_for_mcp_tool(tool: &McpTool) -> PermissionMode {
    let read_only = mcp_annotation_flag(tool, "readOnlyHint");
    let destructive = mcp_annotation_flag(tool, "destructiveHint");
    let open_world = mcp_annotation_flag(tool, "openWorldHint");

    if read_only && !destructive && !open_world {
        PermissionMode::ReadOnly
    } else if destructive || open_world {
        PermissionMode::DangerFullAccess
    } else {
        PermissionMode::WorkspaceWrite
    }
}
```

这段代码的设计要点：

- **只读优先但需无冲突**：只有 `readOnlyHint=true` 且没有破坏性/开放世界 hint，才判为 `ReadOnly`。
- **危险 hint 升级权限**：`destructiveHint` 或 `openWorldHint` 任一为 true，就需要 `DangerFullAccess`。
- **默认中间态**：没有明确只读，也没有明确危险时，落到 `WorkspaceWrite`。

这和我们前面讨论的 Host 过滤工具完全一致：**Server 负责声明能力和 hint，Host 负责把这些 hint 翻译成自己的权限系统。**

---

## 12.5 `GlobalToolRegistry`：MCP 工具如何混入总工具表

（1）runtime tools 与 built-in/plugin 合并

`GlobalToolRegistry` 维护三类工具：内置工具、插件工具、运行时工具。MCP 发现出来的工具属于 runtime tools。

```rust
pub struct GlobalToolRegistry {
    plugin_tools: Vec<PluginTool>,
    runtime_tools: Vec<RuntimeToolDefinition>,
    enforcer: Option<PermissionEnforcer>,
}

pub fn with_runtime_tools(
    mut self,
    runtime_tools: Vec<RuntimeToolDefinition>,
) -> Result<Self, String> {
    let mut seen_names = mvp_tool_specs()
        .into_iter()
        .map(|spec| spec.name.to_string())
        .chain(self.plugin_tools.iter().map(|tool| tool.definition().name.clone()))
        .collect::<BTreeSet<_>>();

    for tool in &runtime_tools {
        if !seen_names.insert(tool.name.clone()) {
            return Err(format!(
                "runtime tool `{}` conflicts with an existing tool name",
                tool.name
            ));
        }
    }

    self.runtime_tools = runtime_tools;
    Ok(self)
}
```

这个 registry 的设计要点：

- **冲突检查**：MCP 工具不能和 built-in/plugin 工具重名。
- **runtime 分层**：MCP 工具不需要写死在内置工具列表里。
- **统一出口**：最终模型拿到的工具定义来自同一个 registry。

（2）传给模型的定义列表

```rust
pub fn definitions(&self, allowed_tools: Option<&BTreeSet<String>>) -> Vec<ToolDefinition> {
    let builtin = mvp_tool_specs()
        .into_iter()
        .filter(|spec| allowed_tools.is_none_or(|allowed| allowed.contains(spec.name)))
        .map(|spec| ToolDefinition {
            name: spec.name.to_string(),
            description: Some(spec.description.to_string()),
            input_schema: spec.input_schema,
        });
    let runtime = self
        .runtime_tools
        .iter()
        .filter(|tool| allowed_tools.is_none_or(|allowed| allowed.contains(tool.name.as_str())))
        .map(|tool| ToolDefinition {
            name: tool.name.clone(),
            description: tool.description.clone(),
            input_schema: tool.input_schema.clone(),
        });

    builtin.chain(runtime).collect()
}
```

这段代码的设计要点：

- **`allowed_tools` 过滤**：即使 MCP 发现了很多工具，Host 仍可只暴露一部分给模型。
- **统一 schema**：内置工具和 MCP 工具最终都变成 `ToolDefinition`。
- **模型无感知来源**：LLM 不需要知道某个工具来自 built-in 还是 MCP，只看到名称、描述、参数 schema。

---

## 12.6 执行路径：模型调用后如何回到 MCP

`CliToolExecutor` 是模型 tool call 的执行入口。它先检查 `--allowedTools`，再判断工具是否是 runtime tool；如果是，就进入 MCP 执行路径。

```rust
impl ToolExecutor for CliToolExecutor {
    fn execute(&mut self, tool_name: &str, input: &str) -> Result<String, ToolError> {
        if self
            .allowed_tools
            .as_ref()
            .is_some_and(|allowed| !allowed.contains(tool_name))
        {
            return Err(ToolError::new(format!(
                "tool `{tool_name}` is not enabled by the current --allowedTools setting"
            )));
        }
        let value = serde_json::from_str(input)
            .map_err(|error| ToolError::new(format!("invalid tool input JSON: {error}")))?;
        let result = if tool_name == "ToolSearch" {
            self.execute_search_tool(value)
        } else if self.tool_registry.has_runtime_tool(tool_name) {
            self.execute_runtime_tool(tool_name, value)
        } else {
            self.tool_registry.execute(tool_name, &value).map_err(ToolError::new)
        };
        result
    }
}
```

这段代码的设计要点：

- **`--allowedTools` 是第一道门**：不在 allowlist 里的工具直接拒绝。
- **runtime tool 优先识别**：MCP 工具通过 `has_runtime_tool` 进入 `execute_runtime_tool`。
- **JSON 参数统一解析**：模型传入的 input 先被解析成 `serde_json::Value`。

`execute_runtime_tool` 再分发到具体 MCP 逻辑：

```rust
fn execute_runtime_tool(
    &self,
    tool_name: &str,
    value: serde_json::Value,
) -> Result<String, ToolError> {
    let Some(mcp_state) = &self.mcp_state else {
        return Err(ToolError::new(format!(
            "runtime tool `{tool_name}` is unavailable without configured MCP servers"
        )));
    };
    let mut mcp_state = mcp_state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    match tool_name {
        "MCPTool" => {
            let input: McpToolRequest = serde_json::from_value(value)?;
            let qualified_name = input
                .qualified_name
                .or(input.tool)
                .ok_or_else(|| ToolError::new("missing required field `qualifiedName`"))?;
            mcp_state.call_tool(&qualified_name, input.arguments)
        }
        "ListMcpResourcesTool" => { /* list resources */ }
        "ReadMcpResourceTool" => { /* read resource */ }
        _ => mcp_state.call_tool(tool_name, Some(value)),
    }
}
```

这个分发器的设计要点：

- **直接 MCP tool**：普通 `mcp__server__tool` 会走 `_ => mcp_state.call_tool(...)`。
- **wrapper MCPTool**：允许用一个通用工具按 `qualifiedName` 调任意 MCP tool。
- **资源工具单独处理**：资源列表/读取不走 `tools/call`，而走 `resources/list` / `resources/read`。

---

## 12.7 ToolSearch 与 pending MCP servers

当某些 MCP Server 启动失败或 transport 不支持时，`claw-code` 不只是报错，还会把 pending 状态传给 `ToolSearch`：

```rust
fn execute_search_tool(&self, value: serde_json::Value) -> Result<String, ToolError> {
    let input: ToolSearchRequest = serde_json::from_value(value)
        .map_err(|error| ToolError::new(format!("invalid tool input JSON: {error}")))?;
    let (pending_mcp_servers, mcp_degraded) =
        self.mcp_state.as_ref().map_or((None, None), |state| {
            let state = state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            (state.pending_servers(), state.degraded_report())
        });
    serde_json::to_string_pretty(&self.tool_registry.search(
        &input.query,
        input.max_results.unwrap_or(5),
        pending_mcp_servers,
        mcp_degraded,
    ))
}
```

这段代码的设计要点：

- **搜索结果带状态**：工具搜索不仅返回匹配工具，也能告诉模型/用户哪些 MCP server 还 pending。
- **degraded 信息上浮**：底层 manager 的失败不是藏在日志里，而是变成工具搜索结果的一部分。
- **面向 Agent 的可恢复性**：模型可以知道“工具暂不可用”，而不是误以为工具不存在。

---

## 12.8 备用路径：`McpToolRegistry` bridge

除了 CLI 主路径，`runtime/src/mcp_tool_bridge.rs` 还提供了一个 `McpToolRegistry`。它像一个全局 registry，服务于 `tools` crate 中的 `MCP`、`McpAuth`、`ListMcpResources` 等工具。

核心调用逻辑是：

```rust
pub fn call_tool(
    &self,
    server_name: &str,
    tool_name: &str,
    arguments: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let inner = self.inner.lock().expect("mcp registry lock poisoned");
    let state = inner
        .get(server_name)
        .ok_or_else(|| format!("server '{}' not found", server_name))?;

    if state.status != McpConnectionStatus::Connected {
        return Err(format!(
            "server '{}' is not connected (status: {})",
            server_name, state.status
        ));
    }

    if !state.tools.iter().any(|t| t.name == tool_name) {
        return Err(format!(
            "tool '{}' not found on server '{}'",
            tool_name, server_name
        ));
    }

    drop(inner);
    let manager = self.manager.get().cloned()
        .ok_or_else(|| "MCP server manager is not configured".to_string())?;

    Self::spawn_tool_call(
        manager,
        mcp_tool_name(server_name, tool_name),
        (!arguments.is_null()).then(|| arguments.clone()),
    )
}
```

这条 bridge 路径的设计要点：

- **先查 registry 状态**：server 必须已注册且 connected。
- **再查工具存在性**：防止调用未声明工具。
- **最后仍回到 manager**：真正调用仍是 `McpServerManager::call_tool`。
- **线程 + runtime 包装**：`spawn_tool_call` 在线程里创建 Tokio runtime，适合从同步工具函数调用 async manager，但相对 CLI 主路径更重。

从教学角度看，我们应该把 CLI 的 `RuntimeMcpState` 视为主线，把 `McpToolRegistry` bridge 视为另一条兼容/工具层路径。

---

## 12.9 运行实例与分析

下面是一段按 CLI 集成路径整理的运行记录：

```text
📚 RuntimeMcpState::new 读取 RuntimeConfig
🔍 McpServerManager.discover_tools_best_effort()
🎯 发现 mcp__playwright__browser_screenshot
🧩 mcp_runtime_tool_definition 生成 RuntimeToolDefinition
✅ name = "mcp__playwright__browser_screenshot"
✅ input_schema = Server 返回的 inputSchema
✅ required_permission = ReadOnly / WorkspaceWrite / DangerFullAccess
🧩 GlobalToolRegistry.with_runtime_tools 合并 MCP 工具
✅ definitions() 将 MCP 工具传给模型
🎬 LLM 输出 tool call：mcp__playwright__browser_screenshot
🔍 CliToolExecutor.execute_runtime_tool
🔍 RuntimeMcpState.call_tool
🔍 McpServerManager.call_tool -> tools/call
👀 MCP Server 返回 result
🎉 CLI 将 pretty JSON 作为 Observation 回写给模型
```

从上面的输出可以看到，MCP tool 进入 Agent 并不是一步完成的：

1. **发现阶段**把 server 工具变成 `ManagedMcpTool`；
2. **注册阶段**把 `ManagedMcpTool` 变成 `RuntimeToolDefinition`；
3. **执行阶段**把模型 tool call 路由回 `McpServerManager::call_tool`。

---

## 12.10 特点、局限性与调试技巧

（1）主要特点

1. **工具来源统一**：built-in、plugin、MCP runtime tools 最终都进入 `GlobalToolRegistry`。
2. **权限有 MCP hint 参与**：`readOnlyHint`、`destructiveHint`、`openWorldHint` 会影响 Host 权限等级。
3. **degraded 信息可被 Agent 感知**：pending server 不只写日志，还会进入 ToolSearch 输出。

（2）固有局限性

1. **发现失败就不会注册工具**：如果 `tools/list` 启动时失败，模型不会看到该 server 的工具。
2. **schema fallback 较宽**：没有 `inputSchema` 时允许任意 object，可能增加模型乱传参风险。
3. **通用 `MCPTool` 权限很高**：它可以按 qualified name 调任意 MCP tool，因此默认是 `DangerFullAccess`。

（3）调试技巧

- **模型看不到工具时**：先看 `discover_tools_best_effort` 是否成功，再看 `with_runtime_tools` 是否发生命名冲突。
- **工具被 allowlist 拒绝时**：检查 `--allowedTools` 是否包含完整 qualified name，而不是 raw tool name。
- **权限过高时**：检查 MCP Server 返回的 `annotations`，尤其是 `destructiveHint` 和 `openWorldHint`。
- **ToolSearch 里有 pending server 时**：回到第十一章按生命周期阶段排查启动、握手、发现或 unsupported transport。

---

## 12.11 总结对比：claw-code MCP Client 的三层结构

| 层次 | 关键文件 | 核心职责 | 典型局限 |
| ---- | ---- | ---- | ---- |
| 配置与 bootstrap | `config.rs` / `mcp_client.rs` / `mcp.rs` | 解析 `mcpServers`、建模 transport、生成工具名前缀 | 远端 transport 只在配置层建模 |
| stdio manager | `mcp_stdio.rs` / `mcp_lifecycle_hardened.rs` | 启动本地 server、握手、发现、调用、错误分类 | 当前只真正支持 stdio |
| Agent 集成 | `rusty-claude-cli/src/main.rs` / `tools/src/lib.rs` | 注册 LLM tool schema、权限判断、执行分发 | 依赖启动时发现结果 |

读完整组三篇，我们可以把 `claw-code` 的 MCP Client 心智模型压缩成一句话：

**配置层可以表达多种 MCP Server，但当前真正可执行的主线是 stdio；成功发现的 MCP tools 会被命名空间化为 `mcp__server__tool`，注册进全局工具表，再由 CLI executor 路由回 `McpServerManager::call_tool`。**

---

## 参考源码

[1] `/Users/fan/workspace/Agent/claw-code/rust/crates/rusty-claude-cli/src/main.rs`

[2] `/Users/fan/workspace/Agent/claw-code/rust/crates/tools/src/lib.rs`

[3] `/Users/fan/workspace/Agent/claw-code/rust/crates/runtime/src/mcp_tool_bridge.rs`
