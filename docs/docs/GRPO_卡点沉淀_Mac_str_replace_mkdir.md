# Mac 侧 str_replace mkdir 兜底补丁（2026-08-11）

> 目标：Mac 上运行的 Environment Server（`executors.py`，不在本仓库，需手动应用）。
> 前置诊断：Docker 日志实测 `str_replace /workspace/src/router.py` —— 目标文件所在
> 目录不存在时 `write_text` 抛 OSError（"Directory nonexistent"）→ ASGI 500 →
> AutoDL 侧 execute 异常 → 整条 rollout ENV_ERROR 死亡（reward 0，0 探索机会）。
> 模型写了个**正确意图**的编辑，被环境直接杀死 —— 这是 harness 问题不是模型问题。

## 现象（日志原文）

```
executors.py:362 write_text ...: Directory nonexistent
```

- 触发条件：`str_replace` 的 `path` 所在**目录**不存在（canonical 说"文件不存在则
  创建"，但没说目录不存在怎么办 —— 目录缺失没有兜底）。
- 后果：ASGI 500 → 客户端 `execute` 抛异常 → AgentLoop 记 ENV_ERROR 终止
  （`verl_agent_loop.py` execute 异常路径）→ 不 export_patch / 不 verify → reward 0。

## 修复（二选一，推荐 A+B）

### A. mkdir -p 兜底（主修复，与 canonical str_replace 语义一致）

在 `executors.py` 的 `write_text`（约 362 行）里，写入前确保父目录存在：

```python
def write_text(self, path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)   # ← 新增这一行
    path.write_text(content)
```

（如果 `write_text` 是模块级函数而非方法，同样处理：写前 `os.makedirs(parent, exist_ok=True)`。）

### B. 结构化错误观察代替 500（防御性，防止同类 500 再杀 rollout）

在 str_replace 执行器入口包一层 try/except，把"文件系统类错误"转成**观察文本**
返回（observation 而不是 HTTP 500），让模型看到错误继续探索：

```python
try:
    ...  # 原 str_replace 执行逻辑
except FileNotFoundError as e:
    return Observation(error=f"str_replace failed: {e}")   # 或等价的结构化返回
```

原则：**只有真正的基础设施故障（容器起不来等）才 500 / ENV_ERROR；文件操作错误
一律是 observation 反馈**，模型应当能从错误中学习。

## 应用步骤（Mac 侧）

1. 找到 Mac 上 server 源码：`executors.py`（`write_text` 约 362 行）。
2. 应用 A（必须）+ B（推荐）。
3. 重启 server：`python -m sweforge.environment.server --port 8500`（重启后
   AutoDL 侧 `curl /health` 确认）。
4. 验证：`str_replace` 一个不存在目录下的路径，应返回观察错误（或创建成功），
   而不是 ASGI 500。

## 影响面

- 只影响 Mac server 执行器；AutoDL 侧无需改动（parse 规范化是另一处修复，
  见 `src/sweforge/rollout/parse.py` 的 `_normalize_arguments`）。
- 不违反任何铁律（不碰 verl / 不伪造 reward / 不依赖 Docker 的 Agent Loop ——
  修复的是 Mac server 内部执行器，Agent Loop 仍只走 EnvironmentBackend）。
