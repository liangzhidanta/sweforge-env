"""Mac Environment Server 补丁：bundle 查找剥离 AutoDL 并发唯一后缀。

背景（docs/GRPO_卡点沉淀_空工作区_task_id后缀.md）:
  AutoDL 侧 verl_agent_loop.py:150 给 task_id 加 `-<12hex>` 后缀（env 并发
  唯一）; Mac server 的 backend.create() 按完整 task_id 精确查找
  `bundles-dir/<task_id>` -> 永远 miss -> 空模板容器 -> search/view 全空
  -> 空 patch -> reward 恒 0。本补丁在 bundle 查找处加剥离回退。

安装（在 Mac 上）:
  1. 定位查找处: grep -rn "bundles" ~/code/SWE_project/src/sweforge/env_server/
     找形如 `bundle = bundles_dir / task.task_id` 的那行。
  2. 把下面 _bundle_path 函数粘到同文件（模块级）。
  3. 把查找行替换为 `bundle = _bundle_path(bundles_dir, task.task_id)`
     （变量名按你实际代码调整）。
  4. 重启 server + 重建隧道（见 docs/GRPO_正式运行手册.md §5）。
  5. 由 AutoDL 侧跑带后缀探针验证（预期: /workspace 有完整代码）。

安全语义: 剥离后查不到就回落空模板（保持原行为）; 不改变原始 task_id 的
查找; env 注册表唯一性不受影响（注册表 key 仍是完整带后缀 task_id）。
"""

import re

_BUNDLE_SUFFIX = re.compile(r"-[0-9a-f]{12}$")


def _bundle_path(bundles_dir, task_id):
    """精确匹配优先; 失败则剥离 `-<12hex>` 后缀再查（AutoDL 并发唯一后缀）。

    Args:
        bundles_dir: --bundles-dir 目录（Path）。
        task_id: backend.create 收到的 task.task_id（可能带后缀）。
    Returns:
        Path: 命中的 bundle 目录; 两者皆无则返回未剥离的路径（触发既有空模板回退）。
    """
    for candidate in (task_id, _BUNDLE_SUFFIX.sub("", task_id)):
        p = bundles_dir / candidate
        if p.exists():
            return p
    return bundles_dir / task_id
