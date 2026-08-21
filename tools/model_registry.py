#!/usr/bin/env python3
"""模型注册表加载器：读取项目根的 model*.yaml，返回每个 .tqwen 的元数据。

每个 YAML 记录：架构、dtype、文件大小、量化方案、推荐配方、实测数字。
控制台 / bench 工具用它在 UI 上自动生成配方按钮、展示模型目录。

用法：
    from model_registry import load_registry
    models = load_registry()  # list[dict]，按 size_mb 排序

作为独立脚本运行时，打印所有已注册模型的摘要信息：
    python tools/model_registry.py

YAML 文件格式示例（model_qwen25_0.5b_f16.yaml）：
    name: qwen25_0.5b_f16
    architecture: qwen2
    dtype: f16
    file: models/qwen25_0.5b_f16.tqwen
    size_mb: 980
    params: 494M
    quantization: null
    recipe:
      matvec_impl: neon_f16
    benchmark:
      android:
        topt_ms: 12.3

本模块提供的功能：
    - load_registry(): 加载所有 model*.yaml 并按文件大小排序。
    - find_recipe(): 按 .tqwen 文件名查找推荐的推理配方。
    - __main__: 打印模型目录摘要。

输入：项目根目录下的 model*.yaml 文件。
输出：Python 字典列表（load_registry）或控制台表格（__main__）。
"""

from __future__ import annotations

# --- 标准库导入 ---
import glob       # 文件通配符匹配（查找 model*.yaml）
from pathlib import Path  # 路径操作

# --- 第三方库导入 ---
import yaml       # YAML 解析

# 项目根目录（model_registry.py 在 tools/ 下，上一级是项目根）
REPO_ROOT = Path(__file__).resolve().parent.parent


def load_registry() -> list[dict]:
    """加载所有 model*.yaml（项目根目录），返回 list[dict]。

    扫描项目根目录下所有以 "model" 开头、以 ".yaml" 结尾的文件，
    解析每个文件的 YAML 内容，提取模型元数据。为每个条目附加两个内部字段：
    - _yaml: 源 YAML 文件名（用于调试/溯源）。
    - _exists: 对应的 .tqwen 文件是否实际存在。

    结果按 size_mb 升序排列（小模型在前）。

    Returns:
        模型元数据字典列表，每个 dict 含 name / architecture / dtype / file /
        size_mb / params / quantization / recipe / benchmark / _yaml / _exists 等字段。
    """
    # 查找项目根下所有 model*.yaml 文件并排序
    files = sorted(glob.glob(str(REPO_ROOT / "model*.yaml")))
    models = []  # 存放解析后的模型元数据
    for f in files:
        with open(f, encoding="utf-8") as fh:
            d = yaml.safe_load(fh)  # 安全解析 YAML（不执行任意 Python 代码）
        if d and "name" in d:  # 跳过空文件或缺少 name 字段的无效条目
            d["_yaml"] = Path(f).name  # 记录源 YAML 文件名
            # 检查 .tqwen 文件是否存在
            tqwen_path = REPO_ROOT / d.get("file", "")  # 拼接 .tqwen 文件的完整路径
            d["_exists"] = tqwen_path.is_file()  # 标记文件是否存在
            models.append(d)
    # 按文件大小升序排序（小模型优先展示）
    models.sort(key=lambda m: m.get("size_mb", 0))
    return models


def find_recipe(model_file: str) -> dict | None:
    """按 .tqwen 文件名查推荐配方。返回 recipe dict 或 None。

    根据 .tqwen 文件名（去掉后缀后与注册表的 name 字段匹配）
    查找该模型的推荐推理配方（matvec_impl 等）。

    Args:
        model_file: .tqwen 文件名（如 "qwen25_0.5b_f16.tqwen"）。

    Returns:
        recipe 字典（含 matvec_impl 等键），未找到返回 None。
    """
    base = model_file.replace(".tqwen", "")  # 去掉 .tqwen 后缀得到模型名
    for m in load_registry():
        if m.get("name") == base:  # 按 name 字段精确匹配
            return m.get("recipe")  # 返回 recipe 子字典
    return None  # 未找到匹配项


if __name__ == "__main__":
    # 作为独立脚本运行：打印所有已注册模型的摘要表格
    for m in load_registry():
        r = m.get("recipe", {})                    # 获取推荐配方
        b = m.get("benchmark", {}).get("android", {})  # 获取 Android 基准测试数据
        ex = "✓" if m["_exists"] else "✗"          # 文件存在标记
        topt = b.get("topt_ms", "?")               # 每 token 推理耗时（ms）
        # 打印一行模型摘要：存在标记、名字、架构、dtype、大小、配方、性能
        print(f"  {ex} {m['name']:<22} {m['architecture']:<8} {m['dtype']:<5} "
              f"{m.get('size_mb','?'):>6}MB  "
              f"recipe={r.get('matvec_impl','?')}  topt={topt}ms")
