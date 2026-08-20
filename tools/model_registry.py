#!/usr/bin/env python3
"""模型注册表加载器：读取项目根的 model*.yaml，返回每个 .tqwen 的元数据。

每个 YAML 记录：架构、dtype、文件大小、量化方案、推荐配方、实测数字。
控制台 / bench 工具用它在 UI 上自动生成配方按钮、展示模型目录。

用法：
    from model_registry import load_registry
    models = load_registry()  # list[dict]，按 size_mb 排序
"""

from __future__ import annotations

import glob
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_registry() -> list[dict]:
    """加载所有 model*.yaml（项目根目录），返回 list[dict]。

    每个 dict 含 name / architecture / dtype / file / size_mb / params /
    quantization / recipe / benchmark 等字段。按 size_mb 升序排。
    """
    files = sorted(glob.glob(str(REPO_ROOT / "model*.yaml")))
    models = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            d = yaml.safe_load(fh)
        if d and "name" in d:
            d["_yaml"] = Path(f).name
            # 检查 .tqwen 文件是否存在
            tqwen_path = REPO_ROOT / d.get("file", "")
            d["_exists"] = tqwen_path.is_file()
            models.append(d)
    models.sort(key=lambda m: m.get("size_mb", 0))
    return models


def find_recipe(model_file: str) -> dict | None:
    """按 .tqwen 文件名查推荐配方。返回 recipe dict 或 None。"""
    base = model_file.replace(".tqwen", "")
    for m in load_registry():
        if m.get("name") == base:
            return m.get("recipe")
    return None


if __name__ == "__main__":
    for m in load_registry():
        r = m.get("recipe", {})
        b = m.get("benchmark", {}).get("android", {})
        ex = "✓" if m["_exists"] else "✗"
        topt = b.get("topt_ms", "?")
        print(f"  {ex} {m['name']:<22} {m['architecture']:<8} {m['dtype']:<5} "
              f"{m.get('size_mb','?'):>6}MB  "
              f"recipe={r.get('matvec_impl','?')}  topt={topt}ms")
