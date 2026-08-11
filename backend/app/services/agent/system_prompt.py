"""三段式 System Prompt 拼接：信息获取 / 分析处理 / 内容生成。

三段分别维护在独立文件里方便迭代，但运行时拼成一份完整的 system prompt——
不是三次独立调用或三个Skill，Claude/LLM在推理时看到的是一份连贯的完整指导
（项目书第五节5.4架构原则）。拼接结果是模块级常量，只在导入时拼一次，
每次请求发送的字符串完全一致——这对 DeepSeek 的自动前缀缓存、以及以后
Claude 适配器上显式的 cache_control 断点都是必要前提。
"""

from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _load(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text().strip()


SYSTEM_PROMPT = "\n\n---\n\n".join(
    [
        _load("retrieval.md"),
        _load("analysis.md"),
        _load("generation.md"),
    ]
)
