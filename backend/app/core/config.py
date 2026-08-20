from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 用绝对路径而不是相对路径".env"——相对路径是相对运行时的cwd解析的，不是相对
# 这个文件的位置，从backend/以外的目录（比如仓库根目录）跑pytest/uvicorn会
# 静默加载不到.env，所有密钥变成空字符串，不会报错提醒，只会在真正调用外部
# API时才失败，且错误信息看不出根因是"目录不对"
ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: str = ""
    deepseek_api_key: str = ""
    sec_edgar_user_agent: str = ""
    polygon_api_key: str = ""
    alpha_vantage_api_key: str = ""

    llm_provider: str = "deepseek"  # "deepseek" 或 "claude"，切换供应商只改这个
    llm_model: str = ""  # 留空则用每个供应商各自的默认模型
    # 留空表示 Best-of-N 的 LLM 裁判跟报告生成用同一个供应商（现状）；填
    # "claude"/"deepseek" 可以让裁判走跟生成不同的供应商，减少"用同一个模型
    # 评自己生成的内容"这种自评偏好嫌疑——需要对应供应商的 API Key 已配置
    judge_llm_provider: str = ""


settings = Settings()
