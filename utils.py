import yaml
import os
from typing import Dict, Any
import re
import json
from langchain_core.messages import AIMessage


def post_process_deepseek_response(response: AIMessage) -> AIMessage:
    """
    处理 DeepSeek-R1 的响应：
    1. 移除 <think> 标签及其内容。
    2. 如果原生 tool_calls 为空，尝试从 content 中的 JSON 代码块提取工具调用。
    """
    content = response.content

    # --- 步骤 1: 移除 <think> 思考过程 ---
    # 使用 re.DOTALL 让 . 匹配换行符
    content_cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    content_cleaned = re.sub(r'</think>.*?</think>', '', content, flags=re.DOTALL).strip()

    # 更新 response.content (只保留最终输出)
    response.content = content_cleaned

    # --- 步骤 2: 提取并回填 tool_calls ---
    # 如果模型没有通过 API 字段返回工具调用，而是写在了 content 里
    if not response.tool_calls and content_cleaned:
        try:
            # 尝试匹配 markdown JSON 代码块 ```json ... ```
            json_match = re.search(r'```json\n(.*?)\n```', content_cleaned, re.DOTALL)

            if not json_match:
                # 如果没有代码块，尝试匹配纯 JSON 对象 {...}
                json_match = re.search(r'(\{.*\})', content_cleaned, re.DOTALL)

            if json_match:
                json_str = json_match.group(1)
                data = json.loads(json_str)

                # 检查是否包含 OpenAI 格式的 tool_calls
                if "tool_calls" in data and isinstance(data["tool_calls"], list):
                    lc_tool_calls = []

                    for tc in data["tool_calls"]:
                        function_data = tc.get("function", {})
                        name = function_data.get("name")
                        arguments_str = function_data.get("arguments")

                        # 解析参数 (通常 arguments 是字符串形式的 JSON)
                        if isinstance(arguments_str, str):
                            try:
                                args = json.loads(arguments_str)
                            except json.JSONDecodeError:
                                args = {}  # 解析失败，设为空
                        else:
                            args = arguments_str or {}

                        if name:
                            lc_tool_calls.append({
                                "name": name,
                                "args": args,
                                "id": tc.get("id", "call_extracted_from_text")
                            })

                    # 【关键】回填到 message 对象中
                    if lc_tool_calls:
                        print(f"--- 🔧 检测到文本 JSON 工具调用，已手动回填 {len(lc_tool_calls)} 个工具 ---")
                        response.tool_calls = lc_tool_calls

        except Exception as e:
            print(f"--- ⚠️ 后处理解析 JSON 失败: {e} ---")
            # 失败了也不报错，让流程继续（可能会触发 Router 的纯文本回退逻辑）
            pass

    return response

def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """
    安全地加载 YAML 配置文件，并返回一个 Python 字典。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"配置文件未找到: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 使用 yaml.safe_load() 确保只加载安全的、标准的 YAML 类型
            config_data = yaml.safe_load(f)

        print(f"✅ YAML 文件 {file_path} 加载成功。")
        return config_data

    except yaml.YAMLError as e:
        print(f"❌ YAML 文件解析错误: {e}")
        # 可以选择重新抛出异常
        raise
    except Exception as e:
        print(f"❌ 加载文件时发生未知错误: {e}")
        raise
