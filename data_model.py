import json
from enum import Enum
from typing import Optional, Tuple, Any
from pydantic import BaseModel, Field, ValidationError
from langchain_core.pydantic_v1 import BaseModel, Field, create_model, validator


# ----------------------------------------------------------------------
# 1. 定义 Agent 的行动类型 (Enum)
# ----------------------------------------------------------------------
class ActionType(str, Enum):
    """定义 Agent 在一轮中可以采取的行动"""
    PROPOSE_NEW = "PROPOSE_NEW"
    ACCEPT = "ACCEPT"


# ----------------------------------------------------------------------
# 2. 定义 Pydantic 解析器 (BaseModel)
# ----------------------------------------------------------------------
class AgentAction(BaseModel):
    """
    一个 Pydantic 模型，用于解析和验证 LLM Agent 的 JSON 输出。
    它取代了手动的 extract_vector 和 extract_speech。
    """

    # Pydantic 会自动将 LLM 输出的字符串 "PROPOSE_NEW" 转换为 ActionType.PROPOSE_NEW
    action: ActionType = Field(
        ...,
        description="Agent 采取的行动类型。"
    )

    # 关键字段：替代 extract_vector
    # 我们使用 Tuple 来严格限制向量必须有 5 个浮点数
    new_proposal_vector: Optional[Tuple[float, float, float, float, float]] = Field(
        default=None,
        description="决策向量 [AR, AC, AO, Delta, Tau]。如果 action 是 'PROPOSE_NEW'，则此项必填。"
    )

    thought_process: str = Field(
        ...,
        description="Agent 的内心独白和思考过程。"
    )

    public_speech: str = Field(
        ...,
        description="Agent 在会议桌上的公开喊话。"
    )

    def check_proposal_logic(self) -> 'AgentAction':
        """
        模型级别的验证：
        如果行动是 PROPOSE_NEW，那么 new_proposal_vector 必须存在。
        """
        if self.action == ActionType.PROPOSE_NEW and self.new_proposal_vector is None:
            raise ValueError(
                "当 'action' 为 'PROPOSE_NEW' 时, 'new_proposal_vector' 不能为空。"
            )
        return self


# ----------------------------------------------------------------------
# 3. 创建一个新的解析函数
# ----------------------------------------------------------------------
def parse_agent_action(llm_response_json: str) -> Optional[AgentAction]:
    """
    尝试将 LLM 的 JSON 字符串响应解析为一个 AgentAction 模型。

    Args:
        llm_response_json: LLM 返回的原始 JSON 字符串。

    Returns:
        一个 AgentAction 对象 (如果解析成功)，否则返回 None。
    """
    try:
        # AgentAction.model_validate_json 会自动完成:
        # 1. JSON 解析 (json.loads)
        # 2. 字段验证 (例如 'action' 必须是那三个值之一)
        # 3. 类型转换 (例如 "0.05" -> 0.05)
        action_model = AgentAction.model_validate_json(llm_response_json)
        return action_model

    except ValidationError as e:
        print(f"❌ Pydantic 验证失败! LLM 输出不符合规范。")
        print("错误详情:")
        print(e)
        return None
    except json.JSONDecodeError:
        print(f"❌ JSON 解析失败! LLM 返回的不是一个有效的 JSON。")
        print("原始响应:", llm_response_json)
        return None
