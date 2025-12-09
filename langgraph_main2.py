import functools
import json
import operator
from typing import TypedDict, Annotated, List, Optional, Literal
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from agent import ReflectiveAgent
from llm_model import llm
from utils import post_process_deepseek_response
from prompt import WORLD_CONTEXT, AGENT_DEV_PROMPT, AGENT_GOV_PROMPT, AGENT_VILLAGE_PROMPT, prompt_template, cfg
import LLM_optimization as math_core

agent_characters = {"Government": WORLD_CONTEXT + AGENT_GOV_PROMPT,
                    "Developer": WORLD_CONTEXT + AGENT_DEV_PROMPT,
                    "Village": WORLD_CONTEXT + AGENT_VILLAGE_PROMPT}

if cfg['target'] == 'WSWM':
    eval_func = math_core.evaluate_proposal
elif cfg['target'] == 'wNBS':
    eval_func = functools.partial(math_core.evaluate_wNBS,
                                  utility_lower_bound=cfg['utility_lb'],
                                  strength=cfg['strength'])
else:
    eval_func = None
    print("[警告]: 未定义目标函数。请检查您的配置文件。")

agents = {name: ReflectiveAgent(name, llm, system_prompt, eval_func) for name, system_prompt in
          agent_characters.items()}


# --- 1. 定义工具 (Agent 的 "手脚") ---
@tool
def evaluate_objective_function(input_vector: List[float], metric: Literal['WSWM', 'wNBS']) -> float:
    """
    调用目标函数 (例如 WSWM) 来评估给定向量的分数。
    分数越高越好。
    """
    print(f"   [工具]: 正在评估向量 {input_vector}...")
    try:
        # 假设我们使用 run_optimization.py 中的逻辑
        # 我们需要从 eval_func 获取分数
        # 注意：您的 eval_func 需要5个浮点数
        if len(input_vector) != 5:
            return -999.9  # 返回一个无效分数

        eval_result = eval_func(
            AR=input_vector[0],
            AC=input_vector[1],
            AO=input_vector[2],
            Delta=input_vector[3],
            Tau=input_vector[4]
        )
        if eval_result["status"] == "ACCEPTED":
            score = float(eval_result["utilities"][metric])
            print(f"   [工具]: 向量 {input_vector} 的 {metric} 分数是: {score}")
            return score
        else:
            print(f"   [工具]: 向量 {input_vector} 因不合规被拒绝。")
            return -999.9  # 返回一个极低的分数
    except Exception as e:
        print(f"   [工具]: 评估失败: {e}")
        return -999.9


# 工具B：LLM 用它来提交新方案
class ProposeNewVector(BaseModel):
    """用于提交一个新优化方案的工具。"""
    input_vector: List[float] = Field(description="当前决策向量。")
    score_old: float = Field(description="当前决策向量的评分。")
    role_name: str = Field(description="当前用户角色。")
    # messages: List[BaseMessage] = Field(description="历史信息。")


@tool(args_schema=ProposeNewVector)
def propose_new_vector(role_name: str, input_vector: List[float], score_old: float):
    """
    选择用户角色，对当前情况提出新的向量方案
    """
    situation_message = prompt_template.format(history=f"当前方案是为评分为{score_old:.4f}的决策向量{input_vector}",
                                               round=round, total_round=5)
    # situation_message = SystemMessage(content=)
    response = agents[role_name].propose([situation_message])
    # todo: add thought process
    if response['evaluation']:
        return json.dumps({'score_new': response['evaluation']['utilities'][cfg['target']],
                           'new_vector': response['action_model'].new_proposal_vector})
    else:
        return "未提出有效方案"


# 工具C：LLM 用它来结束任务
class FinalDecisionSchema(BaseModel):
    """用于做出最终决策并结束优化流程"""
    decision: str = Field(description="'accept_new' (接受新方案) 或 'reject_new' (维持旧方案)")
    justification: str = Field(description="做出此决策的最终理由。")
    final_vector: List[float] = Field(description="最终选定的向量。")


@tool(args_schema=FinalDecisionSchema)
def final_decision(decision: str, justification: str, final_vector: List[float]):
    """
    用于做出最终决策并结束优化流程的工具。
    调用此工具标志着优化任务的完成。
    """
    # 这里是工具被调用时的执行逻辑
    # 在 LangGraph 中，这通常只是打印日志或返回一个确认字符串
    print(f"\n--- 🏁 Agent 提交了最终决策 ---")
    print(f"决策: {decision}")
    print(f"理由: {justification}")
    print(f"最终向量: {final_vector}")

    # 返回值会作为 ToolMessage 回传给 LLM (虽然通常流程在此结束)
    return f"任务结束。决策: {decision}, 最终向量: {final_vector}"


# --- 2. 定义 Agent 和工具执行器 ---

# 注册所有工具
tools = [evaluate_objective_function, propose_new_vector, final_decision]
tools_map = {t.name: t for t in tools}

# 定义 Agent (LLM + 绑定工具)
# 我们将使用 PydanticToolsParser，所以 LLM 可以自由选择工具
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

# --- 3. 定义 Agent 的“大脑” (System Prompt) ---
# 这是最关键的部分，我们在这里定义了您的算法！

OPTIMIZER_SYSTEM_PROMPT = """
你是一个“优化器 Agent”。你的目标是接收一个输入向量，提出一个新向量，并判断新向量是否在“目标函数”上取得了更好的分数。

你必须严格按照以下步骤执行：

1.  **接收输入**：用户会提供一个 `input_vector`。
2.  **评估旧方案**：你 *必须* 首先调用 `evaluate_objective_function` 工具来获取 `input_vector` 的分数（`score_old`）。
3.  **提出新方案**：在获得 `score_old` 后，你 *必须* 分析 `input_vector` 和 `score_old`，然后调用 `propose_new_vector` 。
    该工具会返回 **新向量 (`new_vector`)** 以及 **新分数 (`score_new`)**。
    * 注意：你 **不需要** 再次调用评估工具，直接使用工具返回的 `score_new`。
4.  **做出决策**：在同时拥有 `score_old` 和 `score_new` 后，你 *必须* 比较它们。
    * 如果 `score_new > score_old`，说明新方案更好。
    * 如果 `score_new <= score_old`，说明新方案没有提升或更差。
5.  **结束**：你 *必须* 调用 `final_decision` 工具来结束任务，明确说明你接受还是拒绝新方案，并附上理由。

不要跳过任何步骤。

【final_decision 调用规范】
只有在拥有 `score_old` 和 `score_new` 两个分数并完成比较后，才能调用 `final_decision`。
调用时必须严格填充以下字段：
- decision: 必须是 'accept_new' 或 'reject_new' 字符串。
- final_vector: 必须是一个 5 维浮点数列表。
- justification: 必须包含两个分数的具体数值对比（例如 "新方案分数 0.95 高于旧方案 0.88"）。

【重要格式指令】
1. 你必须且只能生成 Tool Call。
2. **禁止并行调用**：你每次回复 **必须且只能** 调用 **1 个** 工具。
3. **严禁**生成任何自然语言文本或“思考过程”。
4. **严禁**在 content 字段中输出 JSON，必须使用标准的 function calling 协议。
5. 如果你输出纯文本，任务将直接失败。
"""


# --- 4. 定义图的状态 (GraphState) ---

class GraphState(TypedDict):
    """
    GraphState 是图的“内存”。
    'messages' 将包含所有的对话历史。
    """
    messages: Annotated[List[BaseMessage], operator.add]


# --- 5. 定义图的节点和边 ---

def agent_node(state: GraphState):
    """
    Agent 节点：调用 LLM (大脑)
    """
    print(f"--- 🧠 Agent 正在思考... ---")
    llm_forced = llm_with_tools.bind(tool_choice='required')
    response = llm_forced.invoke(state["messages"])
    response = post_process_deepseek_response(response)
    # print(response.content)
    return {"messages": [response]}


def tool_node(state: GraphState):
    """
    Tool 节点：执行工具 (手脚)
    """
    print(f"--- 🛠️ Agent 正在行动... ---")
    # 获取 LLM 刚刚的回复
    last_message = state["messages"][-1]

    # (注意：AIMessage.tool_calls 在 Langchain 0.1. tool_calls 是新标准)
    tool_calls = last_message.tool_calls

    # 执行工具
    tool_messages = []
    for call in tool_calls:
        tool_name = call['name']
        if tool_name not in tools_map:
            content = f"Error: Tool {tool_name} not found."
        elif tool_name == 'propose_new_vector':
            execution_args = call['args'].copy()
            execution_args['messages'] = state["messages"]
            execution_args['role_name'] = role
            output = tools_map[tool_name].invoke(execution_args)
            content = json.dumps(output)

        else:
            output = tools_map[tool_name].invoke(call['args'])  # call 已经是 (name, args) 格式
            content = json.dumps(output)

        tool_messages.append(
            ToolMessage(content=content, tool_call_id=call['id'])
        )

        # 调用工具后的辅助指令
        if tool_name == 'propose_new_vector':
            # 添加系统指令，让LLM对结果进行分析
            instruction_message = HumanMessage(
                content=f"已收到新方案数据：{content}。\n"
                        f"请**仔细对比**新方案的得分 (score_new) 与旧方案得分 (score_old)。\n"
                        f"然后根据对比结果，调用 `final_decision` 结束任务。\n"
                        f"在调用 `final_decision` 时，请务必在 `justification` 字段中详细说明两个分数的对比情况。")
            tool_messages.append(instruction_message)

    return {"messages": tool_messages}


def conditional_router(state: GraphState):
    """
    路由节点：决定下一步做什么 (循环或结束)
    """
    last_message = state["messages"][-1]

    # 如果 LLM 的上一步是 AIMessage 并且它调用了工具
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "call_tools"
    else:
        # 否则 (例如工具执行完毕后)，返回给 Agent 思考
        return "call_agent"


def route_after_tools(state: GraphState):
    """
    工具执行后的路由：判断刚才执行的工具是否是终止工具。
    """
    messages = state["messages"]

    # 倒序查找最近的一条 AI 消息（它触发了当前的工具执行）
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            if msg.tool_calls and any(call['name'] == 'final_decision' for call in msg.tool_calls):
                return END
            break

    # 如果没调用终止工具，则回 agent 继续循环
    return "agent"


class GraphAgent:
    def __init__(self, role: Literal['Government', 'Developer', 'Village'], metric: Literal['WSWM', 'wNBS'] = 'WSWM'):
        self.role = role
        self.metric = metric
        workflow = self._init_workflow()
        self.app = workflow.compile()
        self._optimize_prompt = OPTIMIZER_SYSTEM_PROMPT

    def _init_workflow(self):
        workflow = StateGraph(GraphState)
        # 2. 添加节点
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", self.tool_node)
        # 3. 设置入口
        workflow.set_entry_point("agent")
        # 4. 添加边
        workflow.add_conditional_edges(
            "agent",  # 起点
            conditional_router,  # 路由函数
            {
                "call_tools": "tools",  # 如果路由返回 "call_tools"，则去 "tools" 节点
                "call_agent": "agent",  # 如果路由返回 "call_agent"，则去 "agent" 节点
                END: END  # 如果路由返回 END，则结束
            }
        )
        workflow.add_conditional_edges(
            "tools",  # 起点：工具节点
            route_after_tools,  # 路由函数：执行完工具后判断去哪
            {
                "agent": "agent",  # 情况A：普通工具 -> 回 Agent
                END: END  # 情况B：FinalDecision -> 结束
            }
        )
        return workflow

    def tool_node(self, state: GraphState):
        """
        Tool 节点：执行工具 (手脚)
        """
        print(f"--- 🛠️ Agent 正在行动... ---")
        # 获取 LLM 刚刚的回复
        last_message = state["messages"][-1]

        # (注意：AIMessage.tool_calls 在 Langchain 0.1. tool_calls 是新标准)
        tool_calls = last_message.tool_calls

        # 执行工具
        tool_messages = []
        for call in tool_calls:
            tool_name = call['name']
            if tool_name not in tools_map:
                content = f"Error: Tool {tool_name} not found."
            elif tool_name == 'propose_new_vector':
                execution_args = call['args'].copy()
                execution_args['messages'] = state["messages"]
                execution_args['role_name'] = self.role
                output = tools_map[tool_name].invoke(execution_args)
                content = json.dumps(output)
            elif tool_name == 'evaluate_objective_function':
                execution_args = call['args'].copy()
                execution_args['metric'] = self.metric
                output = tools_map[tool_name].invoke(execution_args)
                content = json.dumps(output)
            else:
                output = tools_map[tool_name].invoke(call['args'])  # call 已经是 (name, args) 格式
                content = json.dumps(output)

            tool_messages.append(
                ToolMessage(content=content, tool_call_id=call['id'])
            )

            # 调用工具后的辅助指令
            if tool_name == 'propose_new_vector':
                # 添加系统指令，让LLM对结果进行分析
                instruction_message = HumanMessage(
                    content=f"已收到新方案数据：{content}。\n"
                            f"请**仔细对比**新方案的得分 (score_new) 与旧方案得分 (score_old)。\n"
                            f"然后根据对比结果，调用 `final_decision` 结束任务。\n"
                            f"在调用 `final_decision` 时，请务必在 `justification` 字段中详细说明两个分数的对比情况。")
                tool_messages.append(instruction_message)

        return {"messages": tool_messages}

    def invoke(self, vector: List[float]):
        initial_messages = [
            SystemMessage(content=self._optimize_prompt),
            HumanMessage(content=f"请优化这个向量: {vector}")
        ]
        final_state = self.app.invoke({"messages": initial_messages})
        latest_vector = vector

        for msg in reversed(final_state['messages']):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                # 检查是否有 final_decision 调用
                final_call = next((call for call in msg.tool_calls if call['name'] == 'final_decision'), None)

                if final_call:
                    args = final_call['args']
                    # 提取 final_vector，如果没有则保持原值
                    if 'final_vector' in args:
                        latest_vector = args['final_vector']
                    break  # 找到后立即停止

        return {'messages': final_state['messages'],
                'latest_vector': latest_vector,
                }


if __name__ == '__main__':
    # --- 6. 构建并运行图 ---

    # 1. 创建图
    workflow = StateGraph(GraphState)

    # 2. 添加节点
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    # 3. 设置入口
    workflow.set_entry_point("agent")

    # 4. 添加边
    workflow.add_conditional_edges(
        "agent",  # 起点
        conditional_router,  # 路由函数
        {
            "call_tools": "tools",  # 如果路由返回 "call_tools"，则去 "tools" 节点
            "call_agent": "agent",  # 如果路由返回 "call_agent"，则去 "agent" 节点
            END: END  # 如果路由返回 END，则结束
        }
    )
    # workflow.add_edge("tools", "agent")  # 工具执行完后，总是返回给 Agent 思考
    workflow.add_conditional_edges(
        "tools",  # 起点：工具节点
        route_after_tools,  # 路由函数：执行完工具后判断去哪
        {
            "agent": "agent",  # 情况A：普通工具 -> 回 Agent
            END: END  # 情况B：FinalDecision -> 结束
        }
    )

    # 5. 编译图
    app = workflow.compile()

    # 6. 运行 Agent！
    print("--- 🚀 优化器 Agent 已启动 ---")

    # 您的输入向量 (来自 agent.py)
    input_vector = [75.91, 8.35, 32.77, 0.0, 0.20]

    # 我们通过输入消息来启动 Agent
    role = "Government"
    round = 1
    total_round = 5
    initial_messages = [
        SystemMessage(content=OPTIMIZER_SYSTEM_PROMPT),
        HumanMessage(content=f"请优化这个向量: {input_vector}")
    ]

    # .stream() 会返回每一步的状态，让您可以看到完整的思考过程
    # .invoke() 只会返回最终状态
    final_state = app.invoke({"messages": initial_messages})

    print("\n--- ✅ 流程结束 ---")
    print("最终的对话历史:")
    for msg in final_state['messages']:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            print(f"AI -> 决定调用工具: {msg.tool_calls[0]['name']}({msg.tool_calls[0]['args']})")
        elif isinstance(msg, ToolMessage):
            print(f"工具 -> 返回: {msg.content}")
