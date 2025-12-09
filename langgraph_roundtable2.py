import operator
import json
import traceback
from typing import TypedDict, Annotated, List, Dict, Union
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph_main2 import GraphAgent, eval_func
from color_print import gov_print, dev_print, village_print
from prompt import cfg
NEXT_ROUND = "NEXT_ROUND"

printer = {"Government": gov_print, "Developer": dev_print, "Village": village_print}

# --- 导入您现有的模块 ---
# 确保这些文件在同一目录下
from agent import ReflectiveAgent
from llm_model import llm
from prompt import (
    WORLD_CONTEXT,
    AGENT_GOV_PROMPT,
    AGENT_DEV_PROMPT,
    AGENT_VILLAGE_PROMPT,
    prompt_template
)

runtime_config = {
    "recursion_limit": cfg['max_node_jump_steps']
}


# --- 1. 定义图的状态 (State) ---
class RoundTableState(TypedDict):
    """
    圆桌会议的全局状态
    """
    # 会议历史：存储所有的发言记录
    history: Annotated[List[BaseMessage], operator.add]
    # 当前轮次
    current_round: int
    # 总轮次限制
    total_round: int
    # 所有人都同意当前方案
    all_agree: bool
    # 最新生效的决策向量 (用于 Prompt 中提示当前状态)
    latest_vector: List[float]
    # 上一位发言者 (用于 Prompt 上下文)
    last_speaker: str
    # 提案日志 (用于最终统计)
    proposals_log: Annotated[List[Dict], operator.add]
    pareto_records: Annotated[List[Dict], operator.add]


# --- 2. 初始化 Agents ---
# 直接复用 agent.py 中的 ReflectiveAgent 类
# 我们在这里实例化它们，作为图的静态资源
agent_configs = {
    "Government": WORLD_CONTEXT + AGENT_GOV_PROMPT,
    "Developer": WORLD_CONTEXT + AGENT_DEV_PROMPT,
    "Village": WORLD_CONTEXT + AGENT_VILLAGE_PROMPT
}

# # 创建三个 ReflectiveAgent 实例
# agents_map = {
#     name: ReflectiveAgent(name, llm, system_prompt, math_core.evaluate_proposal)
#     for name, system_prompt in agent_configs.items()
# }

langgraph_agents = {role: GraphAgent(role, metric=cfg['target']) for role in agent_configs}


# --- 3. 定义通用节点逻辑 ---

def run_role_node(state: RoundTableState, role_name: str):
    printer[role_name](f"\n🎤 --- 轮到 {role_name} 发言 (第 {state['current_round']} 轮) ---")
    printer[role_name](f"[当前决策向量]: {state['latest_vector']}")
    new_messages = langgraph_agents[role_name].invoke(state['latest_vector'])

    # 记录帕累托记录
    try:
        eval_result = eval_func(*new_messages['latest_vector'])
        utilities = eval_result['utilities']
    except KeyError:
        print('缺少utilities字段', eval_result)
        utilities = eval_func(*state['latest_vector'])
    pareto_entry = {
        'utilities': utilities,
        'delta': new_messages['latest_vector'][3]
    }

    updates = dict(history=new_messages['messages'], last_speaker=role_name,
                   latest_vector=new_messages['latest_vector'], pareto_records=[pareto_entry])

    return updates


# def run_role_node(state: RoundTableState, role_name: str):
#     """
#     通用的角色节点执行函数。
#     它负责将图的状态转换为 ReflectiveAgent 需要的输入，并处理输出。
#     """
#     agent = agents_map[role_name]
#
#     print(f"\n🎤 --- 轮到 {role_name} 发言 (第 {state['current_round']} 轮) ---")
#
#     # 1. 准备 Prompt
#     # 我们需要把 BaseMessage 列表转换为 prompt_template 需要的字符串格式
#     # 这里模拟了 agent.py 中 RoundTableLLM._format_history 的逻辑
#     history_text = "\n".join([
#         f"{msg.name if hasattr(msg, 'name') else msg.type}: {msg.content}"
#         for msg in state['history'][-6:]  # 只取最近几条，避免 token 过长
#     ])
#
#     # 添加当前最新的方案信息
#     current_status_str = f"{state.get('last_speaker', '主持人')} 提出的最新决策向量为 {state['latest_vector']}"
#     full_history_str = f"{history_text}\n{current_status_str}"
#
#     # 使用 prompt.py 中的模板格式化
#     formatted_prompt = prompt_template.format(
#         round=state['current_round'],
#         total_round=state['total_round'],
#         history=full_history_str
#     )
#
#     # 2. 调用 Agent
#     # ReflectiveAgent.propose 内部会自动处理 "提议-校验-修正" 循环
#     # 我们将格式化好的 Prompt 包装为 HumanMessage 传入
#     response = agent.propose([HumanMessage(content=formatted_prompt)])
#
#     # 3. 处理结果并更新状态
#     action_model = response['action_model']
#     evaluation = response['evaluation']
#
#     new_messages = []
#     updates = {}
#
#     # 记录 Agent 的公开喊话
#     speech_content = action_model.public_speech
#     new_messages.append(AIMessage(content=speech_content, name=role_name))
#
#     if evaluation:
#         # 如果提出了新方案且通过校验
#         new_vector = action_model.new_proposal_vector
#         updates['latest_vector'] = new_vector
#         updates['proposals_log'] = [response]
#
#         # 添加系统公证信息
#         sys_msg = (
#             f"[系统公证]: {role_name} 提出了新方案 {new_vector}。\n"
#             f"评估指标: WSWM={evaluation['utilities']['WSWM']}, "
#             f"U_G={evaluation['utilities']['U_G']}, "
#             f"U_D={evaluation['utilities']['U_D']}, "
#             f"U_V={evaluation['utilities']['U_V']}"
#         )
#         print(sys_msg)  # 控制台打印
#         new_messages.append(SystemMessage(content=sys_msg))
#     else:
#         # 如果接受了方案或没提新方案
#         print(f"[{role_name}] 未提出有效新方案 (维持现状)")
#
#     updates['history'] = new_messages
#     updates['last_speaker'] = role_name
#
#     return updates


# --- 4. 定义具体节点 ---
# LangGraph 需要具体的函数作为节点

def government_node(state: RoundTableState):
    return run_role_node(state, "Government")


def developer_node(state: RoundTableState):
    return run_role_node(state, "Developer")


def village_node(state: RoundTableState):
    return run_role_node(state, "Village")


def round_manager_node(state: RoundTableState):
    """
    管理轮次的节点
    """
    new_round = state['current_round'] + 1
    return {"current_round": new_round}


def summary_node(state: RoundTableState):
    """
    总结所有人的改动意见，结合角色的话语权，总结新的方案向量
    """
    pass


def check_continuation(state: RoundTableState):
    """
    条件边逻辑：判断是继续还是结束
    """
    if state['all_agree']:
        print("\n🎉 所有人同意，结束会议。")
        return END
    elif state['current_round'] > state['total_round']:
        print("\n🛑 会议达到最大轮次，结束。")
        return END
    else:
        print(f"\n🔄 进入第 {state['current_round']} 轮...")
        return NEXT_ROUND  # 下一轮


# --- 5. 构建图 (Graph) ---

workflow = StateGraph(RoundTableState)

# 添加节点
workflow.add_node("Government", government_node)
workflow.add_node("Developer", developer_node)
workflow.add_node("Village", village_node)
workflow.add_node("RoundManager", round_manager_node)

# 设置入口
workflow.set_entry_point("Government")

# 添加边 (定义发言顺序)
workflow.add_edge("Government", "Developer")
workflow.add_edge("Developer", "Village")
workflow.add_edge("Village", "RoundManager")  # 村民发言后，进入轮次管理

# 添加条件边 (判断循环)
workflow.add_conditional_edges(
    "RoundManager",
    check_continuation,
    {
        NEXT_ROUND: "Government",  # 继续循环
        END: END  # 结束
    }
)

# 编译图
app = workflow.compile()

# --- 6. 运行主程序 ---

if __name__ == "__main__":

    print("--- 🚀 LangGraph 圆桌会议启动 ---")
    baseline_vector = cfg['baseline_vector']

    # 初始状态
    initial_state = {
        "history": [
            SystemMessage(content="主持人: 会议开始。请各方基于基准方案发表意见。")
        ],
        "current_round": 1,
        "total_round": cfg['max_round'],  # 设定讨论轮数
        "latest_vector": baseline_vector,
        "last_speaker": "主持人",
        "proposals_log": [],
        "pareto_records": []
    }

    # 运行图
    # # 使用 .stream() 可以实时看到每一步的输出
    # for s in app.stream(initial_state):
    #     # 这里可以打印每一步的状态更新，用于调试
    #     print(s)
    discuss_result = app.invoke(initial_state, config=runtime_config)

    print("\n--- ✅ 会议结束 ---")
    # 可以从最终状态中提取结果（如果有办法获取最终状态对象）
    # 由于 stream 迭代完，我们可能需要把 final state 存下来，或者只看打印日志
    with open(f'result_{cfg["target"]}.json', 'w') as f:
        if 'history' in discuss_result:
            discuss_result.pop('history')
        json.dump(discuss_result, f)
