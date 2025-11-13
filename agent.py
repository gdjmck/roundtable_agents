import json
import numpy as np
# 假设这是我们上一轮定义的计算核心 (Calculator)
# 包含 evaluate_proposal, utility_G, utility_D, utility_V 等函数
import LLM_optimization as math_core
from typing import List, Dict, Any, Optional
from llm_model import llm
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from data_model import AgentAction, ActionType
from prompt import WORLD_CONTEXT, AGENT_DEV_PROMPT, AGENT_GOV_PROMPT, AGENT_VILLAGE_PROMPT, prompt_template, parser
from langchain_core.output_parsers import PydanticOutputParser
from color_print import gov_print, dev_print, village_print

printer = {'Government': gov_print, 'Developer': dev_print, 'Village': village_print}


def extract_speech(llm_response_json: str) -> str:
    """
    从 LLM 输出的 JSON 字符串中解析 'public_speech'。

    Args:
        llm_response_json: LLM 返回的原始 JSON 字符串。

    Returns:
        公开喊话的字符串。如果失败则返回一个错误信息。
    """
    try:
        data: Dict[str, Any] = json.loads(llm_response_json)
        speech = data.get('public_speech', '错误：找不到 "public_speech" 字段。')
        return str(speech)

    except json.JSONDecodeError:
        print(f"解析错误: 无法解析 LLM 响应 '{llm_response_json}'")
        return "解析 LLM 响应失败。"
    except Exception as e:
        return f"提取喊话时出错: {e}"


class RoundTableLLM:
    def __init__(self, llm, data_model_cls, max_round: int=5):
        self._llm = llm  # 可以直接跳过prompt中指定输出字段的步骤
        self.parser = PydanticOutputParser(pydantic_object=data_model_cls)
        self.max_round = max_round  # 最大的讨论轮数
        self.current_round = 0
        self.history = []  # 全局会议记录
        self.history_length = 6
        self.proposals_log = []  # 记录所有被提出的合法向量
        self.baseline_vector = [75.91, 8.35, 32.77, 0.0, 0.20]
        self.latest_vector = [75.91, 8.35, 32.77, 0.0, 0.20]
        self.last_speaker = '主持人'

        # 定义 Agents
        self.characters = {
            "Government": {
                "system": WORLD_CONTEXT + AGENT_GOV_PROMPT,
                "style": "严肃、讲原则、关注宏观指标"
            },
            "Developer": {
                "system": WORLD_CONTEXT + AGENT_DEV_PROMPT,
                "style": "精明、数据驱动、一直在算账"
            },
            "Village": {
                "system": WORLD_CONTEXT + AGENT_VILLAGE_PROMPT,
                "style": "务实、寸土必争、关注长远分红"
            }
        }
        self.agents = {
            name: ReflectiveAgent(name, llm, info['system'], style=info['style']) for name, info in
            self.characters.items()
        }
        self._init_history()

    def print(self, content: str, agent_name: str = ''):
        if agent_name in printer:
            printer[agent_name](content)
        else:
            print(content)

    def _init_history(self):
        self.history = [
            (
                f"主持人: 各位代表好，会议开始。我们从基准方案出发，基准方案只是大致的比例，并不一定满足所有约束条件 (x0)：\n"
                f"  AR={self.baseline_vector[0]}, AC={self.baseline_vector[1]}, AO={self.baseline_vector[2]}, "
                f"Delta={self.baseline_vector[3]}, Tau={self.baseline_vector[4]}\n"
                f"请政府代表首先根据此方案发表意见。"
            )
        ]

    @property
    def chain(self):
        return self._llm | self.parser

    def _call_llm(self, agent_name: str, context_messages: dict) -> str:
        """
        调用 LLM API
        """
        # todo: 需要把最新的决策向量传入到human message
        sys_msg = SystemMessage(content=self.agents[agent_name]['system'])
        speech_msg = HumanMessage(content=prompt_template.format(history=context_messages['content']))
        response = self.chain.invoke([sys_msg, speech_msg])
        self.print(f"🤖 [{agent_name}] 正在思考...答复:{response.public_speech}\n------------------------", agent_name)
        return response

    def round_discuss(self):
        while self.current_round < self.max_round:
            self.current_round += 1
            self.run_round()

    def run_round(self, speaker_order=["Government", "Developer", "Village"]):
        print(f'开始第{self.current_round}/{self.max_round}轮讨论')
        """运行一轮谈判"""
        for agent_name in speaker_order:
            self.print(f"\n🎤 --- {agent_name} 发言 ---")

            history_message = prompt_template.format(history=self._format_history(),
                                                     round=self.current_round,
                                                     total_round=self.max_round)
            response = self.agents[agent_name].propose([history_message])
            if response['evaluation'] is None:
                self.print(f"{agent_name}: 未提出新的有效方案", agent_name)
                self.history.append(f"{agent_name}: 接受当前方案，未提出新的有效方案")
            else:
                print('发言结果')
                self.print(response['action_model'], agent_name)
                self.print(response['evaluation'], agent_name)
                speech_content = response['action_model'].public_speech
                speech_content += f"\n[系统公证数据]: 方案可行性={response['evaluation']['status']}, WSWM={response['evaluation']['utilities']['WSWM']}, U_G={response['evaluation']['utilities']['U_G']}, U_D={response['evaluation']['utilities']['U_D']}, U_V={response['evaluation']['utilities']['U_V']}"
                self.latest_vector = response['action_model'].new_proposal_vector
                self.proposals_log.append(response)
                self.history.append(f"第{self.current_round}轮{agent_name}的发言: {speech_content}")
                self.last_speaker = agent_name

    def _format_history(self):
        current_proposal = f'{self.last_speaker}提出的最新决策向量为{self.latest_vector}'
        return "\n".join(self.history[-self.history_length:] + [current_proposal])  # 增加当前方案


class RoundTableTooledLLM(RoundTableLLM):
    def __init__(self, llm, data_model_cls):
        super().__init__(llm, data_model_cls)
        self._llm = self._llm.with_structured_output(data_model_cls)

    @property
    def chain(self):
        return self._llm


class ReflectiveAgent:
    """
    一个封装了“提议-评估-修正”循环的博弈 Agent。
    """

    def __init__(self, name, llm, system_prompt, style: str = '', max_retry: int = 3):
        """

        :param name:
        :param llm: langchain ChatOpenAI llm
        :param system_prompt:
        :param style: 人设风格
        """
        self.name = name
        self.max_retry = max_retry
        self.system_prompt = SystemMessage(content=system_prompt)
        self.style = style
        self.proposal_chain = llm | PydanticOutputParser(pydantic_object=AgentAction)
        print(f'{self.name} Agent初始化完毕')

    def propose(self, history_proposal: List[BaseMessage]):
        """

        :param history_proposal: BaseMessage List 可直接作为invoke的输入
        :return: {"action_model": ActionModel, "evaluation": math_core.evaluate_proposal}
        """
        default_messages = [self.system_prompt] + history_proposal
        eval_failure = None  # {'propose_vector': [], 'info': dict}

        for i in range(self.max_retry):
            if eval_failure is None:
                messages = default_messages
            else:
                correction_prompt = (
                        f"[系统修正请求]\n"
                        f"你刚才的提案 {eval_failure['propose_vector']} 因违反以下约束而被拒绝：\n" +
                        "\n".join(f"- {v}" for v in eval_failure['info']['violations']) +
                        "\n请你必须修正上述所有错误，重新思考并提出一个合规的新方案。"
                )
                messages = default_messages + [HumanMessage(content=correction_prompt)]

            printer[self.name](messages)
            action_model: AgentAction = self.proposal_chain.invoke(messages)

            if action_model.action == ActionType.PROPOSE_NEW:
                # 验证新提案
                eval_result = math_core.evaluate_proposal(*action_model.new_proposal_vector)
                if eval_result['status'] == 'ACCEPTED':
                    return {
                        "action_model": action_model,
                        "evaluation": eval_result
                    }
                else:
                    eval_failure = {'propose_vector': action_model.new_proposal_vector, 'info': eval_result}
            elif action_model.action == ActionType.ACCEPT:
                print(f"🟢 {self.name} 接受当前方案")
                return {
                    "action_model": action_model,
                    "evaluation": None  # 没有新方案，无需评估
                }
        # 尝试次数用完，接受当前方案
        return {
            "action_model": action_model,
            "evaluation": None  # 没有新方案，无需评估
        }


if __name__ == "__main__":
    # --- 模拟运行 ---
    table = RoundTableLLM(llm, data_model_cls=AgentAction, max_round=10)
    table.round_discuss()

    print(f'最终方案: {table.latest_vector}')
