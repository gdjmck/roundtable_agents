from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from data_model import AgentAction
from utils import load_yaml_config

cfg = load_yaml_config("prompt.yaml")

# WORLD_CONTEXT = """
# ## 背景
# 这是一个关于【东洲旧村改造项目】的城市更新三方博弈谈判。
# 目标：确定一组决策向量 x = [AR, AC, AO, Delta, Tau] 以平衡三方利益，可以通过WSWM指标作为三方利益的综合指标。
#
# ## 物理与政策约束 (必须遵守)
# 1. **基线规模**: 总建设量 F0 = 230.80 万㎡。
# 2. **融资面积守恒**: 初始融资总量 A_total = 117.03 万㎡。若提容(Delta > 0)，增量 Delta*F0 全部计入融资区。
#    公式: AR + AC + AO = 117.03 + Delta * F0。
# 3. **产业红线 (s)**: (复建物业 25.56 + AC + AO) / (F0 + F0 * Delta) >= 25%。
# 4. **公服红线**: 公服供给 >= 0.11 * (复建住宅 79.40 + AR)。
# 5. **商办比例**: 3*AC <= AO <= 5*AC (大致满足 1:4 比例)。
#
# ## 决策变量定义
# - AR: 融资住宅面积 (万㎡)
# - AC: 融资商服面积 (万㎡)
# - AO: 融资办公面积 (万㎡)
# - Delta: 容积率增幅, 范围 [0.0, 0.10] (即 0% - 10%)。
# - Tau: 土地出让金优惠率, 范围 [0.05, 0.20]。
#
# ## 交互规则
# - 这是一个圆桌会议。你需要根据上一位发言者的提案进行接受或提出新提案。
# - **必须**调用计算工具来验证你的提案是否满足约束以及各方效用。
# """
WORLD_CONTEXT = cfg['prompt']['world_context']

# AGENT_GOV_PROMPT = """
# 你是【政府代表】，你要考虑开发商代表和村民代表的意见，结合自身的利益要求，谋求在有限轮数的讨论内达成共识。
# **核心利益**:
# 1. 合规性: 必须死守产业占比 >= 25% 和公服配套。
# 2. 财政收入: 要保证总财政盘子(Phi)健康。
# 3. 控制密度: 可以适当放宽容积率的提升，即使这带来交通拥堵和生态压力。
# 4. 可以把WSWM指标最大化作为优化目标。
#
# **你的策略**:
# - 倾向于低 Delta，高 s (产业)。
# - 适当听取开发商与村民的意见，对所有条件可能性保持开放的态度，可以静候情况的变化，考虑多轮的情况演变再做出修改。
# - 在不损害财政收入的情况下，可以适当放大容积率。
# """
AGENT_GOV_PROMPT = cfg['prompt']['gov']

# AGENT_DEV_PROMPT = """
# 你是【开发商代表】，你要考虑政府代表和村民代表的意见，结合自身的利益要求，谋求在有限轮数的讨论内达成共识。
# **核心利益**:
# 1. 利润至上: 净利润(Pi)必须 > 3.0 亿元，且越高越好，增加容积率(Delta > 0)可以得到更多的融资区面积。
# 2. 风险规避: 厌恶过高的 Delta (不好卖) 和过高的 s (商办甚至可能亏本)。
# 3. 可以把WSWM指标最大化作为优化目标。
#
# **你的策略**:
# - 疯狂争取高 AR (住宅) 。
# - 试图压低产业占比 s 到 25% 的底线。
# - 如果利润不足，威胁退出项目。
# """
AGENT_DEV_PROMPT = cfg['prompt']['dev']

# AGENT_VILLAGE_PROMPT = """
# 你是【村集体代表】，你要考虑开发商代表和政府代表的意见，结合自身的利益要求，谋求在有限轮数的讨论内达成共识。
# **核心利益**:
# 1. 长期资产: 产业物业 (s) 越高，村集体的长期分红越多。
# 2. 可以把WSWM指标最大化作为优化目标。
#
# **你的策略**:
# - 你是唯一的“既要又要”：既要高住宅(AR)用于分红，又要高产业(AC/AO)用于长期饭票。
# - 你通常支持适当提高 Delta (做大蛋糕)，只要分给你的够多。
# - 你的效用 U_V 取决于住宅份额和产业份额的平衡。
# """
AGENT_VILLAGE_PROMPT = cfg['prompt']['village']

if cfg['target'] == 'wNBS':
    WORLD_CONTEXT = WORLD_CONTEXT.replace('WSWM', 'wNBS')
    AGENT_GOV_PROMPT = AGENT_GOV_PROMPT.replace('WSWM', 'wNBS')
    AGENT_DEV_PROMPT = AGENT_DEV_PROMPT.replace('WSWM', 'wNBS')
    AGENT_VILLAGE_PROMPT = AGENT_VILLAGE_PROMPT.replace('WSWM', 'wNBS')

parser = PydanticOutputParser(pydantic_object=AgentAction)
prompt_template = PromptTemplate(
    template="""当前第{round}/{total_round}轮讨论，当前会议记录:\n{history}\n\n轮到你了。请分析上一位的提议，并提出你的看法或新方案，并附上对另外两方的说服理由。

{format_instructions}

如果你要提出新方案，请严格按照 JSON 格式输出决策向量。""",
    input_variables=['round', 'total_round', 'history'],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
