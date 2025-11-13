import numpy as np
import sys

# ----------------------------------------------------------------------
# 步骤 1: 定义常量与参数 (基于 run_optimization.py)
# ----------------------------------------------------------------------
# 货币单位: 亿元, 面积单位: 万㎡

# A. 基线规模
F0 = 230.80  # 基线总建筑面积
F_res_recon = 79.40  # 复建住宅建面
F_prop_recon = 25.56  # 复建物业
PF_recon = 8.734  # 复建公服
A_total = 117.03  # 融资总量 (基线)

# B. 价格 (亿元/万㎡)
P_R = 7636 / 10000  # 融资住宅单价 0.7636
P_C = 9205 / 10000  # 融资商服单价 0.9205
P_O = 3973 / 10000  # 融资办公单价 0.3973

# C. 监管阈值
s_min = 0.25  # 产业占比下限
r_pf = 0.11  # 公服联动比例
delta_max = 0.10  # 容积率提升上限

# D. 成本与利润参数
C0 = 70.69  # (亿元) 基线总成本常数
pi_min = 3.0  # (亿元) 假设的最低可接受利润

# E. 效用函数参数 (假设)
# U_G (政府)
beta = np.array([0.35, 0.25, 0.20, 0.20])
rho_G = 0.8
lambda_financial_income = 0.2
Phi_min = 1.0
Phi_tar = 74.41 * 0.05
k_ext = 0.5
# U_D (开发商)
gamma = np.array([0.80, 0.15, 0.05])
C_delta = 1.0
C_PF = 0.8
C_land = 0
# U_V (村集体)
eta = np.array([0.5, 0.3, 0.2])
rho_res = 0.67
rho_prop = 0.20
rho_cash = 0.43
epsilon = 1e-6


# ----------------------------------------------------------------------
# 步骤 2 & 3: 中间变量与效用函数 (Helper Functions)
# ----------------------------------------------------------------------

def get_intermediate_vars(x):
    """根据决策向量 x 计算所有派生变量"""
    AR, AC, AO, Delta, Tau = x

    # 总量
    F = F0 * (1 + Delta)
    F_res_total = F_res_recon + AR

    # 产业占比 s
    F_ind = F_prop_recon + AC + AO
    s = F_ind / (F + epsilon)  # F 可能为0? 不，F0 > 0

    # 公服 PF
    PF_req = r_pf * (F_res_recon + AR)
    PF_sup = PF_recon + r_pf * AR
    PF_gap = np.maximum(0, PF_req - PF_sup)

    # 份额 (U_V 用)
    PF_share = PF_sup / (F + epsilon)
    Res_share = np.maximum(0, 1 - s - PF_share)
    Cash_share = rho_cash

    # 收入 Rev (亿元)
    Rev = P_R * AR + P_C * AC + P_O * AO

    # 成本 Cost (亿元)
    Cost_delta = C_delta * (Delta ** 2)
    Cost_PF_penalty = C_PF * PF_gap
    Cost_land = C_land * (1 - Tau) * F
    Cost = C0 + Cost_delta + Cost_PF_penalty + Cost_land

    # 利润 Pi
    Pi = Rev - Cost

    # 政府财政 Phi (亿元)
    L0 = lambda_financial_income * Rev
    tC = 0.01;
    tO = 0.01
    Phi = L0 * (1 - Tau) + tC * P_C * AC + tO * P_O * AO - k_ext * (Delta ** 2)

    return {
        'F': F, 's': s, 'PF_req': PF_req, 'PF_sup': PF_sup,
        'PF_gap': PF_gap, 'Res_share': Res_share, 'Cash_share': Cash_share,
        'Rev': Rev, 'Cost': Cost, 'Pi': Pi, 'Phi': Phi, 'F_res_total': F_res_total
    }


def utility_G(x, vars):
    s, PF_req, PF_sup, Phi = vars['s'], vars['PF_req'], vars['PF_sup'], vars['Phi']
    Delta = x[3]
    z1 = min(1.0, s / s_min)
    z2 = min(1.0, PF_sup / (PF_req + epsilon))
    z3 = 1 - (Delta / delta_max)
    z4 = np.clip((Phi - Phi_min) / (Phi_tar - Phi_min), 0.0, 1.0)
    z = np.array([z1, z2, z3, z4])
    term = np.sum(beta * (z ** rho_G))
    return term ** (1 / rho_G)


def utility_D(x, vars):
    s, Pi = vars['s'], vars['Pi']
    Delta = x[3]
    if (Pi - pi_min) < 0: return -np.inf  # 利润不达标，效用崩盘
    term1 = gamma[0] * np.log(Pi - pi_min + epsilon)
    term2 = gamma[1] * (1 - (Delta / delta_max))
    term3 = gamma[2] * np.maximum(0, s - s_min)
    return term1 + term2 - term3


def utility_V(x, vars):
    s, Res_share, Cash_share = vars['s'], vars['Res_share'], vars['Cash_share']
    term1 = eta[0] * np.log(Res_share / rho_res + epsilon)
    term2 = eta[1] * np.log(s / rho_prop + epsilon)
    term3 = eta[2] * np.log(Cash_share / rho_cash + epsilon)
    return term1 + term2 + term3


def check_constraints(x, vars_dict):
    """检查所有约束是否满足"""
    AR, AC, AO, Delta, Tau = x
    s = vars_dict['s']
    PF_sup = vars_dict['PF_sup']
    F_res_total = vars_dict['F_res_total']

    constraints_met = True
    violations = []

    # 边界
    bounds_min = [0, 0, 0, 0.0, 0.05]
    bounds_max = [A_total, A_total, A_total, delta_max, 0.20]
    if not all(x[i] >= bounds_min[i] and x[i] <= bounds_max[i] for i in range(5)):
        constraints_met = False
        violations.append(f"边界约束: 向量 {x} 超出 {bounds_min} / {bounds_max} 范围。")

    # 1. 融资面积守恒 (来自 `run_optimization.py` 的定义)
    # A_total + Delta*F0 = AR + AC + AO
    con1 = (A_total + Delta * F0) - (AR + AC + AO)
    if not np.isclose(con1, 0, atol=1e-3):  # 允许微小误差
        constraints_met = False
        violations.append(f"融资面积守恒: AR+AC+AO ({AR + AC + AO:.2f}) != A_total+Delta*F0 ({A_total + Delta * F0:.2f})。")

    # 2. 产业比例
    con2 = s - s_min
    if con2 < 0:
        constraints_met = False
        violations.append(f"产业比例: s ({s * 100:.2f}%) < s_min ({s_min * 100}%)。")

    # 3. 商服:办公 (AC:AO)
    con3 = 5 * AC - AO
    if con3 < 0:
        constraints_met = False
        violations.append(f"商办比(1): 5*AC ({5 * AC:.2f}) < AO ({AO:.2f})。")

    # 4. 商服:办公 (AC:AO)
    con4 = AO - 3 * AC
    if con4 < 0:
        constraints_met = False
        violations.append(f"商办比(2): AO ({AO:.2f}) < 3*AC ({3 * AC:.2f})。")

    # 5. 公服要求
    con5 = PF_sup - 0.11 * F_res_total
    if con5 < 0:
        constraints_met = False
        violations.append(f"公服联动: PF_sup ({PF_sup:.2f}) < 0.11 * F_res_total ({0.11 * F_res_total:.2f})。")

    return constraints_met, violations


# ----------------------------------------------------------------------
# 步骤 4: 定义 Agent 可用的主工具
# ----------------------------------------------------------------------

def evaluate_proposal(AR: float, AC: float, AO: float, Delta: float, Tau: float):
    """
    Agent 的核心工具。
    输入决策向量 x = [AR, AC, AO, Delta, Tau]，
    检查约束，然后计算并返回三方效用和关键指标。
    """
    try:
        x = np.array([AR, AC, AO, Delta, Tau])

        # 1. 计算中间变量
        vars_dict = get_intermediate_vars(x)

        # 2. 检查约束
        constraints_met, violations = check_constraints(x, vars_dict)

        if not constraints_met:
            return {
                "status": "REJECTED",
                "message": "方案不可行，违反约束。",
                "violations": violations
            }

        # 3. 检查利润是否达标 (U_D 的硬门槛)
        if (vars_dict['Pi'] - pi_min) < 0:
            return {
                "status": "REJECTED",
                "message": "方案不可行，利润未达最低要求。",
                "violations": [f"利润 Pi ({vars_dict['Pi']:.2f}) < 最低利润 pi_min ({pi_min:.2f})。"]
            }

        # 4. 计算效用
        U_G = utility_G(x, vars_dict)
        U_D = utility_D(x, vars_dict)
        U_V = utility_V(x, vars_dict)
        w_G, w_D, w_V = 0.35, 0.35, 0.3
        WSWM = w_G * U_G + w_D * U_D + w_V * U_V

        return {
            "status": "ACCEPTED",
            "message": "方案可行，效用计算如下。",
            "decision_vector": {
                "AR": AR, "AC": AC, "AO": AO, "Delta": Delta, "Tau": Tau
            },
            "utilities": {
                "U_G": f"{U_G:.4f}",
                "U_D": f"{U_D:.4f}",
                "U_V": f"{U_V:.4f}",
                "WSWM": f"{WSWM:.4f}"
            },
            "key_metrics": {
                "Profit (Pi)": f"{vars_dict['Pi']:.2f} 亿元",
                "Industry Ratio (s)": f"{vars_dict['s'] * 100:.2f}% (Min: {s_min * 100}%)",
                "Total Revenue (Rev)": f"{vars_dict['Rev']:.2f} 亿元",
                "Total Cost (Cost)": f"{vars_dict['Cost']:.2f} 亿元",
                "Gov Finance (Phi)": f"{vars_dict['Phi']:.2f} 亿元",
                "Public Service Gap (PF_gap)": f"{vars_dict['PF_gap']:.4f}"
            }
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "message": f"计算中出现意外错误: {str(e)}"
        }
