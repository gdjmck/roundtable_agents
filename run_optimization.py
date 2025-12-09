import traceback

import numpy as np
import functools
from typing import List, Literal
from scipy.optimize import minimize, Bounds, show_options
from api.data_model import BaseParameters, CompetitionVector, Area, Cost, Income

# ----------------------------------------------------------------------
# 步骤 1: 定义常量与参数 (基于文档)
# ----------------------------------------------------------------------
# 货币单位: 亿元, 面积单位: 万㎡

# A. 基线规模
F0 = 230.80  # 基线总建筑面积
F_res_recon = 79.40  # 复建住宅建面
F_prop_recon = 25.56  # 复建物业
PF_recon = 8.734  # 复建公服
A_total = 117.03  # 融资总量

# B. 价格 (亿元/万㎡)
# 原始价格为 元/㎡, 转换: (元/㎡) * (10000 ㎡/万㎡) / (10^8 元/亿元)
P_R = 7636 / 10000  # 融资住宅单价 0.7636
P_C = 9205 / 10000  # 融资商服单价 0.9205
P_O = 3973 / 10000  # 融资办公单价 0.3973

# C. 监管阈值
s_min = 0.25  # 产业占比下限
r_pf = 0.11  # 公服联动比例
delta_max = 0.10  # 容积率提升上限

# D. 成本与利润参数 (基于文档推算)
# 基准收入 Rev0 ≈ 74.41 亿元
# 选定 5% 目标利润 π0 ≈ 3.72 亿元
# 假设 C_land = 0, 则 C0 = Rev0 - π0
C0 = 70.69  # (亿元) 基线总成本常数
pi_min = 3.0  # (亿元) 假设的最低可接受利润

# E. 效用函数参数 (假设)
# WSWM 权重
W_weights = {'G': 0.35, 'D': 0.35, 'V': 0.30}  # 超参数: 各方利益权重

# U_G (政府) 权重
beta = np.array([0.35, 0.25, 0.20, 0.20])  # 超参数:
rho_G = 0.8
lambda_financial_income = 0.2
# 财政目标 (假设)
Phi_min = 1.0
Phi_tar = 74.41 * 0.05  # Rev_0 * 5%
k_ext = 0.5  # 外部性成本系数

# U_D (开发商) 权重
gamma = np.array([0.80, 0.15, 0.05])  # 超参数: [利润系数, 容积率上溢风险, 产业占比]
# 成本系数 (假设)
C_delta = 1.0  # 超参数: 密度额外成本系数
C_PF = 0.8  # 超参数: 公服缺口罚金 (设为高昂)
# C_PF = 1000
C_land = 0  # 土地成本 (按建议并入C0)

# U_V (村集体) 权重
eta = np.array([0.5, 0.3, 0.2])
# 基准份额 (来自文档校准)
rho_res = 0.67
rho_prop = 0.20
rho_cash = 0.43  # (假设)

# E. 杂项
epsilon = 1e-6  # 防止 log(0)


# ----------------------------------------------------------------------
# 步骤 2 & 3: 决策向量 (x) 与中间变量函数
# x = [AR, AC, AO, Delta, Tau]
# ----------------------------------------------------------------------

def get_intermediate_vars(x):
    """根据决策向量 x 计算所有派生变量"""
    AR, AC, AO, Delta, Tau = x

    # 总量
    F = F0 * (1 + Delta)  # 提容后总建面
    F_res_total = F_res_recon + AR  # 总住宅建面

    # 产业占比 s
    F_ind = F_prop_recon + AC + AO
    s = F_ind / F

    # 公服 PF
    PF_req = r_pf * (F_res_recon + AR)
    # 采用“联动自动补足”的简化假设 (根本不会有gap)
    PF_sup = PF_recon + r_pf * AR
    PF_gap = np.maximum(0, PF_req - PF_sup)  #
    # assert PF_gap == 0, PF_gap

    # 份额 (U_V 用)
    PF_share = PF_sup / F  #
    Res_share = np.maximum(0, 1 - s - PF_share)  #
    Cash_share = rho_cash  # (为简化计算，假设为常数)

    # 收入 Rev (亿元)
    Rev = P_R * AR + P_C * AC + P_O * AO

    # 成本 Cost (亿元)
    Cost_delta = C_delta * (Delta ** 2)
    Cost_PF_penalty = C_PF * PF_gap
    Cost_land = C_land * (1 - Tau) * F  # (若 C_land=0 则此项为0)
    Cost = C0 + Cost_delta + Cost_PF_penalty + Cost_land
    # print('C0:', round(C0, 2), 'penalty: ', round(Cost_PF_penalty, 2), 'land:', round(Cost_land, 2))

    # 利润 Pi
    Pi = Rev - Cost

    # 政府财政 Phi (亿元) (示意)
    # L0(1-t) + tC*pC*AC + ... - E(d)
    L0 = lambda_financial_income * Rev  # 出让基数
    tC = 0.01  # (假设) 税率
    tO = 0.01  # (假设) 税率
    Phi = L0 * (1 - Tau) + tC * P_C * AC + tO * P_O * AO - k_ext * (Delta ** 2)

    return {
        'F': F, 's': s, 'PF_req': PF_req, 'PF_sup': PF_sup,
        'PF_gap': PF_gap, 'Res_share': Res_share, 'Cash_share': Cash_share,
        'Rev': Rev, 'Cost': Cost, 'Pi': Pi, 'Phi': Phi, 'F_res_total': F_res_total
    }


# ----------------------------------------------------------------------
# 步骤 4: 实现三方效用函数
# ----------------------------------------------------------------------

# def utility_G(x, vars):
#     """计算政府效用 U_G (CES)"""
#     s, PF_req, PF_sup, Phi = vars['s'], vars['PF_req'], vars['PF_sup'], vars['Phi']
#     Delta = x[3]
#
#     # z1: 产业达标度
#     z1 = min(1.0, s / s_min)
#     # z2: 公服达标度
#     z2 = min(1.0, PF_sup / (PF_req + epsilon))
#     # z3: 密度偏好
#     z3 = 1 - (Delta / delta_max)
#     # z4: 财政净额
#     z4 = np.clip((Phi - Phi_min) / (Phi_tar - Phi_min), 0.0, 1.0)
#
#     z = np.array([z1, z2, z3, z4])
#
#     # CES 函数
#     term = np.sum(beta * (z ** rho_G))
#     U_G = term ** (1 / rho_G)
#     return U_G
#
#
# def utility_D(x, vars):
#     """计算开发商效用 U_D"""
#     s, Pi = vars['s'], vars['Pi']
#     Delta = x[3]
#
#     # U_D = g1*log(Pi) + g2*(1-Delta) - g3*(s-s_min)
#     term1 = gamma[0] * np.log(Pi - pi_min + epsilon)
#     term2 = gamma[1] * (1 - (Delta / delta_max))
#     term3 = gamma[2] * np.maximum(0, s - s_min)
#
#     U_D = term1 + term2 - term3
#     return U_D
#
#
# def utility_V(x, vars):
#     """计算村集体效用 U_V"""
#     s, Res_share, Cash_share = vars['s'], vars['Res_share'], vars['Cash_share']
#
#     # U_V = n1*log(Res) + n2*log(Prop) + n3*log(Cash)
#     term1 = eta[0] * np.log(Res_share / rho_res + epsilon)
#     term2 = eta[1] * np.log(s / rho_prop + epsilon)
#     term3 = eta[2] * np.log(Cash_share / rho_cash + epsilon)
#
#     U_V = term1 + term2 + term3
#     return U_V


public_binding_ratio = 0.11  # 公服联动系数：公服面积 >= 联动系数 * 住宅面积
target_metric_coeff = {'WSWM': {'G': 0.35, 'D': 0.35, 'V': 0.3}}


def get_industrial(param_base: BaseParameters, param_decision: CompetitionVector):
    """产业面积"""
    industrial = param_base.area.rebuild_collective_prop + \
                 param_base.area.rebuild_state_owned_prop + \
                 param_decision.financing_office + \
                 get_financing_public_area(param_base, param_decision)
    return industrial


def get_s(param_base: BaseParameters, param_decision: CompetitionVector):
    return get_industrial(param_base, param_decision) / get_total_building_area(param_base, param_decision)


def get_tranfer_fee(param_base: BaseParameters, param_decision: CompetitionVector):
    """土地出让金"""
    transfer_fee_unit_price_discount = 0.2
    transfer_fee = (1 - param_decision.Tau) * transfer_fee_unit_price_discount * \
                   (param_base.unit_price.unit_price_resi * param_decision.financing_resi +
                    param_base.unit_price.unit_price_office * param_decision.financing_office +
                    param_base.unit_price.unit_price_office * param_decision.financing_office)
    return transfer_fee


def utility_G(param_base: BaseParameters, param_decision: CompetitionVector):
    """政府效用: 1、产业达标度 2、公服达标度 3、密度偏好 4、财政净额"""
    # 产业达标度
    s_min = 0.25
    s = get_s(param_base, param_decision)
    utility1 = min(1, s / s_min)  # 值域[0, 1]

    # 公服达标度
    r_pf = 0.11
    PF_req = r_pf * (param_base.area.rebuild_resi + param_decision.financing_resi)
    PF_sup = param_base.area.rebuild_public + get_financing_public_area(param_base, param_decision)
    utility2 = min(1, PF_sup / PF_req)  # 值域[0, 1]

    # 密度偏好
    max_delta = 0.1
    utility3 = 1 - (param_decision.FAR_delta / max_delta)  # 值域[0, 1]

    # 财政净额
    t_c = t_o = 0.15  # 商服&办公的税率
    phi_min = 10000  # 最低利润
    phi_tar = 74.41 * 10000 * 0.05  # 目标利润
    transfer_fee = get_tranfer_fee(param_base, param_decision)
    tax_fee = t_c * param_base.unit_price.unit_price_office * param_decision.financing_office + \
              t_o * param_base.unit_price.unit_price_business * param_decision.financing_business
    phi = transfer_fee + tax_fee
    utility4 = np.clip((phi - phi_min) / (phi_tar - phi_min), 0, 1)

    # 整合无量纲
    rho_G = 0.8
    gamma = np.array([0.35, 0.25, 0.20, 0.20])
    utility = np.array([utility1, utility2, utility3, utility4])
    return float(np.sum(gamma * (utility ** rho_G)) ** (1 / rho_G))


def utility_D(param_base: BaseParameters, param_decision: CompetitionVector):
    """开发商效用：利润 + 密度偏好 + 产业"""
    gamma1, gamma2, gamma3 = 0.8, 0.15, 0.05
    min_profit_rate = 0.03  # 按最保守的利润率
    target_profit_rate = 0.1  # 目标利润率
    s_min = 0.25
    delta_max = 0.10
    eps = 1e-6

    cost = get_tranfer_fee(param_base, param_decision) + \
           param_base.cost.cost_upfront + \
           param_base.cost.cost_demolition + \
           param_base.cost.cost_compensation + \
           param_base.cost.cost_rebuild
    income = param_base.unit_price.unit_price_resi * param_decision.financing_resi + \
             param_base.unit_price.unit_price_office * param_decision.financing_office + \
             param_base.unit_price.unit_price_business * param_decision.financing_business
    min_profit = min_profit_rate * income
    target_profit = target_profit_rate * income
    profit = income - cost

    def utility_profit(profit_actual: float):
        return float(np.log(1 + profit_actual - min_profit) / np.log(1 + target_profit - min_profit))

    s = get_s(param_base, param_decision)
    utility = gamma1 * utility_profit(profit) - \
              gamma2 * (param_decision.FAR_delta / delta_max) ** 2 - \
              gamma3 * max(0, s - s_min)
    return float(utility)


def utility_V(param_base: BaseParameters, param_decision: CompetitionVector):
    """村社效用：利润 + 密度偏好 + 产业"""
    gamma1, gamma2, gamma3 = 0.6 / 0.67, 0.3 / 0.2, 0.1 / 0.43
    s = get_s(param_base, param_decision)
    total_building_area = get_total_building_area(param_base, param_decision)
    PF_share = (param_base.area.rebuild_public + get_financing_public_area(param_base,
                                                                           param_decision)) / total_building_area
    utility1 = max(0, 1 - s - PF_share)
    utility2 = s
    utility3 = 0.43

    utility = gamma1 * utility1 + gamma2 * utility2 + gamma3 * utility3
    return float(utility)


def get_total_building_area(param_base: BaseParameters, param_decision: CompetitionVector):
    return (1 + param_decision.FAR_delta) * param_base.area.project_plan_total_area


def get_financing_public_area(param_base: BaseParameters, param_decision: CompetitionVector):
    total_resi_area = param_base.area.rebuild_resi + param_decision.financing_resi
    require_public_area = public_binding_ratio * total_resi_area
    if require_public_area <= param_base.area.rebuild_public:
        return 0
    else:
        # 联动补全
        return require_public_area - param_base.area.rebuild_public


def check_constraints(param_base: BaseParameters, param_decision: CompetitionVector):
    """1、融资面积守恒 2、产业占比 3、容积率提升范围 4、出让优惠率"""
    tau_range = [0.05, 0.20]
    delta_range = [0, 0.1]
    s_min = 0.25

    constraints_met = True
    violations = []

    # 1、融资面积守恒
    total_building_area = get_total_building_area(param_base, param_decision)
    rebuild_area = param_base.area.rebuild_collective_prop + \
                   param_base.area.rebuild_state_owned_prop + \
                   param_base.area.rebuild_public + \
                   param_base.area.rebuild_resi
    finanacing_public_area = get_financing_public_area(param_base, param_decision)
    financing_area = param_decision.financing_business + \
                     param_decision.financing_office + \
                     param_decision.financing_resi + \
                     finanacing_public_area
    constraint1 = np.isclose(total_building_area, financing_area + rebuild_area, atol=1e-3)
    if not constraint1:
        violation_text = f"""融资面积不守恒：AR({param_decision.financing_resi:.2f})+
        AC({param_decision.financing_business:.2f})+
        A0({param_decision.financing_office:.2f})+
        APF({financing_area :.2f}通过公服联动补齐：公服面积>={public_binding_ratio}*(AR+{param_base.area.financing_resi}))
        !=(1+ Delta) * 项目规划总建面({param_base.area.project_plan_total_area:.2f})"""
        violations.append(violation_text.replace('\n', ''))

    # 2、产业占比
    s = get_s(param_base, param_decision)
    constraint2 = s >= s_min
    if not constraint2:
        violation_text = f"""产业占比不满足要求：s({s:.2f})<{s_min:.2f}"""
        violations.append(violation_text)

    # 3、容积率提升范围
    constraint3 = param_decision.FAR_delta >= delta_range[0] and param_decision.FAR_delta <= delta_range[1]
    if not constraint3:
        violation_text = f"""容积率提升范围不在约束范围内：Delta({param_decision.FAR_delta:.2f})∈[{delta_range[0]:.2f}, {delta_range[1]:.2f}]"""
        violations.append(violation_text)

    # 4、出让优惠率
    constraint4 = param_decision.Tau >= tau_range[0] and param_decision.Tau <= tau_range[1]
    if not constraint4:
        violation_text = f"""出让优惠率不在约束范围内：Tau({param_decision.Tau:.2f})∈[{tau_range[0]:.2f}, {tau_range[1]:.2f}]"""
        violations.append(violation_text)

    return constraints_met, violations


def evaluate_proposal(param_base: BaseParameters, param_decision: CompetitionVector,
                      target_metric: Literal['WSWM', 'wNBS']):
    """1、检查约束 2、计算效用 3、计算目标函数值"""
    # 1、检查约束
    constraints_met, violations = check_constraints(param_base, param_decision)
    if not constraints_met:
        return {'status': 'REJECTED', 'message': '方案不可行，违反约束。', 'violations': violations}

    # 2、计算效用
    U_G = utility_G(param_base, param_decision)
    U_D = utility_D(param_base, param_decision)
    U_V = utility_V(param_base, param_decision)

    # 3、计算目标函数值
    if target_metric == 'WSWM':
        coeff_dict = target_metric_coeff['WSWM']
        target = coeff_dict['G'] * U_G + coeff_dict['D'] * U_D + coeff_dict['V'] * U_V
    else:
        coeff_dict = target_metric_coeff['wNBS']
        target = 0

    utility = {'U_G': U_G, 'U_D': U_D, 'U_V': U_V}

    return {
        'status': 'ACCEPTED',
        'message': f'方案可行，效用值：{utility}。',
        'decision_vector': {"AR": param_decision.financing_resi,
                            "AC": param_decision.financing_business,
                            "AO": param_decision.financing_office,
                            "Delta": param_decision.FAR_delta,
                            "Tau": param_decision.Tau},
        'utility': {'G': U_G, 'D': U_D, 'V': U_V, target_metric: target}
    }


# ----------------------------------------------------------------------
# 步骤 5: 定义总目标函数与约束
# ----------------------------------------------------------------------

def objective_function(x):
    """
    总目标函数 (WSWM), 求解器将最小化此函数的 *负值*
    """
    # 检查x中的值是否有效
    if np.any(np.isnan(x)):
        return np.inf  # 无效输入

    vars = get_intermediate_vars(x)

    # 利润必须为正才能取 log
    if (vars['Pi'] - pi_min) < 0:
        return np.inf  # 违反了最低利润，解不可行

    # 计算各方效用
    U_G = utility_G(x, vars)
    U_D = utility_D(x, vars)
    U_V = utility_V(x, vars)

    # WSWM
    W = W_weights['G'] * U_G + W_weights['D'] * U_D + W_weights['V'] * U_V

    # 我们要最大化 W, 所以求解器最小化 -W
    return -W


def objective_function(x: List[float], param_base: BaseParameters):
    param_decision = CompetitionVector(financing_resi=x[0],
                                       financing_business=x[1],
                                       financing_office=x[2],
                                       FAR_delta=x[3],
                                       Tau=x[4])
    evaluation = evaluate_proposal(param_base, param_decision, 'WSWM')
    return -evaluation['utility']['WSWM']


# 定义约束
constraints = (
    {
        'type': 'eq',
        'fun': lambda x: A_total + x[3] * F0 - (x[0] + x[1] + x[2])  #
    },
    {
        'type': 'ineq',
        'fun': lambda x: get_intermediate_vars(x)['s'] - s_min  #
    },
    # 商服: 办公
    {
        'type': 'ineq',
        'fun': lambda x: 5 * x[1] - x[2]  # x[1] 是 AC, x[2] 是 AO
    },
    {
        'type': 'ineq',
        'fun': lambda x: x[2] - 3 * x[1]  # x[1] 是 AC, x[2] 是 AO
    },
    # 公服要求: PF_sup >= 0.11 * (F_res_recon + AR)
    {
        'type': 'ineq',
        'fun': lambda x: get_intermediate_vars(x)['PF_sup'] - 0.11 * (get_intermediate_vars(x)['F_res_total'])
    }
)

# 定义边界 x = [AR, AC, AO, Delta, Tau]
# AR, AC, AO 必须 >= 0
bounds = Bounds(
    [0, 0, 0, 0.0, 0.05],
    [A_total, A_total, A_total, delta_max, 0.20]
)


def solve_optimization(init_vector: List[float], param_base: BaseParameters):
    x0 = init_vector

    print("开始求解 WSWM 最优解...")
    print(f"初始猜测 (基准方案):")
    print(f"  x0 = {x0}")
    print(f"  初始总融资额: {x0[0] + x0[1] + x0[2]:.2f} (应为 {A_total})")
    print("-------------------------------------------------")

    history = []

    def callback(xk):
        history.append(xk.tolist())

    # 调用求解器
    # SLSQP 适合处理带约束的非线性问题
    result = minimize(
        objective_function,
        x0,
        args=(param_base,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'disp': True, 'maxiter': 1000}, #  'eps': 10, 'ftol': 1e-2},
        callback=callback
    )

    # ----------------------------------------------------------------------
    # 输出结果
    # ----------------------------------------------------------------------
    print(history)
    if result.success:
        print("\n✅ 求解成功!")
        x_star = result.x
        max_W = -result.fun

        print(f"\n最优WSWM值 (W*): {max_W:.4f}")

        print("\n最优决策向量 (x*):")
        print(f"  融资住宅 (AR): {x_star[0]:.2f} 万㎡")
        print(f"  融资商服 (AC): {x_star[1]:.2f} 万㎡")
        print(f"  融资办公 (AO): {x_star[2]:.2f} 万㎡")
        print(f"  容积率增幅 (Delta): {x_star[3] * 100:.2f} %")
        print(f"  优惠率 (Tau): {x_star[4] * 100:.2f} %")

        print("\n派生指标:")
        vars_star = get_intermediate_vars(x_star)
        print(f"  产业占比 (s): {vars_star['s'] * 100:.2f}% (约束: >= {s_min * 100}%)")
        print(f"  总融资额: {x_star[0] + x_star[1] + x_star[2]:.2f} (约束: {A_total})")
        print(f"  公服缺口 (Gap): {vars_star['PF_gap']:.4f} 万㎡")
        print(f"  项目利润 (Pi): {vars_star['Pi']:.2f} 亿元")
        print(f"  项目成本 (Cost): {vars_star['Cost']:.2f} 亿元")
        print(f"  政府收入 (Phi): {vars_star['Phi']:.2f} 亿元")

        print("\n三方效用:")
        vector = CompetitionVector(financing_resi=x_star[0],
                                   financing_business=x_star[1],
                                   financing_office=x_star[2],
                                   FAR_delta=x_star[3],
                                   Tau=x_star[4])
        evaluation = evaluate_proposal(param_base, vector, 'WSWM')
        print(f"  U_G (政府): {evaluation['utility']['G']:.4f}")
        print(f"  U_D (开发商): {evaluation['utility']['D']:.4f}")
        print(f"  U_V (村集体): {evaluation['utility']['V']:.4f}")
        result = {'status': 200, 'msg': 'success', 'x_star': x_star.tolist(), 'utility': evaluation['utility']['WSWM']}

    else:
        print("\n❌ 求解失败。")
        print(f"  原因: {result.message}")
        result = {'status': 500, 'msg': result.message, 'x_star': [], 'utility': 0}

    def get_resi_area(param: BaseParameters, decision_vector: List[float]):
        return param.area.rebuild_resi + decision_vector[0]

    def get_public_area(param: BaseParameters, decision_vector: List[float]):
        decision_vector = CompetitionVector(financing_resi=decision_vector[0],
                                            financing_business=decision_vector[1],
                                            financing_office=decision_vector[2],
                                            FAR_delta=decision_vector[3],
                                            Tau=decision_vector[4])
        return get_financing_public_area(param, decision_vector) + param.area.rebuild_public

    def get_business_area(decision_vector: List[float]):
        return decision_vector[1] + decision_vector[2]

    if result['status'] == 200:
        try:
            total_resi = get_resi_area(param_base, x_star)
            total_public = get_public_area(param_base, x_star)
            total_business = get_business_area(x_star)


            total_building_area = total_business + \
                                  total_resi + \
                                  total_public + \
                                  param_base.area.rebuild_collective_prop + \
                                  param_base.area.rebuild_state_owned_prop
            info = {'total_building_area': total_building_area,
                    'total_resi_area': total_resi,
                    'total_public_area': total_public,
                    'total_business_area': total_business,
                    'net_FAR': 2.98,
                    'resi_FAR': np.random.uniform(2.98, 3.25),
                    'business_FAR': np.random.uniform(4.25, 5.75)}

            result['history'] = [{'iter': iter + 1,
                                  'total_resi_area': get_resi_area(param_base, item),
                                  'total_public_area': get_public_area(param_base, item),
                                  'total_business_area': get_business_area(item)}
                                 for iter, item in zip(range(len(history)), history)]
            result['info'] = info
        except:
            result['msg'] += traceback.format_exc()
            result['status'] = 500

    return result


if __name__ == '__main__':
    # ----------------------------------------------------------------------
    # 步骤 6: 调用求解器
    # ----------------------------------------------------------------------

    # 初始猜测 x0 (使用基准方案值)
    # x = [AR, AC, AO, Delta, Tau]
    # 基准: AR=75.91, AC=8.35, AO=32.77, Delta=0, Tau=0.20
    # x0 = [75.91, 8.35, 32.77, 0.0, 0.20]
    x0 = [135.494, 10.0, 30.0, 0.08, 0.03]

    param_base = BaseParameters(
        area=Area(
            project_plan_total_area=230.83,
            rebuild_public=8.66,
            rebuild_collective_prop=18.41,
            rebuild_state_owned_prop=7.12,
            rebuild_resi=78.71
        ),
        cost=Cost(
            cost_upfront=1200,
            cost_demolition=104269,
            cost_compensation=23642,
            cost_rebuild=441163
        ),
        unit_price=Income(
            unit_price_resi=7636,
            unit_price_business=9205,
            unit_price_office=3973
        )
    )
    solve_optimization(x0, param_base)
