import numpy as np
from run_optimization import solve_optimization
from api.data_model import Parameters, BaseParameters, CompetitionVector
from api.app import app


def estimate_cost_and_income(parameters: Parameters):
    """
    成本估算(单位：万元) = 前期费用 + 拆迁费用 + 补充费用 + 复建费用 + 土地出让金 + 其他费用
    收益估算(单位：万元) = 融资住宅金额 + 融资商业金额 + 融资办公金额
    :return
    """
    # 前期费用
    cost_upfront = parameters.financing.upfront_fee
    print(f'前期费用: {cost_upfront / 10000:.4f}亿元')

    # 拆迁费用 = 村集体住宅拆迁费用 + 集体物业拆迁费用 + 国有土地拆迁费用
    ## 村集体住宅
    cost_demolition_resi = parameters.current_situation.collective_resi.area * \
                           parameters.current_situation.collective_resi.unit_price.demolition
    ## 集体物业
    cost_demolition_prop = parameters.current_situation.collective_prop.area * \
                           parameters.current_situation.collective_prop.unit_price.demolition
    ## 国有土地
    cost_demolition_state = parameters.current_situation.state_owned.area * \
                            parameters.current_situation.state_owned.unit_price.demolition
    cost_demolition = cost_demolition_resi + cost_demolition_prop + cost_demolition_state
    print(f'拆迁费用: {cost_demolition / 10000:.4f}亿元')

    # 补偿费用 = 村民住宅补偿费用 + 集体物业补偿费用
    ## 村民住宅补偿费用
    cost_compensation_resi = parameters.current_situation.collective_resi.area * \
                             parameters.current_situation.collective_resi.unit_price.compensation
    ## 集体物业补偿费用
    cost_compensation_prop = parameters.current_situation.collective_prop.area * \
                             parameters.current_situation.collective_prop.unit_price.compensation
    cost_compensation = cost_compensation_resi + cost_compensation_prop
    print(f'补偿费用: {cost_compensation / 10000:.4f}亿元')

    # 复建费用 = 住宅复建费用 + 集体物业复建费用 + 国有物业复建费用 + 公配复建费用
    ## 住宅复建费用
    volume_rebuild_resi = parameters.current_situation.collective_resi.base_area * \
                          parameters.current_situation.rebuild_resi_ratio  # 现状基底面积*系数
    cost_rebuild_resi = volume_rebuild_resi * parameters.current_situation.collective_resi.unit_price.rebuild
    ## 集体物业复建费用
    volume_rebuild_collective_prop = parameters.current_situation.collective_prop.area * \
                                     parameters.current_situation.rebuild_collective_prop_ratio
    cost_rebuild_collective_prop = volume_rebuild_collective_prop * \
                                   parameters.current_situation.collective_prop.unit_price.rebuild
    ## 国有物业复建费用
    volume_rebuild_state_owned_prop = parameters.current_situation.state_owned.area * \
                                      parameters.current_situation.rebuild_state_owned_prop_ratio
    cost_rebuild_state_owned_prop = volume_rebuild_state_owned_prop * \
                                    parameters.current_situation.state_owned.unit_price.rebuild
    ## 公配复建费用
    volume_rebuild_public = volume_rebuild_resi * parameters.current_situation.rebuild_public_ratio
    cost_rebuild_public = volume_rebuild_public * parameters.current_situation.public.unit_price.rebuild
    cost_rebuild = cost_rebuild_resi + cost_rebuild_collective_prop + cost_rebuild_state_owned_prop + cost_rebuild_public
    total_area_rebuild = volume_rebuild_resi + volume_rebuild_collective_prop + volume_rebuild_state_owned_prop + volume_rebuild_public
    print(f'复建总建面: {total_area_rebuild:.2f}万㎡，复建公配: {volume_rebuild_public:.2f}万㎡，复建住宅: {volume_rebuild_resi:.2f}万㎡')
    print(f'复建费用: {cost_rebuild / 10000:.4f}亿元')

    """
    融资总量 = 项目规划总建面 − 复建总建面
    项目规划总建面 = 建新范围用地面积×规划毛容积率
    """
    project_plan_total_area = parameters.current_situation.construction * parameters.financing.net_FAR  # 项目规划总建面
    print(f'项目规划总建面: {project_plan_total_area:.2f}万㎡')
    financing_total_volume = project_plan_total_area - total_area_rebuild  # 融资总量
    print(f'融资总面积: {financing_total_volume:.2f}万㎡')
    volume_industrial = project_plan_total_area * parameters.current_situation.industrial_ratio
    print(
        f'产业总建面: {volume_industrial:.2f}万㎡\n复建产业:集体物业：{volume_rebuild_collective_prop:.2f}万㎡\n国有物业：{volume_rebuild_state_owned_prop:.2f}万㎡')
    volume_industrial_gap = volume_industrial - volume_rebuild_collective_prop - volume_rebuild_state_owned_prop
    volume_financing_business = volume_industrial_gap * parameters.financing.business_ratio
    volume_financing_office = volume_industrial_gap * (1 - parameters.financing.business_ratio)
    volume_financing_resi = financing_total_volume - volume_financing_business - volume_financing_office
    print(
        f'融资住宅总建面: {volume_financing_resi:.2f}万㎡\n融资商服总建面: {volume_financing_business:.2f}万㎡\n融资办公总建面: {volume_financing_office:.2f}万㎡')

    # 土地出让金 = 住宅出让金 + 商服出让金 + 办公出让金
    cost_transfer_fee_resi = volume_financing_resi * parameters.financing.transfer_fee_resi
    cost_transfer_fee_business = volume_financing_business * parameters.financing.transfer_fee_business
    cost_transfer_fee_office = volume_financing_office * parameters.financing.transfer_fee_office
    cost_transfer_fee = cost_transfer_fee_resi + cost_transfer_fee_business + cost_transfer_fee_office
    print(f'土地出让金: {cost_transfer_fee / 10000:.4f}亿元')

    # 其他费用
    cost_other = parameters.financing.other_fee
    print(f'其他费用: {cost_other / 10000:.4f}亿元')

    # 总成本 = 前期费用 + 拆迁费用 + 补偿费用 + 复建费用 + 土地出让金 + 其他费用
    cost = cost_upfront + cost_demolition + cost_compensation + cost_rebuild + cost_transfer_fee + cost_other
    print(f'总成本: {cost / 10000:.4f}亿元')

    sales_resi = parameters.income.unit_price_resi * volume_financing_resi
    sales_office = parameters.income.unit_price_office * volume_financing_office
    sales_business = parameters.income.unit_price_business * volume_financing_business
    income = sales_business + sales_office + sales_resi
    print(f'融资收入: {income / 10000:.4f}亿元')

    return {'cost':
                {'total': cost,
                 'details': {
                     'upfront': cost_upfront,
                     'demolition': {'total': cost_demolition,
                                    'resi': cost_demolition_resi,
                                    'collective_prop': cost_demolition_prop,
                                    'state_owned': cost_demolition_state},
                     'compensation': {'total': cost_compensation,
                                      'resi': cost_compensation_resi,
                                      'collective_prop': cost_compensation_prop},
                     'other': cost_other,
                     'rebuild': {'total': cost_rebuild,
                                 'resi': cost_rebuild_resi,
                                 'collective_prop': cost_rebuild_collective_prop,
                                 'state_owned': cost_rebuild_state_owned_prop,
                                 'public': cost_rebuild_public},
                     'transfer_fee': cost_transfer_fee}
                 },
            'income':
                {'total': income,
                 'details': {'business': sales_business, 'office': sales_office, 'resi': sales_resi}
                 },
            'area':
                {'project_plan_total_area': project_plan_total_area,
                 'financing_total': financing_total_volume,
                 'financing_resi': volume_financing_resi,
                 'financing_office': volume_financing_office,
                 'financing_business': volume_financing_business,
                 'rebuild_total': total_area_rebuild,
                 'rebuild_public': volume_rebuild_public,
                 'rebuild_collective_prop': volume_rebuild_collective_prop,
                 'rebuild_state_owned_prop': volume_rebuild_state_owned_prop,
                 'rebuild_resi': volume_rebuild_resi,
                 },
            'unit_price':
                {'unit_price_resi': parameters.income.unit_price_resi,
                 'unit_price_office': parameters.income.unit_price_office,
                 'unit_price_business': parameters.income.unit_price_business}
            }


@app.post("/cost")
def cost(parameters: Parameters):
    """
    成本估算(单位：万元) = 前期费用 + 拆迁费用 + 补充费用 + 复建费用 + 土地出让金 + 其他费用
    收益估算(单位：万元) = 融资住宅金额 + 融资商业金额 + 融资办公金额
    """

    try:
        estimation = estimate_cost_and_income(parameters)
    except Exception:
        return {'status': 500}

    return {'cost':
                {'total': estimation['cost']['total'],
                 'upfront': estimation['cost']['details']['upfront'],
                 'demolition': estimation['cost']['details']['demolition'],
                 'compensation': estimation['cost']['details']['compensation'],
                 'other': estimation['cost']['details']['other'],
                 'rebuild': estimation['cost']['details']['rebuild'],
                 'transfer_fee': estimation['cost']['details']['transfer_fee']},
            'income':
                {'total': estimation['income']['total'],
                 'business': estimation['income']['details']['business'],
                 'office': estimation['income']['details']['office'],
                 'resi': estimation['income']['details']['resi']},
            'area': estimation['area'],
            'info': {'project_plan_total_area': estimation['area']['project_plan_total_area']},
            'status': 200
            }


@app.post("/strategic_competition")
def strategic_competition(parameters: BaseParameters):
    """
    通过SLSQP算法在初始方案附近采样进行曲线拟合，迭代式进行多目标优化
    :param parameters:
    :return: {'msg': str, 'x_star': List[float], 'utility': Dict}
    """
    # init_vector = [95.494, 10.0, 30.0, 0.08, 0.13]
    init_vector = [np.random.uniform(90, 105),
                   np.random.uniform(5, 25),
                   np.random.uniform(5, 40),
                   np.random.uniform(0.00, 0.1),
                   np.random.uniform(0.05, 0.15)]
    result = solve_optimization(init_vector, parameters)
    return result
