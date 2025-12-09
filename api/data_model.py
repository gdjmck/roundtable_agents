from pydantic import BaseModel, Field, validator
from typing import Any
import json
from enum import Enum
from typing import Optional, Tuple, Any


def ten_thousand_unit(v: Any) -> Any:
    if isinstance(v, (int, float)):
        if v > 10000:
            return v / 10000
    return v


def standard_unit(v: Any) -> Any:
    if isinstance(v, (int, float)):
        if v < 1:
            return v * 10000
    return v


class UnitPrice(BaseModel):
    demolition: float = Field(0, description="拆迁单价")
    compensation: float = Field(0, description="补偿单价，可选")
    rebuild: float = Field(0, description="复建单价")

    @validator('demolition', 'compensation', 'rebuild', pre=True)
    def standard_unit(cls, v: Any) -> float:
        return standard_unit(v)


class CurrentSituationItem(BaseModel):
    area: float = Field(0, description="现状的建筑面积，单位：万㎡")
    unit_price: UnitPrice
    base_area: float = Field(0, description="现状的基底面积，单位：万㎡")  # 只有住宅有

    @validator('area', 'base_area', pre=True)
    def ten_thousand_unit(cls, v: Any) -> float:
        return ten_thousand_unit(v)


"""
CurrentSituation(现状):
    collective_resi(村集体住宅): CurrentSituationItem
    collective_prop(集体物业): CurrentSituationItem
    state_owned(国有土地): CurrentSituationItem
    construction(建新): float
    
Financing(融资):
    
"""


class CurrentSituation(BaseModel):
    """现状：拆迁&补偿"""
    collective_resi: CurrentSituationItem = Field(description="村集体住宅")
    collective_prop: CurrentSituationItem = Field(description="集体物业")
    state_owned: CurrentSituationItem = Field(description="国有土地")
    public: CurrentSituationItem = Field(description="公配")  # todo: 前端界面需添加对应项
    construction: float = Field(description="建新范围用地面积，单位：万㎡")
    # 非必须入参（预留位置）
    rebuild_resi_ratio: float = Field(3.55, description="住宅复建系数：与【现状住宅基底面积】相乘")
    rebuild_collective_prop_ratio: float = Field(0.8, description="集体物业复建系数：与【现状集体物业建筑面积】相乘")
    rebuild_state_owned_prop_ratio: float = Field(1.03, description="国有物业复建系数：与【现状国有物业建筑面积】相乘")
    rebuild_public_ratio: float = Field(0.11, description="公配复建系数：与【住宅复建】相乘")
    industrial_ratio: float = Field(0.25, description="产业占比系数")

    @validator('construction', pre=True)
    def ten_thousand_unit(cls, v: Any) -> float:
        return ten_thousand_unit(v)


class Financing(BaseModel):
    """融资: 成本"""
    net_FAR: float = Field(2.98, description="规划毛容积率")
    transfer_fee_resi: float = Field(description="住宅用地出让金单价")
    transfer_fee_business: float = Field(description="商服用地出让金单价")
    transfer_fee_office: float = Field(description="办公用地出让金单价")
    other_fee: float = Field(description="其他费用")
    upfront_fee: float = Field(description="前期费用")
    # 非必须入参（预留位置）
    business_ratio: float = Field(0.2, description="融资商服占比系数：商服/(商服+办公)")

    @validator('other_fee', 'upfront_fee',
               pre=True)
    def ten_thousand_unit(cls, v: Any) -> float:
        return ten_thousand_unit(v)

    @validator('transfer_fee_resi', 'transfer_fee_business', 'transfer_fee_office', pre=True)
    def standard_unit(cls, v: Any) -> float:
        return standard_unit(v)


class Income(BaseModel):
    """收入"""
    unit_price_resi: float = Field(description="融资住宅单价")
    unit_price_business: float = Field(description="融资商服单价")
    unit_price_office: float = Field(description="融资办公单价")
    earning_rate: float = Field(0.11, description="收益率")  # 暂时没有作用

    @validator('unit_price_resi', 'unit_price_business', 'unit_price_office', pre=True)
    def standard_unit(cls, v: Any) -> float:
        return standard_unit(v)


class Parameters(BaseModel):
    """入参数据结构"""
    current_situation: CurrentSituation
    financing: Financing
    income: Income


####################################### 博弈入参 #########################
class Area(BaseModel):
    project_plan_total_area: float = Field(description="项目规划总建面")
    financing_resi: float = Field(0, description="融资住宅面积")
    financing_business: float = Field(0, description="融资商服面积")
    financing_office: float = Field(0, description="融资办公面积")
    rebuild_public: float = Field(description="复建公配面积")
    rebuild_collective_prop: float = Field(description="复建集体物业面积")
    rebuild_state_owned_prop: float = Field(description="复建国有物业面积")
    rebuild_resi: float = Field(description="复建住宅面积")


class Cost(BaseModel):
    cost_upfront: float = Field(description="前期费用")
    cost_demolition: float = Field(description="拆迁费用")
    cost_compensation: float = Field(description="补偿费用")
    cost_rebuild: float = Field(description="复建费用")


class BaseParameters(BaseModel):
    unit_price: Income  # 融资单价
    area: Area  # 各类固定面积
    cost: Cost


class CompetitionVector(BaseModel):
    financing_resi: float = Field(description="融资住宅面积")
    financing_business: float = Field(description="融资商服面积")
    financing_office: float = Field(description="融资办公面积")
    # financing_public: float = Field(description="融资公服面积")  # 通过联动自动补全
    FAR_delta: float = Field(description="容积率增幅")
    Tau: float = Field(description="土地出让优惠率")


class CompetitionParameters(BaseModel):
    """
    政府效用: 1、产业达标度 2、公服达标度 3、密度偏好 4、财政净额
    产业达标度 = min(1, s / s_min)
        s = 产业 / 提容后总建筑面积
        s_min = 0.25
        产业 = 复建集体物业【固定】 + 复建国有物业【固定】 + 融资商服【博弈变量】 + 融资办公【博弈变量】
        提容后总建筑面积 = (1 + 容积率增幅【博弈变量】) * 项目规划总建面【固定】
    公服达标度 = min(1, PF_sup / PF_req)
        PF_req = r_pf【固定0.11】 * (复建住宅【固定】 + 融资住宅【博弈变量】)
        PF_sup = 复建公服【固定】 + 融资公服【博弈变量】
    密度偏好 = 1 - 容积率增幅【博弈变量】 / 增幅上限【固定0.1】
    财政净额 = clip((财政收入 - 收入下限) / (收入目标 - 收入下限), 0, 1)
    财政收入 = (1-tau) * 出让金 + t_c【固定0.15】 * 融资商服单价【固定】 * 融资商服【博弈变量】 +
                                t_o【固定0.15】 * 融资办公单价【固定】 * 融资办公【博弈变量】


    开发商效用 = gamma1【固定0.8】 * log(利润 - 最低利润【固定】 + eps【固定】) + gamma2【固定0.15] * 密度偏好 - gamma3【固定0.05】 * max(0, s - s_min)
    利润 = 收入 - 成本 （即成本与收益估算的计算公式）

    村民效用 = 0.6 / 0.67 * 住宅份额 + 0.3 / 0.2 * 物业份额 + 0.1 / 0.43 * 货补占比
        住宅份额 = max(0, 1 - s - 公服【固定】 / F0【固定，项目规划总建面】)
        物业份额 = s = 产业
        货补占比 = 0.43 【置为常数，使得货补占比为1】

    """
    project_plan_total_area: float = Field(description="项目规划总建面")
    r_pf: float = Field(0.11, description="公共服务设施联动比例")
