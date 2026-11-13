"""加权分布不平等分析器（洛伦兹曲线、Gini、分组成对分解、分位资源份额）。

输入契约：每条合成统计单元记录含非负人均资源 x、严格为正的人口权 w，
可选 key 与 group。权重表示“代表人口”，不能把大小地区各当一票：

    W = sum(w)
    mu = sum(w * x) / W
    Gini = sum_{i,j} w_i * w_j * abs(x_i - x_j) / (2 * W^2 * mu)

设计说明
--------
1. 小表用显式 O(n^2) 双重求和核对定义式，并保存成对贡献证据；
   大表用排序后的线性累积，避免显式平方展开：
       U = sum_j w_j * (x_j * W_<j - S_<j)，  D = 2 * U
   其中 W_<j、S_<j 是严格先行记录的权重和与资源和；同值项因
   x_j*W_<j - S_<j = 0 而自动无贡献，无需特殊处理并列值。
2. 分组分解在同一次排序遍历中为每个组维护 W_g、S_g 两个累加器
   （额外空间 O(组数)），把每个有序对项按 g_i == g_j 与否划为
   同组 / 跨组两部分，全部按全体共同分母 2*W^2*mu 归一：
       Gini = Gini_同组 + Gini_跨组（恒等，浮点上以同一减法构造）
   跨组项保留分布重叠下的真实绝对差，绝不用组均值差替换。
3. 另给“仅保留组均值”的反事实 Gini：组内全部单元取组均值后重算，
   它与成对跨组项是不同指标（组间分布交错时二者不相等）。
4. 分位人口组在边界处按权重拆分记录，同值单元的资源按拆分权重
   比例分配，不允许整条记录跨过人口阈值。
5. mu = 0（资源全零）时 Gini 按配置输出未定义（None）并单列，
   不谎报 0；不接受负值，也不做未经告知的平移。
6. 仅当配置声明 weight_kind=frequency 且开启重抽样时，才按频数做
   多项抽样 bootstrap；representation 权重不伪造复杂调查置信区间。

本模块只描述合成数据的分布不平等形态，不估计资源配置的因果效果。
"""

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from src.inequality_reader import load

# 小表阈值：不超过该记录数时物化全部成对项作为证据与定义式核对
PAIRWISE_LIMIT = 20
# 权重比较的相对容差，用于分位边界判定
_TOL_FACTOR = 1e-12


class InequalityInputError(ValueError):
    """输入违反契约（负值、非正权重、缺失字段等）。"""


# ---------------------------------------------------------------- 输入校验

def _parse_records(records):
    """把读取器返回的 dict 记录解析为 (keys, x, w, groups)，并严格校验。"""
    if not isinstance(records, list) or not records:
        raise InequalityInputError("输入记录为空，无法构造加权经验分布")
    keys, xs, ws, groups = [], [], [], []
    for idx, row in enumerate(records):
        if not isinstance(row, dict) or "x" not in row or "weight" not in row:
            raise InequalityInputError(f"第 {idx} 行缺少 x 或 weight 字段")
        key = row.get("key", f"row_{idx}")
        raw_x, raw_w = row["x"], row["weight"]
        if raw_x is None or raw_w is None or str(raw_x).strip() == "" or str(raw_w).strip() == "":
            raise InequalityInputError(f"记录 {key} 存在空白数值，空白不等于 0，拒绝隐式填补")
        try:
            xi = float(raw_x)
            wi = float(raw_w)
        except (TypeError, ValueError) as exc:
            raise InequalityInputError(f"记录 {key} 的数值无法解析: {exc}") from exc
        if not math.isfinite(xi) or not math.isfinite(wi):
            raise InequalityInputError(f"记录 {key} 含非有限数值")
        if xi < 0:
            raise InequalityInputError(
                f"记录 {key} 的人均资源为负 ({xi})；平移负值会改变不平等指标，"
                "本分析器不做未告知的平移，请先核对输入"
            )
        if wi <= 0:
            raise InequalityInputError(f"记录 {key} 的人口权必须严格为正，实际为 {wi}")
        keys.append(str(key))
        xs.append(xi)
        ws.append(wi)
        groups.append(str(row.get("group", "all")))
    return keys, xs, ws, groups


# ---------------------------------------------------------------- 矩与洛伦兹

def weighted_moments(x, w):
    """返回 W = sum(w)、mu = sum(w*x)/W。"""
    total_w = math.fsum(w)
    total_r = math.fsum(wi * xi for wi, xi in zip(w, x))
    return total_w, total_r / total_w


def lorenz_curve(x, w):
    """按 x 排序并合并相同值，构造洛伦兹折线点。

    返回点列表：首点 (0, 0)，其后每个合并值级别一个点，含累计人口份额、
    累计资源份额（资源总量为 0 时份额为 None）、该级别 x 值与合并权重。
    """
    total_w = math.fsum(w)
    total_r = math.fsum(wi * xi for wi, xi in zip(w, x))
    levels = []
    for xi, wi in sorted(zip(x, w)):
        if levels and levels[-1][0] == xi:
            levels[-1][1] += wi
        else:
            levels.append([xi, wi])
    points = [{
        "population_share": 0.0,
        "resource_share": 0.0 if total_r > 0 else None,
        "cumulative_weight": 0.0,
        "cumulative_resource": 0.0,
        "value_x": None,
        "merged_weight": 0.0,
    }]
    cum_w, cum_r = 0.0, 0.0
    for xi, wi in levels:
        cum_w += wi
        cum_r += xi * wi
        points.append({
            "population_share": cum_w / total_w,
            "resource_share": (cum_r / total_r) if total_r > 0 else None,
            "cumulative_weight": cum_w,
            "cumulative_resource": cum_r,
            "value_x": xi,
            "merged_weight": wi,
        })
    return points


def gini_from_lorenz(points):
    """洛伦兹折线梯形积分：Gini = 1 - 2 * integral_0^1 L(p) dp。"""
    area = 0.0
    for a, b in zip(points, points[1:]):
        dp = b["population_share"] - a["population_share"]
        area += dp * (a["resource_share"] + b["resource_share"]) / 2.0
    return 1.0 - 2.0 * area


# ---------------------------------------------------------------- Gini 两种算法

def gini_sorted(x, w):
    """O(n log n) 排序线性累积 Gini。

    返回 (gini, D, W, mu)；mu=0 时 gini 为 None。
    D = sum_{i,j} w_i w_j |x_i - x_j| = 2 * U（U 为无序对累积和）。
    """
    total_w, mu = weighted_moments(x, w)
    u_total = 0.0
    w_prior = 0.0
    s_prior = 0.0
    for xi, wi in sorted(zip(x, w)):
        u_total += wi * (xi * w_prior - s_prior)
        w_prior += wi
        s_prior += wi * xi
    d_total = 2.0 * u_total
    if mu == 0.0:
        return None, d_total, total_w, mu
    return d_total / (2.0 * total_w * total_w * mu), d_total, total_w, mu


def gini_double_sum(x, w):
    """定义式 O(n^2) 有序双重求和，仅用于小表核对与成对证据。

    返回 (gini, D, W, mu)；mu=0 时 gini 为 None。
    """
    total_w, mu = weighted_moments(x, w)
    terms = (
        wi * wj * abs(xi - xj)
        for xi, wi in zip(x, w)
        for xj, wj in zip(x, w)
    )
    d_total = math.fsum(terms)
    if mu == 0.0:
        return None, d_total, total_w, mu
    return d_total / (2.0 * total_w * total_w * mu), d_total, total_w, mu


def pairwise_contributions(keys, x, w, groups):
    """物化全部无序对 (i<j) 的成对项，含同组/跨组标记。

    每条给出 w_i*w_j*|dx|（无序项）与 2 倍后的有序贡献；
    同组项与跨组项分别求和，可直接加总为 D。
    """
    n = len(x)
    pairs = []
    sum_unordered_within = 0.0
    sum_unordered_between = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            unordered = w[i] * w[j] * abs(x[i] - x[j])
            same_group = groups[i] == groups[j]
            pairs.append({
                "i": i,
                "j": j,
                "key_i": keys[i],
                "key_j": keys[j],
                "group_i": groups[i],
                "group_j": groups[j],
                "same_group": same_group,
                "abs_diff": abs(x[i] - x[j]),
                "unordered_term_wiwj_absdx": unordered,
                "ordered_pair_contribution_2x": 2.0 * unordered,
            })
            if same_group:
                sum_unordered_within += unordered
            else:
                sum_unordered_between += unordered
    return {
        "n_records": n,
        "pairs": pairs,
        "D_within_from_pairs": 2.0 * sum_unordered_within,
        "D_between_from_pairs": 2.0 * sum_unordered_between,
        "D_total_from_pairs": 2.0 * (sum_unordered_within + sum_unordered_between),
    }


# ---------------------------------------------------------------- 分组分解

def group_decomposition(x, w, groups):
    """一次排序遍历完成总分与同组/跨组成对分解，O(n log n)。

    维护全局先行累加器 W_<、S_< 与每组累加器 W_g、S_g（额外 O(组数)），
    不显式展开任何平方级成对项。跨组 D 由总 D 减同组 D 得到，
    归一化共用全体分母 2*W^2*mu，因此两个分量之和恰为总 Gini。
    """
    total_w, mu = weighted_moments(x, w)
    w_prior = s_prior = 0.0
    w_group_prior = defaultdict(float)
    s_group_prior = defaultdict(float)
    u_total = 0.0
    u_within = 0.0
    u_within_by_group = defaultdict(float)
    for xi, wi, gi in sorted(zip(x, w, groups), key=lambda t: t[0]):
        u_total += wi * (xi * w_prior - s_prior)
        u_in = wi * (xi * w_group_prior[gi] - s_group_prior[gi])
        u_within += u_in
        u_within_by_group[gi] += u_in
        w_prior += wi
        s_prior += wi * xi
        w_group_prior[gi] += wi
        s_group_prior[gi] += wi * xi
    d_total = 2.0 * u_total
    d_within = 2.0 * u_within
    d_between = d_total - d_within  # 同一浮点过程相减，保证加总恒等
    if mu == 0.0:
        return {
            "status": "undefined_zero_mean",
            "D_total": d_total,
            "D_within_group": d_within,
            "D_between_group_pairwise": d_between,
            "D_within_by_group": {g: 2.0 * v for g, v in u_within_by_group.items()},
            "common_denominator_2W2mu": 0.0,
            "gini_within_group": None,
            "gini_between_group_pairwise": None,
            "gini_total": None,
            "components_sum_to_total": None,
        }
    denom = 2.0 * total_w * total_w * mu
    gini_within = d_within / denom
    gini_total = d_total / denom
    return {
        "status": "defined",
        "D_total": d_total,
        "D_within_group": d_within,
        "D_between_group_pairwise": d_between,
        "D_within_by_group": {g: 2.0 * v for g, v in u_within_by_group.items()},
        "common_denominator_2W2mu": denom,
        "gini_within_group": gini_within,
        # 用总 Gini 减同组分量构造，确保浮点上严格恒等
        "gini_between_group_pairwise": gini_total - gini_within,
        "gini_total": gini_total,
        "components_sum_to_total": True,
        "note": (
            "跨组项是全体跨组成对绝对差 w_i*w_j*|x_i-x_j| 按共同分母归一，"
            "包含组间分布重叠；不得用组均值差替换。"
        ),
    }


def counterfactual_group_means(x, w, groups):
    """仅保留组均值的反事实：组内每个单元的 x 替换为该组加权均值。

    权重与组归属不变，在反事实取值上重算 Gini。该指标与分组分解中的
    成对跨组项不同：当两组分布交错（有重叠）时，组均值抹平了组内离散，
    反事实 Gini 一般不等于成对跨组 Gini。
    """
    sum_w = defaultdict(float)
    sum_wx = defaultdict(float)
    for xi, wi, gi in zip(x, w, groups):
        sum_w[gi] += wi
        sum_wx[gi] += wi * xi
    group_means = {g: sum_wx[g] / sum_w[g] for g in sum_w}
    x_counter = [group_means[gi] for gi in groups]
    gini_cf, d_cf, _, mu_cf = gini_sorted(x_counter, w)
    return {
        "group_means": group_means,
        "counterfactual_x": x_counter,
        "gini_group_means_only": gini_cf,
        "D_counterfactual": d_cf,
        "mu_unchanged": mu_cf,
        "note": (
            "反事实 Gini 把同组所有单元压到组均值后计算，衡量组间均值差异；"
            "它不是成对跨组项：分布重叠时两者不相等，勿混用。"
        ),
    }


# ---------------------------------------------------------------- 分位切分

def quantile_population_shares(keys, x, w, quantiles):
    """按人口分位阈值切分，边界处允许把一条记录按权重拆成片段。

    阈值 q 对应累计目标权重 q*W。若某记录跨越阈值，只取其到达阈值所需
    的权重片段进入当前分位箱，剩余片段留给下一箱；同值资源随权重按比例
    拆分，因此不会有整条记录跨过人口阈值。
    """
    order = sorted(range(len(x)), key=lambda i: x[i])
    total_w = math.fsum(w)
    total_r = math.fsum(wi * xi for wi, xi in zip(w, x))
    tol = _TOL_FACTOR * max(1.0, total_w)

    bins = []
    boundary_splits = []
    cumulative = []
    pos = 0
    remaining = w[order[0]]
    consumed = 0.0
    cum_resource = 0.0
    thresholds = list(quantiles) + [1.0]

    for bin_idx, q in enumerate(thresholds):
        end = q * total_w
        need = end - consumed
        fragments = []
        while need > tol:
            rec = order[pos]
            if remaining <= need + tol:
                # 整条记录收入当前箱
                take = remaining
                fragments.append({
                    "key": keys[rec], "x": x[rec], "weight": take,
                    "fragment_of_record_weight": w[rec],
                    "weight_fraction_of_record": take / w[rec],
                    "split": False,
                })
                need -= take
                consumed += take
                cum_resource += take * x[rec]
                pos += 1
                remaining = w[order[pos]] if pos < len(order) else 0.0
            else:
                # 边界落在该记录内部：按权重拆分
                take = need
                fragments.append({
                    "key": keys[rec], "x": x[rec], "weight": take,
                    "fragment_of_record_weight": w[rec],
                    "weight_fraction_of_record": take / w[rec],
                    "split": True,
                })
                boundary_splits.append({
                    "at_quantile": q,
                    "target_cumulative_weight": end,
                    "key": keys[rec],
                    "x": x[rec],
                    "split_off_weight_into_lower_bin": take,
                    "record_weight_remaining_for_higher_bins": remaining - take,
                    "original_record_weight": w[rec],
                })
                consumed += take
                cum_resource += take * x[rec]
                remaining -= take
                need = 0.0
        bin_resource = math.fsum(f["weight"] * f["x"] for f in fragments)
        bin_weight = math.fsum(f["weight"] for f in fragments)
        bins.append({
            "bin_index": bin_idx,
            "lower_quantile": thresholds[bin_idx - 1] if bin_idx else 0.0,
            "upper_quantile": q,
            "population_share": bin_weight / total_w,
            "resource": bin_resource,
            "resource_share": (bin_resource / total_r) if total_r > 0 else None,
            "fragments": fragments,
        })
        cumulative.append({
            "quantile": q,
            "population_share": q,
            "cumulative_resource": cum_resource,
            "resource_share": (cum_resource / total_r) if total_r > 0 else None,
        })

    if pos != len(order) or remaining > tol:
        raise RuntimeError("分位切分未耗尽全部记录，实现存在缺陷")
    return {
        "thresholds": list(quantiles),
        "cumulative_resource_share_at_quantiles": cumulative,
        "bins": bins,
        "boundary_splits": boundary_splits,
        "note": "边界处按权重拆分记录片段，资源随权重比例分配，无整条记录跨越人口阈值。",
    }


# ---------------------------------------------------------------- 频数重抽样

def _as_frequency_counts(w):
    """频数权重须为整数计数；返回总抽样数 N，否则报错而非四舍五入。"""
    rounded = [round(wi) for wi in w]
    for wi, ri in zip(w, rounded):
        if abs(wi - ri) > 1e-9 * max(1.0, abs(wi)):
            raise InequalityInputError(
                f"声明为频数的权重必须接近整数，发现 {wi}；不能把代表性权重当频数抽样"
            )
    return int(math.fsum(rounded))


def frequency_multinomial_bootstrap(x, w, replicates, seed):
    """按频数多项抽样重估 Gini。

    把每条记录视为 w_i 个同值个体，N=sum(w)，每次独立从类别分布
    （类别概率 w_i/N）抽取 N 次（即一次多项分布抽样），以抽中计数为
    新权重重算 Gini，返回最近秩百分位区间。
    """
    n = len(x)
    n_draws = _as_frequency_counts(w)
    rng = random.Random(seed)
    ginis = []
    for _ in range(replicates):
        draws = rng.choices(range(n), weights=w, k=n_draws)
        counts = Counter(draws)
        w_boot = [float(counts.get(i, 0)) for i in range(n)]
        g_boot, _, _, _ = gini_sorted(x, w_boot)
        if g_boot is not None:
            ginis.append(g_boot)
    ordered = sorted(ginis)

    def nearest_rank(p):
        rank = max(1, min(len(ordered), math.ceil(p * len(ordered))))
        return ordered[rank - 1]

    return {
        "status": "ok",
        "method": "frequency_multinomial",
        "n_individuals": n_draws,
        "replicates": replicates,
        "valid_replicates": len(ginis),
        "seed": seed,
        "percentiles": {
            "p2.5": nearest_rank(0.025),
            "p50": nearest_rank(0.5),
            "p97.5": nearest_rank(0.975),
        },
        "note": "仅适用于频数权重；不处理整群、分层等复杂调查设计。",
    }


# ---------------------------------------------------------------- 总入口

def analyze(records, config, *, bootstrap_replicates=500, seed=20261113):
    """执行完整分析，返回可 JSON 序列化的结果字典。"""
    keys, x, w, groups = _parse_records(records)
    n = len(x)
    total_w, mu = weighted_moments(x, w)
    zero_undefined = config.get("zero_mean") == "undefined"
    if not zero_undefined:
        raise InequalityInputError(
            f"不支持的 zero_mean 约定 {config.get('zero_mean')!r}，当前仅支持 'undefined'"
        )

    gini_fast, d_fast, _, _ = gini_sorted(x, w)
    lorenz_points = lorenz_curve(x, w)
    decomposition = group_decomposition(x, w, groups)
    counterfactual = counterfactual_group_means(x, w, groups)
    quantiles = quantile_population_shares(
        keys, x, w, config.get("population_quantiles", [0.25, 0.5, 0.75])
    )

    crosscheck = None
    pairwise = None
    if n <= PAIRWISE_LIMIT:
        gini_ref, d_ref, _, _ = gini_double_sum(x, w)
        crosscheck = {
            "method": "explicit_ordered_double_sum_O(n2)",
            "gini_double_sum": gini_ref,
            "D_double_sum": d_ref,
            "gini_sorted_linear": gini_fast,
            "D_sorted_linear": d_fast,
            "max_abs_diff_gini": (
                None if gini_ref is None else abs(gini_ref - gini_fast)
            ),
        }
        pairwise = pairwise_contributions(keys, x, w, groups)

    if mu == 0.0:
        gini_status = "undefined_zero_mean"
        gini_lorenz = None
    else:
        gini_status = "defined"
        gini_lorenz = gini_from_lorenz(lorenz_points)

    # 重抽样闸门：只有声明频数权重且配置开启时才执行
    weight_kind = config.get("weight_kind")
    bootstrap_enabled = bool(config.get("bootstrap_enabled", False))
    if weight_kind == "frequency" and bootstrap_enabled:
        bootstrap = frequency_multinomial_bootstrap(
            x, w, bootstrap_replicates, seed
        )
    else:
        reasons = []
        if weight_kind != "frequency":
            reasons.append(
                f"weight_kind={weight_kind!r} 为代表性权重而非频数，"
                "多项抽样不适用，亦不伪造复杂调查置信区间"
            )
        if not bootstrap_enabled:
            reasons.append("bootstrap_enabled=false")
        bootstrap = {"status": "not_applicable", "reason": "；".join(reasons)}

    return {
        "n_records": n,
        "W": total_w,
        "mu": mu,
        "gini": gini_fast,
        "gini_status": gini_status,
        "gini_crosscheck": crosscheck,
        "gini_from_lorenz_area": gini_lorenz,
        "lorenz_curve": lorenz_points,
        "quantile_population_groups": quantiles,
        "group_decomposition": decomposition,
        "counterfactual_group_means": counterfactual,
        "bootstrap": bootstrap,
        "disclaimer": (
            "以上指标仅描述合成数据的加权分布不平等形态，"
            "不构成任何资源配置干预的因果效果估计或政策结论。"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="加权分布不平等分析（Gini/洛伦兹/分组分解）")
    parser.add_argument("data_path", help="合成统计单元表（csv/json/jsonl，由 inequality_reader 加载）")
    parser.add_argument("config_path", help="分析配置 JSON（如 fixtures/inequality.json）")
    parser.add_argument("--out", default=".workspace-output", help="证据输出目录")
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261113)
    args = parser.parse_args()

    records = load(args.data_path)
    config = load(args.config_path)
    if not isinstance(config, dict):
        raise InequalityInputError("配置文件必须是 JSON 对象")
    result = analyze(records, config, bootstrap_replicates=args.replicates, seed=args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "inequality_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "quantile_split_evidence.json").write_text(
        json.dumps(result["quantile_population_groups"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if result["n_records"] <= PAIRWISE_LIMIT:
        keys, x, w, groups = _parse_records(records)
        pairwise = pairwise_contributions(keys, x, w, groups)
        (out_dir / "pairwise_contributions.json").write_text(
            json.dumps(pairwise, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps({
        "out_dir": str(out_dir),
        "n_records": result["n_records"],
        "mu": result["mu"],
        "gini": result["gini"],
        "gini_status": result["gini_status"],
        "bootstrap_status": result["bootstrap"]["status"],
        "pairwise_evidence_saved": result["n_records"] <= PAIRWISE_LIMIT,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
