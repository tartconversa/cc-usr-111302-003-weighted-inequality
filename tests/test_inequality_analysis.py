import json
import math
import tempfile
import unittest
from pathlib import Path

from src.inequality_analysis import (
    PAIRWISE_LIMIT,
    InequalityInputError,
    analyze,
    frequency_multinomial_bootstrap,
    gini_double_sum,
    gini_from_lorenz,
    gini_sorted,
    group_decomposition,
    counterfactual_group_means,
    lorenz_curve,
    pairwise_contributions,
    quantile_population_shares,
)
from src.inequality_reader import load

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"
DEFAULT_QUANTILES = [0.25, 0.5, 0.75]


def rows(pairs, groups=None, keys=None):
    """由 (x, w) 对列表构造读取器风格的 dict 记录。"""
    out = []
    for i, (xi, wi) in enumerate(pairs):
        row = {"key": (keys or [f"r{i}" for j in range(len(pairs))])[i], "x": xi, "weight": wi}
        if groups is not None:
            row["group"] = groups[i]
        out.append(row)
    return out


class HandExampleTests(unittest.TestCase):
    """题目给定：x=(0,10)、w=(1,3)，mu=7.5、Gini=0.25。"""

    X, W = [0.0, 10.0], [1.0, 3.0]

    def test_moments(self):
        total_w, mu = gini_sorted(self.X, self.W)[2:]
        self.assertAlmostEqual(total_w, 4.0)
        self.assertAlmostEqual(mu, 7.5)

    def test_gini_sorted_linear(self):
        gini, d, _, _ = gini_sorted(self.X, self.W)
        self.assertAlmostEqual(gini, 0.25)
        # 定义式 D = 2 * 1*3*|0-10| = 60
        self.assertAlmostEqual(d, 60.0)

    def test_gini_double_sum_crosscheck(self):
        gini_ref, d_ref, _, _ = gini_double_sum(self.X, self.W)
        gini_fast, d_fast, _, _ = gini_sorted(self.X, self.W)
        self.assertAlmostEqual(gini_ref, 0.25)
        self.assertAlmostEqual(d_ref, d_fast)
        self.assertLessEqual(abs(gini_ref - gini_fast), 1e-15)

    def test_gini_from_lorenz_area(self):
        points = lorenz_curve(self.X, self.W)
        self.assertAlmostEqual(gini_from_lorenz(points), 0.25)

    def test_quantile_resource_shares(self):
        # 前 25% 人口资源份额 0，前 50% 人口资源份额 1/3
        split = quantile_population_shares(["h1", "h2"], self.X, self.W, [0.25, 0.5, 0.75])
        cumulative = {c["quantile"]: c["resource_share"] for c in split["cumulative_resource_share_at_quantiles"]}
        self.assertAlmostEqual(cumulative[0.25], 0.0)
        self.assertAlmostEqual(cumulative[0.5], 1.0 / 3.0)

    def test_quantile_boundary_splits_record_by_weight(self):
        split = quantile_population_shares(["h1", "h2"], self.X, self.W, [0.25, 0.5, 0.75])
        # 阈值 0.5 落在权重为 3 的记录内部：必须出现按权重拆分证据
        cuts = {s["at_quantile"]: s for s in split["boundary_splits"]}
        self.assertIn(0.5, cuts)
        self.assertEqual(cuts[0.5]["key"], "h2")
        # 拆出 1 单位权重给下半箱，剩余 2 单位权重；原记录权重 3 不变
        self.assertAlmostEqual(cuts[0.5]["split_off_weight_into_lower_bin"], 1.0)
        self.assertAlmostEqual(cuts[0.5]["record_weight_remaining_for_higher_bins"], 2.0)
        self.assertAlmostEqual(cuts[0.5]["original_record_weight"], 3.0)
        # 箱内权重加总等于总人口
        self.assertAlmostEqual(sum(b["population_share"] for b in split["bins"]), 1.0)
        # 资源份额加总为 1
        self.assertAlmostEqual(sum(b["resource_share"] for b in split["bins"]), 1.0)

    def test_full_analyze_on_hand_case(self):
        result = analyze(rows(list(zip(self.X, self.W)), keys=["h1", "h2"]),
                         {"weight_kind": "representation", "population_quantiles": DEFAULT_QUANTILES,
                          "zero_mean": "undefined", "bootstrap_enabled": False})
        self.assertAlmostEqual(result["gini"], 0.25)
        self.assertEqual(result["gini_status"], "defined")
        self.assertIsNotNone(result["gini_crosscheck"])
        self.assertLessEqual(result["gini_crosscheck"]["max_abs_diff_gini"], 1e-15)


class TieAndWeightSplittingTests(unittest.TestCase):
    """并列值合并、以及把权重 3 拆成三条同值权 1 后指标不变。"""

    def test_split_weight_into_three_unit_records_is_invariant(self):
        original = rows([(0.0, 1.0), (10.0, 3.0)], keys=["h1", "h2"])
        expanded = rows([(0.0, 1.0), (10.0, 1.0), (10.0, 1.0), (10.0, 1.0)],
                        keys=["h1", "h2a", "h2b", "h2c"])
        r1 = analyze(original, {"weight_kind": "representation", "population_quantiles": DEFAULT_QUANTILES,
                                "zero_mean": "undefined", "bootstrap_enabled": False})
        r2 = analyze(expanded, {"weight_kind": "representation", "population_quantiles": DEFAULT_QUANTILES,
                                "zero_mean": "undefined", "bootstrap_enabled": False})
        self.assertAlmostEqual(r1["mu"], r2["mu"])
        self.assertAlmostEqual(r1["gini"], r2["gini"])
        self.assertAlmostEqual(r1["gini"], 0.25)
        # 洛伦兹折点必须把三个同值级别合并成一个点
        self.assertEqual(len(r2["lorenz_curve"]), 3)  # (0,0) + 两个值级别
        c1 = {c["quantile"]: c["resource_share"] for c in r1["quantile_population_groups"]["cumulative_resource_share_at_quantiles"]}
        c2 = {c["quantile"]: c["resource_share"] for c in r2["quantile_population_groups"]["cumulative_resource_share_at_quantiles"]}
        for q in DEFAULT_QUANTILES:
            self.assertAlmostEqual(c1[q], c2[q])
        self.assertAlmostEqual(c2[0.25], 0.0)
        self.assertAlmostEqual(c2[0.5], 1.0 / 3.0)

    def test_all_equal_values_gini_zero(self):
        result = analyze(rows([(5.0, 2.0), (5.0, 7.0), (5.0, 1.0)]),
                         {"weight_kind": "representation", "population_quantiles": DEFAULT_QUANTILES,
                          "zero_mean": "undefined", "bootstrap_enabled": False})
        self.assertAlmostEqual(result["gini"], 0.0)
        self.assertAlmostEqual(result["mu"], 5.0)
        # 全同值时洛伦兹线就是对角线
        for point in result["lorenz_curve"][1:]:
            self.assertAlmostEqual(point["population_share"], point["resource_share"])

    def test_ties_dominating_distribution(self):
        # 0,0,10 各带权重 1,2,3：直接核对双重求和与排序算法
        x, w = [0.0, 0.0, 10.0], [1.0, 2.0, 3.0]
        g1, d1, _, _ = gini_sorted(x, w)
        g2, d2, _, _ = gini_double_sum(x, w)
        self.assertAlmostEqual(d1, d2)
        self.assertAlmostEqual(g1, g2)
        # 手算 D：仅有 (0,10) 配对，有序贡献 2*10*(1*3 + 2*3) = 180
        self.assertAlmostEqual(d1, 180.0)


class AllZeroTests(unittest.TestCase):
    """资源全零时 mu=0，Gini 按配置输出未定义并单列，不得报 0。"""

    def test_gini_undefined_not_zero(self):
        result = analyze(rows([(0.0, 1.0), (0.0, 3.0), (0.0, 2.0)]),
                         {"weight_kind": "representation", "population_quantiles": DEFAULT_QUANTILES,
                          "zero_mean": "undefined", "bootstrap_enabled": False})
        self.assertEqual(result["mu"], 0.0)
        self.assertIsNone(result["gini"])
        self.assertEqual(result["gini_status"], "undefined_zero_mean")
        self.assertIsNone(result["gini_from_lorenz_area"])
        self.assertIsNone(result["gini_crosscheck"]["gini_double_sum"])
        self.assertEqual(result["group_decomposition"]["status"], "undefined_zero_mean")
        self.assertIsNone(result["group_decomposition"]["gini_total"])
        # 资源份额无法定义，列 null 而不是 0
        for cum in result["quantile_population_groups"]["cumulative_resource_share_at_quantiles"]:
            self.assertIsNone(cum["resource_share"])
        # D 本身仍是良定义的零，状态与数值分开表达
        self.assertAlmostEqual(result["gini_crosscheck"]["D_double_sum"], 0.0)

    def test_unsupported_zero_mean_convention_rejected(self):
        with self.assertRaises(InequalityInputError):
            analyze(rows([(0.0, 1.0)]),
                    {"weight_kind": "representation", "population_quantiles": DEFAULT_QUANTILES,
                     "zero_mean": "report_zero", "bootstrap_enabled": False})


class ValidationTests(unittest.TestCase):
    def test_negative_x_rejected_without_silent_shift(self):
        with self.assertRaises(InequalityInputError):
            analyze(rows([(-1.0, 1.0), (10.0, 1.0)]),
                    {"weight_kind": "representation", "population_quantiles": DEFAULT_QUANTILES,
                     "zero_mean": "undefined", "bootstrap_enabled": False})

    def test_non_positive_weight_rejected(self):
        for bad_w in (0.0, -2.0):
            with self.subTest(bad_w=bad_w):
                with self.assertRaises(InequalityInputError):
                    analyze(rows([(1.0, 1.0), (2.0, bad_w)]),
                            {"weight_kind": "representation", "population_quantiles": DEFAULT_QUANTILES,
                             "zero_mean": "undefined", "bootstrap_enabled": False})

    def test_blank_is_not_treated_as_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "blank.csv"
            p.write_text("key,x,weight\na,,1\nb,10,1\n", encoding="utf-8")
            with self.assertRaises(InequalityInputError):
                analyze(load(p), {"weight_kind": "representation", "population_quantiles": DEFAULT_QUANTILES,
                                  "zero_mean": "undefined", "bootstrap_enabled": False})

    def test_unweighted_one_vote_per_region_not_used(self):
        # 若错误地把大小地区各当一票（w 全 1），同一组 x 的 Gini 会不同
        weighted = analyze(rows([(0.0, 1.0), (10.0, 3.0)]),
                           {"weight_kind": "representation", "population_quantiles": DEFAULT_QUANTILES,
                            "zero_mean": "undefined", "bootstrap_enabled": False})
        unweighted = analyze(rows([(0.0, 1.0), (10.0, 1.0)]),
                             {"weight_kind": "representation", "population_quantiles": DEFAULT_QUANTILES,
                              "zero_mean": "undefined", "bootstrap_enabled": False})
        self.assertAlmostEqual(weighted["gini"], 0.25)
        self.assertAlmostEqual(unweighted["gini"], 0.5)  # 两票等权的经典结果
        self.assertNotAlmostEqual(weighted["gini"], unweighted["gini"])


class ScaleInvarianceTests(unittest.TestCase):
    def test_overall_scaling_of_x_and_w(self):
        rng_seed_base = [(0.0, 1.0), (2.0, 2.0), (5.0, 0.5), (9.0, 4.0)]
        base = analyze(rows(rng_seed_base),
                       {"weight_kind": "representation", "population_quantiles": DEFAULT_QUANTILES,
                        "zero_mean": "undefined", "bootstrap_enabled": False})
        scaled_x = analyze(rows([(xi * 1000.0, wi) for xi, wi in rng_seed_base]),
                           {"weight_kind": "representation", "population_quantiles": DEFAULT_QUANTILES,
                            "zero_mean": "undefined", "bootstrap_enabled": False})
        scaled_w = analyze(rows([(xi, wi * 1e6) for xi, wi in rng_seed_base]),
                           {"weight_kind": "representation", "population_quantiles": DEFAULT_QUANTILES,
                            "zero_mean": "undefined", "bootstrap_enabled": False})
        self.assertAlmostEqual(base["gini"], scaled_x["gini"])
        self.assertAlmostEqual(base["gini"], scaled_w["gini"])
        # mu 随 x 缩放，不随 w 缩放
        self.assertAlmostEqual(scaled_x["mu"], base["mu"] * 1000.0)
        self.assertAlmostEqual(scaled_w["mu"], base["mu"])


class TinyWeightTests(unittest.TestCase):
    def test_very_small_positive_weights(self):
        data = [(0.0, 1e-12), (4.0, 2e-12), (10.0, 3e-12)]
        result = analyze(rows(data),
                         {"weight_kind": "representation", "population_quantiles": DEFAULT_QUANTILES,
                          "zero_mean": "undefined", "bootstrap_enabled": False})
        self.assertAlmostEqual(result["mu"], (4.0 * 2e-12 + 10.0 * 3e-12) / 6e-12)
        g_ref, _, _, _ = gini_double_sum([x for x, _ in data], [w for _, w in data])
        self.assertAlmostEqual(result["gini"], g_ref)
        self.assertAlmostEqual(sum(b["population_share"] for b in
                                   result["quantile_population_groups"]["bins"]), 1.0)


class GroupDecompositionTests(unittest.TestCase):
    """分组按成对项划分、共同分母归一、和恰为总 Gini。"""

    def test_crossing_subtable_identity_and_pairwise_evidence(self):
        # fixture 把两个案例混在一张表，按 case 列取交错子表
        all_records = load(FIXTURES / "distribution.csv")
        records = [r for r in all_records if r["case"] == "crossing"]
        result = analyze(records, load(FIXTURES / "inequality.json"))
        dec = result["group_decomposition"]
        # 分量加总恒等（同一减法构造，且再独立核对一次）
        self.assertAlmostEqual(
            dec["gini_within_group"] + dec["gini_between_group_pairwise"],
            dec["gini_total"],
        )
        self.assertAlmostEqual(dec["gini_total"], result["gini"])
        # 手算（crossing 四行）：W=7, mu=31/7, D_总=136, Gini=136/434=68/217
        self.assertAlmostEqual(result["mu"], 31.0 / 7.0)
        self.assertAlmostEqual(result["gini"], 68.0 / 217.0)
        # 同组 D=46（g1=28、g2=18），跨组 D=90
        self.assertAlmostEqual(dec["D_within_group"], 46.0)
        self.assertAlmostEqual(dec["D_between_group_pairwise"], 90.0)
        self.assertAlmostEqual(dec["D_total"], 136.0)
        self.assertEqual(dec["D_within_by_group"], {"g1": 28.0, "g2": 18.0})

        keys, x, w, groups = ["g1a", "g1b", "g2a", "g2b"], [1.0, 8.0, 3.0, 6.0], [2.0, 1.0, 1.0, 3.0], ["g1", "g1", "g2", "g2"]
        evidence = pairwise_contributions(keys, x, w, groups)
        self.assertAlmostEqual(evidence["D_within_from_pairs"], dec["D_within_group"])
        self.assertAlmostEqual(evidence["D_between_from_pairs"], dec["D_between_group_pairwise"])
        self.assertAlmostEqual(evidence["D_total_from_pairs"], dec["D_total"])
        # 6 个无序对全部物化
        self.assertEqual(len(evidence["pairs"]), 6)
        self.assertTrue(all(set(p) == {"i", "j", "key_i", "key_j", "group_i", "group_j",
                                       "same_group", "abs_diff", "unordered_term_wiwj_absdx",
                                       "ordered_pair_contribution_2x"} for p in evidence["pairs"]))

    def test_interleaved_groups_counterfactual_differs_from_pairwise_between(self):
        # 组内交错：g1 的值 (1,8) 与 g2 的 (3,6) 互相穿插
        result = analyze(
            rows([(1.0, 2.0), (8.0, 1.0), (3.0, 1.0), (6.0, 3.0)],
                 groups=["g1", "g1", "g2", "g2"], keys=["g1a", "g1b", "g2a", "g2b"]),
            {"weight_kind": "representation", "population_quantiles": DEFAULT_QUANTILES,
             "zero_mean": "undefined", "bootstrap_enabled": False},
        )
        dec = result["group_decomposition"]
        cf = result["counterfactual_group_means"]
        # 组均值 g1=10/3、g2=21/4；反事实 D_cf=46，分母同全体 434，Gini=46/434=23/217
        self.assertAlmostEqual(cf["gini_group_means_only"], 23.0 / 217.0)
        self.assertAlmostEqual(dec["gini_between_group_pairwise"], 45.0 / 217.0)
        # 交错时两个指标不相等
        self.assertNotAlmostEqual(cf["gini_group_means_only"], dec["gini_between_group_pairwise"])
        # 反事实保持总体均值不变
        self.assertAlmostEqual(cf["mu_unchanged"], result["mu"])

    def test_nonoverlapping_groups_counterfactual_equals_pairwise_between(self):
        # 无重叠时（g1 全部低值、g2 全部高值），两指标恰好相等，作为对照锚点
        result = analyze(
            rows([(0.0, 1.0), (10.0, 3.0)], groups=["g1", "g2"], keys=["h1", "h2"]),
            {"weight_kind": "representation", "population_quantiles": DEFAULT_QUANTILES,
             "zero_mean": "undefined", "bootstrap_enabled": False},
        )
        dec = result["group_decomposition"]
        cf = result["counterfactual_group_means"]
        self.assertAlmostEqual(cf["gini_group_means_only"], 0.25)
        self.assertAlmostEqual(dec["gini_between_group_pairwise"], 0.25)
        self.assertAlmostEqual(dec["gini_within_group"], 0.0)
        self.assertAlmostEqual(cf["gini_group_means_only"], dec["gini_between_group_pairwise"])

    def test_large_sample_no_explicit_quadratic_pairs(self):
        # n > PAIRWISE_LIMIT：不物化成对项，排序算法与双重求和在同一数据的小副本上等价
        n = PAIRWISE_LIMIT + 50
        x = [((i * 37) % 101) for i in range(n)]
        w = [1.0 + (i % 7) for i in range(n)]
        groups = [f"g{i % 4}" for i in range(n)]
        result = analyze(rows(list(zip(x, w)), groups=groups),
                         {"weight_kind": "representation", "population_quantiles": DEFAULT_QUANTILES,
                          "zero_mean": "undefined", "bootstrap_enabled": False})
        self.assertIsNone(result["gini_crosscheck"])  # 超阈值不做 O(n^2) 核对
        dec = result["group_decomposition"]
        self.assertAlmostEqual(
            dec["gini_within_group"] + dec["gini_between_group_pairwise"], dec["gini_total"]
        )
        # 独立用前 18 行的小副本做定义式核对，验证排序算法数值正确
        small_x, small_w = x[:18], w[:18]
        g_fast, d_fast, _, _ = gini_sorted(small_x, small_w)
        g_ref, d_ref, _, _ = gini_double_sum(small_x, small_w)
        self.assertAlmostEqual(d_fast, d_ref)
        self.assertAlmostEqual(g_fast, g_ref)

    def test_decomposition_matches_explicit_pair_partition(self):
        x, w, groups = [1.0, 8.0, 3.0, 6.0], [2.0, 1.0, 1.0, 3.0], ["g1", "g1", "g2", "g2"]
        dec = group_decomposition(x, w, groups)
        ev = pairwise_contributions(["a", "b", "c", "d"], x, w, groups)
        self.assertAlmostEqual(dec["D_within_group"], ev["D_within_from_pairs"])
        self.assertAlmostEqual(dec["D_between_group_pairwise"], ev["D_between_from_pairs"])
        self.assertAlmostEqual(dec["D_total"], ev["D_total_from_pairs"])


class BootstrapGateTests(unittest.TestCase):
    def test_representation_weights_no_fake_confidence_interval(self):
        result = analyze(rows([(0.0, 1.0), (10.0, 3.0)]),
                         {"weight_kind": "representation", "population_quantiles": DEFAULT_QUANTILES,
                          "zero_mean": "undefined", "bootstrap_enabled": False})
        self.assertEqual(result["bootstrap"]["status"], "not_applicable")
        self.assertIn("representation", result["bootstrap"]["reason"])
        # 即使配置误开 bootstrap，非频数权重也不抽样
        result2 = analyze(rows([(0.0, 1.0), (10.0, 3.0)]),
                          {"weight_kind": "representation", "population_quantiles": DEFAULT_QUANTILES,
                           "zero_mean": "undefined", "bootstrap_enabled": True})
        self.assertEqual(result2["bootstrap"]["status"], "not_applicable")

    def test_frequency_multinomial_bootstrap_runs_and_shrinks_with_n(self):
        x, w = [0.0, 5.0, 10.0], [100.0, 100.0, 100.0]
        boot = frequency_multinomial_bootstrap(x, w, replicates=400, seed=42)
        self.assertEqual(boot["status"], "ok")
        self.assertEqual(boot["n_individuals"], 300)
        p = boot["percentiles"]
        self.assertTrue(p["p2.5"] <= p["p50"] <= p["p97.5"])
        # 多项抽样插件 Gini 含已知上偏，不以“覆盖真值”为判据；
        # 随总人数 N 增大，抽样区间宽度应收窄（区间含原插件点估计）
        gini_point, _, _, _ = gini_sorted(x, w)
        self.assertAlmostEqual(gini_point, 4.0 / 9.0)
        self.assertLessEqual(p["p2.5"], p["p50"])
        self.assertGreaterEqual(p["p97.5"], p["p50"])
        big = frequency_multinomial_bootstrap([0.0, 5.0, 10.0], [5000.0] * 3, replicates=300, seed=42)
        width_small = p["p97.5"] - p["p2.5"]
        width_big = big["percentiles"]["p97.5"] - big["percentiles"]["p2.5"]
        self.assertLess(width_big, width_small)
        # 中位数随 N 收敛到该经验分布的插件 Gini
        self.assertLess(abs(big["percentiles"]["p50"] - 4.0 / 9.0), 0.01)

    def test_non_integer_frequency_weights_rejected(self):
        with self.assertRaises(InequalityInputError):
            frequency_multinomial_bootstrap([0.0, 10.0], [1.7, 3.0], replicates=10, seed=1)

    def test_seed_makes_bootstrap_reproducible(self):
        x, w = [0.0, 10.0], [50.0, 50.0]
        b1 = frequency_multinomial_bootstrap(x, w, replicates=100, seed=7)
        b2 = frequency_multinomial_bootstrap(x, w, replicates=100, seed=7)
        self.assertEqual(b1["percentiles"], b2["percentiles"])


class FixtureIntegrationTests(unittest.TestCase):
    def test_hand_case_filtered_from_fixture_matches_spec(self):
        # fixture 手算案例两行：x=(0,10)、w=(1,3)
        hand = [r for r in load(FIXTURES / "distribution.csv") if r["case"] == "hand"]
        result = analyze(hand, load(FIXTURES / "inequality.json"))
        self.assertAlmostEqual(result["mu"], 7.5)
        self.assertAlmostEqual(result["gini"], 0.25)
        cumulative = {c["quantile"]: c["resource_share"]
                      for c in result["quantile_population_groups"]["cumulative_resource_share_at_quantiles"]}
        self.assertAlmostEqual(cumulative[0.25], 0.0)
        self.assertAlmostEqual(cumulative[0.5], 1.0 / 3.0)

    def test_full_fixture_pipeline_and_evidence_shape(self):
        records = load(FIXTURES / "distribution.csv")
        config = load(FIXTURES / "inequality.json")
        result = analyze(records, config)
        # 配置：代表性权重、关闭重抽样
        self.assertEqual(result["bootstrap"]["status"], "not_applicable")
        split = result["quantile_population_groups"]
        self.assertEqual([c["quantile"] for c in split["cumulative_resource_share_at_quantiles"]],
                         [0.25, 0.5, 0.75, 1.0])
        # 每个分位箱都有片段证据，且片段权重之和等于箱人口
        for b in split["bins"]:
            self.assertGreater(len(b["fragments"]), 0)
            self.assertAlmostEqual(
                math.fsum(f["weight"] for f in b["fragments"]),
                b["population_share"] * result["W"],
            )
        # 全表权重守恒
        self.assertAlmostEqual(
            math.fsum(f["weight"] for b in split["bins"] for f in b["fragments"]),
            result["W"],
        )
        # 免责声明存在且明确非因果
        self.assertIn("因果", result["disclaimer"])
        # 反事实与跨组项两个键并存，避免混用
        self.assertIn("note", result["counterfactual_group_means"])
        self.assertIn("note", result["group_decomposition"])

    def test_cli_writes_evidence_files(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "out"
            import src.inequality_analysis as mod
            import sys
            old = sys.argv
            sys.argv = ["inequality_analysis", str(FIXTURES / "distribution.csv"),
                        str(FIXTURES / "inequality.json"), "--out", str(out)]
            try:
                mod.main()
            finally:
                sys.argv = old
            files = {p.name for p in out.iterdir()}
            self.assertEqual(
                files,
                {"inequality_result.json", "quantile_split_evidence.json", "pairwise_contributions.json"},
            )
            result = json.loads((out / "inequality_result.json").read_text(encoding="utf-8"))
            evidence = json.loads((out / "pairwise_contributions.json").read_text(encoding="utf-8"))
            # 全 6 行合成表（手算 + 交错两案例混合）的插件 Gini
            self.assertAlmostEqual(result["gini"], 0.36661698956780925)
            self.assertEqual(evidence["n_records"], 6)
            self.assertEqual(len(evidence["pairs"]), 15)


if __name__ == "__main__":
    unittest.main()
