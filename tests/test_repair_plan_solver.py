"""最低成本集合覆盖求解器的穷举对拍测试。

对随机小规模输入，用 2^n 穷举得到的“总成本最小 -> 工包数最少 ->
排序后编号序列字典序最小”最优解与 ``_min_cost_cover`` 的结果逐一比对。
"""

import itertools
import random
import unittest

from app.services import _min_cost_cover


def _brute_force(pkgs, full):
    """穷举全部子集，返回 (排序后选中编号列表, 总成本)；无解返回 None。"""
    n = len(pkgs)
    best_key = None
    best_ids = None
    for r in range(n + 1):
        for combo in itertools.combinations(range(n), r):
            union = 0
            cost = 0
            ids = []
            for i in combo:
                pid, c, mask = pkgs[i]
                union |= mask
                cost += c
                ids.append(pid)
            if union != full:
                continue
            key = (cost, len(ids), tuple(sorted(ids)))
            if best_key is None or key < best_key:
                best_key = key
                best_ids = sorted(ids)
    if best_key is None:
        return None
    return best_ids, best_key[0]


class MinCostCoverSolverTest(unittest.TestCase):
    def test_handcrafted_cases(self):
        # 单包全覆盖优于两包同价组合
        pkgs = [("a", 2, 0b01), ("b", 2, 0b10), ("c", 4, 0b11)]
        self.assertEqual(_min_cost_cover(pkgs, 0b11), (["c"], 4))
        # 同价同数量取字典序
        pkgs = [("zz", 4, 0b11), ("aa", 4, 0b11)]
        self.assertEqual(_min_cost_cover(pkgs, 0b11), (["aa"], 4))
        # 更便宜的多包组合胜过昂贵的单包
        pkgs = [("all", 10, 0b11), ("x", 3, 0b01), ("y", 3, 0b10)]
        self.assertEqual(_min_cost_cover(pkgs, 0b11), (["x", "y"], 6))
        # 空覆盖工包永远不会被选中
        pkgs = [("noop", 1, 0), ("p", 5, 0b11)]
        self.assertEqual(_min_cost_cover(pkgs, 0b11), (["p"], 5))

    def test_randomized_against_brute_force(self):
        rng = random.Random(20260927)
        for trial in range(300):
            n_targets = rng.randint(1, 6)
            full = (1 << n_targets) - 1
            n_pkgs = rng.randint(1, 9)
            pkgs = []
            for i in range(n_pkgs):
                # 随机非空覆盖（保证至少单包有可能覆盖若干目标）
                mask = rng.randint(0, full)
                pkgs.append((f"pkg{i}", rng.randint(1, 9), mask))
            union_all = 0
            for _, _, m in pkgs:
                union_all |= m
            expected = _brute_force(pkgs, full)
            if union_all != full:
                self.assertIsNone(expected)
                continue
            self.assertIsNotNone(expected)
            got = _min_cost_cover(pkgs, full)
            self.assertEqual(got, expected,
                             f"trial={trial} pkgs={pkgs} full={full:b}")

    def test_max_scale_runs_fast(self):
        # 上限规模：20 目标 / 20 工包，全部随机覆盖，验证求解可在合理时间完成
        rng = random.Random(7)
        full = (1 << 20) - 1
        pkgs = [(f"p{i:02d}", rng.randint(1, 100), rng.randint(1, full))
                for i in range(20)]
        # 保证可全覆盖：补一个全覆盖工包
        pkgs.append(("pall", 5000, full))
        ids, cost = _min_cost_cover(pkgs, full)
        self.assertTrue(ids)
        self.assertGreater(cost, 0)


if __name__ == "__main__":
    unittest.main()
