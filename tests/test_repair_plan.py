"""最低成本修补工包优选的纯算法测试（仅标准库，不依赖 FastAPI/pydantic）。"""

import unittest

from app.repair_plan import lex_smaller, optimize


def masks(ids, target_ids):
    """按目标编号清单生成覆盖位掩码。"""
    bit = {t: i for i, t in enumerate(target_ids)}
    m = 0
    for x in ids:
        m |= 1 << bit[x]
    return m


class OptimizeTest(unittest.TestCase):
    def test_empty_targets_returns_empty_plan(self):
        self.assertEqual(optimize([], 0), (0, ()))
        # 即使给了工包，空目标仍为空方案
        self.assertEqual(
            optimize([("p", 1, 0)], 0), (0, ()))

    def test_no_solution_returns_none(self):
        self.assertIsNone(optimize([("p", 5, 0)], 1))
        self.assertIsNone(optimize([("p", 5, 0b01)], 2))
        self.assertIsNone(optimize([], 1))

    def test_minimum_cost_single_vs_combination(self):
        packs = [("all", 10, 0b11), ("p1", 6, 0b01), ("p2", 5, 0b10)]
        self.assertEqual(optimize(packs, 2), (10, ("all",)))
        packs = [("all", 10, 0b11), ("p1", 4, 0b01), ("p2", 5, 0b10)]
        self.assertEqual(optimize(packs, 2), (9, ("p1", "p2")))

    def test_overlap_does_not_double_cover(self):
        targets = ["o1", "o2", "o3"]
        packs = [
            ("a", 6, masks(["o1", "o2"], targets)),
            ("b", 6, masks(["o2", "o3"], targets)),
            ("c", 11, masks(["o1", "o2", "o3"], targets)),
        ]
        self.assertEqual(optimize(packs, 3), (11, ("c",)))

    def test_tie_cost_prefers_fewer_packages(self):
        packs = [("a", 10, 0b11), ("b", 6, 0b01), ("c", 4, 0b10)]
        self.assertEqual(optimize(packs, 2), (10, ("a",)))

    def test_tie_cost_and_count_prefers_lex_sequence(self):
        packs = [
            ("b", 6, 0b01), ("c", 5, 0b10),
            ("a", 6, 0b01), ("d", 5, 0b10),
        ]
        self.assertEqual(optimize(packs, 2), (11, ("a", "c")))

    def test_tie_single_package_lex(self):
        self.assertEqual(
            optimize([("z", 5, 0b1), ("a", 5, 0b1)], 1),
            (5, ("a",)))

    def test_tie_triple_lex_sequence(self):
        packs = [
            ("x", 1, 0b100), ("y", 1, 0b010), ("z", 1, 0b001),
            ("a", 1, 0b100), ("b", 1, 0b010), ("c", 1, 0b001),
        ]
        self.assertEqual(optimize(packs, 3), (3, ("a", "b", "c")))

    def test_zero_mask_packages_never_selected(self):
        self.assertEqual(
            optimize([("e", 1, 0), ("p", 3, 0b1)], 1), (3, ("p",)))

    def test_order_independence(self):
        packs = [
            ("z9", 6, 0b01), ("a1", 5, 0b10),
            ("all", 10, 0b11), ("m5", 6, 0b01),
        ]
        self.assertEqual(optimize(packs, 2), optimize(packs[::-1], 2))

    def test_20_targets_and_packages_singletons(self):
        packs = [(f"p{i:02d}", 1, 1 << i) for i in range(20)]
        cost, chosen = optimize(packs, 20)
        self.assertEqual(cost, 20)
        self.assertEqual(chosen, tuple(f"p{i:02d}" for i in range(20)))

    def test_20_packages_all_cover_all(self):
        packs = [(f"p{i:02d}", 1, (1 << 20) - 1) for i in range(20)]
        self.assertEqual(optimize(packs, 20), (1, ("p00",)))

    def test_lex_smaller_helper(self):
        self.assertTrue(lex_smaller(0b01, 0b10))      # (a) vs (b)
        self.assertFalse(lex_smaller(0b10, 0b01))
        self.assertFalse(lex_smaller(0b11, 0b11))
        # (a,c) vs (b,c)：最低不同位是 a
        self.assertTrue(lex_smaller(0b101, 0b110))


if __name__ == "__main__":
    unittest.main()
