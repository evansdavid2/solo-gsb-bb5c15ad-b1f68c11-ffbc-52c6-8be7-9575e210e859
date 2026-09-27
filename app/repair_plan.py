"""最低成本修补工包优选：小规模（目标、工包各 ≤ 20）的精确加权集合覆盖。

在候选工包子集中求**并集覆盖全部目标**且总成本最小者；成本相同时的
并列裁决依次为：

1. 选中工包数更少；
2. 选中工包编号按字典序排序后，编号序列字典序更小。

精确求解采用**分支定界**（规模上界目标/工包各 20）：

- 预处理：丢弃空覆盖工包；删除被支配工包（另一工包覆盖是其超集且成本
  更低，或同成本且编号更小——替换后在总排序下只会更优）；
- 用贪心可行解快速得到现任解（incumbent）；
- 每次挑选“剩余可覆盖它的工包最少”的未覆盖目标分支，分支动作是
  “选择一个覆盖它的工包”——因此每条搜索路径都是**不断新增覆盖**的
  合法覆盖路径，深度不超过目标数；
- 下界取一组两两不被同一工包共同覆盖的未覆盖目标（独立打包，必须由
  不同工包分别覆盖），其各自最低覆盖成本之和加上已花成本为合法下界，
  严格高于现任最优成本即剪枝；
- 叶节点按（成本, 工包数, 编号序列字典序）的总排序精确裁决并列。
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

# 接口约束：目标与候选工包各不超过 20 个
MAX_TARGETS = 20
MAX_PACKAGES = 20

# 目标位掩码：第 i 位表示覆盖“按编号升序排列的第 i 个目标”
Covered = int


def lex_smaller(selected_a: int, selected_b: int) -> bool:
    """两个选中位集（位序即工包编号升序）排序后编号序列的字典序比较。

    两条已排序序列逐位比较，首个不同的成员就是两个位集的**最低不同位**：
    此前元素完全相同，在该位置上包含该工包的一方，其下一元素更小，
    故包含最低不同位的位集字典序更小。两位集相同返回 ``False``。
    """
    diff = selected_a ^ selected_b
    if diff == 0:
        return False
    low_bit = diff & -diff
    return bool(selected_a & low_bit)


def better(cost_a: int, selected_a: int,
           cost_b: int, selected_b: int) -> bool:
    """方案 b 是否严格优于方案 a：先比成本，再比工包数，最后字典序。"""
    if cost_b != cost_a:
        return cost_b < cost_a
    count_a = selected_a.bit_count()
    count_b = selected_b.bit_count()
    if count_b != count_a:
        return count_b < count_a
    return lex_smaller(selected_b, selected_a)


def _greedy_incumbent(ids: Sequence[str], costs: Sequence[int],
                      masks: Sequence[Covered], full: Covered
                      ) -> Optional[Tuple[int, int]]:
    """贪心构造一个可行覆盖作为初始现任解：每步选“单位新增覆盖成本”最低
    的工包（并列取成本低、编号小）。无法覆盖时返回 None。"""
    remaining = set(range(len(ids)))
    union: Covered = 0
    total = 0
    selected = 0
    while union != full:
        best_i = None
        best_key = None
        for i in remaining:
            added = (masks[i] & ~union).bit_count()
            if added == 0:
                continue
            key = (costs[i] / added, costs[i], ids[i])
            if best_key is None or key < best_key:
                best_key = key
                best_i = i
        if best_i is None:
            return None
        remaining.remove(best_i)
        union |= masks[best_i]
        total += costs[best_i]
        selected |= 1 << best_i
    return total, selected


def optimize(
    packages: Sequence[Tuple[str, int, Covered]],
    target_count: int,
) -> Optional[Tuple[int, Tuple[str, ...]]]:
    """求最低成本覆盖方案。

    :param packages: ``(package_id, cost, covered_mask)`` 三元组序列，
        ``covered_mask`` 第 i 位表示覆盖第 i 个目标（目标序号由调用方
        按目标编号升序分配）；同一工包内重复覆盖同一目标按一次计。
    :param target_count: 去重后的目标数量。
    :returns: ``(总成本, 按编号升序排列的选中编号元组)``；
        无目标时返回 ``(0, ())``；不存在全覆盖方案时返回 ``None``。
    """
    if target_count == 0:
        return 0, ()

    # 位序统一按工包编号升序，保证 lex_smaller 的位序即编号序列顺序
    order = sorted(range(len(packages)), key=lambda i: packages[i][0])
    ids: List[str] = [packages[i][0] for i in order]
    costs: List[int] = [packages[i][1] for i in order]
    masks: List[Covered] = [packages[i][2] & ((1 << target_count) - 1)
                            for i in order]
    size = len(order)

    full: Covered = (1 << target_count) - 1

    # 丢弃空覆盖工包（成本为正整数，永不会出现在最优解中）
    keep = [i for i in range(size) if masks[i] != 0]

    # 支配删除：j 被 k 支配当 mask_k ⊇ mask_j，且
    # 成本更低（替换严格更优）或同成本且编号更小（并列序列字典序更优）
    dominated: set[int] = set()
    for j in keep:
        for k in keep:
            if j == k or k in dominated:
                continue
            if masks[k] & masks[j] == masks[j] and (
                    costs[k] < costs[j]
                    or (costs[k] == costs[j] and ids[k] < ids[j])):
                dominated.add(j)
                break
    keep = [i for i in keep if i not in dominated]

    # 可行性：全部剩余工包的并集都盖不满 -> 无解
    overall: Covered = 0
    for i in keep:
        overall |= masks[i]
    if overall != full:
        return None

    # 重新编号到紧凑位序（保持编号升序），位 i 即 keep[i] 在原序中的位
    compact_ids = [ids[i] for i in keep]
    compact_costs = [costs[i] for i in keep]
    compact_masks = [masks[i] for i in keep]
    n = len(keep)

    # 每个目标被哪些（紧凑位）工包覆盖
    coverers = [
        tuple(i for i in range(n) if compact_masks[i] & (1 << t))
        for t in range(target_count)
    ]

    incumbent = _greedy_incumbent(compact_ids, compact_costs,
                                  compact_masks, full)
    assert incumbent is not None  # overall == full 保证贪心可完成
    best_cost, best_selected = incumbent

    def dfs(available: Tuple[int, ...], union: Covered, spent: int,
            selected: int) -> None:
        nonlocal best_cost, best_selected

        if union == full:
            if better(best_cost, best_selected, spent, selected):
                best_cost, best_selected = spent, selected
            return

        missing = full & ~union

        # 每个未覆盖目标收集当前仍可覆盖它的工包及最低成本；
        # 任何候选为空 -> 该目标再也无法覆盖，剪枝。
        target_options: List[Tuple[int, Tuple[int, ...]]] = []
        for t in range(target_count):
            if not (missing & (1 << t)):
                continue
            candidates = tuple(i for i in coverers[t]
                               if i in available_set)
            if not candidates:
                return
            target_options.append((t, candidates))

        # 合法下界：选一组两两“不被同一工包共同覆盖”的目标（独立打包），
        # 它们必须由不同工包分别覆盖，故各自最低覆盖成本之和是合法下界
        # （同一工包被重复计入只会让求和虚高——独立打包恰好排除该情况）。
        order = sorted(
            target_options,
            key=lambda item: min(compact_costs[i] for i in item[1]),
        )
        blocked: set[int] = set()
        lower_extra = 0
        for t, candidates in order:
            if t in blocked:
                continue
            lower_extra += min(compact_costs[i] for i in candidates)
            for i in candidates:
                mask = compact_masks[i]
                for u, _ in order:
                    if mask & (1 << u):
                        blocked.add(u)
        # 下界严格高于现任最优成本即剪枝；相等时继续，可能工包更少/字典序更小
        if spent + lower_extra > best_cost:
            return

        # 分支：挑选“可覆盖它的工包最少”的未覆盖目标（最难满足），
        # 必须从这些工包中选一个；每个分支都新增覆盖，深度 ≤ 目标数。
        _, hardest = min(
            target_options, key=lambda item: len(item[1]))
        branches = sorted(hardest,
                          key=lambda i: (compact_costs[i], compact_ids[i]))
        for i in branches:
            next_available = tuple(j for j in available if j != i)
            available_set.discard(i)
            dfs(next_available, union | compact_masks[i],
                spent + compact_costs[i], selected | (1 << i))
            available_set.add(i)

    # available 用元组 + 集合双结构：集合供 O(1) 候选判定
    available_tuple = tuple(range(n))
    available_set = set(available_tuple)
    dfs(available_tuple, 0, 0, 0)  # 递归深度 ≤ 目标数（≤ 20）

    chosen = tuple(compact_ids[i] for i in range(n)
                   if (best_selected >> i) & 1)
    return best_cost, chosen
