"""修补工包优化端到端测试（FastAPI TestClient + 临时 SQLite 文件）。

覆盖：
- 精确最低成本选择，及“成本相同 -> 工包数少 -> 编号序列字典序小”的并列裁决；
- 目标为报告快照来源观察去重全集，且只读快照（签发后观察变化不影响）；
- 空目标返回空方案；
- 重复工包编号 / 未知观察 / 越限 / 非法成本 -> 422；报告不存在 -> 404；
- 无法全覆盖 -> 409 并回带未被任何工包覆盖的编号；
- 接口只读，不修改报告或观察。
"""

import os
import tempfile
import unittest


def _obs(oid, r0, r1, c0=1, c1=2, grade=1, conf=0.5, inspection="insp-1"):
    return {
        "observation_id": oid,
        "inspection_id": inspection,
        "radius_start": r0,
        "radius_end": r1,
        "clock_start": c0,
        "clock_end": c1,
        "grade": grade,
        "confidence": conf,
    }


def _pkg(pid, cost, covers):
    return {"package_id": pid, "cost": cost, "covers": covers}


class RepairPlanOptimizeApiTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        db_file = os.path.join(self._tmp.name, "test_plan.db")
        os.environ["CRACK_DB_PATH"] = db_file

        from fastapi.testclient import TestClient
        from app import db as db_mod
        from app.main import app

        db_mod.DB_PATH = db_file
        db_mod.init_db()
        self.client = TestClient(app)
        self.db_mod = db_mod

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, observations, batch="b1", turbine="WT", blade="B"):
        resp = self.client.post("/api/v1/observations/batch", json={
            "batch_id": batch,
            "turbine_id": turbine,
            "blade_id": blade,
            "auto_recompute": False,
            "observations": observations,
        })
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()

    def _issue(self, revision=0, turbine="WT", blade="B", report_id="rep-1"):
        resp = self.client.post("/api/v1/review-reports", json={
            "report_id": report_id,
            "turbine_id": turbine,
            "blade_id": blade,
            "expected_adjudication_revision": revision,
        })
        self.assertEqual(resp.status_code, 201, resp.text)
        return resp.json()

    def _optimize(self, packages, report_id="rep-1"):
        return self.client.post("/api/v1/repair-plans/optimize", json={
            "report_id": report_id,
            "packages": packages,
        })

    # --------------------------------------------------------- 基本求解

    def test_min_cost_exact_selection(self):
        # 三条观察分属两簇，目标为去重后的 {o1, o2, o3}
        self._write([_obs("o1", 0, 5), _obs("o2", 4, 9), _obs("o3", 40, 45)])
        self._issue()

        resp = self._optimize([
            _pkg("pa", 5, ["o1", "o2"]),
            _pkg("pb", 3, ["o3"]),
            _pkg("pc", 9, ["o1", "o2", "o3"]),   # 一包全覆盖但更贵
            _pkg("pd", 4, ["o2", "o3"]),
        ])
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["report_id"], "rep-1")
        self.assertEqual(body["targets"], ["o1", "o2", "o3"])
        self.assertEqual(body["target_count"], 3)
        # pa+pb=8 低于 pc=9、pa+pd=9、pb+pd=7? -> pb+pd={o2,o3}+{o3}... 不含 o1
        self.assertEqual(body["selected_package_ids"], ["pa", "pb"])
        self.assertEqual(body["package_count"], 2)
        self.assertEqual(body["total_cost"], 8)

    def test_single_package_covering_all(self):
        self._write([_obs("o1", 0, 5), _obs("o2", 4, 9)])
        self._issue()
        resp = self._optimize([
            _pkg("all", 10, ["o1", "o2"]),
            _pkg("x", 6, ["o1"]),
            _pkg("y", 6, ["o2"]),
        ])
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["selected_package_ids"], ["all"])
        self.assertEqual(body["total_cost"], 10)

    # ------------------------------------------------------------ 并列裁决

    def test_tie_prefers_fewer_packages(self):
        self._write([_obs("o1", 0, 5), _obs("o2", 4, 9)])
        self._issue()
        resp = self._optimize([
            _pkg("a", 2, ["o1"]),
            _pkg("b", 2, ["o2"]),
            _pkg("c", 4, ["o1", "o2"]),   # 同价 4 但只用 1 个工包
        ])
        body = resp.json()
        self.assertEqual(body["selected_package_ids"], ["c"])
        self.assertEqual(body["total_cost"], 4)
        self.assertEqual(body["package_count"], 1)

    def test_tie_prefers_lexicographically_smaller_sequence(self):
        self._write([_obs("o1", 0, 5), _obs("o2", 4, 9)])
        self._issue()
        resp = self._optimize([
            _pkg("zz", 4, ["o1", "o2"]),
            _pkg("aa", 4, ["o1", "o2"]),
            _pkg("mm", 4, ["o1", "o2"]),
        ])
        body = resp.json()
        self.assertEqual(body["selected_package_ids"], ["aa"])
        self.assertEqual(body["total_cost"], 4)

    def test_tie_sequence_compares_sorted_ids(self):
        # 同价 4 元、同为 2 个工包的全覆盖方案有 ("a","c")、("a","z")、("b","c")，
        # 排序后编号序列字典序最小者为 ("a","c")
        self._write([_obs("o1", 0, 5), _obs("o2", 4, 9), _obs("o3", 40, 45)])
        self._issue()
        resp = self._optimize([
            _pkg("b", 2, ["o1"]),
            _pkg("c", 2, ["o2", "o3"]),
            _pkg("a", 2, ["o1", "o2"]),
            _pkg("z", 2, ["o3"]),
        ])
        body = resp.json()
        self.assertEqual(body["selected_package_ids"], ["a", "c"])
        self.assertEqual(body["total_cost"], 4)

    def test_exact_not_greedy(self):
        # 贪心“单位成本最低”会先拿 big（3 元/2 目标），随后还需 x+y，总 9；
        # 精确解为 x+y+z? 否——x+y=8 覆盖 {o1,o2} 但漏 o3……
        # 构造：big={o1,o2}@3, x={o1}@2, y={o2}@2, z={o3}@2, w={o1,o2,o3}@8
        # 最优 = big+z = 5（贪心若按单位成本：big(1.5) -> z(2) 也得 5；
        # 改为 big={o1,o2}@6, x={o1}@2, y={o2}@2, z={o3}@2 -> 最优 x+y+z=6
        # 与 big+z=8 相比，精确解必须选 x+y+z）
        self._write([_obs("o1", 0, 5), _obs("o2", 4, 9), _obs("o3", 40, 45)])
        self._issue()
        resp = self._optimize([
            _pkg("big", 6, ["o1", "o2"]),
            _pkg("x", 2, ["o1"]),
            _pkg("y", 2, ["o2"]),
            _pkg("z", 2, ["o3"]),
        ])
        body = resp.json()
        self.assertEqual(body["selected_package_ids"], ["x", "y", "z"])
        self.assertEqual(body["total_cost"], 6)

    # ------------------------------------------------------------ 快照语义

    def test_targets_come_from_snapshot_not_live_observations(self):
        self._write([_obs("o1", 0, 5)])
        issued = self._issue()
        self.assertEqual(
            {s["observation_id"] for t in issued["trajectories"]
             for s in t["sources"]},
            {"o1"},
        )
        # 签发后新增观察：现库已有 o2，但优化目标仍只看快照
        self._write([_obs("o2", 40, 45)], batch="b2")

        resp = self._optimize([_pkg("p1", 1, ["o1"])])
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["targets"], ["o1"])
        self.assertEqual(body["selected_package_ids"], ["p1"])
        # 快照外的观察对工包而言是未知观察
        resp2 = self._optimize([_pkg("p1", 1, ["o1", "o2"])])
        self.assertEqual(resp2.status_code, 422)

    def test_targets_deduplicated_across_trajectories(self):
        # must_link 使两条观察同轨；来源清单中各观察只出现一次，
        # 目标去重后仍为 {o1, o2}
        self._write([_obs("o1", 0, 2), _obs("o2", 5, 7)])
        self.client.post("/api/v1/adjudications", json={
            "turbine_id": "WT", "blade_id": "B",
            "observation_a": "o1", "observation_b": "o2",
            "relation": "must_link", "revision": 0,
        })
        self._issue(revision=1)
        resp = self._optimize([_pkg("p1", 3, ["o1", "o2"])])
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["targets"], ["o1", "o2"])

    # ------------------------------------------------------------ 空目标

    def test_empty_target_returns_empty_plan(self):
        # 正常签发的报告必有观察；直接落库一份零轨迹报告模拟空目标
        self._write([_obs("o1", 0, 5)])
        self._issue()
        with self.db_mod.get_conn() as conn:
            conn.execute(
                "UPDATE review_reports SET trajectories_json='[]',"
                " trajectory_count=0 WHERE report_id='rep-1'")
            conn.commit()

        resp = self._optimize([_pkg("p1", 1, [])])
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["targets"], [])
        self.assertEqual(body["target_count"], 0)
        self.assertEqual(body["selected_package_ids"], [])
        self.assertEqual(body["package_count"], 0)
        self.assertEqual(body["total_cost"], 0)

    # ------------------------------------------------------------ 错误分支

    def test_report_not_found_404(self):
        resp = self._optimize([_pkg("p1", 1, ["o1"])], report_id="ghost")
        self.assertEqual(resp.status_code, 404)
        self.assertIn("detail", resp.json())

    def test_duplicate_package_id_422(self):
        self._write([_obs("o1", 0, 5)])
        self._issue()
        resp = self._optimize([
            _pkg("dup", 1, ["o1"]),
            _pkg("dup", 2, ["o1"]),
        ])
        self.assertEqual(resp.status_code, 422)
        self.assertIn("dup", resp.json()["detail"])

    def test_unknown_observation_422(self):
        self._write([_obs("o1", 0, 5)])
        self._issue()
        resp = self._optimize([_pkg("p1", 1, ["o1", "nope"])])
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["unknown_observation_ids"], ["nope"])

    def test_too_many_packages_422(self):
        self._write([_obs("o1", 0, 5)])
        self._issue()
        packages = [_pkg(f"p{i:02d}", 1, ["o1"]) for i in range(21)]
        resp = self._optimize(packages)
        self.assertEqual(resp.status_code, 422)

    def test_too_many_targets_422(self):
        # 21 条互不相交的观察 -> 21 个目标，超过上限 20
        obs = [_obs(f"o{i:02d}", i * 10, i * 10 + 5) for i in range(21)]
        self._write(obs)
        self._issue()
        resp = self._optimize([_pkg("p1", 1, ["o00"])])
        self.assertEqual(resp.status_code, 422)

    def test_illegal_cost_422(self):
        self._write([_obs("o1", 0, 5)])
        self._issue()
        for bad in (0, -3, 2.5, "abc", None):
            resp = self._optimize([_pkg("p1", bad, ["o1"])])
            self.assertEqual(resp.status_code, 422, f"cost={bad!r}")

    def test_uncoverable_409_with_uncovered_ids(self):
        self._write([_obs("o1", 0, 5), _obs("o2", 4, 9), _obs("o3", 40, 45)])
        self._issue()
        resp = self._optimize([
            _pkg("p1", 1, ["o1"]),
            _pkg("p2", 1, ["o1", "o2"]),
        ])
        self.assertEqual(resp.status_code, 409)
        body = resp.json()
        self.assertIn("detail", body)
        self.assertEqual(body["uncovered_observation_ids"], ["o3"])

    def test_empty_packages_nonempty_target_409(self):
        self._write([_obs("o1", 0, 5), _obs("o2", 4, 9)])
        self._issue()
        resp = self._optimize([])
        self.assertEqual(resp.status_code, 409)
        self.assertEqual(resp.json()["uncovered_observation_ids"],
                         ["o1", "o2"])

    def test_validation_errors_422(self):
        self._write([_obs("o1", 0, 5)])
        self._issue()
        # 缺 report_id
        r1 = self.client.post("/api/v1/repair-plans/optimize",
                              json={"packages": []})
        self.assertEqual(r1.status_code, 422)
        # 多余字段（extra="forbid"）
        r2 = self.client.post("/api/v1/repair-plans/optimize", json={
            "report_id": "rep-1", "packages": [], "bogus": 1})
        self.assertEqual(r2.status_code, 422)
        # 工包缺 cost
        r3 = self.client.post("/api/v1/repair-plans/optimize", json={
            "report_id": "rep-1",
            "packages": [{"package_id": "p1", "covers": ["o1"]}]})
        self.assertEqual(r3.status_code, 422)
        # 空 package_id
        r4 = self._optimize([_pkg("", 1, ["o1"])])
        self.assertEqual(r4.status_code, 422)

    # ------------------------------------------------------------ 只读语义

    def test_read_only_does_not_modify_report_or_observations(self):
        self._write([_obs("o1", 0, 5), _obs("o2", 4, 9)])
        issued = self._issue()
        review_before = self.client.get("/api/v1/reviews/WT/B").json()

        packages = [_pkg("p1", 2, ["o1"]), _pkg("p2", 3, ["o1", "o2"])]
        first = self._optimize(packages)
        self.assertEqual(first.status_code, 200, first.text)
        # 多次调用结果一致
        again = self._optimize(packages)
        self.assertEqual(first.json(), again.json())

        # 报告快照原样、现库复核结果不变
        got = self.client.get("/api/v1/review-reports/rep-1").json()
        self.assertEqual(got, issued)
        review_after = self.client.get("/api/v1/reviews/WT/B").json()
        self.assertEqual(review_before, review_after)
        with self.db_mod.get_conn() as conn:
            n_reports = conn.execute(
                "SELECT COUNT(*) AS c FROM review_reports").fetchone()["c"]
            n_obs = conn.execute(
                "SELECT COUNT(*) AS c FROM observations").fetchone()["c"]
        self.assertEqual(n_reports, 1)
        self.assertEqual(n_obs, 2)

    def test_openapi_lists_endpoint(self):
        paths = self.client.get("/openapi.json").json()["paths"]
        self.assertIn("/api/v1/repair-plans/optimize", paths)


if __name__ == "__main__":
    unittest.main()
