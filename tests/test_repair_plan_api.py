"""最低成本修补方案优选的端到端 API 测试。

覆盖：
- 以已签发 report_id 的快照来源观察（去重）为目标，精确求最低成本子集；
- 成本并列时依次取工包数少、排序后编号序列字典序小；
- 目标去重（跨轨迹、同一观察重复出现只计一次），工包内重复编号按一次计；
- 空目标返回空方案；无法全覆盖返回 409 及未被任何工包覆盖的编号；
- 重复工包编号、未知观察、越限（目标/工包各 20）、非法成本 -> 422；
- 报告不存在 -> 404，且报告存在性优先于请求体语义校验；
- 只读：只读取报告快照，报告随后不可变、观察/复核结果不被修改。
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

    def _pkg(self, pid, cost, obs_ids):
        return {"package_id": pid, "cost": cost,
                "observation_ids": obs_ids}

    # ------------------------------------------------------------ 优选

    def test_selects_minimum_cost_combination(self):
        # o1/o2 几何分离 -> 报告含 2 个目标观察
        self._write([_obs("o1", 0, 2), _obs("o2", 5, 7)])
        self._issue()
        resp = self._optimize([
            self._pkg("all", 10, ["o1", "o2"]),
            self._pkg("p1", 4, ["o1"]),
            self._pkg("p2", 5, ["o2"]),
        ])
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["report_id"], "rep-1")
        self.assertEqual(body["selected_package_ids"], ["p1", "p2"])
        self.assertEqual(body["total_cost"], 9)
        self.assertEqual(body["target_observation_ids"], ["o1", "o2"])

    def test_single_all_cover_wins_when_cheaper(self):
        self._write([_obs("o1", 0, 2), _obs("o2", 5, 7)])
        self._issue()
        resp = self._optimize([
            self._pkg("all", 10, ["o1", "o2"]),
            self._pkg("p1", 6, ["o1"]),
            self._pkg("p2", 5, ["o2"]),
        ])
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["selected_package_ids"], ["all"])
        self.assertEqual(body["total_cost"], 10)

    def test_overlapping_packages_exact_union_cover(self):
        self._write([_obs("o1", 0, 2), _obs("o2", 5, 7), _obs("o3", 9, 11)])
        self._issue()
        resp = self._optimize([
            self._pkg("a", 6, ["o1", "o2"]),
            self._pkg("b", 6, ["o2", "o3"]),
            self._pkg("c", 11, ["o1", "o2", "o3"]),
        ])
        # a+b=12 > c=11
        self.assertEqual(resp.json()["selected_package_ids"], ["c"])
        self.assertEqual(resp.json()["total_cost"], 11)

    def test_tie_cost_prefers_fewer_packages(self):
        self._write([_obs("o1", 0, 2), _obs("o2", 5, 7)])
        self._issue()
        resp = self._optimize([
            self._pkg("all", 10, ["o1", "o2"]),
            self._pkg("p1", 6, ["o1"]),
            self._pkg("p2", 4, ["o2"]),
        ])
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["selected_package_ids"], ["all"])

    def test_tie_cost_and_count_prefers_lexicographic_sequence(self):
        self._write([_obs("o1", 0, 2), _obs("o2", 5, 7)])
        self._issue()
        resp = self._optimize([
            self._pkg("b", 6, ["o1"]),
            self._pkg("c", 5, ["o2"]),
            self._pkg("a", 6, ["o1"]),
            self._pkg("d", 5, ["o2"]),
        ])
        # 全部两两组合成本均为 11、均 2 个工包 -> ("a","c") 字典序最小
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["selected_package_ids"], ["a", "c"])
        self.assertEqual(resp.json()["total_cost"], 11)

    def test_result_independent_of_request_order(self):
        self._write([_obs("o1", 0, 2), _obs("o2", 5, 7)])
        self._issue()
        packages = [
            self._pkg("z9", 6, ["o1"]),
            self._pkg("a1", 5, ["o2"]),
            self._pkg("all", 10, ["o1", "o2"]),
            self._pkg("m5", 6, ["o1"]),
        ]
        first = self._optimize(packages).json()
        second = self._optimize(list(reversed(packages))).json()
        self.assertEqual(first, second)
        self.assertEqual(first["selected_package_ids"],
                         second["selected_package_ids"])

    def test_targets_deduplicated_across_trajectory_sources(self):
        # o1/o2 几何相交同簇：报告只有 1 条轨迹，sources 含两个观察
        self._write([_obs("o1", 0, 5), _obs("o2", 4, 9)])
        self._issue()
        resp = self._optimize([self._pkg("p", 3, ["o1", "o2"])])
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["target_observation_ids"], ["o1", "o2"])

    def test_duplicate_observation_in_one_package_counts_once(self):
        self._write([_obs("o1", 0, 2), _obs("o2", 5, 7)])
        self._issue()
        resp = self._optimize([
            self._pkg("p1", 4, ["o1", "o1"]),       # 包内重复按一次
            self._pkg("p2", 5, ["o2"]),
        ])
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["selected_package_ids"], ["p1", "p2"])
        self.assertEqual(resp.json()["total_cost"], 9)

    # ------------------------------------------------------- 409 无法全覆盖

    def test_uncoverable_targets_return_409_with_all_uncovered(self):
        self._write([_obs("o1", 0, 2), _obs("o2", 5, 7), _obs("o3", 9, 11)])
        self._issue()
        resp = self._optimize([
            self._pkg("p1", 4, ["o1"]),
            self._pkg("p2", 5, ["o2"]),
        ])
        self.assertEqual(resp.status_code, 409, resp.text)
        body = resp.json()
        self.assertEqual(body["uncovered_observation_ids"], ["o3"])
        self.assertIn("detail", body)

    def test_uncovered_list_excludes_observations_some_package_covers(self):
        self._write([_obs("o1", 0, 2), _obs("o2", 5, 7), _obs("o3", 9, 11)])
        self._issue()
        # 任何工包都无法单独/组合覆盖 o3，但 o2 至少被某工包覆盖
        resp = self._optimize([
            self._pkg("a", 4, ["o1", "o2"]),
            self._pkg("b", 5, ["o2"]),
        ])
        self.assertEqual(resp.status_code, 409)
        self.assertEqual(resp.json()["uncovered_observation_ids"], ["o3"])

    def test_empty_packages_with_nonempty_targets_returns_409(self):
        self._write([_obs("o1", 0, 2)])
        self._issue()
        resp = self._optimize([])
        self.assertEqual(resp.status_code, 409)
        self.assertEqual(resp.json()["uncovered_observation_ids"], ["o1"])

    # ------------------------------------------------------------ 422

    def test_duplicate_package_ids_return_422(self):
        self._write([_obs("o1", 0, 2)])
        self._issue()
        resp = self._optimize([
            self._pkg("dup", 1, ["o1"]),
            self._pkg("dup", 2, ["o1"]),
        ])
        self.assertEqual(resp.status_code, 422, resp.text)

    def test_unknown_observation_returns_422(self):
        self._write([_obs("o1", 0, 2)])
        self._issue()
        resp = self._optimize([self._pkg("p", 1, ["oX"])])
        self.assertEqual(resp.status_code, 422, resp.text)

    def test_observation_from_other_report_not_a_target_422(self):
        self._write([_obs("o1", 0, 2)], blade="B", batch="b1")
        self._write([_obs("x1", 8, 9)], blade="C", batch="b2",
                    turbine="WT")
        self._issue(report_id="rep-1", blade="B")
        self._issue(report_id="rep-2", blade="C")
        # rep-1 的目标只有 o1；引用另一叶片观察 x1 -> 422
        resp = self._optimize([self._pkg("p", 1, ["o1", "x1"])],
                              report_id="rep-1")
        self.assertEqual(resp.status_code, 422)

    def test_more_than_20_packages_returns_422(self):
        self._write([_obs("o1", 0, 2)])
        self._issue()
        packages = [self._pkg(f"p{i:02d}", 1, ["o1"]) for i in range(21)]
        self.assertEqual(self._optimize(packages).status_code, 422)

    def test_more_than_20_targets_returns_422(self):
        self._write([_obs(f"o{i:02d}", i * 10, i * 10 + 1)
                     for i in range(21)])
        self._issue()
        resp = self._optimize([self._pkg("p00", 1,
                                         [f"o{i:02d}" for i in range(21)])])
        self.assertEqual(resp.status_code, 422, resp.text)

    def test_limit_20_targets_and_packages_accepted(self):
        self._write([_obs(f"o{i:02d}", i * 10, i * 10 + 1)
                     for i in range(20)])
        self._issue()
        packages = [self._pkg(f"p{i:02d}", 1, [f"o{i:02d}"])
                    for i in range(20)]
        resp = self._optimize(packages)
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["total_cost"], 20)
        self.assertEqual(len(body["selected_package_ids"]), 20)
        self.assertEqual(len(body["target_observation_ids"]), 20)

    def test_invalid_cost_returns_422(self):
        self._write([_obs("o1", 0, 2)])
        self._issue()
        for bad_cost in (0, -1):
            resp = self._optimize([self._pkg("p", bad_cost, ["o1"])])
            self.assertEqual(resp.status_code, 422, bad_cost)

    def test_non_integer_cost_returns_422(self):
        self._write([_obs("o1", 0, 2)])
        self._issue()
        resp = self._optimize(
            [{"package_id": "p", "cost": 1.5,
              "observation_ids": ["o1"]}])
        self.assertEqual(resp.status_code, 422)

    def test_missing_fields_and_unknown_fields_422(self):
        self._write([_obs("o1", 0, 2)])
        self._issue()
        # 缺 packages
        resp = self.client.post("/api/v1/repair-plans/optimize",
                                json={"report_id": "rep-1"})
        self.assertEqual(resp.status_code, 422)
        # 多余字段
        resp = self.client.post("/api/v1/repair-plans/optimize", json={
            "report_id": "rep-1",
            "packages": [],
            "bogus": 1,
        })
        self.assertEqual(resp.status_code, 422)
        # 工包缺字段
        resp = self._optimize([{"package_id": "p", "cost": 1}])
        self.assertEqual(resp.status_code, 422)

    # ------------------------------------------------------------ 404

    def test_unknown_report_returns_404(self):
        resp = self._optimize([self._pkg("p", 1, ["o1"])],
                              report_id="ghost")
        self.assertEqual(resp.status_code, 404)
        self.assertIn("detail", resp.json())

    def test_report_not_found_takes_precedence_over_semantic_422(self):
        # 报告不存在时，即使工包越限/编号重复/成本非法，也应先返回 404
        resp = self._optimize(
            [self._pkg("dup", 1, ["o"])] * 21, report_id="ghost")
        self.assertEqual(resp.status_code, 404)

    # ------------------------------------------------------- 空目标

    def test_empty_targets_returns_empty_plan(self):
        # 构造一条轨迹来源为空的报告：直接落库一条自定义快照行
        import json
        from app.services import _canonical_json, utcnow
        with self.db_mod.get_conn() as conn:
            with self.db_mod.transaction(conn):
                conn.execute(
                    """
                    INSERT INTO review_reports(
                        report_id, turbine_id, blade_id,
                        adjudication_revision, observation_fingerprint,
                        ruling_fingerprint, trajectory_count,
                        trajectories_json, issued_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("rep-empty", "WT", "EMPTY", 0, "obs", "rul", 0,
                     _canonical_json([]), utcnow()),
                )
        resp = self._optimize([], report_id="rep-empty")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["selected_package_ids"], [])
        self.assertEqual(body["total_cost"], 0)
        self.assertEqual(body["target_observation_ids"], [])

    # ------------------------------------------------------- 只读语义

    def test_reads_snapshot_not_live_observations(self):
        # 报告签发时只有 o1；之后新增 o2，方案目标仍只有快照中的 o1
        self._write([_obs("o1", 0, 2)])
        issued = self._issue(report_id="rep-1")
        self._write([_obs("o2", 40, 42)], batch="b2")

        # 引用当前库存在但不在报告快照中的 o2 -> 未知观察 422
        resp = self._optimize([self._pkg("p", 1, ["o1", "o2"])])
        self.assertEqual(resp.status_code, 422)

        # 仅覆盖 o1 的工包即可全覆盖（目标仍是快照）
        resp = self._optimize([self._pkg("p", 1, ["o1"])])
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["target_observation_ids"], ["o1"])

        # 报告快照原样不变
        got = self.client.get("/api/v1/review-reports/rep-1")
        self.assertEqual(got.json(), issued)

    def test_optimize_is_read_only(self):
        self._write([_obs("o1", 0, 2), _obs("o2", 5, 7)])
        self._issue()
        self.client.post("/api/v1/reviews/recompute",
                         json={"turbine_id": "WT", "blade_id": "B"})
        review_before = self.client.get("/api/v1/reviews/WT/B").json()
        adj_before = self.client.get("/api/v1/adjudications/WT/B").json()
        report_before = self.client.get(
            "/api/v1/review-reports/rep-1").json()

        for _ in range(2):
            resp = self._optimize([
                self._pkg("a", 4, ["o1"]),
                self._pkg("b", 5, ["o2"]),
            ])
            self.assertEqual(resp.status_code, 200, resp.text)
        # 409 路径同样只读
        self._optimize([self._pkg("a", 4, ["o1"])])

        self.assertEqual(
            self.client.get("/api/v1/reviews/WT/B").json(), review_before)
        self.assertEqual(
            self.client.get("/api/v1/adjudications/WT/B").json(), adj_before)
        self.assertEqual(
            self.client.get("/api/v1/review-reports/rep-1").json(),
            report_before)
        with self.db_mod.get_conn() as conn:
            count = conn.execute(
                "SELECT COUNT(*) AS c FROM observations").fetchone()["c"]
            reports = conn.execute(
                "SELECT COUNT(*) AS c FROM review_reports").fetchone()["c"]
        self.assertEqual(count, 2)
        self.assertEqual(reports, 1)

    def test_openapi_lists_endpoint(self):
        paths = self.client.get("/openapi.json").json()["paths"]
        self.assertIn("/api/v1/repair-plans/optimize", paths)


if __name__ == "__main__":
    unittest.main()
