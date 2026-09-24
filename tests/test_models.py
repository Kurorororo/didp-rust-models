"""CLI regressions; run after `cargo build --workspace --bins`.

Uses only the Python standard library. All generated instances and histories are
kept in a temporary directory. Set MODEL_BIN_DIR to test a different build.
"""
import itertools
import os
from pathlib import Path
import random
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get("MODEL_BIN_DIR", ROOT / "target/debug")).resolve()


def matrix(rows):
    return "\n".join(" ".join(map(str, row)) for row in rows)


def tsplib(distances, demands, capacity, *, cvrp=False):
    return f"""NAME: regression
TYPE: {'CVRP' if cvrp else 'TSP'}
DIMENSION: {len(distances)}
CAPACITY: {capacity}
DEMAND_DIMENSION: 1
EDGE_WEIGHT_TYPE: EXPLICIT
EDGE_WEIGHT_FORMAT: FULL_MATRIX
EDGE_WEIGHT_SECTION
{matrix(distances)}
DEMAND_SECTION
{matrix([(i + 1, d) for i, d in enumerate(demands)])}
""" + ("DEPOT_SECTION\n1\n-1\n" if cvrp else "") + "EOF\n"


class ModelsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def solve(self, binary, data, *options, solver="astar", filename="sample-k2.vrp", expected_expanded=None):
        instance = self.directory / filename
        instance.write_text(data)
        history = self.directory / "history.csv"
        args = [str(BIN / binary), "4" if binary.startswith("golomb") else str(instance),
                "--solver", solver, "--history", str(history), "--time-limit", "10", *options]
        result = subprocess.run(args, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("The solution is invalid", result.stdout)
        if expected_expanded is not None:
            self.assertRegex(result.stdout, rf"(?m)^Expanded: {expected_expanded}$")
        found = re.search(r"^optimal cost: (-?\d+)$", result.stdout, re.MULTILINE)
        if not found:
            self.assertIn("The problem is infeasible.", result.stdout)
            return None
        self.assertIn("The solution is valid", result.stdout)
        self.assertTrue(history.exists())
        if solver == "cabs" and "2" in options:
            self.assertIn("threads: 2" if binary.endswith("_dypdl") else "2 threads", result.stdout)
        return int(found[1])

    def test_all_binaries_serial_and_parallel(self):
        fixtures = {
            "bin-packing": "4 5\n3 2 2 1\n",
            "cvrp": tsplib([[0, 1, 2], [1, 0, 1], [2, 1, 0]], [0, 1, 1], 1, cvrp=True),
            "golomb-ruler": "",
            "graph-clear": "3 2\n1 2 3\n0 1 0\n1 0 2\n0 2 0\n",
            "knapsack": "3 4\n3 2\n4 3\n1 1\n",
            "m-pdtsp": tsplib([[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]], [0, 1, -1, 0], 1),
            "mdkp": "3 2 0\n3 4 1\n2 3 1\n1 2 1\n4 3\n",
            "misp": "p edge 3 2\ne 1 2\ne 2 3\n",
            "mosp": "3 3\n1 1 0\n0 1 1\n1 0 1\n",
            "optw": "0 0 2\n\n0 0 0 0 0 0 10\n1 1 0 0 4 0 10\n2 2 0 0 3 0 10\n",
            "salbp-1": "<number of tasks>\n3\n<cycle time>\n5\n<task times>\n1 3\n2 2\n3 1\n<precedence relations>\n1,3\n<end>\n",
            "talent-scheduling": "sample 3 2\n1 1 0 2\n0 1 1 3\n1 2 1\n",
            "tsptw": "3\n0 1 2\n1 0 1\n2 1 0\n0 10\n0 10\n0 10\n",
            "wt": "3\n2 2 1\n1 2 2\n3 5 1\n",
        }
        for source in sorted(ROOT.glob("*/src/bin/*.rs")):
            with self.subTest(binary=source.stem):
                data = fixtures[source.parts[-4]]
                serial = self.solve(source.stem, data, solver="cabs")
                parallel = self.solve(source.stem, data, "--threads", "2", solver="cabs")
                self.assertIsNotNone(serial)
                self.assertEqual(parallel, serial)
                if source.stem.endswith("_dypdl"):
                    self.assertEqual(self.solve(source.stem, data, "-j", "2", solver="astar"), serial)
                result = subprocess.run([str(BIN / source.stem), "4" if source.stem.startswith("golomb") else "unused", "--threads", "0"],
                                        capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("--threads", result.stderr)

    def test_mdkp_fractional_bounds(self):
        for seed in range(12):
            rng = random.Random(seed)
            n, m = 6, 3
            profit = [rng.randrange(-3, 15) for _ in range(n)]
            weights = [[rng.randrange(5) for _ in range(n)] for _ in range(m)]
            capacities = [rng.randrange(8) for _ in range(m)]
            expected = max(sum(profit[i] for i in range(n) if mask & (1 << i))
                           for mask in range(1 << n)
                           if all(sum(row[i] for i in range(n) if mask & (1 << i)) <= c
                                  for row, c in zip(weights, capacities)))
            data = f"{n} {m} 0\n" + matrix([profit, *weights, capacities])
            with self.subTest(seed=seed):
                for binary in ["mdkp_dypdl", "mdkp_rpid"]:
                    self.assertEqual(self.solve(binary, data), expected)
                self.assertEqual(self.solve("mdkp_dypdl", data, "--threads", "2", solver="cabs"), expected)
        self.assertEqual(self.solve("mdkp_dypdl", "2 1 0\n5 -2\n0 0\n0\n"), 5)
        self.assertEqual(self.solve("mdkp_dypdl", "0 1 0\n0\n"), 0)

    def test_knapsack_dantzig(self):
        for seed in range(8):
            rng = random.Random(seed)
            n, capacity = 6, rng.randrange(10)
            items = [(rng.randrange(-3, 15), rng.randrange(5)) for _ in range(n)]
            expected = max(sum(p for i, (p, _) in enumerate(items) if mask & (1 << i))
                           for mask in range(1 << n)
                           if sum(w for i, (_, w) in enumerate(items) if mask & (1 << i)) <= capacity)
            data = f"{n} {capacity}\n" + matrix(items)
            for binary in ["knapsack_dypdl", "knapsack_rpid"]:
                with self.subTest(seed=seed, binary=binary):
                    self.assertEqual(self.solve(binary, data), expected)

    def test_tsptw_mst_and_depot_windows(self):
        for seed in range(8):
            rng = random.Random(seed)
            n = 5
            distances = [[0 if i == j else rng.randrange(1, 12) for j in range(n)] for i in range(n)]
            opening = [rng.randrange(5) for _ in range(n)]
            closing = [rng.randrange(12, 35) for _ in range(n)]
            feasible = []
            for order in itertools.permutations(range(1, n)):
                time, cost, current = opening[0], 0, 0
                for node in (*order, 0):
                    time = max(time + distances[current][node], opening[node])
                    cost += distances[current][node]
                    current = node
                    if time > closing[node]:
                        break
                else:
                    feasible.append((cost, time))
            data = f"{n}\n" + matrix(distances) + "\n" + matrix(zip(opening, closing))
            for flags in [(), ("--mst",), ("--minimize-makespan",), ("--mst", "--minimize-makespan")]:
                with self.subTest(seed=seed, flags=flags):
                    expected = min((v[1 if "--minimize-makespan" in flags else 0] for v in feasible), default=None)
                    self.assertEqual(self.solve("tsptw_dypdl", data, *flags), expected)
                    self.assertEqual(self.solve("tsptw_dypdl", data, *flags, "-j", "2", solver="cabs"), expected)
                    binary = "tsptw_mst_rpid" if "--mst" in flags else "tsptw_rpid"
                    self.assertEqual(self.solve(binary, data, *(flag for flag in flags if flag != "--mst")), expected)

    def test_tsptw_infinity_primal_bound(self):
        # Preprocessing removes every arc. Both bounds should prove infeasibility
        # at the root instead of exploring states with missing-edge costs of zero.
        disconnected = "3\n0 1 1\n1 0 1\n1 1 0\n0 0\n0 0\n0 0\n"
        # The optimal makespan (101) exceeds n * max_distance (3) due to waiting.
        late_tour = "3\n0 1 1\n1 0 1\n1 1 0\n20 200\n100 200\n0 200\n"
        for flags in [(), ("--mst",), ("--minimize-makespan",), ("--mst", "--minimize-makespan")]:
            for solver, threads in [("astar", ()), ("cabs", ()), ("cabs", ("-j", "2"))]:
                with self.subTest(flags=flags, solver=solver, threads=threads):
                    self.assertIsNone(self.solve(
                        "tsptw_dypdl", disconnected, *flags, *threads,
                        "--simplification-level", "cheap", solver=solver, expected_expanded=0,
                    ))
                    self.assertEqual(self.solve("tsptw_dypdl", late_tour, *flags, *threads, solver=solver),
                                     101 if "--minimize-makespan" in flags else 3)

    def test_cvrp_mst_nonmetric(self):
        for seed in range(8):
            rng = random.Random(seed + 100)
            n, capacity, k = 5, 5, 3
            distances = [[0 if i == j else rng.randrange(1, 20) for j in range(n)] for i in range(n)]
            demand = [0] + [rng.randrange(1, 4) for _ in range(1, n)]
            feasible = []
            for order in itertools.permutations(range(1, n)):
                for splits in itertools.product((False, True), repeat=n - 2):
                    if sum(splits) >= k:
                        continue
                    load, cost, current = 0, 0, 0
                    for pos, node in enumerate(order):
                        if pos and splits[pos - 1]:
                            cost += distances[current][0]
                            load, current = 0, 0
                        load += demand[node]
                        if load > capacity:
                            break
                        cost += distances[current][node]
                        current = node
                    else:
                        feasible.append(cost + distances[current][0])
            with self.subTest(seed=seed):
                for binary in ["cvrp_dypdl", "cvrp_rpid"]:
                    self.assertEqual(self.solve(binary, tsplib(distances, demand, capacity, cvrp=True),
                                                filename="sample-k3.vrp"), min(feasible, default=None))

    def test_mpdtsp_sparse_and_infeasible(self):
        distances = [[0, 3, 2, -1], [-1, 0, 4, 1], [-1, -1, 0, 5], [-1, -1, -1, 0]]
        data = tsplib(distances, [0, 2, -2, 0], 2)
        for binary in ["m_pdtsp_dypdl", "m_pdtsp_rpid"]:
            self.assertEqual(self.solve(binary, data), 12)
        distances[1][2] = -1
        for binary in ["m_pdtsp_dypdl", "m_pdtsp_rpid"]:
            self.assertIsNone(self.solve(binary, tsplib(distances, [0, 2, -2, 0], 2)))

    def test_graph_clear_dominance(self):
        for seed in range(8):
            rng = random.Random(seed)
            n = 5
            weights = [rng.randrange(1, 6) for _ in range(n)]
            edges = [[0] * n for _ in range(n)]
            for i in range(n):
                for j in range(i):
                    edges[i][j] = edges[j][i] = rng.randrange(4)
            costs = []
            for order in itertools.permutations(range(n)):
                clean, cost = set(), 0
                for node in order:
                    step = weights[node] + sum(edges[node])
                    step += sum(edges[i][j] for i in clean for j in range(n) if j not in clean and j != node)
                    cost = max(cost, step)
                    clean.add(node)
                costs.append(cost)
            with self.subTest(seed=seed):
                self.assertEqual(self.solve("graph_clear_dypdl", f"{n} 0\n" + matrix([weights, *edges])), min(costs))
                self.assertEqual(self.solve("graph_clear_dypdl", f"{n} 0\n" + matrix([weights, *edges]), "--threads", "2", solver="cabs"), min(costs))

    def test_talent_subsumption_and_empty_actors(self):
        for seed in range(8):
            rng = random.Random(seed)
            n, m = 5, 3
            players = [[rng.randrange(2) for _ in range(n)] for _ in range(m)]
            # Repeated actor sets also exercise scene concatenation.
            for row in players:
                row[1] = row[0]
            wages = [rng.randrange(1, 6) for _ in range(m)]
            duration = [rng.randrange(1, 4) for _ in range(n)]
            costs = []
            for order in itertools.permutations(range(n)):
                cost = 0
                for row, wage in zip(players, wages):
                    positions = [pos for pos, scene in enumerate(order) if row[scene]]
                    if positions:
                        cost += wage * sum(duration[order[pos]] for pos in range(min(positions), max(positions) + 1))
                costs.append(cost)
            data = f"sample {n} {m}\n" + matrix([row + [wage] for row, wage in zip(players, wages)]) + "\n" + matrix([duration])
            with self.subTest(seed=seed):
                for binary in ["talent_scheduling_dypdl", "talent_scheduling_rpid"]:
                    self.assertEqual(self.solve(binary, data), min(costs))
        self.assertEqual(self.solve("talent_scheduling_dypdl", "sample 2 0\n1 2\n"), 0)

    def test_wt_processing_time_state_function(self):
        for seed in range(8):
            rng = random.Random(seed)
            jobs = [(rng.randrange(1, 6), rng.randrange(1, 20), rng.randrange(1, 6)) for _ in range(5)]
            costs = []
            for order in itertools.permutations(jobs):
                time, cost = 0, 0
                for processing_time, deadline, weight in order:
                    time += processing_time
                    cost += weight * max(time - deadline, 0)
                costs.append(cost)
            data = f"{len(jobs)}\n" + matrix(jobs)
            for binary in ["wt_rpid", "wt_dypdl"]:
                for solver, threads in [("astar", ()), ("cabs", ()), ("cabs", ("-j", "2"))]:
                    with self.subTest(seed=seed, binary=binary, solver=solver, threads=threads):
                        self.assertEqual(self.solve(binary, data, *threads, solver=solver), min(costs))

    def test_bin_packing_quantifier(self):
        self.assertEqual(self.solve("bin_packing_dypdl", "6 6\n4 4 3 3 2 2\n"), 3)


if __name__ == "__main__":
    unittest.main()
