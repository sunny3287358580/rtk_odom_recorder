#!/usr/bin/env python3
import importlib.util
import math
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "align_trajectories.py"
SPEC = importlib.util.spec_from_file_location("align_trajectories", SCRIPT)
ALIGN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ALIGN)


class AlignmentTests(unittest.TestCase):
    def test_recovers_known_se2_and_vertical_offset(self):
        yaw = 0.63
        tx, ty, tz = 8.2, -3.7, 1.25
        cosine, sine = math.cos(yaw), math.sin(yaw)
        rtk = []
        lio = []
        for index in range(41):
            timestamp = index * 0.1
            east = index * 0.25
            north = math.sin(index * 0.2) * 1.5
            up = math.cos(index * 0.13) * 0.2
            world_x = cosine * east - sine * north + tx
            world_y = sine * east + cosine * north + ty
            world_z = up + tz
            rtk.append({
                "timestamp": timestamp,
                "east": east,
                "north": north,
                "up": up,
            })
            lio.append({
                "timestamp": timestamp,
                "x": world_x,
                "y": world_y,
                "z": world_z,
            })

        result = ALIGN.estimate_alignment(rtk, lio, 0.21, 20, 5.0, 1e-6)
        self.assertAlmostEqual(result["yaw_rad"], yaw, places=10)
        self.assertAlmostEqual(result["tx"], tx, places=10)
        self.assertAlmostEqual(result["ty"], ty, places=10)
        self.assertAlmostEqual(result["tz"], tz, places=10)
        self.assertLess(result["rmse_xy_m"], 1e-9)

    def test_interpolates_lio_at_rtk_timestamps(self):
        lio = [
            {"timestamp": 0.0, "x": 0.0, "y": 0.0, "z": 0.0},
            {"timestamp": 1.0, "x": 2.0, "y": 4.0, "z": 6.0},
        ]
        self.assertEqual(
            ALIGN.interpolate_position(lio, 0.25, 1.1),
            (0.5, 1.0, 1.5),
        )
        self.assertIsNone(ALIGN.interpolate_position(lio, 0.25, 0.5))

    def test_rejects_insufficient_motion_and_bad_fit(self):
        rtk = [
            {"timestamp": float(i), "east": float(i), "north": 0.0, "up": 0.0}
            for i in range(5)
        ]
        lio = [
            {"timestamp": float(i), "x": float(i), "y": float(i % 2), "z": 0.0}
            for i in range(5)
        ]
        with self.assertRaisesRegex(ValueError, "at least"):
            ALIGN.estimate_alignment(rtk, lio, 1.1, 10, 1.0, 1.0)
        with self.assertRaisesRegex(ValueError, "RMSE"):
            ALIGN.estimate_alignment(rtk, lio, 1.1, 5, 1.0, 0.01)

    @staticmethod
    def trajectories():
        rtk, lio = [], []
        yaw, tx, ty, tz = 0.63, 8.2, -3.7, 1.25
        scale = 1.0012
        for i in range(121):
            t = i * 0.1
            east, north, up = 0.8 * t, 3.0 * math.sin(t * 0.5), 0.2 * math.cos(t)
            rtk.append({"timestamp": 1792402131.7 + t, "east": east, "north": north,
                        "up": up, "status": 2})
            lio.append({
                "timestamp": 1756224772.2 + scale * t,
                "x": math.cos(yaw) * east - math.sin(yaw) * north + tx,
                "y": math.sin(yaw) * east + math.cos(yaw) * north + ty,
                "z": up + tz, "qx": 0.0, "qy": 0.0,
                "qz": math.sin(yaw / 2), "qw": math.cos(yaw / 2),
            })
        return rtk, lio, scale

    def test_geometry_recovers_transform_and_419_day_clock_difference(self):
        rtk, lio, scale = self.trajectories()
        result, data = ALIGN.estimate_geometry_alignment(
            rtk, lio, samples=121, max_rmse=1e-6, iterations=4)
        np.testing.assert_allclose(ALIGN.transform_points(data["source"], result), data["target"], atol=1e-9)
        self.assertAlmostEqual(result["yaw_rad"], 0.63, places=9)
        self.assertAlmostEqual(result["tx"], 8.2, places=9)
        self.assertAlmostEqual(result["ty"], -3.7, places=9)
        self.assertAlmostEqual(result["tz"], 1.25, places=9)
        self.assertAlmostEqual(result["time_mapping"]["scale"], scale, places=7)
        self.assertLess(result["time_mapping"]["matched_time_residual_rmse_sec"], 1e-6)
        self.assertLess(result["time_synchronized_errors"]["rmse_xy_m"], 1e-5)
        matrix = ALIGN.matrix_from_result(result)
        np.testing.assert_allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(matrix[:3, :3]), 1.0)

    def test_offset_time_fit_uses_correspondences_not_file_starts(self):
        rt = 1792402131.7 + np.arange(120) * 0.2
        lt = 1756224772.2 - 1.35 + np.arange(120) * 0.2
        mapping = ALIGN.infer_time_mapping(rt, lt, 1792402131.7, 1756224772.2, "offset")
        self.assertAlmostEqual(mapping["shift_sec"], -1.35, places=6)
        np.testing.assert_allclose(ALIGN.map_times(rt, mapping), lt, atol=5e-7, rtol=0)

    def test_rejects_straight_or_too_short_geometry(self):
        rtk, lio, _ = self.trajectories()
        with self.assertRaisesRegex(ValueError, "at least"):
            ALIGN.estimate_geometry_alignment(rtk[:4], lio[:4])
        with self.assertRaisesRegex(ValueError, "coverage"):
            ALIGN.estimate_geometry_alignment(rtk, lio, min_displacement=100)
        for a, b in zip(rtk, lio):
            a["north"], b["y"] = 0.0, 0.0
        with self.assertRaisesRegex(ValueError, "nearly straight"):
            ALIGN.estimate_geometry_alignment(rtk, lio)

    def test_rejects_different_trajectory(self):
        rtk, lio, _ = self.trajectories()
        for i, row in enumerate(lio):
            row["x"] += 5 * math.sin(i * 0.2)
        with self.assertRaisesRegex(ValueError, "RMSE"):
            ALIGN.estimate_geometry_alignment(rtk, lio, samples=100, iterations=3, max_rmse=0.1)

    def test_csv_rejects_time_reset_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            path.write_text("timestamp,x\n1,4\n1,5\n2,6\n", encoding="utf-8")
            self.assertEqual(ALIGN.read_rows(path, ("timestamp", "x")),
                             [{"timestamp": 1.0, "x": 5.0}, {"timestamp": 2.0, "x": 6.0}])
            path.write_text("timestamp,x\n2,4\n1,5\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "backwards"):
                ALIGN.read_rows(path, ("timestamp", "x"))

    def test_cli_exports_matrix_matches_errors_tum_and_refuses_overwrite(self):
        rtk, lio, _ = self.trajectories()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for name, rows in (("rtk.csv", rtk), ("lio.csv", lio)):
                with (directory / name).open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            command = [
                sys.executable, str(SCRIPT), "--lio", str(directory / "lio.csv"),
                "--rtk", str(directory / "rtk.csv"), "--output", str(directory / "alignment.yaml"),
                "--samples", "121", "--no-plots",
            ]
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("T_world_enu", completed.stdout)
            report = json.loads((directory / "alignment.report.json").read_text())
            np.testing.assert_allclose(np.loadtxt(directory / "alignment.matrix.txt"), report["T_world_enu"])
            self.assertEqual(np.loadtxt(directory / "alignment.lio.tum").shape, (121, 8))
            self.assertEqual(np.loadtxt(directory / "alignment.rtk_world.tum").shape, (121, 8))
            for name in ("matches", "errors"):
                with (directory / f"alignment.{name}.csv").open() as stream:
                    self.assertGreater(len(list(csv.DictReader(stream))), 20)
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("already exists", completed.stderr)

    def test_cli_invalid_quaternion_leaves_no_usable_alignment(self):
        rtk, lio, _ = self.trajectories()
        lio[0].update(qx=0, qy=0, qz=0, qw=0)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for name, rows in (("rtk.csv", rtk), ("lio.csv", lio)):
                with (directory / name).open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            completed = subprocess.run([
                sys.executable, str(SCRIPT), "--lio", str(directory / "lio.csv"),
                "--rtk", str(directory / "rtk.csv"), "--output", str(directory / "alignment.yaml"),
                "--no-plots",
            ], capture_output=True, text=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse((directory / "alignment.yaml").exists())


if __name__ == "__main__":
    unittest.main()
