#!/usr/bin/env python3
"""Register ordered RTK/LIO trajectories, then infer their timestamp mapping."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist


def read_rows(path, required):
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not set(required).issubset(reader.fieldnames):
            raise ValueError(f"{path}: expected CSV columns {', '.join(required)}")
        rows = []
        for row in reader:
            try:
                values = {name: float(row[name]) for name in required}
            except (TypeError, ValueError):
                continue
            if all(math.isfinite(value) for value in values.values()):
                rows.append(values)
    if any(b["timestamp"] < a["timestamp"] for a, b in zip(rows, rows[1:])):
        raise ValueError(f"{path}: timestamps go backwards; split repeated bag runs first")
    # Keep the last measurement of a duplicate timestamp.
    rows = list({row["timestamp"]: row for row in rows}.values())
    return rows


def interpolate_position(rows, timestamp, max_gap):
    lo = 0
    hi = len(rows)
    while lo < hi:
        mid = (lo + hi) // 2
        if rows[mid]["timestamp"] < timestamp:
            lo = mid + 1
        else:
            hi = mid
    right = lo
    if right < len(rows) and rows[right]["timestamp"] == timestamp:
        return tuple(rows[right][axis] for axis in ("x", "y", "z"))
    if right == 0 or right == len(rows):
        return None
    before = rows[right - 1]
    after = rows[right]
    gap = after["timestamp"] - before["timestamp"]
    if gap <= 0.0 or gap > max_gap:
        return None
    ratio = (timestamp - before["timestamp"]) / gap
    return tuple(
        before[axis] + ratio * (after[axis] - before[axis])
        for axis in ("x", "y", "z")
    )


def estimate_alignment(rtk_rows, lio_rows, max_gap, min_pairs, min_displacement, max_rmse):
    pairs = []
    for rtk in rtk_rows:
        lio = interpolate_position(lio_rows, rtk["timestamp"], max_gap)
        if lio is not None:
            pairs.append(((rtk["east"], rtk["north"], rtk["up"]), lio))
    if len(pairs) < min_pairs:
        raise ValueError(
            f"only {len(pairs)} synchronized pairs; at least {min_pairs} required"
        )

    source = [(pair[0][0], pair[0][1]) for pair in pairs]
    target = [(pair[1][0], pair[1][1]) for pair in pairs]
    coverage = max(
        math.hypot(point[0] - source[0][0], point[1] - source[0][1])
        for point in source
    )
    if coverage < min_displacement:
        raise ValueError(
            f"RTK trajectory coverage {coverage:.3f} m is below "
            f"the {min_displacement:.3f} m minimum"
        )

    source_mean = (
        sum(point[0] for point in source) / len(source),
        sum(point[1] for point in source) / len(source),
    )
    target_mean = (
        sum(point[0] for point in target) / len(target),
        sum(point[1] for point in target) / len(target),
    )
    dot = 0.0
    cross = 0.0
    for src, dst in zip(source, target):
        sx, sy = src[0] - source_mean[0], src[1] - source_mean[1]
        tx, ty = dst[0] - target_mean[0], dst[1] - target_mean[1]
        dot += sx * tx + sy * ty
        cross += sx * ty - sy * tx
    if math.hypot(dot, cross) < 1e-9:
        raise ValueError("trajectory geometry is degenerate; cannot estimate yaw")
    yaw = math.atan2(cross, dot)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    tx = target_mean[0] - (cosine * source_mean[0] - sine * source_mean[1])
    ty = target_mean[1] - (sine * source_mean[0] + cosine * source_mean[1])
    tz = sum(dst[2] - src[2] for src, dst in pairs) / len(pairs)

    residuals = []
    z_residuals = []
    for (east, north, up), (world_x, world_y, world_z) in pairs:
        predicted_x = cosine * east - sine * north + tx
        predicted_y = sine * east + cosine * north + ty
        residuals.append(math.hypot(predicted_x - world_x, predicted_y - world_y))
        z_residuals.append(up + tz - world_z)
    rmse_xy = math.sqrt(sum(value * value for value in residuals) / len(residuals))
    rmse_z = math.sqrt(sum(value * value for value in z_residuals) / len(z_residuals))
    if rmse_xy > max_rmse:
        raise ValueError(
            f"fit XY RMSE {rmse_xy:.3f} m exceeds the {max_rmse:.3f} m limit"
        )

    return {
        "enabled": True,
        "yaw_rad": yaw,
        "tx": tx,
        "ty": ty,
        "tz": tz,
        "pair_count": len(pairs),
        "rtk_coverage_m": coverage,
        "rmse_xy_m": rmse_xy,
        "rmse_z_m": rmse_z,
        "max_xy_error_m": max(residuals),
        "max_time_gap_sec": max_gap,
    }


def fit_rigid(source, target, weights=None):
    """Return a proper planar rotation and translation, using column vectors."""
    weights = np.ones(len(source)) if weights is None else np.asarray(weights)
    weights = weights / weights.sum()
    a = np.sum(source * weights[:, None], axis=0)
    b = np.sum(target * weights[:, None], axis=0)
    u, _, vt = np.linalg.svd(((source - a) * weights[:, None]).T @ (target - b))
    correction = np.eye(2)
    correction[-1, -1] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ correction @ u.T
    return rotation, b - rotation @ a


def trajectory_arrays(rows, axes):
    return (
        np.array([row["timestamp"] for row in rows]),
        np.array([[row[axis] for axis in axes] for row in rows]),
    )


def resample_distance(times, positions, count):
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1))]
    keep = np.r_[True, np.diff(distance) > 1e-8]
    if np.count_nonzero(keep) < 3:
        raise ValueError("trajectory has insufficient motion")
    distance, times, positions = distance[keep], times[keep], positions[keep]
    query = np.linspace(0.0, distance[-1], count)
    points = np.column_stack([np.interp(query, distance, positions[:, k]) for k in range(3)])
    return np.interp(query, distance, times), points


def tangent_vectors(points):
    # A roughly 2% trajectory window reduces the influence of fix jitter.
    window = max(2, len(points) // 100)
    indexes = np.arange(len(points))
    direction = points[np.minimum(indexes + window, len(points) - 1), :2] - points[
        np.maximum(indexes - window, 0), :2]
    return direction / np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-9)


def match_ordered(source, target, band=0.2, direction_weight=0.5):
    """Banded DTW: spatial/tangent cost, no timestamps or unordered nearest points."""
    n, m = len(source), len(target)
    cost = cdist(source[:, :2], target[:, :2])
    if direction_weight:
        cost += direction_weight * cdist(tangent_vectors(source), tangent_vectors(target))
    distance = np.full((n + 1, m + 1), np.inf)
    previous = np.zeros((n, m), dtype=np.uint8)
    distance[0, 0] = 0.0
    for i in range(1, n + 1):
        center = (i - 1) * (m - 1) / (n - 1)
        first = max(1, int(math.floor(center - band * m)) + 1)
        last = min(m, int(math.ceil(center + band * m)) + 1)
        for j in range(first, last + 1):
            choices = (distance[i - 1, j - 1],
                       distance[i - 1, j] + 0.1, distance[i, j - 1] + 0.1)
            step = min(range(3), key=choices.__getitem__)
            distance[i, j] = cost[i - 1, j - 1] + choices[step]
            previous[i - 1, j - 1] = step
    if not np.isfinite(distance[n, m]):
        raise ValueError("no ordered matching path; check trajectory ranges or --match-band")
    pairs = []
    i, j = n, m
    while i > 0 and j > 0:
        pairs.append((i - 1, j - 1))
        step = previous[i - 1, j - 1]
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    return np.asarray(pairs[::-1], dtype=int)


def error_metrics(difference):
    xy = np.linalg.norm(difference[:, :2], axis=1)
    return {
        "pair_count": len(difference),
        "rmse_xy_m": float(np.sqrt(np.mean(xy ** 2))),
        "rmse_z_m": float(np.sqrt(np.mean(difference[:, 2] ** 2))),
        "rmse_3d_m": float(np.sqrt(np.mean(np.sum(difference ** 2, axis=1)))),
        "median_xy_error_m": float(np.median(xy)),
        "p95_xy_error_m": float(np.percentile(xy, 95)),
        "max_xy_error_m": float(xy.max()),
    }


def matrix_from_result(result):
    cosine, sine = math.cos(result["yaw_rad"]), math.sin(result["yaw_rad"])
    return np.array([
        [cosine, -sine, 0.0, result["tx"]],
        [sine, cosine, 0.0, result["ty"]],
        [0.0, 0.0, 1.0, result["tz"]],
        [0.0, 0.0, 0.0, 1.0],
    ])


def transform_points(points, result):
    matrix = matrix_from_result(result)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def infer_time_mapping(rtk_times, lio_times, reference_rtk, reference_lio, model):
    # Relative epochs avoid precision loss when clocks differ by hundreds of days.
    x, y = rtk_times - reference_rtk, lio_times - reference_lio
    if np.ptp(x) < 1.0:
        raise ValueError("matched moving trajectory covers less than one second")
    if model == "offset":
        scale, shift = 1.0, float(np.median(y - x))
    else:
        initial = np.linalg.lstsq(np.column_stack((x, np.ones(len(x)))), y, rcond=None)[0]
        fit = least_squares(lambda p: p[0] * x + p[1] - y, initial,
                            loss="soft_l1", f_scale=0.2, x_scale="jac")
        scale, shift = map(float, fit.x)
    if not math.isfinite(scale) or not 0.5 <= scale <= 2.0:
        raise ValueError("implausible time scale; check whether CSVs describe the same traversal")
    residual = reference_lio + shift + scale * x - lio_times
    return {
        "model": model,
        "formula": "t_lio = lio_reference_sec + shift_sec + scale * (t_rtk - rtk_reference_sec)",
        "rtk_reference_sec": float(reference_rtk),
        "lio_reference_sec": float(reference_lio),
        "scale": scale,
        "shift_sec": shift,
        "lio_minus_rtk_at_reference_sec": float(reference_lio - reference_rtk + shift),
        "inferred_drift_ppm": (scale - 1.0) * 1e6,
        "matched_time_residual_rmse_sec": float(np.sqrt(np.mean(residual ** 2))),
        "matched_time_residual_p95_sec": float(np.percentile(np.abs(residual), 95)),
        "note": "Inferred from spatial matches, not an independent clock calibration.",
    }


def map_times(times, mapping):
    return (mapping["lio_reference_sec"] + mapping["shift_sec"]
            + mapping["scale"] * (times - mapping["rtk_reference_sec"]))


def estimate_geometry_alignment(rtk_rows, lio_rows, min_pairs=20, min_displacement=5.0,
                                max_rmse=1.0, samples=600, band=0.2,
                                direction_weight=0.5, iterations=12, time_model="affine",
                                max_gap=0.2):
    if min(len(rtk_rows), len(lio_rows)) < min_pairs:
        raise ValueError(f"at least {min_pairs} valid rows per trajectory required")
    rt, rp = trajectory_arrays(rtk_rows, ("east", "north", "up"))
    lt, lp = trajectory_arrays(lio_rows, ("x", "y", "z"))
    coverage = float(np.max(np.linalg.norm(rp[:, :2] - rp[0, :2], axis=1)))
    if coverage < min_displacement:
        raise ValueError(f"RTK trajectory coverage {coverage:.3f} m is below {min_displacement:.3f} m")
    count = min(samples, len(rp), len(lp))
    rs_t, rs = resample_distance(rt, rp, count)
    ls_t, ls = resample_distance(lt, lp, count)
    for name, points in (("RTK", rs), ("LIO", ls)):
        eigenvalues = np.linalg.eigvalsh(np.cov(points[:, :2].T))
        if eigenvalues[0] / max(eigenvalues[1], 1e-9) < 0.002:
            raise ValueError(f"{name} trajectory is nearly straight; turns are required for joint time/space registration")
    rotation, translation = fit_rigid(rs[:, :2], ls[:, :2])
    old_pairs = None
    for iteration in range(iterations):
        aligned = rs.copy()
        aligned[:, :2] = rs[:, :2] @ rotation.T + translation
        pairs = match_ordered(aligned, ls, band, direction_weight)
        # Do not let DTW's repeated indices dominate the rigid fit.
        ca = np.bincount(pairs[:, 0], minlength=count)
        cb = np.bincount(pairs[:, 1], minlength=count)
        weights = 0.5 * (1.0 / ca[pairs[:, 0]] + 1.0 / cb[pairs[:, 1]])
        rotation, translation = fit_rigid(rs[pairs[:, 0], :2], ls[pairs[:, 1], :2], weights)
        if old_pairs is not None and np.array_equal(old_pairs, pairs):
            break
        old_pairs = pairs
    source, target = rs[pairs[:, 0]], ls[pairs[:, 1]]
    result = {
        "enabled": True,
        "method": "ordered_geometry_dtw",
        "yaw_rad": float(math.atan2(rotation[1, 0], rotation[0, 0])),
        "tx": float(translation[0]), "ty": float(translation[1]),
        "tz": float(np.average(target[:, 2] - source[:, 2], weights=weights)),
        "rtk_coverage_m": coverage,
        "iterations": iteration + 1,
        "resample_count": count,
        "match_band": band,
        "direction_weight_m": direction_weight,
        "max_time_gap_sec": max_gap,
        "frame_convention": "p_world = T_world_enu @ [east, north, up, 1]",
        "warnings": [
            "Both CSVs must cover the same traversal in the same order; trim partial runs explicitly.",
            "Errors are in-sample trajectory residuals, not independent sensor accuracy.",
            "This is an ENU-to-world frame transform, not a full antenna-to-body extrinsic calibration.",
            "RTK heading is not used; RTK TUM quaternions are identity placeholders.",
            "Use this transform only with the same ENU origin and LIO world frame.",
        ],
    }
    aligned_source = transform_points(source, result)
    metrics = error_metrics(aligned_source - target)
    if metrics["rmse_xy_m"] > max_rmse:
        raise ValueError(f"geometry XY RMSE {metrics['rmse_xy_m']:.3f} m exceeds the {max_rmse:.3f} m limit")
    result.update(metrics)
    result["geometry_errors"] = metrics
    # One time observation per spatial source sample, not per DTW repetition.
    unique = np.unique(pairs[:, 0])
    matched_lt = np.array([np.median(ls_t[pairs[pairs[:, 0] == i, 1]]) for i in unique])
    mapping = infer_time_mapping(rs_t[unique], matched_lt, rt[0], lt[0], time_model)
    # Split-route fits expose timing estimates biased by trajectory deformation.
    if time_model == "affine" and len(unique) >= 40:
        midpoint = len(unique) // 2
        local_drifts = []
        for section in (slice(None, midpoint), slice(midpoint, None)):
            local = infer_time_mapping(
                rs_t[unique][section], matched_lt[section], rt[0], lt[0], time_model)
            local_drifts.append(local["inferred_drift_ppm"])
        mapping["half_trajectory_drift_ppm"] = local_drifts
        if abs(local_drifts[0] - local_drifts[1]) > 500:
            result["warnings"].append(
                "First/second half time-scale estimates disagree by >500 ppm; "
                "the apparent drift may include spatial distortion or correspondence error.")
    result["time_mapping"] = mapping
    corrected = map_times(rt, mapping)
    # Interpolate only within bounded LIO brackets, never across missing data.
    right = np.searchsorted(lt, corrected, side="left")
    clipped = np.clip(right, 1, len(lt) - 1)
    exact = (right < len(lt)) & (lt[np.minimum(right, len(lt) - 1)] == corrected)
    valid = ((corrected >= lt[0]) & (corrected <= lt[-1])
             & ((lt[clipped] - lt[clipped - 1] <= max_gap) | exact))
    world = transform_points(rp, result)
    interpolated = np.column_stack([np.interp(corrected[valid], lt, lp[:, k]) for k in range(3)])
    if np.count_nonzero(valid) < min_pairs:
        raise ValueError("too few time-corrected pairs within --max-gap")
    result["time_synchronized_errors"] = error_metrics(world[valid] - interpolated)
    if result["time_synchronized_errors"]["rmse_xy_m"] > max_rmse:
        result["warnings"].append(
            "Time-synchronized RMSE exceeds --max-rmse: geometry passes, but a single time model "
            "does not explain all spatial matches; inspect errors before treating clocks as aligned.")
    if abs(mapping["inferred_drift_ppm"]) > 1000:
        result["warnings"].append("Inferred time scale differs by >1000 ppm; spatial drift or wrong matches may bias it.")
    diagnostics = {
        "source": source, "target": target, "aligned_source": aligned_source,
        "match_rtk_time": rs_t[pairs[:, 0]], "match_lio_time": ls_t[pairs[:, 1]],
        "rtk_time": rt, "corrected_time": corrected, "world": world,
        "valid": valid, "interpolated_lio": interpolated, "lio_time": lt, "lio_position": lp,
        "time_fit_rtk": rs_t[unique], "time_fit_lio": matched_lt,
    }
    return result, diagnostics


def output_paths(output, geometry=True, plots=True):
    suffixes = [".report.json", ".matrix.txt"]
    if geometry:
        suffixes += [".matches.csv", ".errors.csv", ".lio.tum", ".rtk_world.tum"]
        if plots:
            suffixes += [".overview.png"]
    return [Path(output)] + [Path(output).with_suffix(suffix) for suffix in suffixes]


def write_outputs(result, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        stream.write("rtk_odom_recorder:\n  ros__parameters:\n")
        stream.write("    alignment:\n")
        for key in ("enabled", "yaw_rad", "tx", "ty", "tz"):
            value = result[key]
            stream.write(f"      {key}: {str(value).lower() if isinstance(value, bool) else format(value, '.12g')}\n")
    report = {key: value for key, value in result.items() if key != "enabled"}
    report_path = output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.savetxt(output.with_suffix(".matrix.txt"), matrix_from_result(result), fmt="%.12f",
               header="T_world_enu: column vectors; p_world = T_world_enu @ p_enu")
    return output, report_path


def write_diagnostics(result, data, lio_rows, output, plots=True):
    output = Path(output)
    difference = data["aligned_source"] - data["target"]
    delta = data["match_lio_time"] - data["match_rtk_time"]
    model_delta = map_times(data["match_rtk_time"], result["time_mapping"]) - data["match_rtk_time"]
    matched = np.column_stack((
        data["match_rtk_time"], data["match_lio_time"], delta, model_delta,
        delta - model_delta, data["source"], data["aligned_source"], data["target"],
        difference, np.linalg.norm(difference[:, :2], axis=1),
    ))
    np.savetxt(output.with_suffix(".matches.csv"), matched, delimiter=",", fmt="%.9f",
               comments="", header="rtk_timestamp,lio_timestamp,lio_minus_rtk_sec,model_delta_sec,"
               "time_residual_sec,east,north,up,rtk_world_x,rtk_world_y,rtk_world_z,"
               "lio_x,lio_y,lio_z,dx,dy,dz,error_xy_m")
    valid = data["valid"]
    difference = data["world"][valid] - data["interpolated_lio"]
    errors = np.column_stack((
        data["rtk_time"][valid], data["corrected_time"][valid],
        data["world"][valid], data["interpolated_lio"], difference,
        np.linalg.norm(difference[:, :2], axis=1), np.linalg.norm(difference, axis=1),
    ))
    np.savetxt(output.with_suffix(".errors.csv"), errors, delimiter=",", fmt="%.9f",
               comments="", header="rtk_timestamp,corrected_lio_timestamp,rtk_world_x,rtk_world_y,"
               "rtk_world_z,lio_x,lio_y,lio_z,dx,dy,dz,error_xy_m,error_3d_m")
    if all(all(k in row for k in ("qx", "qy", "qz", "qw")) for row in lio_rows):
        quaternion = np.array([[row[k] for k in ("qx", "qy", "qz", "qw")] for row in lio_rows])
        norm = np.linalg.norm(quaternion, axis=1, keepdims=True)
        if np.any(norm < 1e-9):
            raise ValueError("LIO contains zero quaternions; cannot export TUM")
        quaternion = quaternion / norm
    else:
        quaternion = np.tile([0.0, 0.0, 0.0, 1.0], (len(lio_rows), 1))
    np.savetxt(output.with_suffix(".lio.tum"), np.column_stack((
        data["lio_time"], data["lio_position"], quaternion)), fmt="%.9f",
        header="timestamp x y z qx qy qz qw; original LIO timestamps")
    np.savetxt(output.with_suffix(".rtk_world.tum"), np.column_stack((
        data["corrected_time"], data["world"],
        np.tile([0.0, 0.0, 0.0, 1.0], (len(data["world"]), 1)))), fmt="%.9f",
        header="timestamp x y z qx qy qz qw; mapped LIO time; orientation UNKNOWN (identity placeholder)")
    if plots:
        plot_diagnostics(result, data, output.with_suffix(".overview.png"))


def plot_diagnostics(result, data, output):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/rtk_alignment_matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(data["lio_position"][:, 0], data["lio_position"][:, 1], label="LIO world")
    ax.plot(data["world"][:, 0], data["world"][:, 1], label="RTK transformed", alpha=0.8)
    ax.scatter(*data["lio_position"][0, :2], s=40, marker="s", label="LIO start")
    ax.set(xlabel="world x (m)", ylabel="world y (m)", title="Ordered geometry registration")
    ax.axis("equal")
    ax.legend()
    elapsed = data["match_rtk_time"] - data["rtk_time"][0]
    ax = axes[0, 1]
    residual = data["aligned_source"] - data["target"]
    ax.plot(elapsed, np.linalg.norm(residual[:, :2], axis=1), label="XY")
    ax.plot(elapsed, residual[:, 2], label="signed Z", alpha=0.7)
    ax.set(xlabel="RTK elapsed (s)", ylabel="error (m)", title="Geometry residuals (in-sample)")
    ax.legend()
    ax = axes[1, 0]
    mapping = result["time_mapping"]
    base = mapping["lio_reference_sec"] - mapping["rtk_reference_sec"]
    ax.scatter(data["time_fit_rtk"] - data["rtk_time"][0],
               (data["time_fit_lio"] - mapping["lio_reference_sec"])
               - (data["time_fit_rtk"] - mapping["rtk_reference_sec"]),
               s=5, alpha=0.5, label="geometric match delta")
    x = np.array([0.0, float(data["rtk_time"][-1] - data["rtk_time"][0])])
    ax.plot(x, mapping["shift_sec"] + (mapping["scale"] - 1.0) * x, color="black", label="time fit")
    ax.set(xlabel="RTK elapsed (s)", ylabel=f"LIO - RTK minus ({base:.3f} s)",
           title=f"Inferred time mapping: {mapping['inferred_drift_ppm']:.1f} ppm")
    ax.legend()
    ax = axes[1, 1]
    diff = data["world"][data["valid"]] - data["interpolated_lio"]
    ax.plot(data["rtk_time"][data["valid"]] - data["rtk_time"][0],
            np.linalg.norm(diff[:, :2], axis=1), label="XY after time mapping")
    ax.set(xlabel="RTK elapsed (s)", ylabel="error (m)",
           title="Time-synchronized residuals (same data, no additional spatial fit)")
    ax.legend()
    for ax in axes.flat:
        ax.grid(alpha=0.25)
    fig.suptitle(
        f"yaw={math.degrees(result['yaw_rad']):.5f} deg | "
        f"geometry XY RMSE={result['rmse_xy_m']:.3f} m | "
        f"synchronized XY RMSE={result['time_synchronized_errors']['rmse_xy_m']:.3f} m")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def cropped(rows, start, end):
    if not rows:
        return []
    epoch = rows[0]["timestamp"]
    return [row for row in rows if start <= row["timestamp"] - epoch <= end]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lio", required=True, help="lio_odom.csv from one recording")
    parser.add_argument("--rtk", required=True, help="rtk_fix.csv from the same recording")
    parser.add_argument("--output", required=True, help="alignment ROS parameter YAML")
    parser.add_argument("--method", choices=("geometry", "timestamp"), default="geometry",
                        help="geometry estimates correspondence before time; timestamp uses original clocks")
    parser.add_argument("--samples", type=int, default=600, help="distance samples per trajectory (20..2000)")
    parser.add_argument("--match-band", type=float, default=0.2, help="DTW band as trajectory progress fraction")
    parser.add_argument("--direction-weight", type=float, default=0.5, help="tangent disagreement cost in meters")
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--time-model", choices=("offset", "affine"), default="affine")
    parser.add_argument("--rtk-min-status", type=int, default=2)
    for name in ("rtk", "lio"):
        parser.add_argument(f"--{name}-start", type=float, default=0.0, help="crop start, seconds from first valid row")
        parser.add_argument(f"--{name}-end", type=float, default=math.inf, help="crop end, seconds from first valid row")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--max-gap", type=float, default=0.2,
                        help="maximum LIO interpolation bracket in seconds")
    parser.add_argument("--min-pairs", type=int, default=20)
    parser.add_argument("--min-displacement", type=float, default=5.0,
                        help="minimum RTK distance from its start point in meters")
    parser.add_argument("--max-rmse", type=float, default=1.0,
                        help="maximum geometry XY RMSE; synchronized RMSE above this emits a warning")
    args = parser.parse_args()
    try:
        finite_positive = (args.max_gap, args.min_displacement, args.max_rmse)
        if not all(math.isfinite(x) and x > 0 for x in finite_positive):
            raise ValueError("max-gap, min-displacement and max-rmse must be positive and finite")
        if not (20 <= args.samples <= 2000 and 3 <= args.min_pairs <= args.samples
                and 1 <= args.iterations <= 100 and 0 < args.match_band <= 1
                and math.isfinite(args.direction_weight) and args.direction_weight >= 0):
            raise ValueError("invalid samples/min-pairs/iterations/match-band/direction-weight")
        if any(not (0 <= start < end) for start, end in (
                (args.rtk_start, args.rtk_end), (args.lio_start, args.lio_end))):
            raise ValueError("crop ranges must satisfy 0 <= start < end")
        existing_outputs = [path for path in output_paths(
            args.output, args.method == "geometry", not args.no_plots) if path.exists()]
        if existing_outputs:
            raise ValueError(
                f"{', '.join(map(str, existing_outputs))} already exists; choose a new output path "
                "for this recording so an older calibration cannot be mistaken for this result"
            )
        with open(args.lio, newline="", encoding="utf-8") as stream:
            columns = csv.DictReader(stream).fieldnames or []
        quaternion_columns = ("qx", "qy", "qz", "qw")
        required = ("timestamp", "x", "y", "z")
        if all(key in columns for key in quaternion_columns):
            required += quaternion_columns
        lio_rows = cropped(read_rows(args.lio, required), args.lio_start, args.lio_end)
        rtk_rows = read_rows(args.rtk, ("timestamp", "status", "east", "north", "up"))
        rtk_rows = cropped([row for row in rtk_rows if row["status"] >= args.rtk_min_status],
                           args.rtk_start, args.rtk_end)
        if min(len(rtk_rows), len(lio_rows)) < args.min_pairs:
            raise ValueError(f"at least {args.min_pairs} valid rows per trajectory required")
        if args.method == "geometry":
            result, diagnostics = estimate_geometry_alignment(
                rtk_rows, lio_rows, args.min_pairs, args.min_displacement, args.max_rmse,
                args.samples, args.match_band, args.direction_weight, args.iterations,
                args.time_model, args.max_gap)
        else:
            result = estimate_alignment(
                rtk_rows, lio_rows, args.max_gap, args.min_pairs,
                args.min_displacement, args.max_rmse)
        result["input_lio"] = str(Path(args.lio).resolve())
        result["input_rtk"] = str(Path(args.rtk).resolve())
        result["rtk_min_status"] = args.rtk_min_status
        result["T_world_enu"] = matrix_from_result(result).tolist()
        # Publish the usable YAML only after all reports/exports have succeeded.
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".alignment-", dir=destination.parent) as staging:
            temporary = Path(staging) / destination.name
            write_outputs(result, temporary)
            if args.method == "geometry":
                write_diagnostics(result, diagnostics, lio_rows, temporary, not args.no_plots)
            for artifact in sorted(Path(staging).iterdir(), key=lambda p: p == temporary):
                artifact.rename(destination.parent / artifact.name)
        yaml_path, report_path = destination, destination.with_suffix(".report.json")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"alignment: {yaml_path}")
    print(f"quality report: {report_path}")
    print(
        f"pairs={result['pair_count']} coverage={result['rtk_coverage_m']:.3f} m "
        f"rmse_xy={result['rmse_xy_m']:.3f} m rmse_z={result['rmse_z_m']:.3f} m"
    )
    print(f"yaw={math.degrees(result['yaw_rad']):.9f} deg")
    print("T_world_enu (column vectors, translation in meters):")
    print(np.array2string(matrix_from_result(result), precision=9, suppress_small=True))
    if "time_mapping" in result:
        mapping = result["time_mapping"]
        print(f"time: t_lio = {mapping['lio_reference_sec']:.9f} "
              f"+ ({mapping['shift_sec']:.9f}) "
              f"+ {mapping['scale']:.12f} * (t_rtk - {mapping['rtk_reference_sec']:.9f})")
        print(f"inferred drift={mapping['inferred_drift_ppm']:.3f} ppm; "
              f"synchronized XY RMSE={result['time_synchronized_errors']['rmse_xy_m']:.3f} m")
        for warning in result["warnings"]:
            print(f"NOTE: {warning}")


if __name__ == "__main__":
    main()
