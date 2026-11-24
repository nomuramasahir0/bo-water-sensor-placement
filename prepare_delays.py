"""Build the scenario-by-junction detection-delay matrix with WNTR/EPANET."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir

import numpy as np
import yaml


def _net3_path(wntr, configured_path: str | None) -> Path:
    if configured_path:
        path = Path(configured_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    # WNTR releases have exposed their example networks through different paths.
    try:
        path = Path(wntr.library.networks.get_path("Net3"))
        if path.is_file():
            return path
    except (AttributeError, KeyError, TypeError):
        pass
    package_dir = Path(wntr.__file__).resolve().parent
    matches = list(package_dir.rglob("Net3.inp"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        "Net3.inp was not bundled with this WNTR build; set network_file in config.yaml"
    )


def _seconds(hours: float) -> int:
    return round(hours * 3600)


def _set_time_options(wn, cfg: dict) -> None:
    quality_step = round(cfg["quality_timestep_minutes"] * 60)
    wn.options.time.duration = _seconds(cfg["duration_hours"])
    wn.options.time.hydraulic_timestep = round(cfg["hydraulic_timestep_minutes"] * 60)
    wn.options.time.quality_timestep = quality_step
    wn.options.time.report_timestep = round(cfg["report_timestep_minutes"] * 60)
    wn.options.time.pattern_timestep = quality_step


def select_propagation_capable_sources(wntr, inp_path: Path, cfg: dict, filter_cfg: dict):
    """Select sources with an outward hydraulic path during injection windows."""
    wn = wntr.network.WaterNetworkModel(str(inp_path))
    candidates = list(wn.junction_name_list)
    if filter_cfg.get("mode", "all") == "all":
        return candidates, {name: {"eligible": True} for name in candidates}
    if filter_cfg["mode"] != "propagation_capable":
        raise ValueError(f"Unknown source_filter mode: {filter_cfg['mode']}")

    _set_time_options(wn, cfg)
    with TemporaryDirectory(prefix="graphcombo-epanet-hydraulic-") as temp_dir:
        prefix = str(Path(temp_dir) / "baseline")
        results = wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=prefix)
    flows = results.link["flowrate"]
    graph = wn.to_graph().to_undirected()
    threshold = float(filter_cfg["min_outgoing_flow_m3_s"])
    duration = _seconds(cfg["injection_duration_hours"])
    details = {}

    for source in candidates:
        links = [wn.get_link(name) for name in wn.get_links_for_node(source)]
        maxima = {}
        for start_hour in cfg["injection_start_hours"]:
            start, stop = _seconds(start_hour), _seconds(start_hour) + duration
            window = flows.loc[(flows.index >= start) & (flows.index < stop)]
            max_outgoing = 0.0
            for link in links:
                link_flow = window[link.name].to_numpy()
                if link.start_node_name == source:
                    outward = link_flow
                else:
                    outward = -link_flow
                if len(outward):
                    max_outgoing = max(max_outgoing, float(np.max(outward)))
            maxima[str(start_hour)] = max_outgoing

        flow_checks = [value >= threshold for value in maxima.values()]
        flow_eligible = (
            all(flow_checks)
            if filter_cfg.get("require_all_start_times", True)
            else any(flow_checks)
        )
        degree = int(graph.degree(source))
        eligible = degree >= int(filter_cfg["min_degree"]) and flow_eligible
        details[source] = {
            "eligible": eligible,
            "degree": degree,
            "max_outgoing_flow_m3_s_by_start_hour": maxima,
        }

    selected = [name for name in candidates if details[name]["eligible"]]
    if not selected:
        raise RuntimeError("source_filter excluded every junction")
    return selected, details


def build_candidate_graphs(wntr, inp_path: Path, cfg: dict, graph_cfg: dict):
    """Build binary and representative travel-time candidate graphs."""
    wn = wntr.network.WaterNetworkModel(str(inp_path))
    _set_time_options(wn, cfg)
    with TemporaryDirectory(prefix="graphcombo-epanet-graph-") as temp_dir:
        prefix = str(Path(temp_dir) / "baseline")
        results = wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=prefix)
    flows = results.link["flowrate"]
    candidates = set(wn.junction_name_list)
    min_flow = float(graph_cfg["min_abs_flow_m3_s"])
    edges = {}

    for name in wn.pipe_name_list:
        pipe = wn.get_link(name)
        u, v = pipe.start_node_name, pipe.end_node_name
        if u not in candidates or v not in candidates:
            continue
        valid = np.abs(flows[name].to_numpy()) >= min_flow
        if not np.any(valid):
            continue
        area = np.pi * float(pipe.diameter) ** 2 / 4.0
        speed = np.abs(flows[name].to_numpy()[valid]) / area
        travel_time = float(np.median(float(pipe.length) / speed))
        key = tuple(sorted((u, v)))
        # Parallel pipes are represented by the fastest transport connection.
        edges[key] = min(edges.get(key, float("inf")), travel_time)

    if not edges:
        raise RuntimeError("No candidate-to-candidate pipe edges were constructed")
    scale = float(np.median(list(edges.values())))
    names = np.asarray(list(edges.keys()), dtype=str)
    times = np.asarray(list(edges.values()), dtype=np.float64)
    metadata = {
        "edge_count": len(edges),
        "travel_time_scale_seconds": scale,
        "min_abs_flow_m3_s": min_flow,
        "parallel_pipe_aggregation": "minimum_travel_time",
        "pumps_and_valves": "omitted",
    }
    return names, times, metadata


def simulate_scenario(wntr, inp_path: Path, source: str, start_hour: float, cfg: dict):
    wn = wntr.network.WaterNetworkModel(str(inp_path))
    _set_time_options(wn, cfg)
    duration = int(wn.options.time.duration)
    quality_step = round(cfg["quality_timestep_minutes"] * 60)
    wn.options.quality.parameter = "CHEMICAL"

    start = _seconds(start_hour)
    stop = start + _seconds(cfg["injection_duration_hours"])
    multipliers = [
        1.0 if start <= t < stop else 0.0
        for t in range(0, duration + quality_step, quality_step)
    ]
    pattern_name = "contamination_schedule"
    wn.add_pattern(pattern_name, multipliers)
    wn.add_source(
        "contamination_source",
        source,
        cfg["source_type"],
        cfg["source_strength"],
        pattern_name,
    )

    # WNTR otherwise leaves temp.inp/temp.rpt/temp.bin in the working directory.
    with TemporaryDirectory(prefix="graphcombo-epanet-") as temp_dir:
        prefix = str(Path(temp_dir) / "scenario")
        results = wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=prefix)
    return results.node["quality"], start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--limit-sources", type=int, default=None, help="Smoke-test subset")
    args = parser.parse_args()

    # WNTR imports matplotlib; keep its caches out of the user's home directory.
    cache_root = Path(gettempdir()) / "graphcombo-water-sensor-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

    import wntr

    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    sim_cfg = config["simulation"]
    inp_path = _net3_path(wntr, config.get("network_file"))
    base = wntr.network.WaterNetworkModel(str(inp_path))
    candidates = list(base.junction_name_list)
    sources, source_filter_details = select_propagation_capable_sources(
        wntr, inp_path, sim_cfg, config.get("source_filter", {"mode": "all"})
    )
    edge_names, travel_times, graph_metadata = build_candidate_graphs(
        wntr, inp_path, sim_cfg, config["travel_time_graph"]
    )
    print(f"Selected {len(sources)}/{len(candidates)} propagation-capable sources")
    sources = sources[: args.limit_sources] if args.limit_sources else sources
    scenarios = [(source, hour) for source in sources for hour in sim_cfg["injection_start_hours"]]
    penalty = float(sim_cfg["undetected_penalty_hours"])
    delays = np.full((len(scenarios), len(candidates)), penalty, dtype=np.float32)

    for row, (source, start_hour) in enumerate(scenarios):
        print(f"[{row + 1}/{len(scenarios)}] source={source} start={start_hour:g}h")
        quality, start_seconds = simulate_scenario(wntr, inp_path, source, start_hour, sim_cfg)
        after_start = quality.loc[quality.index >= start_seconds, candidates]
        for col, candidate in enumerate(candidates):
            detected = after_start.index[
                after_start[candidate].to_numpy() >= sim_cfg["detection_threshold"]
            ]
            if len(detected):
                delays[row, col] = (float(detected[0]) - start_seconds) / 3600.0

    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix_file = out_dir / "detection_delays.npz"
    np.savez_compressed(
        matrix_file,
        delays_hours=delays,
        candidate_names=np.asarray(candidates),
        scenario_sources=np.asarray([s for s, _ in scenarios]),
        scenario_start_hours=np.asarray([h for _, h in scenarios]),
        candidate_graph_edges=edge_names,
        travel_times_seconds=travel_times,
    )
    metadata = {
        "network_file": str(inp_path),
        "candidate_count": len(candidates),
        "scenario_count": len(scenarios),
        "shape": list(delays.shape),
        "selected_sources": sources,
        "source_filter_details": source_filter_details,
        "candidate_graph": graph_metadata,
        "config": config,
    }
    (out_dir / "detection_delays.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {matrix_file} with shape {delays.shape}")


if __name__ == "__main__":
    main()
