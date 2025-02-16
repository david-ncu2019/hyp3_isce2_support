#!/usr/bin/env python3
import argparse
import logging
import os
from datetime import datetime

import asf_search
import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import xyzservices.providers as xyz
import yaml

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file '{config_path}' not found.")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def check_required_config(config: dict, keys: list):
    """Ensure all required keys are in the config; if not, raise an error."""
    for key in keys:
        if key not in config:
            raise ValueError(
                f"Missing required configuration parameter: '{key}'. Please input it in the config file."
            )


def check_required_group_fields(group: dict, group_index: int):
    """Ensure each burst_id_group contains the required keys."""
    required = ["iw", "prefix", "range"]
    for key in required:
        if key not in group:
            raise ValueError(
                f"Missing required field '{key}' in burst_id_groups at index {group_index}."
            )
    for subkey in ["start", "end"]:
        if subkey not in group["range"]:
            raise ValueError(
                f"Missing required field 'range.{subkey}' in burst_id_groups at index {group_index}."
            )


def generate_burst_ids_from_groups(relative_orbit, burst_groups: list) -> list:
    """
    Generates a complete burst ID list from the burst groups.

    The full burst ID is constructed as:
      {relative_orbit}_{prefix}{num}_IW{iw}
    The relative orbit is zero-padded to three digits.
    """
    # Convert relative_orbit to a zero-padded string.
    relative_orbit_str = str(relative_orbit).zfill(3)
    burst_ids = []
    for i, group in enumerate(burst_groups):
        check_required_group_fields(group, i)
        iw = str(group["iw"])
        prefix = str(group["prefix"])
        start = int(group["range"]["start"])
        end = int(group["range"]["end"])
        for num in range(start, end + 1):
            burst_ids.append(f"{relative_orbit_str}_{prefix}{num}_IW{iw}")
    return burst_ids


def search_bursts(
    burst_id_list: list,
    start_date: str,
    end_date: str,
    beam_mode: str,
    relative_orbit,
    flight_direction: str,
    polarization: str,
) -> gpd.GeoDataFrame:
    """
    For each full burst ID in burst_id_list, perform an ASF search and concatenate the results.
    """
    all_bursts = gpd.GeoDataFrame()
    for full_burst_id in burst_id_list:
        logger.info(f"Searching for burst with fullBurstID: {full_burst_id}")
        search_result = asf_search.search(
            start=datetime.fromisoformat(start_date),
            end=datetime.fromisoformat(end_date),
            beamMode=beam_mode,
            relativeOrbit=relative_orbit,
            flightDirection=flight_direction,
            polarization=polarization,
            fullBurstID=full_burst_id,
        )
        # Convert the search result to a GeoDataFrame (assuming EPSG:4326)
        single_burst_geodata = gpd.GeoDataFrame.from_features(
            search_result.geojson(), crs=4326
        )
        all_bursts = pd.concat([all_bursts, single_burst_geodata], ignore_index=True)
    return all_bursts


def export_scene_coverage(
    gdf,
    fig_savename="burst_coverage.png",
    column=None,
    alpha=0.3,
    add_basemap=True,
    user_epsg=None,
    figsize=(8.3, 11.7),
):
    if gdf.empty:
        print("Warning: The GeoDataFrame is empty. Nothing to plot.")
        return

    fig, ax = plt.subplots(figsize=figsize)

    # Determine which CRS to use
    original_crs = gdf.crs
    if user_epsg:
        gdf = gdf.to_crs(epsg=user_epsg)
    elif add_basemap and (original_crs is None or original_crs.to_epsg() != 3857):
        gdf = gdf.to_crs(epsg=3857)

    # Plot the GeoDataFrame
    gdf.plot(column=column, ax=ax, alpha=alpha, edgecolor="black")

    # Add basemap if requested
    if add_basemap:
        try:
            ctx.add_basemap(
                ax, crs=gdf.crs, source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.75
            )
        except AttributeError:
            ctx.add_basemap(
                ax, crs=gdf.crs, source=ctx.providers.CartoDB.Positron, alpha=0.75
            )
    # Format plot
    ax.set_axis_off()

    plt.savefig(
        fig_savename,
        dpi=300,
        transparent=False,
        facecolor="w",
        edgecolor="w",
        bbox_inches="tight",
    )
    print(f"Figure saved at: {fig_savename}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Search for Sentinel-1 bursts using parameters from a YAML configuration file."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    # Check that all required parameters are present.
    required_keys = [
        "relative_orbit",
        "start_date",
        "end_date",
        "beam_mode",
        "flight_direction",
        "polarization",
        "burst_id_groups",
    ]
    check_required_config(config, required_keys)

    relative_orbit = config["relative_orbit"]
    start_date = config["start_date"]
    end_date = config["end_date"]
    beam_mode = config["beam_mode"]
    flight_direction = config["flight_direction"]
    polarization = config["polarization"]
    burst_groups = config["burst_id_groups"]

    # Generate full burst ID list from all burst groups.
    burst_id_list = generate_burst_ids_from_groups(relative_orbit, burst_groups)
    logger.info(f"Generated burst ID list: {burst_id_list}")

    # Perform the search for each full burst ID.
    all_bursts = search_bursts(
        burst_id_list,
        start_date,
        end_date,
        beam_mode,
        relative_orbit,
        flight_direction,
        polarization,
    )

    output_burst_df = all_bursts[
        [
            "centerLat",
            "centerLon",
            "stopTime",
            "fileID",
            "flightDirection",
            "pathNumber",
            "processingLevel",
            "startTime",
            "sceneName",
            "platform",
            "orbit",
            "polarization",
            "sensor",
            "groupID",
            "pgeVersion",
            "beamModeType",
            "burst",
        ]
    ]

    output_burst_df = output_burst_df.sort_values(by="stopTime")

    output_savename = f"{flight_direction}_{str(relative_orbit).zfill(3)}_{beam_mode}_{polarization}_{start_date}_{end_date}.xlsx"

    output_burst_df.to_excel(output_savename, index=False)

    logger.info(f"Burst search complete. File saved at {output_savename}")

    acquisition_dates = list(
        set(([ele.split("T")[0] for ele in all_bursts["startTime"]]))
    )
    select_bursts2plot = all_bursts[
        all_bursts["startTime"].str.contains(acquisition_dates[0], na=False)
    ]
    # Plot the data
    export_scene_coverage(
        select_bursts2plot,
        column="fileID",
        fig_savename=output_savename.replace(".xlsx", ".png"),
    )


if __name__ == "__main__":
    main()