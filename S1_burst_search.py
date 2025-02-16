#!/usr/bin/env python3
# -----------------------: Imports
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

# -----------------------: Logging Setup
# Set up basic logging to both console and file with a timestamped log filename.
log_filename = f"S1_burst_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

# Remove existing handlers if rerunning in an interactive environment (e.g., Jupyter)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        # logging.FileHandler(log_filename),  # Log to a file
        logging.StreamHandler(),  # Log to the console
    ],
)

logger = logging.getLogger(__name__)

# -----------------------: Configuration Loading and Validation Functions


def load_config(config_path: str) -> dict:
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration dictionary loaded from the file.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file '{config_path}' not found.")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def check_required_config(config: dict, required_keys: dict):
    """
    Ensure all required sections and keys exist in the configuration.

    Args:
        config (dict): The configuration dictionary.
        required_keys (dict): Dictionary where keys are section names and values are lists of required keys.

    Raises:
        ValueError: If a required section or key is missing.
    """
    for section, keys in required_keys.items():
        if section not in config:
            raise ValueError(
                f"Missing required section: '{section}' in the config file."
            )
        for key in keys:
            if key not in config[section]:
                raise ValueError(
                    f"Missing required key '{key}' in section '{section}'."
                )


def check_required_group_fields(group: dict, group_index: int):
    """
    Ensure each burst_id_group contains the required keys.

    Args:
        group (dict): A burst group dictionary.
        group_index (int): Index of the group in the list.

    Raises:
        ValueError: If a required field or subfield is missing.
    """
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


# -----------------------: Burst ID Generation Function


def generate_burst_ids_from_groups(relative_orbit, burst_groups: list) -> list:
    """
    Generates a complete burst ID list from the burst groups.

    The full burst ID is constructed as:
      {relative_orbit}_{prefix}{num}_IW{iw}
    The relative orbit is zero-padded to three digits.

    Args:
        relative_orbit: Relative orbit number.
        burst_groups (list): List of burst group dictionaries.

    Returns:
        list: List of generated burst IDs.
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


# -----------------------: Burst Search Function


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

    Args:
        burst_id_list (list): List of full burst IDs.
        start_date (str): Start date in ISO format.
        end_date (str): End date in ISO format.
        beam_mode (str): Beam mode.
        relative_orbit: Relative orbit number.
        flight_direction (str): Flight direction.
        polarization (str): Polarization.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing concatenated search results.
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


# -----------------------: Scene Coverage Export Function


def export_scene_coverage(
    gdf,
    fig_savename="burst_coverage.png",
    column=None,
    alpha=0.3,
    add_basemap=True,
    user_epsg=None,
    figsize=(8.3, 11.7),
):
    """
    Exports a plot of the scene coverage from the provided GeoDataFrame.

    Args:
        gdf (GeoDataFrame): GeoDataFrame containing scene data.
        fig_savename (str): Filename for the saved figure.
        column: Column used for plotting.
        alpha (float): Transparency level.
        add_basemap (bool): Whether to add a basemap.
        user_epsg: User specified EPSG code.
        figsize (tuple): Figure size.
    """
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
    logger.info(f"Figure saved at: {fig_savename}")

    plt.close()


# -----------------------: Search Output Summary Function


def summarize_search_output(input_df):
    """
    Summarizes the search output by calculating statistics of acquisition dates.

    Args:
        input_df: DataFrame containing search output.
    """
    # Ensure 'stopTime' is in datetime format (in case it is not)
    input_df["stopTime"] = pd.to_datetime(input_df["stopTime"], errors="coerce")

    # Extract the unique dates by converting 'stopTime' to the date part only
    unique_acquisition_dates = input_df["stopTime"].dt.date.unique()

    # Calculate time intervals (in days) between the unique acquisition dates
    time_intervals_in_days = pd.Series(
        pd.to_datetime(unique_acquisition_dates).diff().dropna()
    ).dt.days

    # Calculate the statistics
    date_count = len(unique_acquisition_dates)
    max_time_interval = time_intervals_in_days.max()
    min_time_interval = time_intervals_in_days.min()

    logger.info(f"Number of acquisitions: {date_count}")
    logger.info(f"First acquisition: {unique_acquisition_dates[0]}")
    logger.info(f"Last acquisition: {unique_acquisition_dates[-1]}")
    logger.info(f"Max time interval: {max_time_interval} days")
    logger.info(f"Min time interval: {min_time_interval} days")


# -----------------------: Main Function


def main():
    # -----------------------: Argument Parsing
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

    # -----------------------: Load Configuration
    config = load_config(args.config)
    # Check that all required parameters are present.
    # Define required sections and keys
    required_keys = {
        "SEARCH": [
            "relative_orbit",
            "start_date",
            "end_date",
            "beam_mode",
            "flight_direction",
            "polarization",
            "burst_id_groups",
        ],
    }
    check_required_config(config, required_keys)

    # -----------------------: Extract SEARCH Parameters
    search_params = config["SEARCH"]
    relative_orbit = search_params["relative_orbit"]
    start_date = search_params["start_date"]
    end_date = search_params["end_date"]
    beam_mode = search_params["beam_mode"]
    flight_direction = search_params["flight_direction"]
    polarization = search_params["polarization"]
    burst_groups = search_params["burst_id_groups"]

    # Validate burst groups
    for index, group in enumerate(burst_groups):
        check_required_group_fields(group, index)

    # -----------------------: Generate Burst IDs
    burst_id_list = generate_burst_ids_from_groups(relative_orbit, burst_groups)
    logger.info(f"Generated burst ID list: {burst_id_list}")

    # -----------------------: Search Bursts
    all_bursts = search_bursts(
        burst_id_list,
        start_date,
        end_date,
        beam_mode,
        relative_orbit,
        flight_direction,
        polarization,
    )

    # -----------------------: Process and Save Search Output
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

    # -----------------------: Plot Scene Coverage
    acquisition_dates = list(
        set(([ele.split("T")[0] for ele in all_bursts["startTime"]]))
    )
    select_bursts2plot = all_bursts[
        all_bursts["startTime"].str.contains(acquisition_dates[0], na=False)
    ]
    export_scene_coverage(
        select_bursts2plot,
        column="fileID",
        fig_savename=output_savename.replace(".xlsx", ".png"),
    )

    # -----------------------: Summarize Search Output
    summarize_search_output(output_burst_df)


if __name__ == "__main__":
    main()