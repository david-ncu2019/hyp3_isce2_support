#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
import yaml


# -----------------------: Logging Setup
def setup_logging() -> logging.Logger:
    log_filename = f"insar_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            # Uncomment the next line to log to file:
            # logging.FileHandler(log_filename),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


logger = setup_logging()


# -----------------------: Configuration Functions
def load_config(config_path: str) -> dict:
    logger.info(f"Loading configuration from {config_path}...")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file '{config_path}' not found.")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info("Configuration loaded successfully.")
    return config


def check_required_config(config: dict, required_keys: dict):
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


# -----------------------: Data Loading Functions
def load_csv_data(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"File not found: {filepath}")
    logger.info(f"Loading CSV data from {filepath}...")
    return pd.read_csv(filepath)


def load_excel_data(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"File not found: {filepath}")
    logger.info(f"Loading Excel data from {filepath}...")
    return pd.read_excel(filepath)


# -----------------------: Helper Functions
def extract_date_from_string(text: str, pos: int) -> str:
    return text.split("_")[pos].split("T")[0]


def prepare_output_folder(OUTPUT_FOLDER: str) -> str:
    if not OUTPUT_FOLDER:
        OUTPUT_FOLDER = "batch_scripts"
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    logger.info(f"Output folder is set to: {OUTPUT_FOLDER}")
    return OUTPUT_FOLDER


# -----------------------: Batch Script Generation Functions
def generate_batch_script(
    reference_scenes: list[str],
    secondary_scenes: list[str],
    output_file: str = "run_insar_tops_burst.txt",
    looks: str = "20x4",
) -> None:
    if not reference_scenes or not secondary_scenes:
        raise ValueError("Both reference and secondary scene lists must be non-empty.")

    reference_command = "--reference \\\n" + " \\\n".join(
        [f"        {scene}" for scene in reference_scenes]
    )
    secondary_command = "--secondary \\\n" + " \\\n".join(
        [f"        {scene}" for scene in secondary_scenes]
    )
    watermask_command = "--apply-water-mask True"
    looks_command = f"--looks {looks}"

    command = f"""#!/bin/bash

insar_tops_burst \\
{reference_command} \\
{secondary_command} \\
{looks_command} \\
{watermask_command}
"""

    try:
        with open(output_file, "w") as file:
            file.write(command)
        # logger.info(f"Batch script successfully written to {output_file}")
    except Exception as e:
        logger.error(
            f"Failed to write batch script to {output_file}: {e}", exc_info=True
        )


def generate_batchscript_filename(burst_byRefDate, burst_bySecDate) -> str:
    ref_mission = burst_byRefDate["platform"].unique()[0][-1]
    sec_mission = burst_bySecDate["platform"].unique()[0][-1]

    ref = (
        burst_byRefDate["stopTime"]
        .apply(lambda x: pd.to_datetime(x.split("T")[0]))
        .unique()[0]
    )
    ref_str = ref.strftime("%Y%m%d")

    sec = (
        burst_bySecDate["stopTime"]
        .apply(lambda x: pd.to_datetime(x.split("T")[0]))
        .unique()[0]
    )
    sec_str = sec.strftime("%Y%m%d")

    days_of_separation = (sec - ref).days

    foldername = f"S1{ref_mission}{sec_mission}_{ref_str}_{sec_str}_VVP{str(days_of_separation).zfill(3)}"
    return foldername


# -----------------------: Helper Functions for SEARCH Section
def extract_search_params(config: dict) -> dict:
    required_search_keys = [
        "relative_orbit",
        "start_date",
        "end_date",
        "beam_mode",
        "flight_direction",
        "polarization",
    ]
    for key in required_search_keys:
        if key not in config["SEARCH"]:
            raise ValueError(f"Missing required key '{key}' in SEARCH section.")
    search_params = config["SEARCH"]
    return search_params


def generate_burst_search_filename(search_params: dict) -> str:
    relative_orbit = search_params["relative_orbit"]
    start_date = search_params["start_date"]
    end_date = search_params["end_date"]
    beam_mode = search_params["beam_mode"]
    flight_direction = search_params["flight_direction"]
    polarization = search_params["polarization"]
    filename = f"{flight_direction}_{str(relative_orbit).zfill(3)}_{beam_mode}_{polarization}_{start_date}_{end_date}.xlsx"
    logger.info(
        f"No input from user, automatically generate burst search filename: {filename}"
    )
    return filename


# -----------------------: Processing Functions
def process_sbas(
    SBAS_PAIRS_FILE: str,
    burst_search_file: str,
    OUTPUT_FOLDER: str,
    number_of_looks: str,
):
    logger.info("Starting SBAS processing mode...")
    if not SBAS_PAIRS_FILE:
        raise ValueError("Error: CSV file for SBAS pairs must be provided.")

    sbas_df = load_csv_data(SBAS_PAIRS_FILE)
    sbas_df.columns = [ele.strip() for ele in sbas_df.columns]

    all_scenes = pd.concat(
        [sbas_df[col] for col in ["Reference", "Secondary"]], axis=0, ignore_index=True
    )
    unique_sbas_acquisition_dates = all_scenes.apply(
        lambda x: extract_date_from_string(x, 5)
    ).unique()
    logger.info(
        f"Unique acquisition dates from ASF SBAS pairs data: {len(unique_sbas_acquisition_dates)} found."
    )

    burst_search_df = load_excel_data(burst_search_file)
    burst_search_unique_acquisition = (
        burst_search_df["fileID"]
        .apply(lambda x: extract_date_from_string(x, 3))
        .unique()
    )
    logger.info(
        f"Unique acquisition dates from burst search output: {len(burst_search_unique_acquisition)} found."
    )

    matching_acquisition_dates = sorted(
        set(unique_sbas_acquisition_dates).intersection(
            set(burst_search_unique_acquisition)
        )
    )
    logger.info(
        f"Matching acquisition dates between ASF SBAS pairs and Burst Search: {len(matching_acquisition_dates)} found."
    )

    OUTPUT_FOLDER = prepare_output_folder(OUTPUT_FOLDER)

    for select_reference_date in matching_acquisition_dates:
        burst_search_byReference = burst_search_df[
            burst_search_df["fileID"].str.contains(select_reference_date)
        ]
        reference_bursts = burst_search_byReference["fileID"].tolist()

        sbas_df_query_byReference = sbas_df[
            sbas_df["Reference"].str.contains(select_reference_date)
        ]
        sbas_df_Secondary_byReference = sbas_df_query_byReference["Secondary"]
        sbas_df_Secondary_date = sbas_df_Secondary_byReference.apply(
            lambda x: extract_date_from_string(x, 5)
        ).unique()

        for secondary_date in sbas_df_Secondary_date:
            burst_search_bySecondary = burst_search_df[
                burst_search_df["fileID"].str.contains(secondary_date)
            ]
            if len(burst_search_bySecondary)>0:
                secondary_bursts = burst_search_bySecondary["fileID"].tolist()

                output_filename = generate_batchscript_filename(
                    burst_search_byReference, burst_search_bySecondary
                )
                output_savepath = os.path.join(OUTPUT_FOLDER, output_filename + ".txt")
                generate_batch_script(
                    reference_scenes=reference_bursts,
                    secondary_scenes=secondary_bursts,
                    output_file=output_savepath,
                    looks=number_of_looks,
                )


def process_sequential(
    sequential: int, burst_search_file: str, OUTPUT_FOLDER: str, number_of_looks: str
):
    logger.info(
        f"Starting sequential processing mode with maximum {sequential} secondary images per master..."
    )
    burst_search_df = load_excel_data(burst_search_file)
    burst_search_unique_acquisition = (
        burst_search_df["fileID"]
        .apply(lambda x: extract_date_from_string(x, 3))
        .unique()
    )
    logger.info(
        f"Unique acquisition dates from burst search output: {len(burst_search_unique_acquisition)} found."
    )

    matching_acquisition_dates = sorted(burst_search_unique_acquisition)
    OUTPUT_FOLDER = prepare_output_folder(OUTPUT_FOLDER)

    for select_reference_date in matching_acquisition_dates:
        burst_search_byReference = burst_search_df[
            burst_search_df["fileID"].str.contains(select_reference_date)
        ]
        reference_bursts = burst_search_byReference["fileID"].tolist()

        current_reference_date_idx = np.where(
            burst_search_unique_acquisition == select_reference_date
        )[0][0]
        end_idx = current_reference_date_idx + sequential
        secondary_date_list = burst_search_unique_acquisition[
            current_reference_date_idx + 1 : end_idx + 1
        ]
        if len(secondary_date_list) > 0:
            for secondary_date in secondary_date_list:
                burst_search_bySecondary = burst_search_df[
                    burst_search_df["fileID"].str.contains(secondary_date)
                ]
                secondary_bursts = burst_search_bySecondary["fileID"].tolist()

                output_filename = generate_batchscript_filename(
                    burst_search_byReference, burst_search_bySecondary
                )
                output_savepath = os.path.join(OUTPUT_FOLDER, output_filename + ".txt")
                generate_batch_script(
                    reference_scenes=reference_bursts,
                    secondary_scenes=secondary_bursts,
                    output_file=output_savepath,
                    looks=number_of_looks,
                )


# -----------------------: Main Workflow
def main():
    parser = argparse.ArgumentParser(description="InSAR burst processing script")
    parser.add_argument("--config", required=True, help="Path to config file")
    args = parser.parse_args()

    config_path = args.config
    config = load_config(config_path)

    check_required_config(
        config,
        {
            "SEARCH": [
                "relative_orbit",
                "start_date",
                "end_date",
                "beam_mode",
                "flight_direction",
                "polarization",
            ],
            "PAIR": [
                "method",
                "asf_sbas_file",
                "burst_search_file",
                "sequential",
                "looks_num",
            ],
        },
    )

    search_params = extract_search_params(config)

    burst_search_file = config["PAIR"]["burst_search_file"].strip()
    if not burst_search_file:
        burst_search_file = generate_burst_search_filename(search_params)

    pair_params = config["PAIR"]
    number_of_looks = pair_params["looks_num"]
    pair_method = pair_params["method"].strip().lower()
    SBAS_PAIRS_FILE = pair_params["asf_sbas_file"].strip()
    OUTPUT_FOLDER = pair_params["batch_outfolder"].strip()
    sequential_val = pair_params["sequential"]

    logger.info("Starting InSAR burst processing workflow...")
    if pair_method == "sbas":
        process_sbas(SBAS_PAIRS_FILE, burst_search_file, OUTPUT_FOLDER, number_of_looks)
    elif pair_method == "sequential":
        process_sequential(
            sequential_val, burst_search_file, OUTPUT_FOLDER, number_of_looks
        )
    else:
        raise ValueError("Input error: the pair method must be 'sbas' or 'sequential'.")


if __name__ == "__main__":
    main()