import pandas as pd
from datetime import datetime, date
import argparse
import logging
import os
import yaml
from burst2safe.burst2safe import burst2safe  # Assuming burst2safe is properly imported

# -----------------------: Logging Setup
log_filename = f"S1_download_burst_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

# Remove existing handlers if rerunning in an interactive environment (e.g., Jupyter)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        # logging.FileHandler(log_filename),  
        logging.StreamHandler(),  
    ],
)

logger = logging.getLogger(__name__)

# -----------------------: Configuration Loading and Validation Functions

def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file '{config_path}' not found.")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def check_required_config(config: dict, required_keys: dict):
    """Ensure all required sections and keys exist in the configuration."""
    for section, keys in required_keys.items():
        if section not in config:
            raise ValueError(f"Missing required section: '{section}' in the config file.")
        for key in keys:
            if key not in config[section]:
                raise ValueError(f"Missing required key '{key}' in section '{section}'.")

# -----------------------: Burst Download Function

def download_burst(burst_list):
    """Download and convert a list of burst scenes into SAFE format."""
    try:
        if not burst_list:
            logger.warning("Received an empty burst list. Skipping.")
            return
        
        logger.info(f"Processing burst list: {burst_list}")
        burst2safe(burst_list)  
        logger.info(f"Completed downloading burst list: {burst_list}")
    except Exception as e:
        logger.error(f"Error processing burst list {burst_list}: {e}", exc_info=True)

# -----------------------: Existing Date Check Function

def check_existed_date():
    """
    Check and return a sorted list of unique acquisition dates (as datetime.date objects)
    from files in the current directory. This function examines:
      - TIFF files: expects the date at index 3 (e.g., S1_224318_IW2_20250123T215254_VV_9383-BURST.tiff)
      - XML files: expects the date at index 5 (e.g., S1A_IW_SLC__1SDV_20250123T215253_20250123T215321_057577_0717D2_9383_VV.xml)
      - SAFE files: expects the date at index 5 (e.g., S1A_IW_SLC__1SSV_20250111T215252_20250111T215304_057402_0710E0_3AC7.SAFE)
    
    Returns:
        A sorted list of unique datetime.date objects representing acquisition dates.
    """
    from datetime import datetime
    existed_dates = []
    
    for f in os.listdir():
        try:
            if f.endswith(".tiff"):
                # e.g., S1_224318_IW2_20250123T215254_VV_9383-BURST.tiff
                f_date_str = f.split("_")[3].split("T")[0]  # "20250123"
            elif f.endswith(".xml") or f.endswith(".SAFE"):
                # e.g., S1A_IW_SLC__1SDV_20250123T215253_20250123T215321_057577_0717D2_9383_VV.xml
                # or    S1A_IW_SLC__1SSV_20250111T215252_20250111T215304_057402_0710E0_3AC7.SAFE
                f_date_str = f.split("_")[5].split("T")[0]  # "20250123" or "20250111"
            else:
                continue  # Skip files that do not match our patterns
            
            # Parse the date string (format YYYYMMDD)
            parsed_date = datetime.strptime(f_date_str, "%Y%m%d").date()
            existed_dates.append(parsed_date)
        except (IndexError, ValueError) as e:
            logger.warning(f"Skipping file with unexpected format: {f}. Error: {e}")
    
    return sorted(set(existed_dates))

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

    required_keys = {
        "SEARCH": ["relative_orbit", "start_date", "end_date", "beam_mode", "flight_direction", "polarization"],
        "DOWNLOAD": ["specific_burst_file", "storage_dir"]
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

    # -----------------------: Extract DOWNLOAD Parameters
    download_params = config["DOWNLOAD"]
    user_burst_file = download_params.get("specific_burst_file", "") 
    output_savefld = download_params.get("storage_dir")
    if not output_savefld:
        logger.error("Storage directory ('storage_dir') is not specified in the configuration file. Please provide a valid storage directory.")
        return

    # Determine search output filename
    if not user_burst_file:
        search_burst_output_excel = f"{flight_direction}_{str(relative_orbit).zfill(3)}_{beam_mode}_{polarization}_{start_date}_{end_date}.xlsx"
    else:
        search_burst_output_excel = user_burst_file

    if not os.path.exists(search_burst_output_excel):
        logger.error(f"Excel file {search_burst_output_excel} not found. Please ensure the file exists.")
        return

    try:
        # Read the Excel file
        df = pd.read_excel(search_burst_output_excel)

        # Ensure 'stopTime' is in datetime format (convert to date)
        df["stopTime"] = pd.to_datetime(df["stopTime"], errors="coerce").dt.date

        # Extract unique acquisition dates (ensuring it's a list of datetime.date objects)
        unique_acquisition_dates = df["stopTime"].dropna().unique()

        # Ensure output directory exists before changing
        os.makedirs(output_savefld, exist_ok=True)
        os.chdir(output_savefld)

        # -----------------------: Announce Existing Dates
        existing_dates = check_existed_date()
        logger.info("Already downloaded acquisition dates: " + ", ".join(str(d) for d in existing_dates))

        # -----------------------: Process Bursts Sequentially
        for select_date in unique_acquisition_dates:
            if select_date in existing_dates:
                logger.info(f"Skipping download for date: {select_date} because it already exists.")
                continue

            subset = df[df["stopTime"] == select_date]
            subset_burst = subset["fileID"].tolist()

            if subset_burst:
                download_burst(subset_burst)
            else:
                logger.warning(f"No bursts found for date: {select_date}")

        logger.info("All burst downloads completed.")

    except pd.errors.EmptyDataError:
        logger.error("The Excel file is empty.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()
