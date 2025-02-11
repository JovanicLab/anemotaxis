import pandas as pd
import os
import re
import numpy as np
from math import sqrt, asin

def parse_protocol(protocol_str):
    """Parses the protocol section of the filename.
    
    Args:
        protocol_str (str): Protocol string from the filename.
    
    Returns:
        dict: Parsed protocol metadata.
    """
    parts = protocol_str.split("_")

    if len(parts) < 2:
        return {"stimulus_type": protocol_str, "raw_protocol": protocol_str}

    stimulus_type = parts[0]
    stimulus_specifications = parts[1]
    
    # Extract timing details
    timing_match = re.match(r"(\d+[a-zA-Z]+)(\d+)x(\d+[a-zA-Z]+)(\d+[a-zA-Z]*)", parts[2]) if len(parts) > 2 else None

    if timing_match:
        prestimulus_duration = timing_match.group(1)
        repetitions = int(timing_match.group(2))
        stimulus_duration = timing_match.group(3)
        interval_between_repetitions = timing_match.group(4) if timing_match.group(4) else "0s"
    else:
        prestimulus_duration, repetitions, stimulus_duration, interval_between_repetitions = None, None, None, None

    return {
        "stimulus_type": stimulus_type,
        "stimulus_specifications": stimulus_specifications,
        "prestimulus_duration": prestimulus_duration,
        "number_of_repetitions": repetitions,
        "stimulus_duration": stimulus_duration,
        "interval_between_repetitions": interval_between_repetitions,
        "raw_protocol": protocol_str  # Keep original for reference
    }

def parse_filename(file_name):
    """Parses the filename to extract metadata including experiment details and larva number.
    
    Args:
        file_name (str): The name of the .dat file.
    
    Returns:
        dict: Metadata containing date, genotype, effector, tracker, protocol details, and larva number.
    """
    parts = file_name.split("@")
    
    if len(parts) < 6:
        raise ValueError(f"Unexpected filename format: {file_name}")
    
    metadata = {
        "date": parts[0],         # Date of experiment
        "genotype": parts[1],     # Genotype
        "effector": parts[2],     # Effector
        "tracker": parts[3],      # Tracker system used
    }

    # Parse protocol section
    protocol_metadata = parse_protocol(parts[4])
    metadata.update(protocol_metadata)

    # Extract larva number (last part after '@', before ".dat")
    larva_number = os.path.splitext(parts[-1])[0]  

    return metadata, larva_number

def extract_larva_data(file_path):
    """Extracts time, speed, length, and curvature from a single .dat file.
    
    Args:
        file_path (str): Path to the .dat file.
    
    Returns:
        dict: A dictionary with keys ["time", "speed", "length", "curvature"] and corresponding values.
    """
    columns = ["time", "speed", "length", "curvature"]
    df = pd.read_csv(file_path, sep=r"\s+", names=columns, usecols=[0, 1, 2, 3], header=None)
    return df

def compute_summary(df):
    """Computes summary statistics for each variable in the larva data.
    
    Args:
        df (pd.DataFrame): DataFrame containing time, speed, length, curvature.
    
    Returns:
        dict: Summary statistics including size, mean, max, and min.
    """
    summary = {}
    for column in df.columns:
        summary[column] = {
            "size": len(df[column]),
            "mean": np.nanmean(df[column]),  # Handle potential NaNs
            "max": np.nanmax(df[column]),
            "min": np.nanmin(df[column])
        }
    return summary

def extract_all_larvae(data_folder):
    """Extracts larva data from all .dat files in a given folder.
    
    Args:
        data_folder (str): Path to the folder containing .dat files.
    
    Returns:
        dict: A dictionary where keys are larva numbers and values contain metadata, data, and summary.
    """
    larvae_data = {}
    
    for file in os.listdir(data_folder):
        if file.endswith(".dat"):
            file_path = os.path.join(data_folder, file)
            metadata, larva_number = parse_filename(file)
            df = extract_larva_data(file_path)
            summary = compute_summary(df)
            
            larvae_data[larva_number] = {
                "metadata": metadata,
                "data": df.to_dict(orient="list"),  # Store data as lists
                "summary": summary
            }
    
    return larvae_data

def array_with_nan(lst):
    return np.array([np.nan if x is None else x for x in lst])

def compute_v_and_axis(larvae_data):
    for larva_id, larva in larvae_data.items():
        speed = larva["data"]["speed"]
        curvature = larva["data"]["curvature"]
        time = larva["data"]["time"]

        # Normalize speed and curvature by the mean speed
        mean_speed = np.nanmean(array_with_nan(speed))
        larva["data"]["speed_normalized"] = array_with_nan(speed) / mean_speed
        larva["data"]["curvature_normalized"] = array_with_nan(curvature) / mean_speed

    return larvae_data

def compute_navigational_index(larvae_data):
    for larva_id, larva in larvae_data.items():
        larva["data"]["mean_speed"] = np.nanmean(larva["data"]["speed_normalized"])
        larva["data"]["mean_curvature"] = np.nanmean(larva["data"]["curvature_normalized"])

    all_mean_speeds = [larva["data"]["mean_speed"] for larva in larvae_data.values()]
    all_mean_curvatures = [larva["data"]["mean_curvature"] for larva in larvae_data.values()]

    mean_speed = np.mean(all_mean_speeds)
    mean_curvature = np.mean(all_mean_curvatures)

    navigational_index = mean_curvature / mean_speed
    return navigational_index, navigational_index  # Return a tuple with two values