import warnings
warnings.filterwarnings('ignore')

import os
import subprocess 

#################################################################################
def convert_to_wav(input_file, output_file):
    """
    Convert an audio file to WAV format using ffmpeg.

    Parameters:
        input_file (str): Path to the input audio file (any format supported by ffmpeg).
        output_file (str): Path to save the converted WAV file.

    Notes:
        - Uses a specific ffmpeg executable path.
        - Raises a CalledProcessError if the conversion fails.
    """
    
    command = ["/home/l083319/ffmpeg-7.0.2-amd64-static/ffmpeg", "-i", input_file, output_file]
    subprocess.run(command, check=True)

#################################################################################
# Check if an indicator str is in list_all_files
def file_mapping(indicator, list_all_files):
    """
    Filter file names that contain a given indicator string.

    Parameters:
        indicator (str): Substring used to identify relevant files.
        list_all_files (list of str): List of file names to search.

    Returns:
        list of str: File names that contain the indicator.
    """
    
    list_mapped = [x for x in list_all_files if indicator in x]
    return list_mapped

#################################################################################
# Round data based on threshold == 0.5
def custom_round(num):
    """
    Apply a custom rounding rule to a numeric value.

    Values greater than or equal to 0.5 are rounded to 1,
    while values below 0.5 are rounded to 0.

    Parameters:
        num (float): Input numeric value.

    Returns:
        int: Rounded value (0 or 1).
    """
    
    if num >= 0.5:
        return 1
    else:
        return 0