import warnings
warnings.filterwarnings('ignore')

import os
import subprocess 

#################################################################################
def convert_to_wav(input_file, output_file):
    command = ["/home/l083319/ffmpeg-7.0.2-amd64-static/ffmpeg", "-i", input_file, output_file]
    subprocess.run(command, check=True)

#################################################################################
# Check if an indicator str is in list_all_files
def file_mapping(indicator, list_all_files):
    list_mapped = [x for x in list_all_files if indicator in x]
    return list_mapped

#################################################################################
# Round data based on threshold == 0.5
def custom_round(num):
    if num >= 0.5:
        return 1
    else:
        return 0