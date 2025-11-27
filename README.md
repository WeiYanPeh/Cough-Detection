# Cough Detection Project

## Overview
This project detects cough sounds from audio recordings using traditional machine learning models, the Whisper model, and convolutional neural networks (CNN).


## Code
**Analyze Datasets**
- main_0_load_data.ipynb
    - Load and prepare dataset for standardization
- Datasets:
    - Coswara: [Link Coswara](https://github.com/iiscleap/Coswara-Data)
    - Virufy: [Link Virufy](https://github.com/virufy/virufy-data/tree/main/clinical)
    - COUGHVID: [Link COUGHVID](https://zenodo.org/records/7024894)
    - ESC-50: [Link ESC-50](https://github.com/karolpiczak/ESC-50?tab=readme-ov-file#download)
    - FSDKaggle2018: [Link FSDKaggle2018](https://zenodo.org/records/2552860#.XwscUud7kaE)

<br>

**Classification using Traditional Machine Learning (ML) Models**
- main_1_ML_analyze_data.ipynb
    - Extract time and frequency features for ML classification
- main_2_ML_prediction.ipynb
    - Classify audio features using ML models.
<br>

**Classification using OpenAI Whisper**
- main_3_whisper_online_code.ipynb
    - Temporary OpenAI Whisper model code from online.
    - [Link OpenAI Whisper](https://www.daniweb.com/programming/computer-science/tutorials/540802/fine-tuning-openai-whisper-model-for-audio-classification-in-pytorch)
- main_4_whisper_wav_extractor.ipynb
    - Convert MP3 to WAV format, crop audio into n-seconds segments for OpenAI Whisper classification.
    - OpenAI Whisper [Link OpenAI Whisper](https://github.com/openai/whisper)
    - Resample to 16kHz
- main_5_whisper_prediction.ipynb
    - Classify audio using OpenAI Whisper.
<br>

**Classification using 2D CNN and MFCC**
- main_6_CNN_analyze_data.ipynb
    - Extract 2D MFCC features for 2D CNN classification.
- main_7_CNN_prediction.ipynb
    - Classify audio using 2D CNN.
<br>

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/WeiYanPeh/Cough-Detection
   cd Cough-Detection
   ```

2. Clone the repository:
   ```bash
    pip install -r requirements.txt
   ```

Usage Guide
Run the notebooks in the following order:
- main_0_load_data.ipynb
- main_1_ML_analyze_data.ipynb
- main_2_ML_prediction.ipynb
- main_3_whisper_online_code.ipynb (optional)
- main_4_whisper_wav_extractor.ipynb
- main_5_whisper_prediction.ipynb
- main_6_CNN_analyze_data.ipynb
- main_7_CNN_prediction.ipynb

This ensures a smooth workflow from data preparation to final classification.