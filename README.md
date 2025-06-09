# Cough Detection Project

## Overview
This project detects cough sounds from audio recordings using traditional machine learning models, the Whisper model, and convolutional neural networks (CNN).

## Project Structure
**main_0_load_data.ipynb**: 
- Load and prepare dataset for standardization

**main_1_ML_analyze_data.ipynb**: 
- Extract time and frequency features for ML classification

**main_2_ML_prediction.ipynb**: 
- Classify audio features using ML models.

**main_3_whisper_online_code.ipynb**: 
- Temporary OpenAI Whisper model code from online.

**main_4_whisper_wav_extractor.ipynb**: 
- Convert MP3 to WAV format, crop audio into n-seconds segments.

**main_5_whisper_prediction.ipynb**:
- Classify audio using OpenAI Whisper.

**main_6_CNN_analyze_data.ipynb**:
- Extract 2D MFCC features CNN.

**main_7_CNN_prediction.ipynb**:
- Classify audio using CNN.

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/WeiYanPeh/Cough-Detection
   cd Cough-Detection
   ```