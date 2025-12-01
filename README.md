# Cough Detection Project

## Overview
This project detects cough sounds from audio recordings using traditional machine learning models, the Whisper model, and convolutional neural networks (CNN).


## Code
**Analyze Datasets**
- Load and prepare dataset for standardization
- Datasets:
    - Coswara: [Link Coswara](https://github.com/iiscleap/Coswara-Data)
    - Virufy: [Link Virufy](https://github.com/virufy/virufy-data/tree/main/clinical)
    - COUGHVID: [Link COUGHVID](https://zenodo.org/records/7024894)
    - ESC-50: [Link ESC-50](https://github.com/karolpiczak/ESC-50?tab=readme-ov-file#download)
    - FSDKaggle2018: [Link FSDKaggle2018](https://zenodo.org/records/2552860#.XwscUud7kaE)
<br>

**Classification using Traditional Machine Learning (ML) Models**
- Extract time and frequency features for ML classification
- Classify audio features using ML models.
<br>

**Classification using OpenAI Whisper**
- Temporary OpenAI Whisper model code from online.
- [Link OpenAI Whisper](https://www.daniweb.com/programming/computer-science/tutorials/540802/fine-tuning-openai-whisper-model-for-audio-classification-in-pytorch)
- Convert MP3 to WAV format, crop audio into n-seconds segments for OpenAI Whisper classification.
- OpenAI Whisper [Link OpenAI Whisper](https://github.com/openai/whisper)
- Resample to 16kHz
- Classify audio using OpenAI Whisper.
<br>

**Classification using 2D CNN and MFCC**
- Extract 2D MFCC features for 2D CNN classification.
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

## Usage Guide
Run notebooks:

1. Load Audio Data
    ```bash
    main_0_load_data.ipynb
    ```

2. Extract ML features and perform predictions - For Binary Detection
    ```bash
    main_1_binary_ML_features.ipynb
    main_2_binary_ML_prediction.ipynb
    ```

3. Extract MFCC features and perform predictions - For Binary Detection
    ```bash
    main_3_binary_CNN_features.ipynb
    main_4_binary_CNN_prediction.ipynb
    ```

4. Extract Whisper wav format and perform predictions - For Binary Detection
    ```bash
    main_5_binary_whisper_wav_extractor.ipynb
    main_6_binary_whisper_prediction.ipynb
    ```

5. Plot audio for onset detection, and perform automatic cough label extraction - For Onset Detection
    ```bash
    main_7_onset_plot_audio.ipynb
    ```

6. Extract ML features and perform predictions - For Onset Detection
    ```bash
    main_8_onset_ML_features.ipynb
    main_9_onset_ML_prediction.ipynb
    ```

7. Extract MFCC features and perform predictions - For Onset Detection
    ```bash
    main_A0_onset_CNN_features.ipynb
    main_A1_onset_CNN_prediction.ipynb
    ```

8. Extract Whisper wav format and perform predictions - For Onset Detection
    ```bash
    main_A2_onset_whisper_extraction.ipynb
    main_A3_onset_whisper_prediction.ipynb
    ```

9. Plot audio for onset detection, with onset model evaluation - For Onset Detection
    ```bash
    main_A4_onset_display.ipynb
    main_A5_onset_plot_audio.ipynb
    ```

10. Check if GPU is detected in TF, Keras, Torch
    ```bash
    main_check_gpu.ipynb
    ```

This ensures a smooth workflow from data preparation to final classification.

