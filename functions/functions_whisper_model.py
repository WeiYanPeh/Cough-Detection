import warnings
warnings.filterwarnings('ignore')

import time

import pandas as pd
import numpy as np

from tqdm.notebook import tqdm
from collections import Counter
from pprint import pprint

from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    roc_curve,
    roc_auc_score, 
    precision_recall_curve,
    auc,
    precision_score, 
    recall_score, 
    f1_score
    )

import torch.utils.data
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset, DatasetDict,  Audio

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, accuracy_score

from transformers import WhisperModel, WhisperFeatureExtractor, AdamW

#################################################################################
# Audio Dataset
#################################################################################
class SpeechClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, audio_data,  text_processor, encoder):
        self.audio_data = audio_data
        self.text_processor = text_processor
        self.encoder = encoder

    def __len__(self):
        return len(self.audio_data)

    def __getitem__(self, index):

        inputs = self.text_processor(
            self.audio_data[index]["audio"]["array"],
            return_tensors="pt",
            sampling_rate=self.audio_data[index]["audio"]["sampling_rate"]
        )

        input_features = inputs.input_features
        decoder_input_ids = torch.tensor([[1, 1]]) * self.encoder.config.decoder_start_token_id

        labels = np.array(self.audio_data[index]['labels'])

        return input_features, decoder_input_ids, torch.tensor(labels)


#################################################################################
# Whisper Classifier
#################################################################################
class SpeechClassifier(nn.Module):
    def __init__(self, num_labels, encoder):
        super(SpeechClassifier, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_labels)
        )

    def forward(self, input_features):
        outputs = self.encoder(input_features=input_features)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        logits = self.classifier(pooled_output)
        return logits


#################################################################################
# Define the training function
#################################################################################
def train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, path_model_save):
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        start = time.time()

        for i, batch in enumerate(train_loader):
            input_features, decoder_input_ids, labels = batch

            input_features = input_features.squeeze().to(device)
            labels = labels.view(-1).to(device)

            optimizer.zero_grad()
            logits = model(input_features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 20 == 0:
                end = time.time()
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(train_loader)}, '
                      f'Train Loss: {loss.item():.4f}, Run-time: {round(end - start, 3)}s')
                start = time.time()

        val_loss, val_accuracy, val_f1, _, _, _ = evaluate(model, val_loader, optimizer, criterion, device)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), path_model_save)


        print("========================================================================================")
        print(f'Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}, Best Accuracy: {best_accuracy:.4f}')
        print("========================================================================================")


#################################################################################
# Evaluate
#################################################################################
def evaluate(model, data_loader, optimizer, criterion, device):
    all_labels = []
    all_preds = []
    all_probs = []
    total_loss = 0.0

    model.eval()
    with torch.no_grad():
        for i, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc="Evaluating"):
            input_features, decoder_input_ids, labels = batch

            input_features = input_features.squeeze().to(device)
            labels = labels.view(-1).to(device)

            optimizer.zero_grad()

            logits = model(input_features)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.nn.functional.softmax(logits, dim=1)
            _, preds = torch.max(logits, 1)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)

    loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')

    return loss, accuracy, f1, all_labels, all_preds, all_probs