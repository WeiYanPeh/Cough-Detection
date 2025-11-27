import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=UserWarning, message=".*saving your model as an HDF5 file.*")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix, 
    roc_curve,
    precision_recall_curve,
    auc,
    )

from tensorflow import keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout, BatchNormalization, MaxPooling2D
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.callbacks import Callback


#################################################################################
def get_CNN_model(input_shape):
    model = Sequential()
    
    # Convolutional layers with padding='same'
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
    model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
              
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(2, activation='softmax'))
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

#################################################################################
def get_NN_model(input_length):
    model = Sequential()
    
    model.add(Dense(512, activation='relu', input_shape=(input_length,)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(2, activation='softmax'))
    
    model.compile(
        optimizer= keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
        )
    
    return model

#################################################################################
def history_loss_acc(history):
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
    
    # Summarize history for accuracy
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Test'], loc='upper left')
    
    # Summarize history for loss
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Test'], loc='upper left')
    plt.tight_layout()    
    # plt.show()
    plt.close()
    return None

#################################################################################
def evaluate_matrix(y_test, y_predict):
    cm = confusion_matrix(y_test, y_predict)
    cm_df = pd.DataFrame(
        cm, 
        index=["Negative", "Positive"], 
        columns=["Negative", "Positive"]
        )

    plt.figure(figsize=(4, 4))

    sns.set(font_scale=1)

    ax = sns.heatmap(
        cm_df, 
        annot=True, 
        square=True, 
        fmt='d', 
        linewidths=.2, 
        cbar=0, 
        cmap=plt.cm.Blues
        )
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    plt.ylabel("True labels")
    plt.xlabel("Predicted labels")
    plt.tight_layout()
    plt.title('Confusion Matrix')
    # plt.show()
    plt.close()
    
    return cm


#################################################################################
def ROC_PR_curve(y_test, predictions):
    # Calculate roc curves
    lr_fpr, lr_tpr, _ = roc_curve(y_test, predictions[:,1])
    roc_auc = auc(lr_fpr, lr_tpr)
    
    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, predictions[:,1])
    pr_auc = auc(recall, precision)
    
    # Plot the curves for the model
    lw = 2
    plt.figure(figsize=(8, 3))
    
    # ROC Curve
    plt.subplot(1, 2, 1)
    plt.plot(
        lr_fpr, lr_tpr, 
        color="darkorange", 
        lw=lw, 
        label=f"ROC curve (area = {round(roc_auc, 3)})"
        )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    
    # Precision-Recall Curve
    plt.subplot(1, 2, 2)
    plt.plot(
        recall, precision, 
        color="blue", 
        lw=lw, 
        label=f"PR curve (area = {round(pr_auc, 3)})"
        )
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.tight_layout()
    # plt.show()
    plt.close()
    
    return roc_auc, pr_auc










