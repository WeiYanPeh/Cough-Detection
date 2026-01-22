import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", 
                        category=UserWarning, 
                        message=".*saving your model as an HDF5 file.*")

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
from tensorflow.keras.layers import (
    Conv2D, 
    Flatten, 
    Dense,
    Dropout, 
    BatchNormalization, 
    MaxPooling2D
    )


#################################################################################
def get_CNN_model(input_shape):
    """
    Build and compile a Convolutional Neural Network (CNN) for binary classification.

    Architecture:
        - Two convolutional blocks:
            Block 1: Conv2D(32) -> Conv2D(32) -> MaxPooling2D
            Block 2: Conv2D(64) -> Conv2D(64) -> MaxPooling2D
        - Flatten layer to convert 2D feature maps to 1D
        - Fully connected (dense) layers: 256 -> 128
        - Output layer with 2 units and softmax activation
        - L2 regularization on dense layers

    Parameters:
        input_shape (tuple): Shape of input data (height, width, channels).

    Returns:
        keras.models.Sequential: Compiled CNN model ready for training.
    """
    
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
    """
    Build and compile a fully connected (dense) neural network for binary classification.

    Architecture:
        - 4 hidden layers with decreasing size: 512 -> 256 -> 128 -> 64
        - ReLU activation for hidden layers
        - Batch normalization after each hidden layer
        - Dropout (0.3) for regularization
        - L2 weight regularization on some hidden layers
        - Output layer with 2 units and softmax activation for binary classification

    Parameters:
        input_length (int): Number of input features (size of input vector).

    Returns:
        keras.models.Sequential: Compiled Keras model ready for training.
    """
    
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
    """
    Plot training and validation accuracy and loss over epochs for a Keras model.

    Parameters:
        history (keras.callbacks.History): History object returned by model.fit().
                                           Contains training and validation metrics.

    Returns:
        None: The function generates plots but does not return any values.
    """
    
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
    """
    Compute and visualize the confusion matrix for binary classification.

    Parameters:
        y_test (array-like): True binary labels (0 or 1).
        y_predict (array-like): Predicted binary labels (0 or 1).

    Returns:
        numpy.ndarray: Confusion matrix as a 2x2 array.
                       [[TN, FP],
                        [FN, TP]]
    """
    
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
    """
    Compute and plot ROC and Precision-Recall curves for a binary classifier.

    Parameters:
        y_test (array-like): True binary labels (0 or 1).
        predictions (numpy array): Predicted probabilities for each class. 
                                   Shape should be (n_samples, 2) with column 1 being positive class probability.

    Returns:
        tuple: 
            roc_auc (float): Area under the ROC curve.
            pr_auc (float): Area under the Precision-Recall curve.
    """
    
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
    plt.close()
    
    return roc_auc, pr_auc










