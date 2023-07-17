import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def plot_confusion_matrix(y_true, y_pred, classes, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")


def classification_report_with_accuracy(y_true, y_pred, classes, save_path=None):
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.drop(["macro avg", "weighted avg"], axis=0)
    report_df = report_df.drop(["support"], axis=1)
    report_df = report_df.round(3)  # Round all numbers to 3 digits

    if save_path:
        plt.figure(figsize=(8, 4))
        sns.heatmap(report_df.iloc[:-1, :].T, annot=True, cmap="YlGnBu", fmt=".3f", linewidths=0.5)
        plt.xticks(rotation=45)
        plt.title("Classification Report")
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.show()

    return report_df


# Example usage:
if __name__ == "__main__":
    labels = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    predictions = [0, 2, 1, 0, 2, 1, 0, 1, 2]
    class_names = ["Class 0", "Class 1", "Class 2"]

    plot_confusion_matrix(labels, predictions, class_names, save_path="confusion_matrix.png")
    report_df = classification_report_with_accuracy(labels, predictions, class_names, save_path="report.png")

