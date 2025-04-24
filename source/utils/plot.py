import pandas as pd
import matplotlib.pyplot as plt


def plot_training_curves(csv_file, save_prefix="plot"):
    df = pd.read_csv(csv_file)
    epochs = df["Epoch"]

    plt.figure()
    plt.plot(epochs, df["TrainLoss"], label="Train Loss")
    plt.plot(epochs, df["ValLoss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_prefix}_loss.png")

    plt.figure()
    plt.plot(epochs, df["TrainAcc"], label="Train Acc")
    plt.plot(epochs, df["ValAcc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_prefix}_acc.png")

    plt.figure()
    plt.plot(epochs, df["DurationSeconds"], label="Epoch Duration")
    plt.xlabel("Epoch")
    plt.ylabel("Time (s)")
    plt.title("Epoch Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_prefix}_time.png")

    print(f"✅ 曲线图已保存为: {save_prefix}_loss.png, {save_prefix}_acc.png, {save_prefix}_time.png")