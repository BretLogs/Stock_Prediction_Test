import csv
import pandas as pd
import os
import sys

def generate_plot_data(directory, stock, start, end, rolling):
    df = pd.read_csv(os.path.join(directory, stock))

    df = df[df["Date"] > start]
    df = df[df["Date"] < end]
    df = df.set_index("Date")

    rolled = df.Close.rolling(rolling).mean()

    data = list(zip(rolled))
    file_name = "test_plot_data.csv"

    with open(file_name, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["pred"])
        writer.writerows(data)


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python predict.py <directory> <stock> <start> <end> <rolling>")
    else:
        directory = sys.argv[1]
        stock = sys.argv[2]
        start = sys.argv[3]
        end = sys.argv[4]
        rolling = int(sys.argv[5])

        generate_plot_data(directory, stock, start, end, rolling)
