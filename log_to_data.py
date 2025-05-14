import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


keep_phrases = ["Epoch", "Kendall`s Tau"]  # Liste von Phrasen

os.chdir("experiment")
experiments = os.listdir()
experiments.remove('.DS_Store')  # Entfernen von .DS_Store, falls vorhanden
for exp in experiments:
    os.chdir(exp)
    important = []
    with open("out.log") as f:
        f = f.readlines()
        for line in f:
            if any(phrase in line for phrase in keep_phrases):  # Überprüfen, ob eine der Phrasen in der Zeile enthalten ist
                important.append(line)
        float_values = []
        tau_values = []
        for log in important:
            match = re.search(r"Train Error: ([0-9.]+), Valid Error: ([0-9.]+), Train Acc: ([0-9.]+), Valid Acc: ([0-9.]+)", log)
            match_kendall = re.search(r"Kendall`s Tau: ([0-9.]+)", log)
            if match:
                float_values.append([float(match.group(1)), float(match.group(2)), float(match.group(3)), float(match.group(4))])
            if match_kendall:
                tau_values.append(float(match_kendall.group(1)))

        df = pd.DataFrame(float_values, columns=['Train Error', 'Valid Error', 'Train Acc', 'Valid Acc'])
        df['tau'] = np.nan
        # Fügen Sie die Werte aus tau_values an den gewünschten Indizes ein
        for i, value in enumerate(tau_values):
            index = i * 5  # Indizes 0, 5, 10, 15, ...
            if index < len(df):  # Überprüfen, ob der Index im Bereich des DataFrames liegt
                df.at[index, 'tau'] = value

        if not df.empty:
            df.to_pickle('dataframe.pkl')
            df.interpolate(method='linear', inplace=True)
            df.plot(legend=True)
            plt.xlabel("Epochs")
            # plt.ylabel("Accuracy | Loss")
            # plt.savefig('plot.svg')
            # plt.close()
            fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(12, 6))

            ax1.set_xlabel("Epochs")
            ax1.set_title("Train/Valid Error and Accuracy")
            ax1.set_ylabel("Error")
            ax1.plot(df.index, df['Train Error'], label='Train Error', color='tab:blue')
            ax1.plot(df.index, df['Valid Error'], label='Valid Error', color='tab:orange')
            ax1.tick_params(axis='y')

            ax2 = ax1.twinx()
            ax2.set_ylabel("Accuracy")
            ax2.plot(df.index, df['Train Acc'], label='Train Acc', color='tab:green')
            ax2.plot(df.index, df['Valid Acc'], label='Valid Acc', color='tab:red')
            ax2.tick_params(axis='y')

            handles_ax1, labels_ax1 = ax1.get_legend_handles_labels()
            handles_ax2, labels_ax2 = ax2.get_legend_handles_labels()
            ax1.legend(handles_ax1 + handles_ax2, labels_ax1 + labels_ax2, loc='upper left')

            ax3.set_xlabel("Epochs")
            ax3.set_ylabel("Kendall's Tau")
            ax3.plot(df.index, df['tau'], label='Tau', color='tab:purple')
            ax3.legend()
            ax3.set_title("Kendall's Tau")

            fig.tight_layout()
            # fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
            plt.savefig('plot.pdf', format='pdf')
            plt.close()
    os.chdir("..")