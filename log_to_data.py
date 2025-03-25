import os
import re
import pandas as pd
import matplotlib.pyplot as plt

keep_phrase = "Epoch"

os.chdir("experiment")
experiments = os.listdir()
for exp in experiments:
    os.chdir(exp)
    important = []
    with open("out.log") as f:
        f = f.readlines()
        for line in f:
            if keep_phrase in line:
                important.append(line)
        float_values = []
        for log in important:
            match = re.search(r"Train Error: ([0-9.]+), Valid Error: ([0-9.]+), Train Acc: ([0-9.]+), Valid Acc: ([0-9.]+)", log)
            if match:
                float_values.append([float(match.group(1)), float(match.group(2)), float(match.group(3)), float(match.group(4))])
        df = pd.DataFrame(float_values, columns=['Train Error', 'Valid Error', 'Train Acc', 'Valid Acc'])
        if not df.empty:
            df.to_pickle('dataframe.pkl')
            df.plot(legend=True)
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy / Loss")
            plt.savefig('plot.svg')
    os.chdir("..")