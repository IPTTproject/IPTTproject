import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("data1.csv")
df["房地總價"] = df["房地總價"].replace(",", "", regex = True).astype(float)
print(df.dtypes)

