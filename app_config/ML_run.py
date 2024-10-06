import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
clf = load("MLmodel.joblib")
pred = clf.predict()
print(pred)