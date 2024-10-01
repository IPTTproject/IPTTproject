import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("data.csv")
df["房地總價"] = df["房地總價"].replace(",", "", regex = True).astype(int)
df["代表建號層次"] = df["代表建號層次"].str.split(";", expand = True)[0]
df.loc[df["代表建號層次"].isin(["0", "見其他登記事項", "地下一層", "地下層", "騎樓", "地下三層", "地下二層"]), "代表建號層次"] = "其它"
df["代表建號層次"] = df["代表建號層次"].apply(lambda x: "其它" if x in "其它" 
                                  else "低樓層" if x in ["一層", "二層", "三層", "四層", "五層"]
                                  else "中樓層" if x in ["六層", "七層", "八層", "九層", "十層",
                                                      "十一層", "十二層", "十三層", "十四層", "十五層"]
                                                      else "高樓層")
floor_dict = {"其它": 0, "低樓層": 1, "中樓層": 2, "高樓層": 3}
df["樓層_Lebel"] = df["代表建號層次"].map(floor_dict)
df = pd.get_dummies(df, columns = ["行政區", "交易標的種類", "地段"])
df = df.select_dtypes(exclude = "object")
df = df.astype(int)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
y = df.房地總價
X = df.drop("房地總價", axis = 1)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
model = RandomForestRegressor(n_estimators = 100, random_state = 42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
score = r2_score(y_test, y_pred)
print("準確度為：", score, sep = "")