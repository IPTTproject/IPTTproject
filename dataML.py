import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder as OHE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
import joblib
df = pd.read_csv("data.csv")
#將交易日期的值改成年份
df["交易日期"] = df["交易日期"]//10000
df = df.rename(columns = {"交易日期": "交易年份"})
#將房價數值轉為整數
df["房地總價"] = df["房地總價"].replace(",", "", regex = True).astype(int)
#將房屋層次分為低、中、高樓層
df["代表建號層次"] = df["代表建號層次"].str.split(";", expand = True)[0]
df.loc[df["代表建號層次"].isin(["0", "見其他登記事項", "地下一層", "地下層", "騎樓", "地下三層", "地下二層"]), "代表建號層次"] = "其它"
df["代表建號層次"] = df["代表建號層次"].apply(lambda x: "其它" if x in "其它" 
                                  else "低樓層" if x in ["一層", "二層", "三層", "四層", "五層"]
                                  else "中樓層" if x in ["六層", "七層", "八層", "九層", "十層",
                                                      "十一層", "十二層", "十三層", "十四層", "十五層"]
                                                      else "高樓層")
floor_dict = {"其它": 0, "低樓層": 1, "中樓層": 2, "高樓層": 3}
df["樓層_Lebel"] = df["代表建號層次"].map(floor_dict)
one = OHE(sparse_output = False)
onehot = one.fit_transform(df[["行政區"]])
行政區_df = pd.DataFrame(onehot, columns = one.get_feature_names_out(["行政區"])).astype(int)
type_dict = {"建物": 0, "房地(土地+建物)": 1, "房地(土地+建物)+車位": 2}
df["交易種類_Lebel"] = df["交易標的種類"].map(type_dict)
df = df.drop(["交易標的種類", "代表建號層次", "行政區"], axis = 1)
df = df.astype(int)
final_df = pd.concat([df, 行政區_df], axis = 1)
for i in ["幾房", "建物移轉面積", "土地移轉面積"]:
    s = final_df[i].describe()
    IQR = s["75%"] - s["25%"]
    upper_lim = s["75%"] + (IQR * 1.5)
    lower_lim = s["25%"] - (IQR * 1.5)
    final_df = final_df[final_df[i] < upper_lim]
X = final_df.drop(["房地總價"], axis = 1)
y = final_df["房地總價"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train.values, y_train)
joblib.dump(model, "/MLmodel.joblib")
