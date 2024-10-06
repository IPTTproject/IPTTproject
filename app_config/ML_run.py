import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
test_input = np.array(
[111, #交易年份
 20, #代表建號屋齡
 3, #幾房
 1, #幾廳
 2, #幾衛
 2, #車位數量
 20,#土地移轉面積(平方公尺)
 300,#建物移轉面積(平方公尺)
 20, #車位總持分面積(平方公尺)
 2,  #樓層_Lebel  {"其它": 0, "低樓層": 1, "中樓層": 2, "高樓層": 3}
 2,  #交易種類_Lebel {"建物": 0, "房地(土地+建物)": 1, "房地(土地+建物)+車位": 2}
 0,  #三名區
 0,  #仁武區
 0,  #內門區
 0,  #六龜區
 0,  #前金區
 0,  #前鎮區
 0,  #大寮區
 0,  #大樹區
 0,  #大社區
 1,  #小港區
 0,  #岡山區
 0,  #左營區
 0,  #彌陀區
 0,  #新興區
 0,  #旗山區
 0,  #旗津區
 0,  #杉林區
 0,  #林園區
 0,  #桃源區
 0,  #梓官區
 0,  #楠梓區
 0,  #橋頭區
 0,  #永安區
 0,  #湖內區
 0,  #燕巢區
 0,  #田寮區
 0,  #甲仙區
 0,  #美濃區
 0,  #苓雅區
 0,  #茂林區
 0,  #茄萣區
 0,  #路竹區
 0,  #那瑪夏區
 0,  #阿蓮區
 0,  #鳥松區
 0,  #鳳山區
 0,  #鹽埕區
 0,  #鼓山區
 ]).reshape(1,49)
clf = load("MLmodel.joblib")
price_pred = clf.predict(test_input)[0]//10000
price_pred = int(price_pred)
print(f"預估價格為: {price_pred}萬")