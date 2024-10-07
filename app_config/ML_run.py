from app_config import app
import numpy as np
from joblib import load
from flask import request, render_template
@app.route("/pred_house_price", methods = ["post"])
def ML():
    age = int(request.form["age"])
    rooms = int(request.form["rooms"])
    living_rooms = int(request.form["living_rooms"])
    bath_rooms = int(request.form["bath_rooms"])
    parking_spots = int(request.form["parking_spots"])
    land_area = int(request.form["land_area"])
    building_area = int(request.form["building_area"])
    parking_spots_area = int(request.form["parking_spots_area"])
    floor = int(request.form["floor"])
    
    if parking_spots == 0 and parking_spots_area > 0:
        return "<h1>輸入錯誤，請確認參數無誤。<h1>"
    elif parking_spots_area == 0 and parking_spots > 0:
        return "<h1>輸入錯誤，請確認參數無誤。<h1>"
    if land_area == 0:
        trade_type = 0
    elif parking_spots == 0:
        trade_type = 1
    else:
        trade_type = 2
    district = int(request.form["district"])
    input_array = np.array(
        [113, #交易年份
        age, #代表建號屋齡
        rooms, #幾房
        living_rooms, #幾廳
        bath_rooms, #幾衛
        parking_spots, #車位數量
        land_area,#土地移轉面積(平方公尺)
        building_area,#建物移轉面積(平方公尺)
        parking_spots_area, #車位總持分面積(平方公尺)
        floor,  #樓層_Lebel  {"其它": 0, "低樓層": 1, "中樓層": 2, "高樓層": 3}
        trade_type])  #交易種類_Lebel {"建物": 0, "房地(土地+建物)": 1, "房地(土地+建物)+車位": 2}
    for i in range(district):
        input_array = np.append(input_array, 0)
    input_array = np.append(input_array, 1)
    for i in range(37 - district):
        input_array = np.append(input_array, 0)
    input_array = input_array.reshape(1, 49)
    model = load("MLmodel.joblib")
    pred_price = int(model.predict(input_array)[0]//10000)
    return render_template("price.html", pred_price = pred_price)
if __name__ == "__main__":
    app.run()
