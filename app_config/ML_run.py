from app_config import app
import numpy as np
from joblib import load
from flask import Flask, request
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
    ans_floor = request.form["floor"]
    arr = np.zeros(38)
    Qianjin, Xinxing, Yancheng, Zuoying, Nanzih, Gushan, Qijin, Lingya, Sanmin, Qianzhen, Xiaogang, Fengshan, Niaosong, Dashe, Renwu, Dasu, Gangshan, Yanshao, Zihguan, Yongan, Mitou, Qiaotou, Tianliao, Qieding, Alian, Luzhu, Hunei, Namaxia, Taoyuan, Maolin, Liugui, Meinong, Qishan, Jiaxian, Neimen, Sanlin, Linyuan, Daliao = arr
    match ans_floor:
        case "low":
            floor = 1
        case "medium":
            floor = 2
        case "high":
            floor = 3
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
    district = request.form["district"]
    match district:
        case "Qianjin":
            Qianjin = 1
        case "Xinxing":
            Xinxing = 1
        case "Yancheng":
            Yancheng = 1
        case "Zuoying":
            Zuoying = 1
        case "Nanzih":
            Nanzih = 1
        case "Gushan":
            Gushan = 1
        case "Qijin":
            Qijin = 1
        case "Lingya":
            Lingya = 1
        case "Sanmin":
            Sanmin = 1
        case "Qianzhen":
            Qianzhen = 1
        case "Xiaogang":
            Xiaogang = 1
        case "Fengshan":
            Fengshan = 1
        case "Niaosong":
            Niaosong = 1
        case "Dashe":
            Dashe = 1
        case "Renwu":
            Renwu = 1
        case "Dasu":
            Dasu = 1
        case "Gangshan":
            Gangshan = 1
        case "Yanshao":
            Yanshao = 1
        case "Zihguan":
            Zihguan = 1
        case "Yongan":
            Yongan = 1
        case "Mitou":
            Mitou = 1
        case "Qiaotou":
            Qiaotou = 1
        case "Tianliao":
            Tianliao = 1
        case "Qieding":
            Qieding = 1
        case "Alian":
            Alian = 1
        case "Luzhu":
            Luzhu = 1
        case "Hunei":
            Hunei = 1
        case "Namaxia":
            Namaxia = 1
        case "Taoyuan":
            Taoyuan = 1
        case "Maolin":
            Maolin = 1
        case "Liugui":
            Liugui = 1
        case "Meinong":
            Meinong = 1
        case "Qishan":
            Qishan = 1
        case "Jiaxian":
            Jiaxian = 1
        case "Neimen":
            Neimen = 1
        case "Sanlin":
            Sanlin = 1
        case "Linyuan":
            Linyuan = 1
        case "Daliao":
            Daliao = 1
            

    input = np.array(
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
        trade_type,  #交易種類_Lebel {"建物": 0, "房地(土地+建物)": 1, "房地(土地+建物)+車位": 2}
        Sanmin,  #三名區
        Renwu,  #仁武區
        Neimen,  #內門區
        Liugui,  #六龜區
        Qianjin,  #前金區
        Qianzhen,  #前鎮區
        Daliao,  #大寮區
        Dasu,  #大樹區
        Dashe,  #大社區
        Xiaogang,  #小港區
        Gangshan,  #岡山區
        Zuoying,  #左營區
        Mitou,  #彌陀區
        Xinxing,  #新興區
        Qishan,  #旗山區
        Qijin,  #旗津區
        Sanlin,  #杉林區
        Linyuan,  #林園區
        Taoyuan,  #桃源區
        Zihguan,  #梓官區
        Nanzih,  #楠梓區
        Qiaotou,  #橋頭區
        Yongan,  #永安區
        Hunei,  #湖內區
        Yanshao,  #燕巢區
        Tianliao,  #田寮區
        Jiaxian,  #甲仙區
        Meinong,  #美濃區
        Lingya,  #苓雅區
        Maolin,  #茂林區
        Qieding,  #茄萣區
        Luzhu,  #蘆竹區
        Namaxia,  #那瑪夏區
        Alian,  #阿蓮區
        Niaosong,  #鳥松區
        Fengshan,  #鳳山區
        Yancheng,  #鹽埕區
        Gushan,  #鼓山區
        ]).reshape(1,49)
    model = load("MLmodel.joblib")
    price_pred = int(model.predict(input)[0]//10000)
    return f"<h1>我們的模型預測您的房價為{price_pred}萬元。</h1>"
if __name__ == "__main__":
    app.run()