from app_config import app
from flask import render_template
@app.route("/")
def home():
    content = "你好，這是我們專案的首頁。"
    html = render_template("homepage.html", msg = content)
    return(html)