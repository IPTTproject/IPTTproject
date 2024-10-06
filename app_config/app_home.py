from app_config import app
from flask import render_template
@app.route("/")
def home():
    html = render_template("test.html")
    return(html)