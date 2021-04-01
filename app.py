from flask import Flask, jsonify,request
from flask_cors import CORS
import time
app = Flask(__name__);
CORS(app)
@app.route("/bot", methods=["POST"])

#response
def response():
    query = dict(request.form)['query']
    res = query + " " 
    return jsonify({"response" : res})
     
if __name__=="__main__":
    app.run(host="0.0.0.0",)