import subprocess
from autoscaler import *
from flask import Flask, jsonify, request
import os
# Flask stuff
app = Flask(__name__)


@app.route("/toggle", methods=["GET"])
def toggle():
    """Toggle state of autoscaler. 
    """
    try:
        with open(os.path.join(os.getcwd(), "autoscaler", "database.txt"), "r") as file:
            i = file.read()
        
        print(i)
        if i == "true":
            i = "false"
        else: 
            i = "true"
        
        with open(os.path.join(os.getcwd(), "autoscaler", "database.txt"), "w") as file:
            file.write(i)
        
        if i == "true":
            cmd = subprocess.Popen("python test.py", shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            

        return jsonify({"result" : "", "status" : ""})
    except:
        return jsonify({"result" : "error", "status" : ""})


@app.route("/hyperparameters", methods = ["POST"])
def hyp():
    body = request.get_json()
    with open("config.json", "w") as infile:
        json.dump(body, infile)
        
    return jsonify({"result" : "", "status" : ""})



# Start autoscaler

if __name__ == "__main__":
    # hyperParameters = {
    #     "epochs" : 100,
    #     "csv" : os.path.join(os.getcwd(), "data", "dataset.csv"),
    #     "timeStep" : 100,
    #     "trainFraction" : 0.9,
    #     "algorithm" : "BiLSTM_V1",
    #     # "batchSize" : 64
    #     "batchSize" : 16
    # }
    # print(hyperParameters)
    # lstm = BiLSTMModel(**hyperParameters)
    app.run(debug=True)