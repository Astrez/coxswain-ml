from time import sleep
import requests
import json
import requests
import os
url = ""


from .bilstmModel import BiLSTMModel
from .gru import GRUModel
from .lstmModel import LSTMModel
import os

# model = LSTMModel()
threshold = 6



def autoscale(**kwargs):
    with open(os.path.join(os.getcwd(), "config.json"), "r") as cred:
        hyp = json.load(cred)
    
    model = LSTMModel(**hyp)
    queue = []
    step = 100
    while True:
        with open(os.path.join(os.getcwd(), "autoscaler", "database.txt"), "r") as file:
            i = file.read()

        if i == "false":
            break
        else:
            queue.append(json.loads(requests.get(url=url + "/autometrics").text).get("result"))
            if len(queue) >= step:
                queue.pop(0)
                upscaleValue = model.predict(queue) // threshold
                if upscaleValue != json.loads(requests.get(url=url + "/replicas").text).get("result"):
                    requests.post(url=url + "/update/replicas", json={"replicas" : upscaleValue, "deployment" : kwargs.get("deployment"), "namespace" : kwargs.get("namespace")})
        
        sleep(1)
    
    

    

    
    pass


# Recieve metric from server per second