import requests
# from utils.func import extract_before_parenthesis
import os
from requests.exceptions import SSLError
import time
import sys
import pandas as pd
import numpy as np


def extract_before_parenthesis(s):
    match = re.search(r'^(.*?)\([^)]*\)', s)
    return match.group(1) if match else s
# This tool refers to the "DistanceMatrix" in the paper. Considering this data obtained from Google API, we consistently use this name in the code. 
# Please be assured that this will not influence the experiment results shown in the paper. 

class GoogleDistanceMatrix:
    def __init__(self, subscription_key: str="") -> None:
        self.gplaces_api_key: str = subscription_key
        self.data =  pd.read_csv('/home/hadoop-kg-llm-ddpt/dolphinfs_hdd_hadoop-kg-llm-ddpt/chengzihao02/Travel-R1/verl/utils/reward_score/database/googleDistanceMatrix/distance.csv')
        # print("GoogleDistanceMatrix loaded.")

    def run(self, origin, destination, mode='driving'):
        origin = extract_before_parenthesis(origin)
        destination = extract_before_parenthesis(destination)
        info = {"origin": origin, "destination": destination,"cost": None, "duration": None, "distance": None}
        response = self.data[(self.data['origin'] == origin) & (self.data['destination'] == destination)]
        if len(response) > 0:
                if response['duration'].values[0] is None or response['distance'].values[0] is None or response['duration'].values[0] is np.nan or response['distance'].values[0] is np.nan:
                    return "No valid information."
                info["duration"] = response['duration'].values[0]
                info["distance"] = response['distance'].values[0]
                if 'driving' in mode:
                    info["cost"] = int(eval(info["distance"].replace("km","").replace(",","")) * 0.05)
                elif mode == "taxi":
                    info["cost"] = int(eval(info["distance"].replace("km","").replace(",","")))
                if 'day' in info["duration"]:
                    return "No valid information."
                return f"{mode}, from {origin} to {destination}, duration: {info['duration']}, distance: {info['distance']}, cost: {info['cost']}"

        return f"{mode}, from {origin} to {destination}, no valid information."   
    
    def run_for_evaluation(self, origin, destination, mode='driving'):
        origin = extract_before_parenthesis(origin)
        destination = extract_before_parenthesis(destination)
        info = {"origin": origin, "destination": destination,"cost": None, "duration": None, "distance": None}
        response = self.data[(self.data['origin'] == origin) & (self.data['destination'] == destination)]
        if len(response) > 0:
                if response['duration'].values[0] is None or response['distance'].values[0] is None or response['duration'].values[0] is np.nan or response['distance'].values[0] is np.nan:
                    return info
                info["duration"] = response['duration'].values[0]
                info["distance"] = response['distance'].values[0]

                if 'day' not in info["duration"]:
                    if 'driving' in mode:
                        info["cost"] = int(eval(info["distance"].replace("km","").replace(",","")) * 0.05)
                    elif mode == "taxi":
                        info["cost"] = int(eval(info["distance"].replace("km","").replace(",","")))

                return info

        return info 


    def run_online(self, origin, destination, mode="driving"):
        # mode in ['driving','taxi','walking', 'distance','transit']
        endpoint = "https://maps.googleapis.com/maps/api/distancematrix/json"

        params = {
            "origins": origin,
            "destinations": destination,
            "mode": mode if mode=="taxi" else "driving",
            "key": self.gplaces_api_key
        }

        while True:
            try:
                response = requests.get(endpoint, params=params)
                break
            except SSLError:
                time.sleep(30)

        data = response.json()
        info = {"origin": origin, "destination": destination,"cost": None, "duration": None, "distance": None}
        if data['status'] == "OK":
            element = data['rows'][0]['elements'][0]
            if element['status'] == "OK":
                info["duration"] = element['duration']['text']
                info["distance"] = element['distance']['text']
                if 'driving' in mode:
                    info["cost"] = int(eval(info["distance"].replace("km","").replace(",","")) * 0.05)
                elif mode == "taxi":
                    info["cost"] = int(eval(info["distance"].replace("km","").replace(",","")))
                # if 'day' in info["duration"]:
                #     return "No valid information."
                return f"{mode}, from {origin} to {destination}, duration: {info['duration']}, distance: {info['distance']}, cost: {info['cost']}"

        return "No valid information."   
    
    def run_for_annotation(self, origin, destination, mode="driving"):
        # mode in ['driving','taxi','walking', 'distance','transit']
        endpoint = "https://maps.googleapis.com/maps/api/distancematrix/json"

        params = {
            "origins": extract_before_parenthesis(origin),
            "destinations": extract_before_parenthesis(destination),
            "mode": mode if mode!="taxi" else "driving",
            "key": self.gplaces_api_key
        }
        
        response = requests.get(endpoint, params=params)
        data = response.json()
        info = {}
        if data['status'] == "OK":
            element = data['rows'][0]['elements'][0]
            if element['status'] == "OK":
                info["duration"] = element['duration']['text']
                info["distance"] = element['distance']['text']
                info["cost"] = None
                if 'driving' in mode:
                    info["cost"] = int(eval(info["distance"].replace("km","").replace(",","")) * 0.05)
                elif mode == "taxi":
                    info["cost"] = int(eval(info["distance"].replace("km","").replace(",","")))
        else:
            info = {"duration": "N/A", "distance": "N/A", "cost": "N/A", "Hint":"Please check the input."}
        return info
    
