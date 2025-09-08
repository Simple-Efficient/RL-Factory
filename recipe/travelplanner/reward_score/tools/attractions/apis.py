import pandas as pd
from pandas import DataFrame
from typing import Optional
# from utils.func import extract_before_parenthesis
def extract_before_parenthesis(s):
    match = re.search(r'^(.*?)\([^)]*\)', s)
    return match.group(1) if match else s


class Attractions:
    def __init__(self, path="/home/hadoop-kg-llm-ddpt/dolphinfs_hdd_hadoop-kg-llm-ddpt/chengzihao02/Travel-R1/verl/utils/reward_score/database/attractions/attractions.csv"):
        self.path = path
        self.data = pd.read_csv(self.path).dropna()[['Name','Latitude','Longitude','Address','Phone','Website',"City"]]
        # print("Attractions loaded.")

    def load_db(self):
        self.data = pd.read_csv(self.path)

    def run(self,
            city: str,
            ) -> DataFrame:
        """Search for Accommodations by city and date."""
        results = self.data[self.data["City"] == city]
        # the results should show the index
        results = results.reset_index(drop=True)
        if len(results) == 0:
            return "There is no attraction in this city."
        return results  
      
    def run_for_annotation(self,
            city: str,
            ) -> DataFrame:
        """Search for Accommodations by city and date."""
        results = self.data[self.data["City"] == extract_before_parenthesis(city)]
        # the results should show the index
        results = results.reset_index(drop=True)
        return results