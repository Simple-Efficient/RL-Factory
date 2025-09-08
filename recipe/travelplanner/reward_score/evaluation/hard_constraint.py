# from utils.func import get_valid_name_city,extract_before_parenthesis,extract_numbers_from_filenames
from ..tools.flights.apis import Flights
from ..tools.accommodations.apis import Accommodations
from ..tools.restaurants.apis import Restaurants
from ..tools.googleDistanceMatrix.apis import GoogleDistanceMatrix
from ..tools.attractions.apis import Attractions
import math
import json
import re
import numpy as np
import os
import sys
from tqdm import tqdm
import argparse
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
# os.chdir(os.path.dirname(os.path.abspath(__file__)))


flight = Flights()
accommodation = Accommodations()
restaurants = Restaurants()
googleDistanceMatrix = GoogleDistanceMatrix()
attractions = Attractions()

def extract_numbers_from_filenames(directory):
    # Define the pattern to match files
    pattern = r'annotation_(\d+).json'

    # List all files in the directory
    files = os.listdir(directory)

    # Extract numbers from filenames that match the pattern
    numbers = [int(re.search(pattern, file).group(1)) for file in files if re.match(pattern, file)]

    return numbers

def extract_before_parenthesis(s):
    match = re.search(r'^(.*?)\([^)]*\)', s)
    return match.group(1) if match else s

def get_valid_name_city(info):
    # Modified the pattern to preserve spaces at the end of the name
    pattern = r'(.*?),\s*([^,]+)(\(\w[\w\s]*\))?$'
    match = re.search(pattern, info)
    if match:
        return match.group(1).strip(), extract_before_parenthesis(match.group(2).strip()).strip()
    else:
        print(f"{info} can not be parsed, '-' will be used instead.")
        return "-","-"
    
    
def load_line_json_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.read().strip().split('\n'):
            unit = json.loads(line)
            data.append(unit)
    return data


def convert_bool_values(item):
    if isinstance(item, dict):
        # If the item is a dictionary, recurse on each value
        return {key: convert_bool_values(value) for key, value in item.items()}
    elif isinstance(item, list):
        # If the item is a list, recurse on each item in the list
        return [convert_bool_values(value) for value in item]
    elif isinstance(item, tuple):
        # If the item is a tuple, recurse on each item in the tuple and repackage as a tuple
        return tuple(convert_bool_values(value) for value in item)
    elif isinstance(item, np.bool_):  # Here we check for numpy's bool_ type
        # If the item is a numpy bool_, convert it to a standard Python bool
        return bool(item)
    else:
        # If the item is any other type, return it unchanged
        return item




def extract_from_to(text: str):
    """
    Extracts 'A' and 'B' from the format "from A to B" in the given text, with B ending at a comma or the end of the string.
    
    Args:
    - text (str): The input string.
    
    Returns:
    - tuple: A tuple containing 'A' and 'B'. If no match is found, returns (None, None).
    """
    pattern = r"from\s+(.+?)\s+to\s+([^,]+)(?=[,\s]|$)"
    matches = re.search(pattern, text)
    return matches.groups() if matches else (None, None)


def get_total_cost(question, tested_data):
    total_cost = 0
    for i in range(min(question['days'],len(tested_data))):
        unit = tested_data[i]
        # transporation 
        if unit['transportation'] and  unit['transportation'] != '-':
            value = unit['transportation']
            org_city, dest_city = extract_from_to(value)
            if org_city == None or dest_city == None:
                org_city, dest_city = extract_from_to(unit['current_city'])
            
            if org_city == None or dest_city == None:
                pass
            else:
                if 'flight number' in value.lower():
                    res = flight.data[flight.data['Flight Number'] == value.split('Flight Number: ')[1].split(',')[0]]
                    if len(res) > 0:
                        total_cost += res['Price'].values[0] * question['people_number']
                
                elif 'self-driving' in value.lower() or 'taxi' in value.lower():
                    if 'self-driving' in value.lower():
                        # print(org_city,dest_city)
                        cost = googleDistanceMatrix.run_for_evaluation(org_city,dest_city,'self-driving')['cost']
                        total_cost += cost * math.ceil(question['people_number'] * 1.0 / 5)
                    else:
                        cost = googleDistanceMatrix.run_for_evaluation(org_city,dest_city,'taxi')['cost']
                        total_cost += cost * math.ceil(question['people_number'] * 1.0 / 4)
        
        # breakfast
        if unit['breakfast'] and unit['breakfast'] != '-':
            name, city = get_valid_name_city(unit['breakfast'])
            res = restaurants.data[(restaurants.data['Name'].astype(str).str.contains(re.escape(name))) & (restaurants.data['City'] == city)]
            if len(res) > 0:
                total_cost += res['Average Cost'].values[0] * question['people_number']

            
        # lunch
        if unit['lunch'] and unit['lunch'] != '-':
            name, city = get_valid_name_city(unit['lunch'])
            res = restaurants.data[(restaurants.data['Name'].astype(str).str.contains(re.escape(name))) & (restaurants.data['City'] == city)]
            if len(res) > 0:
                total_cost += res['Average Cost'].values[0] * question['people_number']
        
        # dinner
        if unit['dinner'] and unit['dinner'] != '-':
            name, city = get_valid_name_city(unit['dinner'])
            res = restaurants.data[(restaurants.data['Name'].astype(str).str.contains(re.escape(name))) & (restaurants.data['City'] == city)]
            if len(res) > 0:
                total_cost += res['Average Cost'].values[0] * question['people_number']
        
        # accommodation
        if unit['accommodation'] and unit['accommodation'] != '-':
            name, city = get_valid_name_city(unit['accommodation'])
            res = accommodation.data[(accommodation.data['NAME'].astype(str).str.contains(re.escape(name))) & (accommodation.data['city'] == city)]
            if len(res) > 0:
                total_cost += res['price'].values[0] * math.ceil(question['people_number'] * 1.0 / res['maximum occupancy'].values[0])
    # print(total_cost)
    return total_cost


def is_valid_room_rule(question, tested_data):

    if question['local_constraint']['house rule'] is None:
        return None,None
    
    for i in range(min(question['days'],len(tested_data))):
        unit = tested_data[i]
        if unit['accommodation'] and unit['accommodation'] != '-':
            name, city = get_valid_name_city(unit['accommodation'])
            res = accommodation.data[(accommodation.data['NAME'].astype(str).str.contains(re.escape(name))) & (accommodation.data['city'] == city)]
            if len(res) > 0:
                if question['local_constraint']['house rule'] == 'smoking' and 'No smoking' in str(res['house_rules'].values[0]):
                    return False, f"The house rule should be {question['local_constraint']['house rule']}."
                if question['local_constraint']['house rule'] == 'parties' and 'No parties' in str(res['house_rules'].values[0]):
                    return False, f"The house rule should be {question['local_constraint']['house rule']}."
                if question['local_constraint']['house rule'] == 'children under 10' and 'No children under 10' in str(res['house_rules'].values[0]):
                    return False, f"The house rule should be {question['local_constraint']['house rule']}."
                if question['local_constraint']['house rule'] == 'visitors' and 'No visitors' in str(res['house_rules'].values[0]):
                    return False, f"The house rule should be {question['local_constraint']['house rule']}."
                if question['local_constraint']['house rule'] == 'pets' and 'No pets' in str(res['house_rules'].values[0]):
                    return False, f"The house rule should be {question['local_constraint']['house rule']}."
                
            
    return True, None



def is_valid_cuisine(question, tested_data):
    cuisine_set = set()
    if question['local_constraint']['cuisine']:
        for i in range(min(question['days'],len(tested_data))):
            unit = tested_data[i]

            if unit['breakfast'] and unit['breakfast'] != '-':
                name, city = get_valid_name_city(unit['breakfast'])
                if city == question['org']:
                    continue
                res = restaurants.data[(restaurants.data['Name'].astype(str).str.contains(re.escape(name))) & (restaurants.data['City'] == city)]
                if len(res) > 0:       
                    for cuisine in question['local_constraint']['cuisine']:
                        if cuisine in res.iloc[0]['Cuisines']:
                            cuisine_set.add(cuisine)

            if unit['lunch'] and unit['lunch'] != '-':
                name, city = get_valid_name_city(unit['lunch'])
                if city == question['org']:
                    continue
                res = restaurants.data[(restaurants.data['Name'].astype(str).str.contains(re.escape(name))) & (restaurants.data['City'] == city)]
                if len(res) > 0:
                    for cuisine in question['local_constraint']['cuisine']:
                        if cuisine in res.iloc[0]['Cuisines']:
                            cuisine_set.add(cuisine)

            if unit['dinner'] and unit['dinner'] != '-':
                name, city = get_valid_name_city(unit['dinner'])
                if city == question['org']:
                    continue
                res = restaurants.data[(restaurants.data['Name'].astype(str).str.contains(re.escape(name))) & (restaurants.data['City'] == city)]
                if len(res) > 0:
                    for cuisine in question['local_constraint']['cuisine']:
                        if cuisine in res.iloc[0]['Cuisines']:
                            cuisine_set.add(cuisine)

        if len(cuisine_set) == len(question['local_constraint']['cuisine']):
            return True, None
        else:
            # judge which cuisine is not satisfied
            for cuisine in question['local_constraint']['cuisine']:
                if cuisine not in cuisine_set:
                    return False, f"The cuisine {cuisine} is not satisfied."
            # return False, f"The cuisine should be {question['local_constraint']['cuisine']}."
    else:
        return None,None
    

def is_valid_transportation(question, tested_data):
    if question['local_constraint']['transportation'] is None:
        return None,None
    for i in range(min(question['days'],len(tested_data))):
        unit = tested_data[i]
        if unit['transportation'] and unit['transportation'] != '-':
            value = unit['transportation']
            if question['local_constraint']['transportation'] == 'no flight' and 'Flight' in value:
                return False, f"The transportation should not be {question['local_constraint']['transportation']}."
            elif question['local_constraint']['transportation'] == 'no self-driving' and 'Self-driving'  in value:
                return False, f"The transportation should not be {question['local_constraint']['transportation']}."
            
    return True, None


def is_valid_room_type(question, tested_data):
    if question['local_constraint']['room type'] is None:
        return None,None
    for i in range(min(question['days'],len(tested_data))):
        unit = tested_data[i]
        if unit['accommodation'] and unit['accommodation'] != '-':
            name, city = get_valid_name_city(unit['accommodation'])
            res = accommodation.data[(accommodation.data['NAME'].astype(str).str.contains(re.escape(name))) & (accommodation.data['city'] == city)]
            if len(res) > 0:
                if question['local_constraint']['room type'] == 'not shared room' and res['room type'].values[0] == 'Shared room':
                    return False, f"The room type should be {question['local_constraint']['room type']}."
                # "shared room", "not shared room", "private room", "entire room"
                elif question['local_constraint']['room type'] == 'shared room' and res['room type'].values[0] != 'Shared room':
                    return False, f"The room type should be {question['local_constraint']['room type']}."

                elif question['local_constraint']['room type'] == 'private room' and res['room type'].values[0] != 'Private room':
                    return False, f"The room type should be {question['local_constraint']['room type']}."

                elif question['local_constraint']['room type'] == 'entire room' and res['room type'].values[0] != 'Entire home/apt':
                    return False, f"The room type should be {question['local_constraint']['room type']}."

    return True, None


def evaluation(query_data, tested_data):
    return_info = {}
    return_info['valid_cuisine'] = is_valid_cuisine(query_data, tested_data)
    return_info['valid_room_rule'] = is_valid_room_rule(query_data, tested_data)
    return_info['valid_transportation'] = is_valid_transportation(query_data, tested_data)
    return_info['valid_room_type'] = is_valid_room_type(query_data, tested_data)
    return_info['valid_cost'] = (bool(get_total_cost(query_data, tested_data) <= query_data['budget']), None)
    return return_info

def boolean_evaluation(query_data, tested_data):
    return_info = {}
    return_info['valid_cuisine'] = is_valid_cuisine(query_data, tested_data)
    return_info['valid_room_rule'] = is_valid_room_rule(query_data, tested_data)
    return_info['valid_transportation'] = is_valid_transportation(query_data, tested_data)
    return_info['valid_room_type'] = is_valid_room_type(query_data, tested_data)
    return_info['valid_cost'] = (bool(get_total_cost(query_data, tested_data) <= query_data['budget']), None)
    for key in return_info:
        if return_info[key][0] == False:
            print(key)
            return False
    return True

