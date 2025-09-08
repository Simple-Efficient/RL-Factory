# from utils.func import get_valid_name_city,extract_before_parenthesis,extract_numbers_from_filenames
import sys
sys.path.append("..")
from ..tools.flights.apis import Flights
from ..tools.accommodations.apis import Accommodations
from ..tools.restaurants.apis import Restaurants
from ..tools.googleDistanceMatrix.apis import GoogleDistanceMatrix
from ..tools.attractions.apis import Attractions
import math
import json
import re   
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

city_state_set = open('/home/hadoop-kg-llm-ddpt/dolphinfs_hdd_hadoop-kg-llm-ddpt/chengzihao02/Travel-R1/verl/utils/reward_score/database/background/citySet_with_states.txt','r').read().split('\n')
city_state_map = {x:y for x,y in [unit.split('\t') for unit in city_state_set]}

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


def count_consecutive_values(lst):
    if not lst:
        return []

    result = []
    current_string = lst[0]
    count = 1

    for i in range(1, len(lst)):
        if lst[i] == current_string:
            count += 1
        else:
            result.append((current_string, count))
            current_string = lst[i]
            count = 1

    result.append((current_string, count))  # Add the last group of values
    return result


def transportation_match(text: str):

    if 'taxi' in text.lower():
        return 'Taxi'
    
    elif 'self-driving' in text.lower():
        return 'Self-driving'
    
    elif 'flight' in text.lower():
        return 'Flight'


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



def is_valid_city_sequence(city_list):
    """
    Checks if the city sequence is valid. A valid sequence has every city (except the first and last) 
    appearing consecutively, and no city should appear again once its sequence is over.
    
    Args:
    - city_list (list): List of cities.
    
    Returns:
    - bool: True if the sequence is valid, False otherwise.
    """
    
    # If the list has less than 3 cities, it's invalid.
    if len(city_list) < 3:
        return False
    
    # Set to keep track of visited cities
    visited_cities = set()
    
    i = 0
    while i < len(city_list):
        city = city_list[i]
        
        # If the city was already visited, it's invalid.
        if city in visited_cities and (i != 0 and i != len(city_list) - 1):
            return False
        
        # Count the consecutive occurrences of the city
        count = 0
        while i < len(city_list) and city_list[i] == city:
            count += 1
            i += 1
        
        # If the city appeared only once in the medium, it's invalid.
        if count == 1 and 0 < i - 1 < len(city_list) - 1:
            return False
        
        visited_cities.add(city)
    
    return True



def is_reasonable_visiting_city(question, tested_data):

    city_list = []
    
    # print(tested_data)
    for i in range(min(question['days'],len(tested_data))):
        city_value = tested_data[i]['current_city']

        if 'from' in city_value:
            city1, city2 = extract_from_to(city_value)
            city1 = extract_before_parenthesis(city1)
            city2 = extract_before_parenthesis(city2)
            if i==0 and  city1 != question['org']:
                return False, f"The first day's city should be {question['org']}."

            city_list += [city1, city2]

        else:
            city_list.append(extract_before_parenthesis(city_value))
    
    if city_list[0] != city_list[-1]:
        return False, "The trip should be a closed circle."

    if not is_valid_city_sequence(city_list):
        return False, "The city sequence is invalid."
    
    for idx, city in enumerate(city_list):
        if city not in city_state_map:
            return False, f"{city} is not a valid city."
        if idx not in [0,len(city_list)-1] and question['days'] >3 and city_state_map[city] != question['dest']:
            return False, f"{city} is not in {question['dest']}."
    
    return True, None


def is_valid_restaurants(question, tested_data):

    restaurants_list = []

    for i in range(min(question['days'],len(tested_data))):
        unit = tested_data[i]

        if 'breakfast' in unit and unit['breakfast'] and unit['breakfast'] != '-':
            if unit['breakfast'] not in restaurants_list:
                restaurants_list.append(unit['breakfast'])
            else:
                return False, f"The restaurant in day {i+1} breakfast is repeated."
        # elif 'breakfast' not in unit :
        #     return False, f"No Breakfast Info."
            
        if 'lunch' in unit and unit['lunch'] and unit['lunch'] != '-':
            if unit['lunch'] not in restaurants_list:
                restaurants_list.append(unit['lunch'])
            else:
                return False, f"The restaurant in day {i+1} lunch {unit['lunch']} is repeated."
        # elif 'lunch' not in unit:
        #     return False, f"No Lunch Info."
        
        if 'dinner' in unit and unit['dinner'] and unit['dinner'] != '-':
            if unit['dinner'] not in restaurants_list:
                restaurants_list.append(unit['dinner'])
            else:
                return False, f"The restaurant in day {i+1} dinner is repeated."
        # elif 'dinner' not in unit:
        #     return False, f"No Dinner Info."

    return True, None
            
def is_valid_attractions(question, tested_data):

    attractions_list = []

    for i in range(min(question['days'],len(tested_data))):
        unit = tested_data[i]

        if 'attraction' in unit and unit['attraction'] and unit['attraction'] != '-':
            for attraction in unit['attraction'].split(';')[:-1]:
                if attraction not in attractions_list:
                    attractions_list.append(attraction)
                else:
                    return False, f"The attraction '{attraction}' in day {i+1} is repeated."
                
        # elif 'attraction' not in unit:
        #     return False, f"No Attraction Info."
        
    return True, None

def is_valid_transportation(question, tested_data):
    
    if tested_data[0]['transportation'] and tested_data[0]['transportation'] != '-':
        transportation_list = [transportation_match(tested_data[0]['transportation'])]
    
    else:
        return False, "The transportation in day 1 should not be empty."

    for i in range(min(question['days'],len(tested_data))):
        unit = tested_data[i]

        if 'transportation' in unit and unit['transportation'] and unit['transportation'] != '-':
            transportation_list.append(transportation_match(unit['transportation']))
        # elif 'transportation' not in unit:
        #     return False, f"No Transportation Info."
    
    if (('Self-driving' in transportation_list) and ('Flight' in transportation_list)) or (('Taxi' in transportation_list) and ('Self-driving' in transportation_list)):
        return False, "The transportation is conflicting."

    return True, None

def is_valid_information_in_current_city(question, tested_data):

    for i in range(min(question['days'],len(tested_data))):
        unit = tested_data[i]
        current_city = unit['current_city']
        final_city_list = []

        if 'from' in current_city:
            city1, city2 = extract_from_to(current_city)
            city1 = extract_before_parenthesis(city1)
            city2 = extract_before_parenthesis(city2)
            final_city_list = [city1, city2]
        else:
            final_city_list = extract_before_parenthesis(current_city)

        if 'transportation' in unit and unit['transportation'] and unit['transportation'] != '-':
            for city in final_city_list:
                if city not in unit['transportation']:
                    # print(city)
                    return False, f"The transportation in day {i+1} is invalid city choice."
        # elif 'transportation' not in unit:
        #     return False, f"No Transportation Info."
        
        if 'breakfast' in unit and unit['breakfast'] and unit['breakfast'] != '-':

            flag = False

            for city in final_city_list:
                if city  in unit['breakfast']:
                    flag = True

            if not flag:
                return False, f"The breakfast in day {i+1} is invalid city choice."
        # elif 'breakfast' not in unit:
        #     return False, f"No Breakfast Info."
        
        if 'lunch' in unit and unit['lunch'] and unit['lunch'] != '-':
            flag = False

            for city in final_city_list:
                if city  in unit['lunch']:
                    flag = True
            
            if not flag:
                return False, f"The lunch in day {i+1} is invalid city choice."
        # elif 'lunch' not in unit:
        #     return False, f"No Lunch Info."
            
        if 'dinner' in unit and unit['dinner'] and unit['dinner'] != '-':
            flag = False

            for city in final_city_list:
                if city  in unit['dinner']:
                    flag = True
            
            if not flag:
                return False, f"The dinner in day {i+1} is invalid city choice."
        # elif 'dinner' not in unit:
        #     return False, f"No Dinner Info."
        
        if 'attraction' in unit and unit['attraction'] and unit['attraction'] != '-':
            
            attraction_list = unit['attraction'].split(';')[:-1]

            for attraction in attraction_list:
                flag = False
                for city in final_city_list:
                    if city  in attraction:
                        flag = True
                if not flag:
                    return False, f"The attraction in day {i+1} is invalid city choice."
                
        # elif 'attraction' not in unit:
        #     return False, f"No Attraction Info."
            
            
        if 'accommodation' in unit and unit['accommodation'] and unit['accommodation'] != '-':
            
            if final_city_list[-1] not in unit['accommodation']:
                return False, f"The accommodation in day {i+1} is invalid city choice."
            
        # elif 'accommodation' not in unit:
        #     return False, f"No Accommodation Info."
    
    return True, None
        
# hallucination 
def is_valid_information_in_sandbox(question, tested_data):
    
    for i in range(min(question['days'],len(tested_data))):
        unit = tested_data[i]
        
        if unit['transportation'] and unit['transportation'] != '-':
            value = unit['transportation']
            org_city, dest_city = extract_from_to(value)
            if org_city == None or dest_city == None:
                org_city, dest_city = extract_from_to(unit['current_city'])
            if 'flight number' in value.lower():
                try:
                    org_city = extract_before_parenthesis(org_city)
                    dest_city = extract_before_parenthesis(dest_city)
                except TypeError:
                    raise ValueError("The transportation {} in day {} can not be parsed.".format(value,i+1))
                # print(value)
                if len(flight.data[(flight.data['Flight Number'] == value.split('Flight Number: ')[1].split(',')[0]) & (flight.data['OriginCityName']==org_city) & (flight.data['DestCityName']==dest_city)]) < 1:
                     return False, f"The flight number in day {i+1} is invalid in the sandbox."
            
            elif 'self-driving' in value.lower() or 'taxi' in value.lower():
                try:
                    org_city = extract_before_parenthesis(org_city)
                    dest_city = extract_before_parenthesis(dest_city)
                except TypeError:
                    org_city = '-'
                    dest_city = '-'
                    print("The transportation {} in day {} can not be parsed and '-' will be used instead.".format(value,i+1))
                
                if 'self-driving' in value.lower():
                    if googleDistanceMatrix.run_for_evaluation(org_city, dest_city, mode='self-driving')['cost'] == None:
                        return False, f"The self-driving in day {i+1} is invalid in the sandbox."
                else:
                    if googleDistanceMatrix.run_for_evaluation(org_city, dest_city, mode='taxi')['cost'] == None:
                        return False, f"The taxi in day {i+1} is invalid in the sandbox."

        if 'breakfast' in unit and unit['breakfast'] and unit['breakfast'] != '-':
            name, city = get_valid_name_city(unit['breakfast'])
            if len(restaurants.data[(restaurants.data['Name'].astype(str).str.contains(re.escape(name))) & (restaurants.data['City'] == city)]) < 1:
                return False, f"The breakfast in day {i+1} is invalid in the sandbox."
        # elif 'breakfast' not in unit:
        #     return False, f"No Breakfast Info."
        
        if 'lunch' in unit and unit['lunch'] and unit['lunch'] != '-':
            name, city = get_valid_name_city(unit['lunch'])
            if len(restaurants.data[(restaurants.data['Name'].astype(str).str.contains(re.escape(name))) & (restaurants.data['City'] == city)]) < 1:
                return False, f"The lunch in day {i+1} is invalid in the sandbox."
        # elif 'lunch' not in unit:
        #     return False, f"No Lunch Info."
        
        if 'dinner' in unit and unit['dinner'] and unit['dinner'] != '-':
            name, city = get_valid_name_city(unit['dinner'])
            if len(restaurants.data[(restaurants.data['Name'].astype(str).str.contains(re.escape(name))) & (restaurants.data['City'] == city)]) < 1:
                return False, f"The dinner in day {i+1} is invalid in the sandbox."
        # elif 'dinner' not in unit:
        #     return False, f"No Dinner Info."
            
        if 'attraction' in unit and unit['attraction'] and unit['attraction'] != '-':
            attractions_list = unit['attraction'].split(';')[:-1]
            for attraction in attractions_list:
                name, city = get_valid_name_city(attraction)
                if len(attractions.data[(attractions.data['Name'].astype(str).str.contains(re.escape(name))) & (attractions.data['City'] == city)]) < 1:
                    return False, f"The attraction {attraction} in day {i+1} is invalid in the sandbox."
        # elif 'attraction' not in unit:
        #     return False, f"No Attraction Info."
                
        if 'accommodation' in unit and unit['accommodation'] and unit['accommodation'] != '-':
            name, city = get_valid_name_city(unit['accommodation'])
            # print(name,city)
            # print(accommodation.data[accommodation.data['NAME'].astype(str).str.contains(re.escape(name))])
            if len(accommodation.data[(accommodation.data['NAME'].astype(str).str.contains(re.escape(name))) & (accommodation.data['city'] == city)]) < 1:
                return False, f"The accommodation in day {i+1} is invalid in the sandbox."
        # elif 'accommodation' not in unit:
        #     return False, f"No Accommodation Info."
        
    return True, None


def is_valid_accommodaton(question, tested_data):
    data = []
    for i in range(min(question['days'],len(tested_data))):
        unit = tested_data[i]

        if 'accommodation' not in unit:
            return False, f"No Accommodation Info."
        
        data.append(unit['accommodation'])
    # data = [unit['accommodation'] for unit in tested_data]
    consectutive_accommodation = count_consecutive_values(data)
    for unit in consectutive_accommodation:
        # print(unit)
        if unit and unit[0] not in  ['-',''] :
            name, city = get_valid_name_city(unit[0])
            # print(unit[0],name,city)
            # try:
            if len(accommodation.data[(accommodation.data['NAME'].astype(str).str.contains(re.escape(name))) & (accommodation.data['city'] == city)]) == 1 and unit[1] <  accommodation.data[(accommodation.data['NAME'].astype(str).str.contains(re.escape(name))) & (accommodation.data['city'] == city)].iloc[0]['minimum nights']:
                return False, f"The accommodation {unit[0]} do not obey the minumum nights rule."
            # can not parse data
            # except re.error:
            #     continue
            
    return True, None

def is_valid_visiting_city_number(question, tested_data):

    city_set = set()
    

    for i in range(min(question['days'],len(tested_data))):
        city_value = tested_data[i]['current_city']

        if 'from' in city_value:
            city1, city2 = extract_from_to(city_value)
            city1 = extract_before_parenthesis(city1)
            city2 = extract_before_parenthesis(city2)
            if i==0 and  city1 != question['org']:
                return False, f"The first day's city should be {question['org']}."

            city_set.add(city1)
            city_set.add(city2)

        else:
            city_set.add(extract_before_parenthesis(city_value))
    
    city_set.discard(question['org'])

    if len(city_set) != question['visiting_city_number']:
        return False, f"The number of visiting cities should be {question['visiting_city_number']}."
    
    return True, None

def is_valid_days(question, tested_data):
    lens = 0
    for i in range(min(question['days'],len(tested_data))):
        if tested_data[i] != {} and tested_data[i]['current_city'] != "You don't need to fill in the information for this or later days.":
            lens += 1
        
    if lens != question['days']:
        # print(lens)
        return False, f"The number of days should be {question['days']}."
    else:
        return True, None

def is_not_absent(question, tested_data):
    needed_info = 6 * question['days']
    total_valid_info = 0

    if not is_valid_days(question, tested_data)[0]:
        return False, "Invalid Days"
    
    if not is_valid_visiting_city_number(question, tested_data)[0]:
        return False, "Invalid City Number"

    for i in range(min(question['days'],len(tested_data))):
        unit = tested_data[i]

        if 'transportation' not in unit:
            return False, f"No Transportation Info."
        
        if 'breakfast' not in unit:
            return False, f"No Breakfast Info."
        
        if 'lunch' not in unit:
            return False, f"No Lunch Info."
        
        if 'dinner' not in unit:
            return False, f"No Dinner Info."
        
        if 'attraction' not in unit:
            return False, f"No Attraction Info."
        
        if 'accommodation' not in unit:
            return False, f"No Accommodation Info."
        
        if ('from ' in unit['current_city'] or 'to ' in unit['current_city']) and unit['transportation'] in ['','-']:
            return False, f"No transportation in day {i+1} is not allowed."
        
        if ('from ' not in unit['current_city'] and  ' to ' not in unit['current_city']) and unit['attraction'] in ['','-']:
            return False, f"No attaction in day {i+1} is not allowed."

        if i != question['days'] - 1 and unit['accommodation'] in ['','-']:
            return False, f"No accommodation in day {i+1} is not allowed."

        if (unit['breakfast'] in ['','-'] or unit['lunch'] in ['','-'] or unit['dinner'] in ['','-']) and 'from ' not in unit['current_city']:
            return False, f"No meal in day {i+1} is not allowed."
        

        for key in unit:
            if unit[key] and unit[key] != '-':
                total_valid_info += 1


    if total_valid_info * 1.0 / needed_info < 0.5:
        return False, f"The absent information is more than 50%."
    
    return True, None


def evaluation(query_data, tested_data):
    return_info = {}
    return_info['is_reasonable_visiting_city'] = is_reasonable_visiting_city(query_data, tested_data)
    return_info['is_valid_restaurants'] = is_valid_restaurants(query_data, tested_data)
    return_info['is_valid_attractions'] = is_valid_attractions(query_data, tested_data)
    return_info['is_valid_accommodation'] = is_valid_accommodaton(query_data, tested_data)
    return_info['is_valid_transportation'] = is_valid_transportation(query_data, tested_data)
    return_info['is_valid_information_in_current_city'] = is_valid_information_in_current_city(query_data, tested_data)
    return_info['is_valid_information_in_sandbox'] = is_valid_information_in_sandbox(query_data, tested_data)
    return_info['is_not_absent'] = is_not_absent(query_data, tested_data)
    return return_info

def boolean_evaluation(query_data, tested_data):
    return_info = {}
    return_info['is_reasonable_visiting_city'] = is_reasonable_visiting_city(query_data, tested_data)
    return_info['is_valid_restaurants'] = is_valid_restaurants(query_data, tested_data)
    return_info['is_valid_accommodation'] = is_valid_accommodaton(query_data, tested_data)
    return_info['is_valid_attractions'] = is_valid_attractions(query_data, tested_data)
    return_info['is_valid_transportation'] = is_valid_transportation(query_data, tested_data)
    return_info['is_valid_information_in_current_city'] = is_valid_information_in_current_city(query_data, tested_data)
    return_info['is_valid_information_in_sandbox'] = is_valid_information_in_sandbox(query_data, tested_data)
    return_info['is_not_absent'] = is_not_absent(query_data, tested_data)
    for key in return_info:
        if return_info[key][0] == False:
            print(return_info[key][1])
            return False
    return True
