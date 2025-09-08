from tools.flights.apis import Flights
from tools.accommodations.apis import Accommodations
from tools.restaurants.apis import Restaurants
from tools.googleDistanceMatrix.apis import GoogleDistanceMatrix
from tools.attractions.apis import Attractions
from evaluation.hard_constraint import extract_from_to,get_valid_name_city
import math


class ReactEnv:
    def __init__(self):
        
        self.flight = Flights()
        self.accommodation = Accommodations()
        self.restaurants = Restaurants()
        self.googleDistanceMatrix = GoogleDistanceMatrix()
        self.attractions = Attractions()
    
    def run(self, tested_data):

        total_cost = 0
        unit = tested_data
        people_number = tested_data['people_number']
        returned_info = []

        if 'transportation' in unit and unit['transportation'] and  unit['transportation'] != '-':
            value = unit['transportation']
            org_city, dest_city = extract_from_to(value)
            if org_city == None or dest_city == None:
                org_city, dest_city = extract_from_to(unit['current_city'])
            if 'flight number' in value.lower():
                    try:
                        res = self.flight.data[self.flight.data['Flight Number'] == value.split('Flight Number: ')[1].split(',')[0]]
                        if len(res) > 0:
                            total_cost += res['Price'].values[0] * people_number
                        else:
                            returned_info.append('The filght information is not valid')
                    except:
                        returned_info.append('The filght information is not valid')

            elif 'self-driving' in value.lower() or 'taxi' in value.lower():
                try:
                    if 'self-driving' in value.lower():
                        # print(org_city,dest_city)
                        cost = self.googleDistanceMatrix.run_for_evaluation(org_city,dest_city,'self-driving')['cost']
                        if cost == None:
                            returned_info.append('The transporation information is not valid, please check.')
                        else:
                            total_cost += cost * math.ceil(people_number * 1.0 / 5)
                    else:
                        cost = self.googleDistanceMatrix.run_for_evaluation(org_city,dest_city,'taxi')['cost']
                        if cost == None:
                            returned_info.append('The transporation information is not valid, please check.')
                        else:
                            total_cost += cost * math.ceil(people_number * 1.0 / 4)
                except:
                    returned_info.append('The transporation information is not valid, please check. You have to make sure there are two cities (from A to B) in your transportation plan.')

        if 'breakfast' in unit and unit['breakfast'] and unit['breakfast'] != '-':
            name, city = get_valid_name_city(unit['breakfast'])
            if name != '-' and city != '-':
                res = self.restaurants.data[(self.restaurants.data['Name'] == name) & (self.restaurants.data['City'] == city)]
                if len(res) > 0:
                    total_cost += res['Average Cost'].values[0] * people_number
                else:
                    returned_info.append('The breakfase information is not valid, please check.')

        if 'lunch' in unit and  unit['lunch'] and unit['lunch'] != '-':
            name, city = get_valid_name_city(unit['lunch'])
            if name != '-' and city != '-':
                res = self.restaurants.data[(self.restaurants.data['Name'] == name) & (self.restaurants.data['City'] == city)]
                if len(res) > 0:
                    total_cost += res['Average Cost'].values[0] * people_number
                else:
                    returned_info.append('The lunch information is not valid, please check.')

        if 'dinner' in unit and unit['dinner'] and unit['dinner'] != '-':
            name, city = get_valid_name_city(unit['dinner'])
            if name != '-' and city != '-':
                res = self.restaurants.data[(self.restaurants.data['Name'] == name) & (self.restaurants.data['City'] == city)]
                if len(res) > 0:
                    total_cost += res['Average Cost'].values[0] * people_number
                else:
                    returned_info.append('The dinner information is not valid, please check.')

        if 'accommodation' in unit and unit['accommodation'] and unit['accommodation'] != '-':
            name, city = get_valid_name_city(unit['accommodation'])
            if name != '-' and city != '-':
                res = self.accommodation.data[(self.accommodation.data['NAME'] == name) & (self.accommodation.data['city'] == city)]
                if len(res) > 0:
                    total_cost += res['price'].values[0] * math.ceil(people_number * 1.0 / res['maximum occupancy'].values[0])
                else:
                    returned_info.append('The accommodation information is not valid, please check.')
        
        if len(returned_info) == 0:
            return "The cost of your plan is " + str(total_cost) + " dollars."
        else:
            message = "Sorry, the cost of your plan is not available because of the following reasons:"
            for idx, info in enumerate(returned_info):
                message += str(idx + 1) + ". " + info + " " + '\t'
            return message
        
class ReactReflectEnv(ReactEnv):
    def __init__(self):
        super().__init__()
        self.is_terminated = False
        self.max_retry_step = 3
        self.retry_step = 0

    def reset(self):
        self.is_terminated = False
        self.retry_step = 0

    def run(self, tested_data):
        total_cost = 0
        unit = tested_data
        people_number = tested_data['people_number']
        returned_info = []

        if 'transportation' in unit and unit['transportation'] and  unit['transportation'] != '-':
            value = unit['transportation']
            org_city, dest_city = extract_from_to(value)
            if org_city == None or dest_city == None:
                org_city, dest_city = extract_from_to(unit['current_city'])
                
            
            if org_city == None or dest_city == None:
                returned_info.append('The transporation information is not valid, please check.')

            else:    
                if 'flight number' in value.lower():
                        try:
                            res = self.flight.data[self.flight.data['Flight Number'] == value.split('Flight Number: ')[1].split(',')[0]]
                            if len(res) > 0:
                                total_cost += res['Price'].values[0] * people_number
                            else:
                                returned_info.append('The filght information is not valid')
                        except:
                            returned_info.append('The filght information is not valid')

                elif 'self-driving' in value.lower() or 'taxi' in value.lower():
                        if 'self-driving' in value.lower():
                            cost = self.googleDistanceMatrix.run_for_evaluation(org_city,dest_city,'self-driving')['cost']
                            if cost == None:
                                returned_info.append('The transporation information is not valid, please check.')
                            else:
                                total_cost += cost * math.ceil(people_number * 1.0 / 5)
                        else:
                            cost = self.googleDistanceMatrix.run_for_evaluation(org_city,dest_city,'taxi')['cost']
                            if cost == None:
                                returned_info.append('The transporation information is not valid, please check.')
                            else:
                                total_cost += cost * math.ceil(people_number * 1.0 / 4)

        if 'breakfast' in unit and unit['breakfast'] and unit['breakfast'] != '-':
            name, city = get_valid_name_city(unit['breakfast'])
            if name != '-' and city != '-':
                res = self.restaurants.data[(self.restaurants.data['Name'] == name) & (self.restaurants.data['City'] == city)]
                if len(res) > 0:
                    total_cost += res['Average Cost'].values[0] * people_number
                else:
                    returned_info.append('The breakfase information is not valid, please check.')

        if 'lunch' in unit and  unit['lunch'] and unit['lunch'] != '-':
            name, city = get_valid_name_city(unit['lunch'])
            if name != '-' and city != '-':
                res = self.restaurants.data[(self.restaurants.data['Name'] == name) & (self.restaurants.data['City'] == city)]
                if len(res) > 0:
                    total_cost += res['Average Cost'].values[0] * people_number
                else:
                    returned_info.append('The lunch information is not valid, please check.')

        if 'dinner' in unit and unit['dinner'] and unit['dinner'] != '-':
            name, city = get_valid_name_city(unit['dinner'])
            if name != '-' and city != '-':
                res = self.restaurants.data[(self.restaurants.data['Name'] == name) & (self.restaurants.data['City'] == city)]
                if len(res) > 0:
                    total_cost += res['Average Cost'].values[0] * people_number
                else:
                    returned_info.append('The dinner information is not valid, please check.')

        if 'accommodation' in unit and unit['accommodation'] and unit['accommodation'] != '-':
            name, city = get_valid_name_city(unit['accommodation'])
            if name != '-' and city != '-':
                res = self.accommodation.data[(self.accommodation.data['NAME'] == name) & (self.accommodation.data['city'] == city)]
                if len(res) > 0:
                    total_cost += res['price'].values[0] * math.ceil(people_number * 1.0 / res['maximum occupancy'].values[0])
                else:
                    returned_info.append('The accommodation information is not valid, please check.')
        
        if len(returned_info) == 0:
            self.retry_step = 0
            self.is_terminated = False
            return "The cost of your plan is " + str(total_cost) + " dollars."
        else:
            message = "Sorry, the cost of your plan is not available because of the following reasons:"
            for idx, info in enumerate(returned_info):
                message += str(idx + 1) + ". " + info + " " + '\t'
            self.retry_step += 1
            if self.retry_step >= self.max_retry_step:
                self.is_terminated = True
            return message
        
