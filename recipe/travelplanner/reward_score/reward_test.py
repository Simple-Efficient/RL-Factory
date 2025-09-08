import sys
import os

# 添加必要的导入
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
import re
import math
import sys
import os


# 添加必要的导入
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from search_r1.tool.tools.flights.apis import Flights
from search_r1.tool.tools.accommodations.apis import Accommodations
from search_r1.tool.tools.restaurants.apis import Restaurants
from search_r1.tool.tools.googleDistanceMatrix.apis import GoogleDistanceMatrix
from search_r1.tool.tools.attractions.apis import Attractions

# 初始化工具实例
flight = Flights()
accommodation = Accommodations()
restaurants = Restaurants()
googleDistanceMatrix = GoogleDistanceMatrix()
attractions = Attractions()

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

def compute_hard_constraint(final_answer: dict) -> float:
    """
    根据 final_answer 计算硬约束奖励
    
    参数:
    - final_answer: 包含旅行计划的字典，必须包含'tested_data'字段，可能包含'hard_constraint_data'字段
    
    返回:
    - float: 硬约束奖励分数
      - 如果所有约束都满足，返回1.0
      - 如果至少满足一个约束，返回0.1
      - 如果一个约束都不满足，返回0.0
    """
    try:
        # 检查final_answer是否包含必要的字段
        if 'tested_data' not in final_answer:
            print("DEBUG: final_answer缺少tested_data字段")
            return 0.0
            
        tested_data = final_answer['tested_data']
        
        # 如果提供了hard_constraint_data字段，使用它；否则返回0.0
        if 'hard_constraint_data' not in final_answer:
            print("DEBUG: final_answer缺少hard_constraint_data字段")
            return 0.0
            
        hard_constraint_data = final_answer['hard_constraint_data']
        
        # 使用evaluation函数评估所有硬约束
        constraint_results = evaluation(hard_constraint_data, tested_data)
        
        # 计算满足的约束数量
        satisfied_constraints = 0
        total_constraints = 0
        
        for key, (is_valid, _) in constraint_results.items():
            # 只考虑有明确约束的情况（不是None）
            if is_valid is not None:
                total_constraints += 1
                if is_valid:
                    satisfied_constraints += 1
        
        if total_constraints == 0:
            return 0.0  # 没有约束，返回0
        elif satisfied_constraints == 0:
            return 0.0  # 一个约束都不满足，返回0
        elif satisfied_constraints == total_constraints:
            return 1.0  # 所有约束都满足，返回1.0
        else:
            return 0.1  # 至少满足一个约束但不是全部，返回0.1
            
    except Exception as e:
        print(f"DEBUG: 计算硬约束奖励时出错: {e}")
        return 0.0 

# 测试数据
basic_test_data = [
    {
        "current_city": "New York",
        "transportation": "Self-driving from New York to Boston",
        "breakfast": "Cafe Central",
        "lunch": "Pizza Place",
        "dinner": "Italian Restaurant",
        "accommodation": "Hotel ABC"
    }
]

print("开始测试...\n")

# 测试1：没有约束条件
print("测试1：没有约束条件")
final_answer = {
    "tested_data": basic_test_data,
    "hard_constraint_data": {
        "house rule": None,
        "cuisine": None,
        "transportation": None,
        "room type": None
    }
}
print("结果:", compute_hard_constraint(final_answer))

# 测试2：所有约束满足
print("\n测试2：所有约束满足")
final_answer = {
    "tested_data": basic_test_data,
    "hard_constraint_data": {
        "house rule": "smoking",
        "cuisine": ["Italian"],
        "transportation": "no flight",
        "room type": "private room"
    }
}
print("结果:", compute_hard_constraint(final_answer))

# 测试3：部分约束满足
print("\n测试3：部分约束满足")
final_answer = {
    "tested_data": basic_test_data,
    "hard_constraint_data": {
        "house rule": "smoking",
        "cuisine": ["Italian"],
        "transportation": None,
        "room type": None
    }
}
print("结果:", compute_hard_constraint(final_answer))

# 测试4：没有约束满足
print("\n测试4：没有约束满足")
final_answer = {
    "tested_data": basic_test_data,
    "hard_constraint_data": {
        "house rule": "no smoking",
        "cuisine": ["Japanese"],
        "transportation": "no flight",
        "room type": "shared room"
    }
}
print("结果:", compute_hard_constraint(final_answer))

print("\n测试完成!") 