import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from .commonsense_constraint import evaluation as commonsense_eval
from .hard_constraint import evaluation as hard_eval
# def convert_json_plan_to_string_format(json_plan):
#     """
#     将JSON格式的旅行计划转换为原始的字符串格式
    
#     Args:
#         json_plan (list): JSON格式的旅行计划
        
#     Returns:
#         list: 原始字符串格式的旅行计划
#     """
#     string_plan = []
    
#     for day in json_plan:
#         day_dict = {
#             "days": day["days"]
#         }
        
#         # 处理current_city
#         if isinstance(day.get("current_city"), dict):
#             if "from" in day["current_city"] and "to" in day["current_city"]:
#                 day_dict["current_city"] = f"from {day['current_city']['from']} to {day['current_city']['to']}"
#             else:
#                 # 如果格式不完整，则使用原始值或空字符串
#                 day_dict["current_city"] = str(day["current_city"])
#         else:
#             day_dict["current_city"] = str(day.get("current_city", ""))
        
#         # 处理transportation
#         trans = day.get("transportation", "-")
#         if isinstance(trans, dict):
#             if trans.get("type") == "flight":
#                 day_dict["transportation"] = (
#                     f"Flight Number: {trans.get('flight_number', '')}, "
#                     f"from {trans.get('from', '')} to {trans.get('to', '')}, "
#                     f"Departure Time: {trans.get('departure_time', '')}, "
#                     f"Arrival Time: {trans.get('arrival_time', '')}"
#                 )
#             elif trans.get("type") == "taxi":
#                 day_dict["transportation"] = f"Taxi; Cost: {trans.get('cost', 0)}"
#             elif trans.get("type") == "self-driving":
#                 day_dict["transportation"] = f"Self-driving; Cost: {trans.get('cost', 0)}"
#             else:
#                 day_dict["transportation"] = str(trans)
#         else:
#             day_dict["transportation"] = str(trans)
        
#         # 处理餐饮信息
#         for meal in ["breakfast", "lunch", "dinner"]:
#             meal_info = day.get(meal, "-")
#             if isinstance(meal_info, dict):
#                 cost_str = f"; Cost: {meal_info.get('cost', '')}" if "cost" in meal_info else ""
#                 day_dict[meal] = f"{meal_info.get('name', '')}, {meal_info.get('city', '')}{cost_str}"
#             else:
#                 day_dict[meal] = str(meal_info)
        
#         # 处理景点
#         attractions = day.get("attraction", [])
#         if isinstance(attractions, list) and len(attractions) > 0:
#             attraction_str = ""
#             for attr in attractions:
#                 if isinstance(attr, dict):
#                     attraction_str += f"{attr.get('name', '')}, {attr.get('city', '')};"
#                 else:
#                     attraction_str += f"{str(attr)};"
#             day_dict["attraction"] = attraction_str
#         else:
#             day_dict["attraction"] = str(attractions)
        
#         # 处理住宿
#         accom = day.get("accommodation", "-")
#         if isinstance(accom, dict):
#             nights_str = f" for {accom.get('nights', '')} nights" if "nights" in accom else ""
#             cost_str = f"; Cost: {accom.get('cost', '')}" if "cost" in accom else ""
#             day_dict["accommodation"] = f"{accom.get('name', '')}, {accom.get('city', '')}{cost_str}{nights_str}"
#         else:
#             day_dict["accommodation"] = str(accom)
        
#         string_plan.append(day_dict)
    
#     return string_plan


def evaluate_plan(query_data: dict, plan: list, weight: list=[1,1,1,1,1,1]) -> dict:
    """
    评估单个旅行计划是否满足给定的约束条件

    Args:
        query_data (dict): 约束条件
            {
                "level": "easy/medium/hard",
                "days": 3/5/7,
                "org": "出发城市",
                "dest": "目的地州",
                "budget": 预算金额,
                "local_constraint": {
                    "house rule": "房间规则约束" 或 None,
                    "cuisine": "餐厅类型约束" 或 None,
                    "room type": "房间类型约束" 或 None,
                    "transportation": "交通方式约束" 或 None
                }
            }
        
        plan (list): 待评估的旅行计划，每个元素是一天的行程
            [
                {
                    "days": 天数,
                    "current_city": "当前城市",
                    "transportation": "交通信息",
                    "breakfast": "早餐信息",
                    "attraction": "景点信息",
                    "lunch": "午餐信息",
                    "dinner": "晚餐信息",
                    "accommodation": "住宿信息"
                },
                ...
            ]

    Returns:
        dict: 评估结果，只包含五个关键指标
    """
    # 评估常识性约束
    commonsense_info_box = commonsense_eval(query_data, plan)
    
    # 如果通过基本检查，评估硬性约束
    hard_info_box = None
    if (commonsense_info_box and 
        commonsense_info_box['is_not_absent'][0] and 
        commonsense_info_box['is_valid_information_in_sandbox'][0]):
        hard_info_box = hard_eval(query_data, plan)

    # 构建评估结果
    result = {
        'Delivery Rate': 1 if plan else 0
    }

    # 处理常识性约束结果
    if commonsense_info_box:
        commonsense_pass = True
        commonsense_count = 0
        commonsense_pass_count = 0
        
        for key, value in commonsense_info_box.items():
            if value[0] is not None:
                commonsense_count += 1
                if value[0]:
                    commonsense_pass_count += 1
                else:
                    commonsense_pass = False
        
        result['Commonsense Constraint Micro Pass Rate'] = (
            commonsense_pass_count / commonsense_count if commonsense_count > 0 else 0
        )
        result['Commonsense Constraint Macro Pass Rate'] = 1 if commonsense_pass else 0
    else:
        result['Commonsense Constraint Micro Pass Rate'] = 0
        result['Commonsense Constraint Macro Pass Rate'] = 0

    # 处理硬性约束结果
    if hard_info_box:
        hard_pass = True
        hard_count = 0
        hard_pass_count = 0
        
        for key, value in hard_info_box.items():
            if value[0] is not None:
                hard_count += 1
                if value[0]:
                    hard_pass_count += 1
                else:
                    hard_pass = False
        
        result['Hard Constraint Micro Pass Rate'] = (
            hard_pass_count / hard_count if hard_count > 0 else 0
        )
        result['Hard Constraint Macro Pass Rate'] = 1 if hard_pass else 0
    else:
        result['Hard Constraint Micro Pass Rate'] = 0
        result['Hard Constraint Macro Pass Rate'] = 0
    
    # 计算最终通过状态
    commonsense_pass = result['Commonsense Constraint Macro Pass Rate'] == 1
    hard_pass = result['Hard Constraint Macro Pass Rate'] == 1
    result['Final Pass Rate'] = 1 if (commonsense_pass and hard_pass) else 0
    reward = 0
    for i, (key, value) in enumerate(result.items()):
        reward += weight[i] * value
    return reward


def evaluate_plan_infer(query_data: dict, plan: list, weight: list=[1,1,1,1,1,1]) -> dict:
    """
    评估单个旅行计划是否满足给定的约束条件

    Args:
        query_data (dict): 约束条件
            {
                "level": "easy/medium/hard",
                "days": 3/5/7,
                "org": "出发城市",
                "dest": "目的地州",
                "budget": 预算金额,
                "local_constraint": {
                    "house rule": "房间规则约束" 或 None,
                    "cuisine": "餐厅类型约束" 或 None,
                    "room type": "房间类型约束" 或 None,
                    "transportation": "交通方式约束" 或 None
                }
            }
        
        plan (list): 待评估的旅行计划，每个元素是一天的行程
            [
                {
                    "days": 天数,
                    "current_city": "当前城市",
                    "transportation": "交通信息",
                    "breakfast": "早餐信息",
                    "attraction": "景点信息",
                    "lunch": "午餐信息",
                    "dinner": "晚餐信息",
                    "accommodation": "住宿信息"
                },
                ...
            ]

    Returns:
        dict: 评估结果，只包含五个关键指标
    """
    # 评估常识性约束
    commonsense_info_box = commonsense_eval(query_data, plan)
    
    # 如果通过基本检查，评估硬性约束
    hard_info_box = None
    if (commonsense_info_box and 
        commonsense_info_box['is_not_absent'][0] and 
        commonsense_info_box['is_valid_information_in_sandbox'][0]):
        hard_info_box = hard_eval(query_data, plan)

    # 构建评估结果
    result = {
        'Delivery Rate': 1 if plan else 0
    }

    # 处理常识性约束结果
    if commonsense_info_box:
        commonsense_pass = True
        commonsense_count = 0
        commonsense_pass_count = 0
        
        for key, value in commonsense_info_box.items():
            if value[0] is not None:
                commonsense_count += 1
                if value[0]:
                    commonsense_pass_count += 1
                else:
                    commonsense_pass = False
        
        result['Commonsense Constraint Micro Pass Rate'] = (
            commonsense_pass_count / commonsense_count if commonsense_count > 0 else 0
        )
        result['Commonsense Constraint Macro Pass Rate'] = 1 if commonsense_pass else 0
    else:
        result['Commonsense Constraint Micro Pass Rate'] = 0
        result['Commonsense Constraint Macro Pass Rate'] = 0

    # 处理硬性约束结果
    if hard_info_box:
        hard_pass = True
        hard_count = 0
        hard_pass_count = 0
        
        for key, value in hard_info_box.items():
            if value[0] is not None:
                hard_count += 1
                if value[0]:
                    hard_pass_count += 1
                else:
                    hard_pass = False
        
        result['Hard Constraint Micro Pass Rate'] = (
            hard_pass_count / hard_count if hard_count > 0 else 0
        )
        result['Hard Constraint Macro Pass Rate'] = 1 if hard_pass else 0
    else:
        result['Hard Constraint Micro Pass Rate'] = 0
        result['Hard Constraint Macro Pass Rate'] = 0
    
    # 计算最终通过状态
    commonsense_pass = result['Commonsense Constraint Macro Pass Rate'] == 1
    hard_pass = result['Hard Constraint Macro Pass Rate'] == 1
    result['Final Pass Rate'] = 1 if (commonsense_pass and hard_pass) else 0
    reward = 0
    for i, (key, value) in enumerate(result.items()):
        reward += weight[i] * value
    return reward, result

# 使用示例
if __name__ == "__main__":
    # 示例输入数据
    query_data_example = {
        "level": "medium",
        "days": 7,
        "org": "Myrtle Beach",
        "dest": "Michigan",
        "budget": 8300,
        "local_constraint": {
            "house rule": None,
            "cuisine": ["French", "American"],
            "room type": None,
            "transportation": None
        }
    }

#     plan_example = [
#   {
#     "day": 1,
#     "current_city": "from Ithaca to Charlotte",
#     "transportation": "Flight Number: F3633413, from Ithaca to Charlotte, Departure Time: 05:38, Arrival Time: 07:46",
#     "breakfast": "Nagaland's Kitchen, Charlotte",
#     "attraction": "The Charlotte Museum of History, Charlotte",
#     "lunch": "Cafe Maple Street, Charlotte",
#     "dinner": "Bombay Vada Pav, Charlotte",
#     "accommodation": "Affordable Spacious Refurbished Room in Bushwick!, Charlotte"
#   },
#   {
#     "day": 2,
#     "current_city": "Charlotte",
#     "transportation": "-",
#     "breakfast": "Olive Tree Cafe, Charlotte",
#     "attraction": "The Mint Museum, Charlotte;Romare Bearden Park, Charlotte",
#     "lunch": "Birbal Ji Dhaba, Charlotte",
#     "dinner": "Pind Balluchi, Charlotte",
#     "accommodation": "Affordable Spacious Refurbished Room in Bushwick!, Charlotte"
#   },
#   {
#     "day": 3,
#     "current_city": "from Charlotte to Ithaca",
#     "transportation": "Flight Number: F3786167, from Charlotte to Ithaca, Departure Time: 21:42, Arrival Time: 23:26",
#     "breakfast": "Subway, Charlotte",
#     "attraction": "Books Monument, Charlotte",
#     "lunch": "Olive Tree Cafe, Charlotte",
#     "dinner": "Kylin Skybar, Charlotte",
#     "accommodation": "-"
#   }
# ] 
    test_file = "/home/hadoop-kg-llm-ddpt/dolphinfs_hdd_hadoop-kg-llm-ddpt/shanyingyu/TravelPlanner/data/query/final_annotation_easy_test.jsonl"
    with open(test_file, 'r') as f:
        query_ls = json.load(f)
    
    output_path = "/home/hadoop-kg-llm-ddpt/dolphinfs_hdd_hadoop-kg-llm-ddpt/chengzihao02/Travel-R1/output1.json"
    with open(output_path, 'r') as f:
        plan_ls = json.load(f)

    # 执行评估
    for query, plan in zip(query_ls, plan_ls):
        final_answer_reward = evaluate_plan(query, plan['answer'])
        print(final_answer_reward)