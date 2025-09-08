import os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from commonsense_constraint import evaluation as commonsense_eval
from hard_constraint import evaluation as hard_eval
import json
from tqdm import tqdm
from datasets import load_dataset
import argparse

def load_line_json_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.read().strip().split('\n'):
            unit = json.loads(line)
            data.append(unit)
    return data

def count_true_false(data):
    """Count the number of true and false values in a list."""
    true_count = data.count(True)
    false_count = data.count(False)
    return true_count, false_count

def statistics(commonsense_statistic):
    """Generate statistics for each level and day in the given data with a different structure."""
    result = {level: {day: {} for day in commonsense_statistic[level]} for level in commonsense_statistic}
    
    for level, days in commonsense_statistic.items():
        for day, dicts in days.items():
            for dct in dicts:
                if dct:
                    for key, data in dct.items():
                        true_count, false_count = count_true_false(data)
                        if key not in result[level][day]:
                            result[level][day][key] = {"true": 0, "false": 0}
                        result[level][day][key]["true"] += true_count
                        result[level][day][key]["false"] += false_count
                
    return result

def paper_term_mapping(commonsense_constraint_record, hard_constraint_record):
    mapping_dict = {
        'is_valid_information_in_current_city':'Within Current City',
        'is_valid_information_in_sandbox':'Within Sandbox',
        'is_reasonable_visiting_city':'Reasonable City Route',
        'is_valid_restaurants':'Diverse Restaurants',
        'is_valid_transportation':'Non-conf. Transportation',
        'is_valid_attractions':'Diverse Attractions',
        'is_valid_accommodation':'Minimum Nights Stay',
        'is_not_absent':'Complete Information',
        'valid_cost':'Budget',
        'valid_room_rule':'Room Rule',
        'valid_cuisine':'Cuisine',
        'valid_room_type':'Room Type',
        'valid_transportation':'Transportation'
    }
    remap_commonsense_constraint_record = {level:{day:{} for day in [3,5,7]} for level in ['easy','medium','hard']} 
    remap_hard_constraint_record = {level:{day:{} for day in [3,5,7]} for level in ['easy','medium','hard']} 
    for level in commonsense_constraint_record:
        for day in commonsense_constraint_record[level]:
            remap_commonsense_constraint_record[level][day] = {mapping_dict[key] : val for key,val in commonsense_constraint_record[level][day].items()}
            remap_hard_constraint_record[level][day] = {mapping_dict[key] : val for key,val in hard_constraint_record[level][day].items()}
    return remap_commonsense_constraint_record, remap_hard_constraint_record


def eval_score(set_type: str, file_path: str, single_eval: bool = False, eval_index: int = 0):
    """
    评估函数
    Args:
        set_type: 数据集类型 ('train', 'validation', 'test')
        file_path: 评估文件路径
        single_eval: 是否只评估单条数据
        eval_index: 要评估的数据索引
    """
    if set_type == 'train':
        query_data_list = load_dataset('osunlp/TravelPlanner','train')['train']
    elif set_type == 'validation':
        query_data_list = load_dataset('osunlp/TravelPlanner','validation')['validation']
    
    query_data_list = [x for x in query_data_list]
    tested_plans = load_line_json_data(file_path)
    
    # 如果是单条评估，只保留指定索引的数据
    if single_eval:
        if eval_index >= len(query_data_list):
            raise ValueError(f"eval_index {eval_index} 超出数据范围 [0, {len(query_data_list)-1}]")
        query_data_list = [query_data_list[eval_index]]
        tested_plans = [tested_plans[eval_index]]

    hardConstraint_statistic = {level:{day:[] for day in [3,5,7]} for level in ['easy','medium','hard']} 
    commonsenseConstraint_statistic = {level:{day:[] for day in [3,5,7]} for level in ['easy','medium','hard']} 
    delivery_cnt = 0
    plan_constraint_store = []

    for idx in tqdm(range(0,len(query_data_list))):
        query_data = query_data_list[idx]
        tested_plan = tested_plans[idx]
        if type(query_data) == str:
            query_data = eval(query_data)
        if type(tested_plan) == str:
            tested_plan = eval(tested_plan)
        if type(query_data['local_constraint']) == str:
            query_data['local_constraint'] = eval(query_data['local_constraint'])

        if tested_plan['plan']:
            delivery_cnt += 1
            commonsense_info_box = commonsense_eval(query_data,tested_plan['plan'])
        else:
            commonsense_info_box = None

        if commonsense_info_box and commonsense_info_box['is_not_absent'][0] and commonsense_info_box['is_valid_information_in_sandbox'][0]:
            hard_info_box = hard_eval(query_data,tested_plan['plan'])
        else:
            hard_info_box = None

        plan_constraint_store.append({'commonsense_constraint':commonsense_info_box,'hard_constraint':hard_info_box})
        commonsenseConstraint_statistic[query_data['level']][query_data['days']].append(commonsense_info_box)
        hardConstraint_statistic[query_data['level']][query_data['days']].append(hard_info_box)

    constraint_record = {key: {day: {'house rule':0, 'cuisine':0, 'room type':0, 'transportation':0} for day in [3,5,7]} for key in ['medium','hard']}
    constraint_mapping = {'house rule':'valid_room_rule','cuisine':'valid_cuisine','room type':'valid_room_type','transportation':'valid_transportation'}
    mapping_constraint_record = {key: {day: {'valid_room_rule':0, 'valid_cuisine':0, 'valid_room_type':0, 'valid_transportation':0} for day in [3,5,7]} for key in ['medium','hard']}
    count_record = {key:{day:0 for day in [3,5,7]} for key in ['easy','medium','hard']}

    for unit in query_data_list:
        count_record[unit['level']][unit['days']] += 1
        for key in constraint_record['medium'][3]:
            if unit['local_constraint'][key] != None:
                constraint_record[unit['level']][unit['days']][key] += 1
                mapping_constraint_record[unit['level']][unit['days']][constraint_mapping[key]] += 1
    
    commonsenseConstraint_statistic_processed = statistics(commonsenseConstraint_statistic)
    hardConstraint_statistic_processed = statistics(hardConstraint_statistic)

    data_record = {key:{day:[] for day in [3,5,7]} for key in ['easy','medium','hard']}
    constraint_dis_record = {"commonsense":{"pass":0,"total":0},"hard":{"pass":0,"total":0}}
    constraint_count = {key:{day:{} for day in [3,5,7]} for key in ['easy','medium','hard']}

    for constraint in ['commonsense','hard']:
        if constraint == 'commonsense':
            constraint_statistic = commonsenseConstraint_statistic_processed
        elif constraint == 'hard':
            constraint_statistic = hardConstraint_statistic_processed

        key_dict = {
            'commonsense':[
                'is_valid_information_in_current_city',
                'is_valid_information_in_sandbox',
                'is_reasonable_visiting_city',
                'is_valid_restaurants',
                'is_valid_transportation',
                'is_valid_attractions',
                'is_valid_accommodation',
                'is_not_absent'
            ],
            'hard':[
                'valid_cost',
                'valid_room_rule',
                'valid_cuisine',
                'valid_room_type',
                'valid_transportation'
            ]
        }
        
        for key in constraint_statistic:
            for key2 in constraint_statistic[key]:
                if key2 == -1:
                    print(constraint_statistic[key])
                    exit(0)
                for key3 in key_dict[constraint]:
                    data_record[key][key2].append('0/0')
                    if key3 in constraint_statistic[key][key2]:
                        constraint_dis_record[constraint]['pass'] += constraint_statistic[key][key2][key3]['true']
                        if constraint == 'hard':
                            if key == 'hard' and key3 in ['valid_room_rule','valid_cuisine','valid_room_type','valid_transportation']:
                                data_record[key][key2][-1] = f"{constraint_statistic[key][key2][key3]['true']}/{mapping_constraint_record[key][key2][key3]}"
                                constraint_dis_record[constraint]['total'] += mapping_constraint_record[key][key2][key3]
                                hardConstraint_statistic_processed[key][key2][key3]['total'] = mapping_constraint_record[key][key2][key3]
                            elif key == 'medium' and key3 in ['valid_room_rule','valid_cuisine','valid_room_type']:
                                data_record[key][key2][-1] = f"{constraint_statistic[key][key2][key3]['true']}/{mapping_constraint_record[key][key2][key3]}"
                                constraint_dis_record[constraint]['total'] += mapping_constraint_record[key][key2][key3]
                                hardConstraint_statistic_processed[key][key2][key3]['total'] = mapping_constraint_record[key][key2][key3]
                            else:
                                data_record[key][key2][-1] = f"{constraint_statistic[key][key2][key3]['true']}/{count_record[key][key2]}"
                                if key3 in ['valid_cost','valid_visitng_city_number','valid_days']:
                                    constraint_dis_record[constraint]['total'] += count_record[key][key2]
                                    constraint_count[key][key2][key3] = count_record[key][key2]
                                    hardConstraint_statistic_processed[key][key2][key3]['total'] = count_record[key][key2]
                        else:
                            data_record[key][key2][-1] = f"{constraint_statistic[key][key2][key3]['true']}/{count_record[key][key2]}"
                            constraint_dis_record[constraint]['total'] += count_record[key][key2]
                            constraint_count[key][key2][key3] = count_record[key][key2]
                            commonsenseConstraint_statistic_processed[key][key2][key3]['total'] =  count_record[key][key2]

    final_all_cnt = 0
    final_commonsense_cnt = 0
    final_hardConstraint_cnt = 0
    final_all_cnt_map = {level:0 for level in ['easy','medium','hard']}

    for idx in (range(0,len(query_data_list))):
        if plan_constraint_store[idx]['commonsense_constraint']:
            final_commonsense_pass = True
            final_hardConstraint_pass = True
            for item in plan_constraint_store[idx]['commonsense_constraint']:
                if plan_constraint_store[idx]['commonsense_constraint'][item][0] is not None and not plan_constraint_store[idx]['commonsense_constraint'][item][0]:
                    final_commonsense_pass = False
                    break
            if plan_constraint_store[idx]['hard_constraint'] is None:
                continue
            for item in plan_constraint_store[idx]['hard_constraint']:
                if plan_constraint_store[idx]['hard_constraint'][item][0] is not None and  plan_constraint_store[idx]['hard_constraint'][item][0] == False:
                    final_hardConstraint_pass = False
                    break
                
            if final_commonsense_pass:
                final_commonsense_cnt += 1
            if final_hardConstraint_pass:
                final_hardConstraint_cnt += 1
            if final_commonsense_pass and final_hardConstraint_pass:
                final_all_cnt += 1
                final_all_cnt_map[query_data_list[idx]['level']] += 1

    result = {}
    remap_commonsense_constraint_record, remap_hard_constraint_record = paper_term_mapping(commonsenseConstraint_statistic_processed, hardConstraint_statistic_processed)

    if single_eval:
        # 单条数据的评估结果，使用与多条数据相同的指标格式
        result['Delivery Rate'] = delivery_cnt  # 1 或 0
        result['Commonsense Constraint Micro Pass Rate'] = constraint_dis_record['commonsense']['pass'] / 8  # 单条数据有8个常识约束
        result['Commonsense Constraint Macro Pass Rate'] = final_commonsense_cnt  # 1 或 0
        result['Hard Constraint Micro Pass Rate'] = constraint_dis_record['hard']['pass'] / 5  # 单条数据有5个硬性约束
        result['Hard Constraint Macro Pass Rate'] = final_hardConstraint_cnt  # 1 或 0
        result['Final Pass Rate'] = final_all_cnt  # 1 或 0
        
        # 添加详细的评估信息作为附加信息
        query_data = query_data_list[0]
        level = query_data['level']
        days = query_data['days']
        
        result['Details'] = {
            'Query Data': query_data,
            'Commonsense Constraints': commonsenseConstraint_statistic_processed[level][days],
            'Hard Constraints': hardConstraint_statistic_processed[level][days],
            'Constraint Details': {
                'commonsense_constraints': plan_constraint_store[0]['commonsense_constraint'],
                'hard_constraints': plan_constraint_store[0]['hard_constraint']
            } if plan_constraint_store[0]['commonsense_constraint'] else None
        }
    else:
        if set_type == 'train':
            result['Delivery Rate'] = delivery_cnt / 45
            result['Commonsense Constraint Micro Pass Rate'] = constraint_dis_record['commonsense']['pass'] / 360
            result['Commonsense Constraint Macro Pass Rate'] = final_commonsense_cnt / 45
            result['Hard Constraint Micro Pass Rate'] = constraint_dis_record['hard']['pass'] / 105
            result['Hard Constraint Macro Pass Rate'] = final_hardConstraint_cnt / 45
            result['Final Pass Rate'] = final_all_cnt / 45
        elif set_type == 'validation':
            result['Delivery Rate'] = delivery_cnt / 180
            result['Commonsense Constraint Micro Pass Rate'] = constraint_dis_record['commonsense']['pass'] / 1440
            result['Commonsense Constraint Macro Pass Rate'] = final_commonsense_cnt / 180
            result['Hard Constraint Micro Pass Rate'] = constraint_dis_record['hard']['pass'] / 420
            result['Hard Constraint Macro Pass Rate'] = final_hardConstraint_cnt / 180
            result['Final Pass Rate'] = final_all_cnt / 180
        elif set_type == 'test':
            result['Delivery Rate'] = delivery_cnt / 1000
            result['Commonsense Constraint Micro Pass Rate'] = constraint_dis_record['commonsense']['pass'] / 8000

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_type', type=str, required=True, help='数据集类型：train, validation, test')
    parser.add_argument('--evaluation_file_path', type=str, required=True, help='评估文件路径')
    parser.add_argument('--single_eval', action='store_true', help='是否只评估单条数据')
    parser.add_argument('--eval_index', type=int, default=0, help='要评估的数据索引')
    args = parser.parse_args()
    
    result = eval_score(args.set_type, args.evaluation_file_path, args.single_eval, args.eval_index)
    print(json.dumps(result, indent=2, ensure_ascii=False))