import torch
import json
import os
from .metrics import flatten_structure, TYPE_LIST
from tabulate import tabulate

def postprocess_seg_result(result):
    metrics_j = {'total': 0} 
    metrics_f = {'total': 0} 
    number = {'total': 0}

    for d in result:
        metrics_j['total'] += d['j']
        metrics_f['total'] += d['f']
        number['total'] += 1
        if d['type'] not in metrics_j:
            metrics_j[d['type']] = 0
            metrics_f[d['type']] = 0
            number[d['type']] = 0
        metrics_j[d['type']] += d['j']
        metrics_f[d['type']] += d['f']
        number[d['type']] += 1

    metrics_jf = {}
    for k, v in number.items():
        j = metrics_j[k]
        f = metrics_f[k]
        metrics_j[k] = j / v
        metrics_f[k] = f / v
        metrics_jf[k] = (j + f) / 2 / v
    
    # draw table
    headers = ['Metric'] + list(number.keys())
    all_num_row = ['all num']
    j_row = ['j']
    f_row = ['f']
    jf_row = ['j&f']

    for tp in headers[1:]:
        all_num_row.append(number[tp])
        j_row.append(f"{metrics_j[tp]:.2}")
        f_row.append(f"{metrics_f[tp]:.2}")
        jf_row.append(f"{metrics_jf[tp]:.2}")

    table_data = [all_num_row, j_row, f_row, jf_row]

    print("####### Results Summary #######")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    metrics = [metrics_j, metrics_f, metrics_jf]
    result.insert(0, metrics)
    return result


def postprocess_prop_result(infos):
    task_sum = {"all":0}
    task_right_num = {"all": 0}
    class_sum = {}
    class_right_num = {}
    idx_list = []
    for info in infos:
        if info is None:
            continue
        if info["idx"] not in idx_list:
            idx_list.append(info["idx"])
            if info["class_name"] is not None:
                class_name = info['class_name']
                class_name = class_name.replace(".", "")
                if class_name not in class_sum:
                    class_sum[class_name] = 0
                    class_right_num[class_name] = 0
                class_sum[class_name] += 1
                class_right_num[class_name] += info['score']

            task_type = info["type"]
            if task_type not in task_sum:
                task_sum[task_type] = 0
                task_right_num[task_type] = 0
            task_right_num[task_type] += info['score']
            task_sum[task_type] += 1

            task_sum["all"] += 1
            task_right_num["all"] += info['score']

    metrics = {}
    for tp in task_sum.keys():
        metrics[tp] = task_right_num[tp]/(task_sum[tp]+1e-6)
    
    headers = ['Metric'] + list(task_sum.keys())

    # draw table
    all_num_row = ['all num']
    accuracy_row = ['accuracy']

    for tp in headers[1:]:
        all_n = task_sum[tp]
        right_n = task_right_num[tp]
        accuracy = right_n / all_n if all_n > 0 else 0
        
        all_num_row.append(all_n)
        accuracy_row.append(f"{accuracy:.2%}")

    table_data = [all_num_row, accuracy_row]

    print("####### Results Summary #######")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    for cl in class_sum.keys():
        metrics[cl] = class_right_num[cl]/(class_sum[cl]+1e-6)
    
    infos.insert(0, metrics)

    return infos



def postprocess_spatial_result(infos):
    type_list = flatten_structure(TYPE_LIST)
    all_sum = {"all":0}
    right_num = {"all": 0}
    idx_list = []
    for info in infos:
        if info is None:
            continue
        if info["idx"] not in idx_list:
            idx_list.append(info["idx"])
            if info["class_name"] is not None:
                class_name = info['class_name']
                class_name = class_name.replace(".", "")
                if class_name not in all_sum:
                    all_sum[class_name] = 0
                    right_num[class_name] = 0
                all_sum[class_name] += 1
                right_num[class_name] += info['score']

            for k, v in type_list.items():
                if info["type"] in v:
                    major_type = k
                    if major_type not in all_sum:
                        all_sum[major_type] = 0
                        right_num[major_type] = 0
                    right_num[major_type] += info['score']
                    all_sum[major_type] += 1
            all_sum["all"] += 1
            right_num["all"] += info['score']

    metrics = {}
    for tp in all_sum.keys():
        metrics[tp] = right_num[tp]/(all_sum[tp]+1e-6)
    
    headers = ['Metric'] + list(right_num.keys())

    # draw table
    all_num_row = ['all num']
    accuracy_row = ['accuracy']

    for tp in headers[1:]:
        all_n = all_sum[tp]
        right_n = right_num[tp]
        accuracy = right_n / all_n if all_n > 0 else 0
        
        all_num_row.append(all_n)
        accuracy_row.append(f"{accuracy:.2%}")

    table_data = [all_num_row, accuracy_row]

    print("####### Results Summary #######")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    infos.insert(0, metrics)

    return infos


def save_results(data, save_path):
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if save_path.endswith(".json"):
        with open(save_path, "w") as f:
            json.dump(data, f, indent=4)
    elif save_path.endswith(".jsonl"):
        with open(save_path, "w") as f:
            for info in data:
                f.write(json.dumps(info) + "\n")
    else:
        raise ValueError("Unsupported file format.")
    print(f"Answer saved at:{save_path}")
