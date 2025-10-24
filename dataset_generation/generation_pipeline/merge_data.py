import json
import random

random.seed(0)


skip_words = ['"', '\n', "'", '*', ':', '：']


with open('data0.jsonl', 'r', encoding='utf-8') as file:
    t_data = json.loads('[' + ','.join(file.readlines()) + ']')

    filtered_data = [i for i in t_data if all(word not in i for word in skip_words)]
    random.shuffle(filtered_data)
    o_problem_1_dataset: list = filtered_data[:2000]
random.shuffle(o_problem_1_dataset)


with open('find_object_en_new_yolo_9.jsonl', 'r', encoding='utf-8') as file:
    t_data = json.loads('[' + ','.join(file.readlines()) + ']')

    filtered_data = [i for i in t_data if all(word not in i for word in skip_words)]
    random.shuffle(filtered_data)
    o_problem_2_dataset: list = filtered_data[:5000]
with open('find_object_zh_new_yolo_9.jsonl', 'r', encoding='utf-8') as file:
    t_data = json.loads('[' + ','.join(file.readlines()) + ']')

    filtered_data = [i for i in t_data if all(word not in i for word in skip_words)]
    random.shuffle(filtered_data)
    o_problem_2_dataset.extend(filtered_data[:5000])
random.shuffle(o_problem_2_dataset)


with open('data_B.jsonl', 'r', encoding='utf-8') as file:
    t_data = json.loads('[' + ','.join(file.readlines()) + ']')

    filtered_data = [i for i in t_data if all(word not in i for word in skip_words)]
    random.shuffle(filtered_data)
    o_problem_3_dataset: list = filtered_data[:10000]
random.shuffle(o_problem_3_dataset)


with open('fly_control.jsonl', 'r', encoding='utf-8') as file:
    t_data = json.loads('[' + ','.join(file.readlines()) + ']')

    filtered_data = [i for i in t_data if all(word not in i for word in skip_words)]
    random.shuffle(filtered_data)
    o_problem_4_dataset: list = filtered_data[:5000]
with open('fly_control_en.jsonl', 'r', encoding='utf-8') as file:
    t_data = json.loads('[' + ','.join(file.readlines()) + ']')

    filtered_data = [i for i in t_data if all(word not in i for word in skip_words)]
    random.shuffle(filtered_data)
    o_problem_4_dataset.extend(filtered_data[:5000])
random.shuffle(o_problem_4_dataset)


with open('train_prompt.json', 'r', encoding='utf-8') as file:
    train_prompt = json.loads(''.join(file.readlines()))


problem_1_dataset = []
problem_2_dataset = []
problem_3_dataset = []
problem_4_dataset = []


for i in o_problem_1_dataset:
    t_query = train_prompt['problem_1']['prompt'][0].replace('{}', i['word'])
    t_response = f"{i['type']}."
    problem_1_dataset.append({'query': t_query, 'response': t_response})

for i in o_problem_2_dataset[:2000]:
    t_query = train_prompt['problem_1']['prompt'][0].replace('{}', i['words'])
    t_response = f"A."
    problem_1_dataset.append({'query': t_query, 'response': t_response})

for i in o_problem_3_dataset[:2000]:
    t_query = train_prompt['problem_1']['prompt'][0].replace('{}', i['user input'])
    t_response = f"B."
    problem_1_dataset.append({'query': t_query, 'response': t_response})

for i in o_problem_4_dataset[:2000]:
    t_query = train_prompt['problem_1']['prompt'][0].replace('{}', i['words'])
    t_response = f"C."
    problem_1_dataset.append({'query': t_query, 'response': t_response})


for i in o_problem_2_dataset:
    t_query = train_prompt['problem_2']['prompt'][0].replace('{}', i['words'])
    t_response = f"{i['key_objects']}"
    problem_2_dataset.append({'query': t_query, 'response': t_response})


for i in o_problem_3_dataset:
    t_query = train_prompt['problem_3']['prompt'][0].replace('{}', i['user input'])
    t_i_r = '; '.join(i['flight control command'].split('\n')) + '.'
    t_response = f"{t_i_r}"
    problem_3_dataset.append({'query': t_query, 'response': t_response})


for i in o_problem_4_dataset:
    t_query = train_prompt['problem_4']['prompt'][0].replace('{}', i['words'])
    t_response = f"{i['key_objects']}."
    problem_4_dataset.append({'query': t_query, 'response': t_response})


train_dataset = []
val_dataset = []


def split_dataset(dataset, train_list, val_list):
    for i, d in enumerate(dataset):
        if i % 5 == 0:
            val_list.append(d)
        else:
            train_list.append(d)


datasets = [problem_1_dataset, problem_2_dataset, problem_3_dataset, problem_4_dataset]

for dataset in datasets:
    split_dataset(dataset, train_dataset, val_dataset)

random.shuffle(train_dataset)
random.shuffle(val_dataset)


with open('../train_dataset_swift_4_type_new_yolo_9.jsonl', 'w', encoding='utf-8') as f:
    for entry in train_dataset:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
with open('../val_dataset_swift_4_type_new_yolo_9.jsonl', 'w', encoding='utf-8') as f:
    for entry in val_dataset:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
