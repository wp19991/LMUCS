import json

with open('./before/llama3.2_3b_instruct_before.jsonl', 'r', encoding='utf-8') as file:
    val_res_data = json.loads('[' + ','.join(file.readlines()) + ']')

print(val_res_data[0])


def is_true(t_data: dict):
    true_ans = [i.replace('.', '').strip() for i in t_data['response'].split(';')]
    model_ans = t_data['model_ans']
    all_in = True
    for i in true_ans:
        if i not in model_ans:
            all_in = False
            break
    if all_in:
        return True
    return False


true_count = 0
all_true = 0
for i in val_res_data:
    if i['response'] in i['model_ans']:
        all_true += 1
    if is_true(i):
        true_count += 1

print(true_count)
print('包含答案：', true_count / len(val_res_data))
print('完全相同：', all_true / len(val_res_data))