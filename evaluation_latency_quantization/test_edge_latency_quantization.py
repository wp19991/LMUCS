import requests
import json
import time
import os
import re

OLLAMA_API_URL = "http://192.168.137.37:11434/api/generate"

DATASET_FILE = "val_dataset_swift_4_type_new_yolo_9.jsonl"

LOG_FILE = "test_progress.jsonl"

MAX_SIZE_GB = 3.0

OLLAMA_LIST_OUTPUT = """
NAME                                     ID            SIZE      MODIFIED      
qwen2.5_0.5b_drone_f16:latest            f8f9bcb1d4c7  994 MB    15 minutes ago  
qwen2.5_0.5b_drone_q4:latest             ba97be3e2241  352 MB    15 minutes ago  
qwen2.5_1.5b_drone_f16:latest            62346d1bce76  3.1 GB    16 minutes ago  
qwen2.5_1.5b_drone_q4:latest             08f13bd54af4  934 MB    17 minutes ago  
qwen2.5_3b_drone_f16:latest              d516e991aedb  6.2 GB    17 minutes ago  
qwen2.5_3b_drone_q4:latest               f85bfd6767ce  1.8 GB    18 minutes ago  
phi3.5-mini_drone_f16:latest             432d686c4d90  7.6 GB    19 minutes ago  
phi3.5-mini_drone_q4:latest              8f45feb5d73e  2.2 GB    20 minutes ago  
llama3.2_1b_drone_f16:latest             35d510a75394  2.5 GB    21 minutes ago  
llama3.2_1b_drone_q4:latest              527a11c957b0  770 MB    21 minutes ago  
llama3.2_3b_drone_f16:latest             cbfaa7e6f389  6.4 GB    26 minutes ago  
llama3.2_3b_drone_q4:latest              81fe6c882114  1.9 GB    27 minutes ago  
gemma2-2b_drone_f16:latest               2cd7936f3730  5.2 GB    28 minutes ago  
gemma2-2b_drone_q4:latest                b3918034fb2d  1.6 GB    29 minutes ago  
deepseek-r1-qwen2.5-1.5b_drone_f16:latest  fa1dd670e06e  3.6 GB    29 minutes ago  
deepseek-r1-qwen2.5-1.5b_drone_q4:latest   c802c1a9151b  1.1 GB    30 minutes ago
"""


def ollama(prompt="1+1=?", model="xxx", system=''):
    t_json = {"model": model, "prompt": prompt, 'stream': False, "keep_alive": -1}
    if system:
        t_json['system'] = system
    r = requests.post(OLLAMA_API_URL, timeout=600, json=t_json)
    r.raise_for_status()
    return r.json()['response']


def parse_ollama_list(output):
    models = []
    lines = output.strip().split('\n')
    for line in lines[1:]:
        match = re.match(r'(\S+)\s+[a-f0-9]+\s+([\d.]+)\s+(MB|GB)', line)
        if match:
            name, size_str, unit = match.groups()
            size_gb = float(size_str) / 1024 if unit == 'MB' else float(size_str)
            models.append({'name': name, 'size_gb': size_gb})
    return models


def load_dataset(filename):
    if not os.path.exists(filename):
        print(f"错误: 数据集文件 '{filename}' 未找到。")
        exit()
    with open(filename, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def load_progress(log_file):
    completed = set()
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    log_entry = json.loads(line)
                    completed.add((log_entry['model_name'], log_entry['index']))
                except json.JSONDecodeError:
                    print(f"警告: 日志文件中发现无效行: {line.strip()}")
    return completed


def main():
    print("--- Ollama 模型性能与错误分析测试 ---")

    all_models = parse_ollama_list(OLLAMA_LIST_OUTPUT)
    eligible_models = [m for m in all_models if m['size_gb'] <= MAX_SIZE_GB]
    print(f"\n[1/5] 筛选出 {len(eligible_models)} 个小于等于 {MAX_SIZE_GB} GB 的模型...")
    for model in all_models:
        status = "符合" if model in eligible_models else "跳过"
        print(f"  - [{status}] {model['name']} ({model['size_gb']:.2f} GB)")

    if not eligible_models:
        print("\n没有找到符合大小限制的模型，程序退出。")
        return

    print(f"\n[2/5] 正在加载数据集 '{DATASET_FILE}'...")
    dataset = load_dataset(DATASET_FILE)
    print(f"  - 加载完成，共 {len(dataset)} 条数据。")

    print(f"\n[3/5] 正在检查进度日志 '{LOG_FILE}'...")
    completed_tasks = load_progress(LOG_FILE)
    print(f"  - 发现 {len(completed_tasks)} 条已完成记录，将自动跳过。")

    print("\n[4/5] 开始执行测试...")
    results = {}

    with open(LOG_FILE, 'a', encoding='utf-8') as log_f:
        for model_info in eligible_models:
            model_name = model_info['name']
            print(f"\n--- 开始测试模型: {model_name} ---")

            try:
                print(f"  - [预热] 正在加载模型 {model_name}...")
                _ = ollama(prompt="Hello", model=model_name)
                print("  - [预热] 模型加载完成。")
            except Exception as e:
                print(f"  - [错误] 模型 {model_name} 预热失败: {e}。将跳过此模型。")
                results[model_name] = {'failed_warmup': True}
                continue

            model_results = {
                'latencies': [], 'response_lengths': [],
                'correct_count': 0, 'total_count': 0,
                'failed_count': 0, 'failed_cases': []
            }

            for i, item in enumerate(dataset):
                if (model_name, i) in completed_tasks:
                    print(f"  - [跳过] 数据项 {i + 1}/{len(dataset)} (已在日志中)")
                    continue

                query, expected_response = item['query'], item['response']

                log_entry = {'model_name': model_name, 'index': i}

                try:
                    start_time = time.time()
                    actual_response = ollama(prompt=query, model=model_name)
                    latency = time.time() - start_time

                    response_len = len(actual_response)
                    is_correct = actual_response.strip() == expected_response.strip()

                    model_results['latencies'].append(latency)
                    model_results['response_lengths'].append(response_len)
                    model_results['total_count'] += 1

                    log_entry.update({
                        'latency': latency,
                        'is_correct': is_correct,
                        'response_length': response_len
                    })

                    if is_correct:
                        model_results['correct_count'] += 1
                        print(
                            f"  - [测试] 数据项 {i + 1}/{len(dataset)} | 状态: 正确 | 延迟: {latency:.2f}s | 长度: {response_len}")
                    else:
                        expected_semicolons = expected_response.count(';')
                        actual_semicolons = actual_response.count(';')

                        failure_details = {
                            "expected_response": expected_response,
                            "actual_response": actual_response,
                            "expected_semicolons": expected_semicolons,
                            "actual_semicolons": actual_semicolons
                        }
                        model_results['failed_cases'].append({'index': i, **failure_details})
                        log_entry['failure_details'] = failure_details

                        print(
                            f"  - [测试] 数据项 {i + 1}/{len(dataset)} | 状态: 错误 | 延迟: {latency:.2f}s | 长度: {response_len}")
                        print(f"    - 期望: '{expected_response}' ({expected_semicolons} commands)")
                        print(f"    - 得到: '{actual_response}' ({actual_semicolons} commands)")

                    # 将完整的 log_entry 写入文件
                    log_f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                    log_f.flush()

                except Exception as e:
                    print(f"  - [错误] 数据项 {i + 1}/{len(dataset)} | 请求失败: {e}")
                    model_results['failed_count'] += 1
                    if model_results['failed_count'] > 5:
                        print("  - 模型已连续失败多次，跳过该模型的剩余测试。")
                        break

            results[model_name] = model_results
            print(f"--- 模型 {model_name} 测试完成 ---")

    print("\n[5/5] 所有测试完成，生成总结报告...")
    print("\n==================================== 测试总结 ====================================")
    print(f"{'模型名称':<40} | {'平均延迟 (s)':<15} | {'Avg. Resp. Length':<20} | {'准确率':<15} | {'测试数':<10}")
    print("-" * 115)

    for model_name, res in results.items():
        if res.get('failed_warmup'):
            print(f"{model_name:<40} | {'N/A':<15} | {'N/A':<20} | {'N/A':<15} | {'预热失败':<10}")
            continue

        if res['total_count'] > 0:
            avg_latency = f"{sum(res['latencies']) / len(res['latencies']):.2f}"
            avg_len = f"{sum(res['response_lengths']) / len(res['response_lengths']):.0f}"
            accuracy = f"{(res['correct_count'] / res['total_count']) * 100:.2f}%"
            count_str = f"{res['correct_count']}/{res['total_count']}"

            print(f"{model_name:<40} | {avg_latency:<15} | {avg_len:<20} | {accuracy:<15} | {count_str:<10}")
        else:
            print(f"{model_name:<40} | {'N/A':<15} | {'N/A':<20} | {'N/A':<15} | {'无有效测试':<10}")

    print("=" * 115)

    print("\n================================== 详细失败分析 ==================================")
    has_failures = False
    for model_name, res in results.items():
        if res.get('failed_cases'):
            has_failures = True
            print(f"\n--- 模型: {model_name} ---")
            for case in res['failed_cases']:
                print(f"  - 数据项 #{case['index'] + 1}:")
                print(f"    - 期望 ({case['expected_semicolons']} commands): '{case['expected_response']}'")
                print(f"    - 得到 ({case['actual_semicolons']} commands): '{case['actual_response']}'")

    if not has_failures:
        print("\n所有模型在测试中均未出现错误。")
    print("=" * 82)
    print(f"\n详细测试日志已保存在 '{LOG_FILE}' 文件中。")


if __name__ == "__main__":
    main()