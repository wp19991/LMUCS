import json
import time
import os
from openai import OpenAI


API_KEY = "sk-xxxxxx"  # IMPORTANT: Replace with your actual key
BASE_URL = "https://api.deepseek.com"

MODELS_TO_TEST = [
    "deepseek-chat"
]

DATASET_FILE = "val_dataset_swift_4_type_new_yolo_9.jsonl"

LOG_FILE = "cloud_api_test_progress.jsonl"

try:
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
except Exception as e:
    print(f"初始化 OpenAI 客户端失败: {e}")
    client = None


def query_cloud_api(prompt="1+1=?", model="deepseek-chat"):
    if not client:
        raise ConnectionError("API 客户端未成功初始化。")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            stream=False,
            timeout=600,
        )
        return response.choices[0].message.content
    except Exception as e:
        raise ConnectionError(f"API 请求失败: {e}")


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
    print("--- Cloud API 模型性能与错误分析测试 ---")

    if not client:
        print("\n由于 API 客户端初始化失败，程序退出。请检查您的 API 密钥和网络连接。")
        return

    eligible_models = [{'name': name} for name in MODELS_TO_TEST]
    print(f"\n[1/5] 准备测试 {len(eligible_models)} 个指定的云端模型...")
    for model in eligible_models:
        print(f"  - [待测试] {model['name']}")

    if not eligible_models:
        print("\n没有在配置中指定要测试的模型，程序退出。")
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
                print(f"  - [预检] 正在测试 API 连通性 ({model_name})...")
                _ = query_cloud_api(prompt="Hello", model=model_name)
                print("  - [预检] API 连通性正常。")
            except Exception as e:
                print(f"  - [错误] 模型 {model_name} API 连通性测试失败: {e}。将跳过此模型。")
                results[model_name] = {'failed_warmup': True, 'error_message': str(e)}
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
                    actual_response = query_cloud_api(prompt=query, model=model_name)
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
                        print(f"  - 期望: '{expected_response}' ({expected_semicolons} commands)")
                        print(f"  - 得到: '{actual_response}' ({actual_semicolons} commands)")

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
            print(f"{model_name:<40} | {'N/A':<15} | {'N/A':<20} | {'N/A':<15} | {'API连通性失败':<15}")
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
    print(f"\n详细测试日志已保存在 '{LOG_FILE}' 文件中。")


if __name__ == "__main__":
    main()
