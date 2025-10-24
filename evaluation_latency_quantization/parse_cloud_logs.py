import json
from collections import defaultdict
import numpy as np

LOG_FILE = 'cloud_api_test_progress.jsonl'


def analyze_logs(log_file_path):
    total_tests = 0
    correct_predictions = 0
    latencies = []
    response_lengths = []
    failure_cases = []

    failure_reasons = defaultdict(int)

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)

                    total_tests += 1
                    latencies.append(data.get('latency', 0))
                    response_lengths.append(data.get('response_length', 0))

                    if data.get('is_correct', False):
                        correct_predictions += 1
                    else:
                        failure_case_info = {
                            'index': data.get('index'),
                            'model': data.get('model_name'),
                            'details': data.get('failure_details', {})
                        }
                        failure_cases.append(failure_case_info)

                        if 'expected_response' in failure_case_info['details']:
                            failure_reasons['content_mismatch'] += 1
                        else:
                            failure_reasons['unknown'] += 1

                except json.JSONDecodeError:
                    print(f"Warning: Could not decode line: {line.strip()}")

    except FileNotFoundError:
        print(f"Error: Log file not found at '{log_file_path}'")
        return None


    accuracy = (correct_predictions / total_tests) * 100

    avg_latency_s = np.mean(latencies)
    std_latency_s = np.std(latencies)

    p95_latency_s = np.percentile(latencies, 95)  # 95分位延迟
    max_latency_s = np.max(latencies)
    inferences_per_second = 1 / avg_latency_s if avg_latency_s > 0 else 0

    results = {
        "total_tests": total_tests,
        "correct_predictions": correct_predictions,
        "incorrect_predictions": total_tests - correct_predictions,
        "accuracy_percent": accuracy,
        "average_latency_s": avg_latency_s,
        "std_dev_latency_s": std_latency_s,
        "p95_latency_s": p95_latency_s,
        "max_latency_s": max_latency_s,
        "inferences_per_second (IPS)": inferences_per_second,
    }

    return results


def display_results(results):
    if not results:
        return
    all_metrics = {'deepseek v3 670b': results}
    for model_name, metrics in all_metrics.items():
        print(f"\n--- 模型名称: {model_name} ---")
        print("-" * (len(model_name) + 16))

        print("\n[ 指令解析精度 ]")
        print(f"  总测试样本数: {metrics['total_tests']}")
        print(f"    - 正确解析:   {metrics['correct_predictions']}")
        print(f"    - 解析错误:   {metrics['incorrect_predictions']}")
        print(f"  整体准确率: {metrics['accuracy_percent']:.4f}%")

        print("\n[ 性能与延迟 (单位: 秒) ]")
        print(f"  平均推理延迟: {metrics['average_latency_s']:.3f} s (标准差: {metrics['std_dev_latency_s']:.3f} s)")
        print(f"  P95 分位延迟: {metrics['p95_latency_s']:.3f} s  (95%的请求延迟低于此值)")
        print(f"  最大推理延迟: {metrics['max_latency_s']:.3f} s")
        print(f"  平均吞吐量 (IPS): {metrics['inferences_per_second (IPS)']:.4f} inferences/second")
    print("Detailed Failure Cases:")
    print("=" * 50)


if __name__ == "__main__":
    analysis_results = analyze_logs(LOG_FILE)
    if analysis_results:
        display_results(analysis_results)