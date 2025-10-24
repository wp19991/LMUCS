import json
from collections import defaultdict
import numpy as np
import os

LOG_FILE = 'test_progress.jsonl'


def analyze_jetson_logs(log_file_path):
    results_by_model = defaultdict(lambda: {
        'latencies': [],
        'response_lengths': [],
        'is_correct_list': [],
        'failure_cases': []
    })

    if not os.path.exists(log_file_path):
        print(f"错误: 日志文件 '{log_file_path}' 不存在。")
        return None

    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                model_name = data.get('model_name')
                if not model_name:
                    continue

                results_by_model[model_name]['latencies'].append(data.get('latency', 0))
                results_by_model[model_name]['response_lengths'].append(data.get('response_length', 0))

                is_correct = data.get('is_correct', False)
                results_by_model[model_name]['is_correct_list'].append(is_correct)

                if not is_correct:
                    failure_info = {
                        'index': data.get('index'),
                        'details': data.get('failure_details', {})
                    }
                    results_by_model[model_name]['failure_cases'].append(failure_info)

            except json.JSONDecodeError:
                print(f"警告: 无法解析行: {line.strip()}")

    final_metrics = {}
    for model_name, data in results_by_model.items():
        total_tests = len(data['is_correct_list'])
        if total_tests == 0:
            continue

        correct_predictions = sum(data['is_correct_list'])

        accuracy = (correct_predictions / total_tests) * 100

        latencies = data['latencies']
        avg_latency_s = np.mean(latencies)
        std_latency_s = np.std(latencies)

        p95_latency_s = np.percentile(latencies, 95)
        max_latency_s = np.max(latencies)
        inferences_per_second = 1 / avg_latency_s if avg_latency_s > 0 else 0

        final_metrics[model_name] = {
            "total_tests": total_tests,
            "correct_predictions": correct_predictions,
            "incorrect_predictions": total_tests - correct_predictions,
            "accuracy_percent": accuracy,
            "average_latency_s": avg_latency_s,
            "std_dev_latency_s": std_latency_s,
            "p95_latency_s": p95_latency_s,
            "max_latency_s": max_latency_s,
            "inferences_per_second (IPS)": inferences_per_second,
            "failure_cases": data['failure_cases']
        }

    return final_metrics


def display_results(all_metrics):
    if not all_metrics:
        print("未找到可分析的数据。")
        return

    print("=" * 80)
    print("      Jetson Nano 端侧大语言模型性能实测分析报告")
    print("=" * 80)

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

        print("-" * 80)


if __name__ == "__main__":
    analysis_results = analyze_jetson_logs(LOG_FILE)
    if analysis_results:
        display_results(analysis_results)