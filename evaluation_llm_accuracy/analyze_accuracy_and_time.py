import os
import json
import numpy as np
from collections import defaultdict
import pandas as pd


def is_true(data: dict) -> bool:
    model_ans_raw = data.get("model_ans", "")
    ground_truth = data.get("response", "")

    if not ground_truth:
        return False

    true_ans_objects = [obj.replace('.', '').strip() for obj in ground_truth.split(';')]

    return all(obj in model_ans_raw for obj in true_ans_objects)


def extract_response_from_model_ans(model_ans, model_name) -> str:
    model_name_lower = model_name.lower()

    if 'deepseek_r1' in model_name_lower:
        response = model_ans.split('<｜Assistant｜><think>\n')[-1]
        response = response.replace('<｜end▁of▁sentence｜>', '')
    elif 'qwen2.5' in model_name_lower:
        response = model_ans.split('<|im_start|>assistant\n')[-1]
        response = response.replace('<|im_end|>', '')
    elif 'llama3.2' in model_name_lower:
        response = model_ans.split('assistant<|end_header_id|>\n\n')[-1]
        response = response.replace('<|eot_id|>', '')
    elif 'gemma2' in model_name_lower:
        response = model_ans.split('<start_of_turn>model\n')[-1]
        response = response.replace('<end_of_turn><eos>', '')
    elif 'phi3.5' in model_name_lower:
        response = model_ans.split(' <|end|><|assistant|> ')[-1]
        response = response.replace('<|end|>', '')
    else:
        if '<start_of_turn>model\n' in model_ans:
            response = model_ans.split('<start_of_turn>model\n')[-1]
            response = response.replace('\n<end_of_turn>', '').replace('<end_of_turn>', '')
        else:
            response = ""

    return response.strip()


def analyze_files(file_paths: list):
    stats = defaultdict(lambda: {
        "latencies": [],
        "throughputs": [],
        "exact_matches": 0,
        "contains_matches": 0,
        "total_count": 0
    })

    for file_path in file_paths:
        model_name = os.path.basename(file_path).replace('.jsonl', '')

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)

                        ground_truth = data.get("response", "").strip()
                        model_ans_raw = data.get("model_ans", "")
                        sp_time = data.get("sp_time")

                        if not all([ground_truth, model_ans_raw, sp_time and sp_time > 0]):
                            continue

                        model_response_cleaned = extract_response_from_model_ans(model_ans_raw, model_name)
                        response_len = len(model_response_cleaned)

                        stats[model_name]["total_count"] += 1
                        stats[model_name]["latencies"].append(sp_time)

                        if sp_time > 0:
                            throughput = response_len / sp_time
                            stats[model_name]["throughputs"].append(throughput)

                        if model_response_cleaned == ground_truth:
                            stats[model_name]["exact_matches"] += 1

                        if is_true(data):
                            stats[model_name]["contains_matches"] += 1

                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"Skipping malformed line in {file_path}: {e}")
        except FileNotFoundError:
            print(f"Warning: File not found at {file_path}. Skipping.")

    return stats


def calculate_and_present_results(stats: dict):
    results = []
    for model_name, data in stats.items():
        count = data["total_count"]
        if count == 0:
            continue

        latencies = np.array(data["latencies"])
        throughputs = np.array(data["throughputs"]) if data["throughputs"] else np.array([0])

        result_row = {
            "Model": model_name,
            "Total Samples": count,
            "Exact Match Acc (%)": (data["exact_matches"] / count) * 100,
            "Contains Acc (%)": (data["contains_matches"] / count) * 100,
            "Avg Latency (s)": np.mean(latencies),
            "std_dev_latency_s":np.std(latencies),
            "inferences_per_second (IPS)":1 / np.mean(latencies) if np.mean(latencies) > 0 else 0,
            "P95 Latency (s)": np.percentile(latencies, 95),
            "Max Latency (s)": np.max(latencies),
            "Avg Throughput (chars/s)": np.mean(throughputs)
        }
        results.append(result_row)

    if not results:
        print("No data processed. Please check your file paths and content.")
        return

    df = pd.DataFrame(results)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    df = df.sort_values(by="Model").reset_index(drop=True)
    df.to_excel('a.xlsx', index=False)

    float_cols = [
        'Exact Match Acc (%)', 'Contains Acc (%)', 'Avg Latency (s)',
        'P95 Latency (s)', 'Max Latency (s)', 'Avg Throughput (chars/s)'
    ]
    for col in float_cols:
        df[col] = df[col].map('{:.4f}'.format)

    print("\n--- Model Performance Analysis ---\n")
    print(df)


if __name__ == "__main__":
    if not os.path.isdir('./before') or not os.path.isdir('./after'):
        print("Error: Make sure 'before' and 'after' directories exist in the same location as the script.")
    else:
        before_model_json_file_path = ['./before/' + i for i in os.listdir('./before') if i.endswith('.jsonl')]
        after_model_json_file_path = ['./after/' + i for i in os.listdir('./after') if i.endswith('.jsonl')]
        all_files = before_model_json_file_path + after_model_json_file_path

        if not all_files:
            print("No '.jsonl' files found in 'before' or 'after' directories.")
        else:
            raw_stats = analyze_files(all_files)
            calculate_and_present_results(raw_stats)