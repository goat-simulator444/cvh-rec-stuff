import subprocess

for i in range(1, 11):
    fold_name = f"fold{i}"
    cmd = ["python", "main.py", "--dataset", fold_name]
    metric_cmd = ["python", "metrics_single.py", "--dataset", fold_name]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    metric = subprocess.run(metric_cmd, capture_output=True, text=True)

    print("----- Output -----")
    print(metric.stdout)
    if result.stderr:
        print(result.stderr)

all_metric_cmd = ["python", "metric.py"]
all_metric = subprocess.run(all_metric_cmd, capture_output=True, text=True)
print(all_metric.stdout)
