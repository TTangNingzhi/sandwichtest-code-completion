# SaNDwich&TEST

This Docker container generates our context files for the private evaluation stage of the JetBrains Code Completion competition.

## How to Run

### For Python

```bash
docker run --rm \
  -v $(pwd)/data:/data \
  -v $(pwd)/predictions:/predictions \
  sandwich-test-context python.py \
    --dataset_dir /data \
    --output_path /predictions/sandwich_test_python_results.jsonl \
    --stage private \
    --lang python \
    --strategy sandwich-test
```

### For Kotlin

```bash
docker run --rm \
  -v $(pwd)/data:/data \
  -v $(pwd)/predictions:/predictions \
  sandwich-test-context kotlin.py \
    --dataset_dir /data \
    --output_path /predictions/sandwich_test_kotlin_results.jsonl \
    --stage private \
    --lang kotlin \
    --strategy sandwich-test
```