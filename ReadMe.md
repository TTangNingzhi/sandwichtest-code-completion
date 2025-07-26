# SaNDwich&TEST

This Docker container generates our context files for the private evaluation stage of the JetBrains Code Completion competition.

Pull the latest image:

```bash
docker pull ningzhitang/sandwich-test-context:latest
```

## Dataset Structure & Assumptions

The data is expected to follow the **default structure** after preprocessing using the `prepare_data.sh` script from `ase2025-starter-kit`. By default, the container is executed from the **root directory of `ase2025-starter-kit`**, with the following folders mounted:
* `/data`: preprocessed dataset
* `/predictions`: output directory for generated context files

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

## Team Information

**Team name**: SaNDwich&TEST

**Team members & emails**:
  * Ningzhi Tang `<ntang@nd.edu>` (equal contribution)
  * Junwen An `<junwenan@u.nus.edu>` (equal contribution)
  * David Meininger `<dmeining@nd.edu>`
  * Gelei Xu `<gxu4@nd.edu>`
  * Huihui Huang `<hh.huang.2024@phdcs.smu.edu.sg>`

## Public Submission Results

Public submission numbers on EvalAI:

* **Python**: Submission #5
  ```json
  {'Mellum ChrF': 0.5076098847812635, 'Codestral ChrF': 0.5727674308716586, 'Qwen-Coder ChrF': 0.5311432030409533, 'Average ChrF': 0.5371735062312918}
  ```
* **Kotlin**: Submission #3
  ```json
  {'Mellum ChrF': 0.6286589737594024, 'Codestral ChrF': 0.6771627968502668, 'Qwen-Coder ChrF': 0.6060614958043855, 'Average ChrF': 0.6372944221380182}
  ```
