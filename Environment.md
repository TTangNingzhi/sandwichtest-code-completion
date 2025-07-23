```
conda create -n ast25-jb-completion python=3.11

pip install jsonlines argparse tqdm rank-bm25 faiss-cpu transformers torch datasketch codegen-metrics vllm ipykernel
```

Download data into any location and set up the absolute path in `.env`. After preprocessing, the data folder should look like this:
```
$ ls
answers-kotlin-practice.jsonl  python-practice.zip
answers-kotlin-public.jsonl    python-public.jsonl
answers-python-practice.jsonl  python-public.zip
answers-python-public.jsonl    python-start.jsonl
answers-python-start.jsonl     python-start.zip
kotlin-practice.jsonl          repositories-kotlin-practice
kotlin-practice.zip            repositories-kotlin-public
kotlin-public.jsonl            repositories-python-practice
kotlin-public.zip              repositories-python-public
python-practice.jsonl          repositories-python-start
```

**Reference:**
- https://github.com/JetBrains-Research/ase2025-starter-kit
- https://datalore.jetbrains.com/notebook/HKlb0uMd9xwAK4NOaOEEi7/9pyInbBI9gUXCYPNbfiBSE/