# SaNDwich&TEST: Context Collection for Code Completion

We participated in the **[JetBrains Research Context Collection Competition](https://lp.jetbrains.com/research/context-collection-competition)**, co-located with **ASE 2025**.
Our team (SaNDwich&TEST) **advanced to the final stage in both Python and Kotlin tracks** (invited to present at the ASE 2025 Context Collection Workshop), outperforming official baselines and ranking **5th place** in both tracks' final leaderboards.

## Team Information

- Ningzhi Tang `<ntang@nd.edu>` (equal contribution)
- Junwen An `<junwenan@u.nus.edu>` (equal contribution)
- David Meininger `<dmeining@nd.edu>`
- Gelei Xu `<gxu4@nd.edu>`
- Huihui Huang `<hh.huang.2024@phdcs.smu.edu.sg>`

## Solution Overview

Our solution, **SaNDwich&TEST**, implements an intelligent **multi-task context collection strategy** for LLM-based code completion. The approach systematically analyzes both the code prefix and suffix around the completion point to gather the most relevant context from the repository.

> **Note**: To view our other exploration code and evaluation code organization, please check the [`main` branch](https://github.com/TTangNingzhi/sandwichtest-code-completion/tree/main).

### Core Strategy: Three-Task Context Collection

**Task 1: Import File Resolution**

- **Language-adaptive parsing**: Uses AST parsing for Python and regex patterns for Kotlin to extract import statements from the code prefix
- **Smart file matching**: Maps import names to actual repository files using language-specific file extensions (`.py` for Python, `.kt` for Kotlin)
- **Quality control**: Limits imported files to 500 lines (Python) or 300 lines (Kotlin) maximum and caps at 5 (Python) or 3 (Kotlin) import files to prevent context bloat
- **Unique mapping**: Only includes imports with exactly one matching file to ensure accuracy

**Task 2: Function Definition Extraction**

- **Dual-source analysis**: Extracts function names from both prefix and suffix code segments
- **Repository-wide search**: Scans all language-specific files in the repository to find function definitions
- **Language-specific extraction**: Uses AST parsing for Python and regex + brace counting for Kotlin to precisely locate function boundaries
- **Context enrichment**: Includes function definitions with file location information for better understanding

**Task 3: Class Definition Discovery**

- **Comprehensive class detection**: Identifies class references in both prefix and suffix
- **Definition retrieval**: Searches repository for class implementations using language-appropriate parsing methods
- **Structured context**: Provides complete class definitions with source file information

### Technical Features

- **Language agnostic**: Supports both Python (`.py`) and Kotlin (`.kt`) through configurable extensions and language-specific parsing strategies
- **Robust parsing**: Combines AST parsing (Python) with regex fallbacks (Kotlin) for maximum reliability across different languages
- **Fallback strategy**: Includes intelligent fallback mechanisms when no relevant context is found
- **Docker containerization**: Ensures reproducible execution across different environments

### Context Composition

The final context is structured as a concatenated string of relevant code segments, each prefixed with descriptive headers indicating the source and type of information. This approach provides LLMs with comprehensive, well-organized context that balances local code details with broader repository understanding, leading to improved code completion accuracy across multiple programming languages.

## Quick Start

### Pull Docker Image

```bash
docker pull ningzhitang/sandwich-test-context:latest
```

### Dataset Structure & Assumptions

The data is expected to follow the **default structure** after preprocessing using the `prepare_data.sh` script from `ase2025-starter-kit`. By default, you should run the `docker run` command from within the `ase2025-starter-kit` project directory. The following folders from your local machine will be mounted into the container:

- `/data`: preprocessed dataset
- `/predictions`: output directory for generated context files

### Usage

#### For Python

```bash
docker run --rm \
  -v $(pwd)/data:/data \
  -v $(pwd)/predictions:/predictions \
  ningzhitang/sandwich-test-context python.py \
    --dataset_dir /data \
    --output_path /predictions/sandwich_test_python_results.jsonl \
    --stage private \
    --lang python \
    --strategy sandwich-test
```

#### For Kotlin

```bash
docker run --rm \
  -v $(pwd)/data:/data \
  -v $(pwd)/predictions:/predictions \
  ningzhitang/sandwich-test-context kotlin.py \
    --dataset_dir /data \
    --output_path /predictions/sandwich_test_kotlin_results.jsonl \
    --stage private \
    --lang kotlin \
    --strategy sandwich-test
```

## Performance Results

### Public Phase Results

Public submission numbers on EvalAI:

- **Python**: Submission #5
  ```python
  {'Mellum ChrF': 0.5076098847812635, 'Codestral ChrF': 0.5727674308716586, 'Qwen-Coder ChrF': 0.5311432030409533, 'Average ChrF': 0.5371735062312918}
  ```
- **Kotlin**: Submission #3
  ```python
  {'Mellum ChrF': 0.6286589737594024, 'Codestral ChrF': 0.6771627968502668, 'Qwen-Coder ChrF': 0.6060614958043855, 'Average ChrF': 0.6372944221380182}
  ```

### Final Private Phase Results - Finalist

#### 🐍 Python Track

| Team              | Mellum | Codestral | Qwen  | Average |
| ----------------- | ------ | --------- | ----- | ------- |
| NoMoreActimel     | 0.656  | 0.820     | 0.725 | 0.734   |
| SpareCodeComplete | 0.695  | 0.766     | 0.713 | 0.725   |
| REALISE Lab       | 0.613  | 0.710     | 0.608 | 0.644   |
| WSPR_NCSU         | 0.582  | 0.710     | 0.615 | 0.636   |
| Baseline: BM25    | 0.585  | 0.659     | 0.585 | 0.610   |
| **SaNDwich&TEST** | 0.590  | 0.661     | 0.578 | 0.610   |
| Baseline: Recent  | 0.576  | 0.657     | 0.587 | 0.606   |

#### ☕ Kotlin Track

| Team              | Mellum | Codestral | Qwen  | Average |
| ----------------- | ------ | --------- | ----- | ------- |
| SpareCodeComplete | 0.723  | 0.769     | 0.753 | 0.748   |
| NoMoreActimel     | 0.684  | 0.791     | 0.719 | 0.731   |
| WSPR_NCSU\*       | 0.616  | 0.709     | 0.653 | 0.660   |
| REALISE Lab\*     | 0.652  | 0.688     | 0.637 | 0.659   |
| **SaNDwich&TEST** | 0.633  | 0.658     | 0.613 | 0.635   |
| Baseline: BM25    | 0.627  | 0.652     | 0.621 | 0.634   |
| Wu Wei            | 0.624  | 0.648     | 0.609 | 0.627   |
| Baseline: Recent  | 0.618  | 0.636     | 0.605 | 0.620   |
