import os
import ast
import argparse
import jsonlines
import numpy as np
from typing import List, Dict

from datasketch import MinHash, MinHashLSH


def extract_code_segments(py_file: str) -> List[Dict]:
    with open(py_file, "r", encoding="utf-8") as f:
        source = f.read()
    try:
        tree = ast.parse(source)
    except Exception:
        return []
    segments = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno
            end = getattr(node, "end_lineno", None)
            if end is None:
                end = start
                for child in ast.iter_child_nodes(node):
                    if hasattr(child, "end_lineno"):
                        end = max(end, child.end_lineno)
                    elif hasattr(child, "lineno"):
                        end = max(end, child.lineno)
            code_lines = source.splitlines()[start - 1 : end]
            code = "\n".join(code_lines)
            segments.append(
                {
                    "type": "class" if isinstance(node, ast.ClassDef) else "function",
                    "name": node.name,
                    "code": code,
                    "file": py_file,
                    "lineno": start,
                    "end_lineno": end,
                }
            )
    return segments


def collect_all_segments(root_dir: str, extension: str = ".py") -> List[Dict]:
    all_segments = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(extension):
                py_file = os.path.join(dirpath, filename)
                segments = extract_code_segments(py_file)
                all_segments.extend(segments)
    return all_segments


def get_shingles(text: str, k: int = 5) -> set:
    tokens = text.split()
    return (
        set([" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)])
        if len(tokens) >= k
        else set([" ".join(tokens)])
    )


def build_lsh_index(segments: List[Dict], num_perm: int = 128, k: int = 5):
    lsh = MinHashLSH(threshold=0.3, num_perm=num_perm)
    minhashes = []
    for idx, seg in enumerate(segments):
        shingles = get_shingles(seg["code"], k)
        m = MinHash(num_perm=num_perm)
        for sh in shingles:
            m.update(sh.encode("utf8"))
        lsh.insert(str(idx), m)
        minhashes.append(m)
    return lsh, minhashes


def jaccard_sim(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def trim_prefix(prefix: str):
    prefix_lines = prefix.split("\n")
    if len(prefix_lines) > 10:
        prefix = "\n".join(prefix_lines[-10:])
    return prefix


def trim_suffix(suffix: str):
    suffix_lines = suffix.split("\n")
    if len(suffix_lines) > 10:
        suffix = "\n".join(suffix_lines[:10])
    return suffix


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--stage", type=str, default="practice")
    argparser.add_argument("--lang", type=str, default="python")
    argparser.add_argument("--strategy", type=str, default="lsh-only")
    argparser.add_argument("--trim-prefix", action="store_true")
    argparser.add_argument("--trim-suffix", action="store_true")
    argparser.add_argument("--lsh-topk", type=int, default=3)
    args = argparser.parse_args()

    if args.lang == "python":
        extension = ".py"
    elif args.lang == "kotlin":
        extension = ".kt"
    else:
        raise ValueError(f"Unsupported language: {args.lang}")

    stage = args.stage
    language = args.lang
    completion_points_file = os.path.join("../data", f"{language}-{stage}.jsonl")
    prediction_file_name = f"{language}-{stage}-{args.strategy}"
    if args.trim_prefix:
        prediction_file_name += "-short-prefix"
    if args.trim_suffix:
        prediction_file_name += "-short-suffix"
    predictions_file = os.path.join("predictions", f"{prediction_file_name}.jsonl")

    FILE_SEP_SYMBOL = "<|file_sep|>"
    FILE_COMPOSE_FORMAT = "{file_sep}{file_name}\n{file_content}"

    from tqdm import tqdm

    with jsonlines.open(completion_points_file, "r") as reader, jsonlines.open(
        predictions_file, "w"
    ) as writer:
        datapoints = list(reader)
        for datapoint in tqdm(datapoints, desc="Processing queries"):
            prefix = datapoint["prefix"]
            if args.trim_prefix:
                prefix = trim_prefix(prefix)
            # Dynamically determine repo path for this datapoint
            repo_path = datapoint["repo"].replace("/", "__")
            repo_revision = datapoint["revision"]
            root_directory = os.path.join(
                "../data",
                f"repositories-{language}-{stage}",
                f"{repo_path}-{repo_revision}",
            )
            tqdm.write(f"Processing repo: {repo_path}-{repo_revision}")
            if not os.path.isdir(root_directory):
                tqdm.write(f"Repo directory not found: {root_directory}")
                # If repo not found, skip or write empty context
                submission = {"context": ""}
                if args.trim_prefix:
                    submission["prefix"] = prefix
                if args.trim_suffix and "suffix" in datapoint:
                    submission["suffix"] = trim_suffix(datapoint["suffix"])
                writer.write(submission)
                continue
            segments = collect_all_segments(root_directory, extension=extension)
            tqdm.write(
                f"Collected {len(segments)} segments from repo: {repo_path}-{repo_revision}"
            )
            if not segments:
                tqdm.write(f"No segments found in repo: {repo_path}-{repo_revision}")
                submission = {"context": ""}
                if args.trim_prefix:
                    submission["prefix"] = prefix
                if args.trim_suffix and "suffix" in datapoint:
                    submission["suffix"] = trim_suffix(datapoint["suffix"])
                writer.write(submission)
                continue
            lsh, minhashes = build_lsh_index(segments, num_perm=128, k=5)
            tqdm.write(
                f"LSH hashed {len(segments)} segments for repo: {repo_path}-{repo_revision}"
            )
            shingles = get_shingles(prefix, k=5)
            m_query = MinHash(num_perm=128)
            for sh in shingles:
                m_query.update(sh.encode("utf8"))
            lsh_candidates = list(lsh.query(m_query))
            if len(lsh_candidates) == 0:
                lsh_candidates = [
                    str(i)
                    for i in np.random.choice(
                        len(segments), args.lsh_topk, replace=False
                    )
                ]
            elif len(lsh_candidates) < args.lsh_topk:
                extra = [
                    str(i)
                    for i in np.random.choice(
                        len(segments),
                        args.lsh_topk - len(lsh_candidates),
                        replace=False,
                    )
                ]
                lsh_candidates += extra
            else:
                lsh_candidates = lsh_candidates[: args.lsh_topk]
            candidate_idxs = [int(idx) for idx in lsh_candidates]
            candidate_sims = [
                jaccard_sim(shingles, get_shingles(segments[idx]["code"], k=5))
                for idx in candidate_idxs
            ]
            sorted_idx = np.argsort(candidate_sims)[::-1]
            top_idxs = [candidate_idxs[i] for i in sorted_idx[: args.lsh_topk]]
            context_parts = []
            for idx in top_idxs:
                seg = segments[idx]
                rel_file = os.path.relpath(seg["file"], root_directory)
                context_parts.append(
                    FILE_COMPOSE_FORMAT.format(
                        file_sep=FILE_SEP_SYMBOL,
                        file_name=rel_file,
                        file_content=seg["code"],
                    )
                )
            context = "".join(context_parts)
            submission = {"context": context}
            if args.trim_prefix:
                submission["prefix"] = prefix
            if args.trim_suffix and "suffix" in datapoint:
                submission["suffix"] = trim_suffix(datapoint["suffix"])
            writer.write(submission)
