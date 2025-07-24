import re
import os
import jsonlines
import argparse
import random
from tqdm import tqdm
from dotenv import load_dotenv

# ==========================
# Kotlin Static Analysis Script
# ==========================
# This script extracts import statements, function and class definitions, and their usages from Kotlin (.kt) files.
# It uses regular expressions and indentation/braces heuristics, as Kotlin does not have a built-in AST module for Python.
# All comments are in English.

load_dotenv()
DATA_DIR = os.getenv("DATA_DIR")

argparser = argparse.ArgumentParser()
argparser.add_argument("--stage", type=str, default="public")
argparser.add_argument("--lang", type=str, default="kotlin")
argparser.add_argument("--strategy", type=str, default="ningzhi")
argparser.add_argument("--trim-prefix", action="store_true")
argparser.add_argument("--trim-suffix", action="store_true")
args = argparser.parse_args()

stage = args.stage
language = args.lang
strategy = args.strategy

if language == "kotlin":
    extension = ".kt"
else:
    raise ValueError(f"Unsupported language: {language}")

FILE_SEP_SYMBOL = "<|file_sep|>"
FILE_COMPOSE_FORMAT = "{file_sep}{file_name}\n{file_content}"

# ======================= Helper Functions =======================


def trim_prefix(prefix: str):
    """Trim prefix to last 10 lines if too long."""
    prefix_lines = prefix.split("\n")
    if len(prefix_lines) > 10:
        prefix = "\n".join(prefix_lines[-10:])
    return prefix


def trim_suffix(suffix: str):
    """Trim suffix to first 10 lines if too long."""
    suffix_lines = suffix.split("\n")
    if len(suffix_lines) > 10:
        suffix = "\n".join(suffix_lines[:10])
    return suffix


def parse_imports_from_prefix(prefix: str, log_method=None):
    """
    Extract import statements from Kotlin code using regex.
    """
    imports = set()
    import_pattern = re.compile(
        r"^\s*import\s+([a-zA-Z0-9_\.]+)", re.MULTILINE)
    for match in import_pattern.findall(prefix):
        imports.add(match)
    if log_method is not None:
        log_method(f"Import extraction method: regex")
    return imports


def extract_function_names_from_file(src: str, log_method=None):
    """
    Extract function names and count their calls in Kotlin code using regex.
    """
    funcs = {}  # {name: call_count}
    func_pattern = re.compile(r"^\s*fun\s+([a-zA-Z0-9_]+)\s*\(", re.MULTILINE)
    call_pattern = re.compile(r"([a-zA-Z0-9_]+)\s*\(", re.MULTILINE)
    for match in func_pattern.findall(src):
        funcs[match] = 1
    for match in call_pattern.findall(src):
        if match in funcs:
            funcs[match] += 1
    if log_method is not None:
        log_method(f"Function extraction method: regex")
    return funcs


def extract_class_names_from_file(src: str, log_method=None):
    """
    Extract class names and count their calls in Kotlin code using regex.
    """
    classes = {}  # {name: call_count}
    class_pattern = re.compile(r"^\s*class\s+([a-zA-Z0-9_]+)", re.MULTILINE)
    call_pattern = re.compile(r"([a-zA-Z0-9_]+)\s*\(", re.MULTILINE)
    for match in class_pattern.findall(src):
        classes[match] = 1
    for match in call_pattern.findall(src):
        if match in classes:
            classes[match] += 1
    if log_method is not None:
        log_method(f"Class extraction method: regex")
    return classes


def _extract_block_by_braces(lines, start_idx):
    """
    Extract a code block starting at start_idx using brace counting.
    Returns the block as a string.
    """
    block_lines = []
    open_braces = 0
    started = False
    for i in range(start_idx, len(lines)):
        line = lines[i]
        # Count braces
        open_braces += line.count("{")
        open_braces -= line.count("}")
        block_lines.append(line)
        if "{" in line:
            started = True
        if started and open_braces == 0:
            break
    return "".join(block_lines)


def _extract_function_source(file_path, func_name):
    """
    Extract the full source code of a Kotlin function by name using regex and brace counting.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source_lines = f.readlines()
            source = "".join(source_lines)
    except Exception:
        return None, None

    # Regex to find function definition line
    func_def_pattern = re.compile(
        rf"^\s*fun\s+{re.escape(func_name)}\s*\(.*", re.MULTILINE
    )
    for idx, line in enumerate(source_lines):
        if func_def_pattern.match(line):
            # Extract block using braces
            func_code = _extract_block_by_braces(source_lines, idx)
            return func_code, file_path
    return None, None


def extract_function_def_from_repo(repo_path, target_func, log_method=None):
    """
    Search all .kt files in the repo for the function definition.
    """
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".kt"):
                full_path = os.path.join(root, file)
                func_code, found_file = _extract_function_source(
                    full_path, target_func)
                if func_code:
                    if log_method is not None:
                        log_method(f"\n📍 Found in: {full_path}\n")
                        log_method(func_code)
                    return func_code, found_file
    if log_method is not None:
        log_method(f"❌ Function '{target_func}' not found in repo.")
    return None, None


def _extract_class_source(file_path, class_name):
    """
    Extract the full source code of a Kotlin class by name using regex and brace counting.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source_lines = f.readlines()
            source = "".join(source_lines)
    except Exception:
        return None, None

    # Regex to find class definition line
    class_def_pattern = re.compile(
        rf"^\s*class\s+{re.escape(class_name)}\b.*", re.MULTILINE
    )
    for idx, line in enumerate(source_lines):
        if class_def_pattern.match(line):
            # Extract block using braces
            class_code = _extract_block_by_braces(source_lines, idx)
            return class_code, file_path
    return None, None


def extract_class_def_from_repo(repo_path, class_name, log_method=None):
    """
    Search all .kt files in the repo for the class definition.
    """
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".kt"):
                full_path = os.path.join(root, file)
                class_code, found_file = _extract_class_source(
                    full_path, class_name)
                if class_code:
                    if log_method is not None:
                        log_method(f"\n📍 Found in: {full_path}\n")
                        log_method(class_code)
                    return class_code, found_file
    if log_method is not None:
        log_method(f"❌ Class '{class_name}' not found in repo.")
    return None, None


def collect_all_kt_files(repo_root: str):
    """
    Recursively collect all .kt files in the repository.
    """
    kt_files = []
    for dirpath, _, filenames in os.walk(repo_root):
        for filename in filenames:
            if filename.endswith(".kt"):
                rel_path = os.path.relpath(
                    os.path.join(dirpath, filename), repo_root)
                kt_files.append(rel_path)
    return kt_files


def match_import_to_ktfiles(import_name: str, kt_files):
    """
    Match an import name to a .kt file path.
    """
    import_path = import_name.replace(".", "/") + ".kt"
    matches = []
    for ktfile in kt_files:
        if ktfile.endswith(import_path):
            matches.append(ktfile)
    return matches


def find_random_file(root_dir: str, min_lines: int = 10) -> str:
    """
    Select a random file in the given language and directory with at least min_lines.
    """
    code_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(extension):
                file_path = os.path.join(dirpath, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        if len(lines) >= min_lines:
                            code_files.append(file_path)
                except Exception:
                    pass
    return random.choice(code_files) if code_files else None


def find_random_recent_file(
    root_dir: str, recent_filenames: list, min_lines: int = 10
) -> str:
    """
    Select a random recent file in the given language and directory with at least min_lines.
    """
    code_files = []
    for filename in recent_filenames:
        if filename.endswith(extension):
            file_path = os.path.join(root_dir, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    if len(lines) >= min_lines:
                        code_files.append(file_path)
            except Exception:
                pass
    return random.choice(code_files) if code_files else None


# ======================= Main Processing =======================

completion_points_file = os.path.join(DATA_DIR, f"{language}-{stage}.jsonl")
prediction_file_name = f"{language}-{stage}-{strategy}"
if args.trim_prefix:
    prediction_file_name += "-short-prefix"
if args.trim_suffix:
    prediction_file_name += "-short-suffix"
predictions_file = os.path.join("predictions", f"{prediction_file_name}.jsonl")

with jsonlines.open(completion_points_file, "r") as reader, jsonlines.open(
    predictions_file, "w"
) as writer:
    datapoints = list(reader)
    for datapoint in tqdm(datapoints, desc="Processing queries"):
        repo_path = datapoint["repo"].replace("/", "__")
        repo_revision = datapoint["revision"]
        root_directory = os.path.join(
            DATA_DIR,
            f"repositories-{language}-{stage}",
            f"{repo_path}-{repo_revision}",
        )
        tqdm.write("======= Task 1: Import files =======")

        # Collect all .kt files in repo
        kt_files = collect_all_kt_files(root_directory)

        # New: Separate context files by type
        readme_files = []
        func_class_files = []
        import_files = []
        fallback_files = []
        used_files = set()

        # 1. Try to find README in repo root (only check readme.md or readme, case-insensitive)
        readme_files_in_root = [
            f
            for f in os.listdir(root_directory)
            if os.path.isfile(os.path.join(root_directory, f))
        ]
        readme_file = None
        for candidate in ["readme.md", "readme"]:
            for f in readme_files_in_root:
                if f.lower() == candidate:
                    readme_file = f
                    break
            if readme_file:
                break
        if readme_file:
            readme_path = os.path.join(root_directory, readme_file)
            with open(readme_path, "r", encoding="utf-8") as f:
                readme_content = f.read()
            readme_files.append(
                (f"{readme_file} [Repository README]", readme_content))
            tqdm.write(f"Added README file: {readme_file}")

        # Parse imports from prefix
        prefix = datapoint["prefix"]
        suffix = datapoint.get("suffix", "")
        if args.trim_prefix:
            prefix = trim_prefix(prefix)
        import_names = parse_imports_from_prefix(prefix)

        # Add imported files if present in repo using a helper function with line count check
        imported_files_included = []
        import_files_added = 0
        max_import_files = 5

        for import_name in import_names:
            matches = match_import_to_ktfiles(import_name, kt_files)
            # Only add if exactly one match (unique mapping)
            if len(matches) == 1 and import_files_added < max_import_files:
                rel_import_file = matches[0]
                abs_import_file = os.path.abspath(
                    os.path.join(root_directory, rel_import_file)
                )
                if abs_import_file not in used_files:
                    with open(
                        os.path.join(root_directory, rel_import_file),
                        "r",
                        encoding="utf-8",
                    ) as f:
                        import_content_lines = f.readlines()
                    if len(import_content_lines) > 300:
                        tqdm.write(
                            f"Discard file {rel_import_file} exceeds 300 lines.")
                        continue
                    else:
                        import_content = "".join(import_content_lines)
                    # Add file name and "Imported file" string
                    import_files.append(
                        (f"{rel_import_file} [Imported file]", import_content)
                    )
                    used_files.add(abs_import_file)
                    imported_files_included.append(rel_import_file)
                    import_files_added += 1
                    tqdm.write(f"Added imported file: {rel_import_file}")

        tqdm.write("======= Task 2: Function definitions =======")
        # Add function definitions
        func_prefix = extract_function_names_from_file(prefix)
        func_suffix = extract_function_names_from_file(suffix)
        for func_name in func_prefix:
            func_def, func_file = extract_function_def_from_repo(
                root_directory, func_name
            )
            if func_def and func_file:
                rel_func_file = os.path.relpath(func_file, root_directory)
                func_class_files.append(
                    (f"Function: {func_name} in {rel_func_file}", func_def)
                )
                used_files.add(os.path.abspath(func_def))
        for func_name in func_suffix:
            func_def, func_file = extract_function_def_from_repo(
                root_directory, func_name
            )
            if func_def and func_file:
                rel_func_file = os.path.relpath(func_file, root_directory)
                func_class_files.append(
                    (f"Function: {func_name} in {rel_func_file}", func_def)
                )
                used_files.add(os.path.abspath(func_def))

        tqdm.write("======= Task 3: Class definitions =======")
        # Add class definitions
        class_prefix = extract_class_names_from_file(prefix)
        class_suffix = extract_class_names_from_file(suffix)
        for class_name in class_prefix:
            class_def, class_file = extract_class_def_from_repo(
                root_directory, class_name
            )
            if class_def and class_file:
                rel_class_file = os.path.relpath(class_file, root_directory)
                func_class_files.append(
                    (f"Class: {class_name} in {rel_class_file}", class_def)
                )
                used_files.add(os.path.abspath(class_def))
        for class_name in class_suffix:
            class_def, class_file = extract_class_def_from_repo(
                root_directory, class_name
            )
            if class_def and class_file:
                rel_class_file = os.path.relpath(class_file, root_directory)
                func_class_files.append(
                    (f"Class: {class_name} in {rel_class_file}", class_def)
                )
                used_files.add(os.path.abspath(class_def))

        # ===== Fallback: If nothing found, use recent file =====
        if not (readme_files or func_class_files or import_files):
            tqdm.write(
                "No context files found, falling back to recent file strategy.")
            recent_filenames = datapoint.get("modified", [])
            fallback_file = find_random_recent_file(
                root_directory, recent_filenames)
            if fallback_file is None:
                fallback_file = find_random_file(root_directory)
            if fallback_file is not None:
                with open(fallback_file, "r", encoding="utf-8") as f:
                    fallback_content = f.read()
                clean_file_name = os.path.relpath(
                    fallback_file, root_directory)
                log_message = f"[Fallback recent file]"
                tqdm.write(f"Fallback file used: {clean_file_name}")
                fallback_files.append(
                    (f"{clean_file_name} {log_message}", fallback_content)
                )
            else:
                tqdm.write("No suitable fallback file found.")

        # Compose context in order: function/class, import, README, fallback
        context = ""
        for files in [func_class_files, import_files, readme_files, fallback_files]:
            for fname, fcontent in files:
                context += FILE_COMPOSE_FORMAT.format(
                    file_sep=FILE_SEP_SYMBOL,
                    file_name=fname,
                    file_content=fcontent,
                )

        submission = {"context": context}
        if args.trim_prefix:
            submission["prefix"] = prefix
        if args.trim_suffix and "suffix" in datapoint:
            submission["suffix"] = trim_suffix(datapoint["suffix"])
        writer.write(submission)
