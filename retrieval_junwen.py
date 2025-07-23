import re
import os
import jsonlines
import random
import argparse
import ast
import textwrap
import chardet
from tqdm import tqdm

argparser = argparse.ArgumentParser()
argparser.add_argument("--stage", type=str, default="public")
argparser.add_argument("--lang", type=str, default="python")
argparser.add_argument("--strategy", type=str, default="junwen")
argparser.add_argument("--trim-prefix", action="store_true")
argparser.add_argument("--trim-suffix", action="store_true")
args = argparser.parse_args()

stage = args.stage
language = args.lang
strategy = args.strategy

if language == "python":
    extension = ".py"
elif language == "kotlin":
    extension = ".kt"
else:
    raise ValueError(f"Unsupported language: {language}")

FILE_SEP_SYMBOL = "<|file_sep|>"
FILE_COMPOSE_FORMAT = "{file_sep}{file_name}\n{file_content}"


def find_random_recent_file(
    root_dir: str, recent_filenames: list[str], min_lines: int = 10
) -> str:
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


def find_random_file(root_dir: str, min_lines: int = 10) -> str:
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


def parse_imports_from_prefix(prefix: str, log_method=None):
    imports = set()
    used_method = "ast"
    try:
        tree = ast.parse(prefix)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
    except Exception:
        # Fallback to regex extraction
        used_method = "regex"
        import_pattern = re.compile(
            r"^\s*import\s+([a-zA-Z0-9_\.]+)", re.MULTILINE)
        from_pattern = re.compile(
            r"^\s*from\s+([a-zA-Z0-9_\.]+)\s+import", re.MULTILINE
        )
        for match in import_pattern.findall(prefix):
            imports.add(match)
        for match in from_pattern.findall(prefix):
            imports.add(match)
    if log_method is not None:
        log_method(f"Import extraction method: {used_method}")
    return imports

def extract_function_names_from_file(src: str, log_method=None):
    funcs = {} # {name: call_count}
    used_method = "ast"		
    try:
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                funcs[node.name] = 1
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in funcs:
                        funcs[node.func.id] += 1
    except Exception:
        used_method = "regex"
        func_pattern = re.compile(r"^\s*def\s+([a-zA-Z0-9_]+)\s*\(", re.MULTILINE)
        call_pattern = re.compile(r"([a-zA-Z0-9_]+)\s*\(", re.MULTILINE)
        for match in func_pattern.findall(src):
            funcs[match] = 1
        for match in call_pattern.findall(src):
            if match in funcs:
                funcs[match] += 1
    if log_method is not None:
        log_method(f"Function extraction method: {used_method}")
    return funcs

def extract_class_names_from_file(src: str, log_method=None):
    classes = {} # {name: call_count}
    used_method = "ast"
    try:
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes[node.name] = 1
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in classes:
                    classes[node.func.id] += 1
    except Exception:
        used_method = "regex"
        class_pattern = re.compile(r"^\s*class\s+([a-zA-Z0-9_]+)\s*:", re.MULTILINE)
        call_pattern = re.compile(r"([a-zA-Z0-9_]+)\s*\(", re.MULTILINE)
        for match in class_pattern.findall(src):
            classes[match] = 1
        for match in call_pattern.findall(src):
            if match in classes:
                classes[match] += 1
    if log_method is not None:
        log_method(f"Class extraction method: {used_method}")
    return classes


def _extract_function_source(file_path, func_name):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source_lines = f.readlines()
            source = ''.join(source_lines)
    except Exception as e:
        return None

    try:
        node = ast.parse(source)
        for item in ast.walk(node):
            if isinstance(item, ast.FunctionDef) and item.name == func_name:
                start_line = item.lineno - 1
                end_line = getattr(item, 'end_lineno', None)

                if end_line is None:
                    # Fallback based on indentation
                    indent = len(source_lines[start_line]) - len(source_lines[start_line].lstrip())
                    for i in range(start_line + 1, len(source_lines)):
                        line = source_lines[i]
                        if line.strip() and len(line) - len(line.lstrip()) <= indent:
                            end_line = i
                            break
                    if end_line is None:
                        end_line = len(source_lines)

                func_code = ''.join(source_lines[start_line:end_line])
                return textwrap.dedent(func_code)

    except Exception as e:
        # Fallback: use regex to find the function
        pattern = rf"^def {re.escape(func_name)}\(.*?\):([\s\S]+?)(?=^\S|\Z)"
        matches = re.finditer(pattern, source, flags=re.MULTILINE)
        for match in matches:
            func_def = f"def {func_name}{match.group(0).split('def')[1]}"
            return textwrap.dedent(func_def)

    return None

def extract_function_def_from_repo(repo_path, target_func, log_method=None):
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                func_code = _extract_function_source(full_path, target_func)
                if func_code:
                    if log_method is not None:
                        log_method(f"\n📍 Found in: {full_path}\n")
                        log_method(func_code)
                    return func_code
    if log_method is not None:
        log_method(f"❌ Function '{target_func}' not found in repo.")
    return None


def _extract_class_source(file_path, class_name):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source_lines = f.readlines()
            source = ''.join(source_lines)
    except Exception as e:
        return None

    try:
        node = ast.parse(source)
        for item in ast.walk(node):
            if isinstance(item, ast.ClassDef) and item.name == class_name:
                start_line = item.lineno - 1
                end_line = getattr(item, 'end_lineno', None)

                if end_line is None:
                    indent = len(source_lines[start_line]) - len(source_lines[start_line].lstrip())
                    for i in range(start_line + 1, len(source_lines)):
                        line = source_lines[i]
                        if line.strip() and len(line) - len(line.lstrip()) <= indent:
                            end_line = i
                            break
                    if end_line is None:
                        end_line = len(source_lines)

                class_code = ''.join(source_lines[start_line:end_line])
                return textwrap.dedent(class_code)

    except Exception as e:
        # Regex fallback
        pattern = rf"^class {re.escape(class_name)}\(?.*?\)?:([\s\S]+?)(?=^\S|\Z)"
        matches = re.finditer(pattern, source, flags=re.MULTILINE)
        for match in matches:
            class_def = f"class {class_name}{match.group(0).split('class')[1]}"
            return textwrap.dedent(class_def)

    return None

def extract_class_def_from_repo(repo_path, class_name, log_method=None):
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                class_code = _extract_class_source(full_path, class_name)
                if class_code:
                    if log_method is not None:
                        log_method(f"\n📍 Found in: {full_path}\n")
                        log_method(class_code)
                    return class_code
    if log_method is not None:
        log_method(f"❌ Class '{class_name}' not found in repo.")
    return None

def collect_all_py_files(repo_root: str):
    py_files = []
    for dirpath, _, filenames in os.walk(repo_root):
        for filename in filenames:
            if filename.endswith(".py"):
                rel_path = os.path.relpath(
                    os.path.join(dirpath, filename), repo_root)
                py_files.append(rel_path)
    return py_files


def match_import_to_pyfiles(import_name: str, py_files):
    import_path = import_name.replace(".", "/") + ".py"
    init_path = import_name.replace(".", "/") + "/__init__.py"
    matches = []
    for pyfile in py_files:
        if pyfile.endswith(import_path) or pyfile.endswith(init_path):
            matches.append(pyfile)
    return matches


def trim_file_content(content: str, head_lines: int = 20) -> str:
    """
    Keep the first `head_lines` lines, and all class/function definition lines (parsed by ast), with ... in between omitted sections.
    """
    import ast

    lines = content.splitlines()
    n = len(lines)
    # 1. First head_lines lines
    head = lines[:head_lines]

    # 2. Find all class/function definition lines using ast (including nested)
    def get_def_lines(tree):
        def_lines = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # lineno is 1-based
                if node.lineno > head_lines:
                    def_lines.add(node.lineno - 1)
        return sorted(def_lines)

    try:
        tree = ast.parse(content)
        def_line_indices = get_def_lines(tree)
    except Exception:
        def_line_indices = []

    # 3. Compose output
    output = []
    output.extend(head)
    last_idx = head_lines - 1
    for idx in def_line_indices:
        # Insert ... if there is omitted content
        if idx > last_idx + 1:
            output.append("...")
        output.append(lines[idx])
        last_idx = idx
    # If there is omitted content after the last def/class, add ...
    if def_line_indices and last_idx < n - 1:
        output.append("...")
    return "\n".join(output)


completion_points_file = os.path.join("./data", f"{language}-{stage}.jsonl")
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
            "./data",
            f"repositories-{language}-{stage}",
            f"{repo_path}-{repo_revision}",
        )
        tqdm.write(f"Processing repo: {repo_path}-{repo_revision}")

        # Collect all .py files in repo
        py_files = collect_all_py_files(root_directory)

        recent_filenames = datapoint.get("modified", [])
        file_name = find_random_recent_file(root_directory, recent_filenames)
        if file_name is None:
            file_name = find_random_file(root_directory)
        if file_name is None:
            submission = {"context": ""}
            if args.trim_prefix:
                submission["prefix"] = trim_prefix(datapoint["prefix"])
            if args.trim_suffix and "suffix" in datapoint:
                submission["suffix"] = trim_suffix(datapoint["suffix"])
            writer.write(submission)
            continue

        context_files = []
        used_files = set()

        # Parse imports from prefix
        prefix = datapoint["prefix"]
        suffix = datapoint.get("suffix", "")
        if args.trim_prefix:
            prefix = trim_prefix(prefix)
        import_names = parse_imports_from_prefix(prefix, log_method=tqdm.write)
        tqdm.write(f"Found import statements: {sorted(import_names)}")

        # Add imported files if present in repo
        imported_files_included = []
        import_files_added = 0
        max_import_files = 5
        for import_name in import_names:
            matches = match_import_to_pyfiles(import_name, py_files)
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
                        import_content = f.read()
                    # strategy=recent-imports-trim: trim import file content
                    if strategy == "recent-imports-trim":
                        import_content = trim_file_content(
                            import_content, head_lines=20
                        )
                    context_files.append((rel_import_file, import_content))
                    used_files.add(abs_import_file)
                    imported_files_included.append(rel_import_file)
                    import_files_added += 1

        # Add function definitions
        func_prefix = extract_function_names_from_file(prefix, log_method=tqdm.write)
        func_suffix = extract_function_names_from_file(suffix, log_method=tqdm.write)
        tqdm.write(f"Found functions in prefix: {func_prefix}")
        tqdm.write(f"Found functions in suffix: {func_suffix}")
        # find the definitions of the functions in the repo
        for func_name in func_prefix:
            func_def = extract_function_def_from_repo(root_directory, func_name, log_method=tqdm.write)
            if func_def:
                context_files.append((f"Function: {func_name}", func_def))
                used_files.add(os.path.abspath(func_def))
        for func_name in func_suffix:
            func_def = extract_function_def_from_repo(root_directory, func_name, log_method=tqdm.write)
            if func_def:
                context_files.append((f"Function: {func_name}", func_def))
                used_files.add(os.path.abspath(func_def))


        # Add class definitions
        class_prefix = extract_class_names_from_file(prefix, log_method=tqdm.write)
        class_suffix = extract_class_names_from_file(suffix, log_method=tqdm.write)
        tqdm.write(f"Found classes in prefix: {class_prefix}")
        tqdm.write(f"Found classes in suffix: {class_suffix}")
        # find the definitions of the classes in the repo
        for class_name in class_prefix:
            class_def = extract_class_def_from_repo(root_directory, class_name, log_method=tqdm.write)
            if class_def:
                context_files.append((f"Class: {class_name}", class_def))
                used_files.add(os.path.abspath(class_def))
        for class_name in class_suffix:
            class_def = extract_class_def_from_repo(root_directory, class_name, log_method=tqdm.write)
            if class_def:
                context_files.append((f"Class: {class_name}", class_def))
                used_files.add(os.path.abspath(class_def))

        # Add recent file after import files
        clean_file_name = file_name[len(root_directory) + 1:]
        with open(file_name, "r", encoding="utf-8") as f:
            file_content = f.read()
        context_files.append((clean_file_name, file_content))
        used_files.add(os.path.abspath(file_name))

        tqdm.write(
            f"Imported files included in context: {imported_files_included}")

        # Compose context
        context = ""
        for i, (fname, fcontent) in enumerate(context_files):
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
