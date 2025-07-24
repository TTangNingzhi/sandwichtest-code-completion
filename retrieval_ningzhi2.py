import re
import os
import jsonlines
import argparse
import ast
import textwrap
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
DATA_DIR = os.getenv("DATA_DIR")

argparser = argparse.ArgumentParser()
argparser.add_argument("--stage", type=str, default="public")
argparser.add_argument("--lang", type=str, default="python")
argparser.add_argument("--strategy", type=str, default="ningzhi")
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


# ======================= Task 1: Import files =======================
# Helper function to add an imported file to context_files.
def add_import_file_to_context(
    rel_import_file,
    root_directory,
    used_files,
    context_files,
    imported_files_included,
):
    """
    Add an imported file to context_files.
    """
    abs_import_file = os.path.abspath(
        os.path.join(root_directory, rel_import_file))
    if abs_import_file not in used_files:
        with open(
            os.path.join(root_directory, rel_import_file),
            "r",
            encoding="utf-8",
        ) as f:
            import_content_lines = f.readlines()
        if len(import_content_lines) > 500:
            tqdm.write(f"Discard file {rel_import_file} exceeds 500 lines.")
            return False
        else:
            import_content = "".join(import_content_lines)
        # Add file name and "Imported file" string
        context_files.append(
            (f"{rel_import_file} [Imported file]", import_content))
        used_files.add(abs_import_file)
        imported_files_included.append(rel_import_file)
        tqdm.write(f"Added imported file: {rel_import_file}")
        return True
    return False


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
    funcs = {}  # {name: call_count}
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
        func_pattern = re.compile(
            r"^\s*def\s+([a-zA-Z0-9_]+)\s*\(", re.MULTILINE)
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
    classes = {}  # {name: call_count}
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
        class_pattern = re.compile(
            r"^\s*class\s+([a-zA-Z0-9_]+)\s*:", re.MULTILINE)
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
        with open(file_path, "r", encoding="utf-8") as f:
            source_lines = f.readlines()
            source = "".join(source_lines)
    except Exception as e:
        return None, None

    try:
        node = ast.parse(source)
        for item in ast.walk(node):
            if isinstance(item, ast.FunctionDef) and item.name == func_name:
                start_line = item.lineno - 1
                end_line = getattr(item, "end_lineno", None)

                if end_line is None:
                    # Fallback based on indentation
                    indent = len(source_lines[start_line]) - len(
                        source_lines[start_line].lstrip()
                    )
                    for i in range(start_line + 1, len(source_lines)):
                        line = source_lines[i]
                        if line.strip() and len(line) - len(line.lstrip()) <= indent:
                            end_line = i
                            break
                    if end_line is None:
                        end_line = len(source_lines)

                func_code = "".join(source_lines[start_line:end_line])
                return textwrap.dedent(func_code), file_path

    except Exception as e:
        # Fallback: use regex to find the function
        pattern = rf"^def {re.escape(func_name)}\(.*?\):([\s\S]+?)(?=^\S|\Z)"
        matches = re.finditer(pattern, source, flags=re.MULTILINE)
        for match in matches:
            func_def = f"def {func_name}{match.group(0).split('def')[1]}"
            return textwrap.dedent(func_def), file_path

    return None, None


def extract_function_def_from_repo(repo_path, target_func, log_method=None):
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".py"):
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
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source_lines = f.readlines()
            source = "".join(source_lines)
    except Exception as e:
        return None, None

    try:
        node = ast.parse(source)
        for item in ast.walk(node):
            if isinstance(item, ast.ClassDef) and item.name == class_name:
                start_line = item.lineno - 1
                end_line = getattr(item, "end_lineno", None)

                if end_line is None:
                    indent = len(source_lines[start_line]) - len(
                        source_lines[start_line].lstrip()
                    )
                    for i in range(start_line + 1, len(source_lines)):
                        line = source_lines[i]
                        if line.strip() and len(line) - len(line.lstrip()) <= indent:
                            end_line = i
                            break
                    if end_line is None:
                        end_line = len(source_lines)

                class_code = "".join(source_lines[start_line:end_line])
                return textwrap.dedent(class_code), file_path

    except Exception as e:
        # Regex fallback
        pattern = rf"^class {re.escape(class_name)}\(?.*?\)?:([\s\S]+?)(?=^\S|\Z)"
        matches = re.finditer(pattern, source, flags=re.MULTILINE)
        for match in matches:
            class_def = f"class {class_name}{match.group(0).split('class')[1]}"
            return textwrap.dedent(class_def), file_path

    return None, None


def extract_class_def_from_repo(repo_path, class_name, log_method=None):
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".py"):
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

        # Collect all .py files in repo
        py_files = collect_all_py_files(root_directory)

        # Separate context files by type for Python code completion context
        readme_files = []
        func_class_files = []
        import_files = []
        fallback_files = []
        used_files = set()

        # 1. Try to find README in repo root (only check readme.md or readme, case-insensitive)
        # This ensures README is always at the front of the context if present.
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

        # 2. Parse imports from prefix and collect imported files (import_files)
        prefix = datapoint["prefix"]
        suffix = datapoint.get("suffix", "")
        if args.trim_prefix:
            prefix = trim_prefix(prefix)
        import_names = parse_imports_from_prefix(prefix)
        imported_files_included = []
        import_files_added = 0
        max_import_files = 5

        for import_name in import_names:
            matches = match_import_to_pyfiles(import_name, py_files)
            # Only add if exactly one match (unique mapping)
            if len(matches) == 1 and import_files_added < max_import_files:
                rel_import_file = matches[0]
                if add_import_file_to_context(
                    rel_import_file,
                    root_directory,
                    used_files,
                    import_files,
                    imported_files_included,
                ):
                    import_files_added += 1

        # 3. Collect function and class definitions (func_class_files)
        tqdm.write("======= Task 2: Function definitions =======")
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

        # 4. Fallback: If nothing found, use recent file (fallback_files)
        if not (readme_files or func_class_files or import_files):
            tqdm.write(
                "No context files found, falling back to recent file strategy.")
            recent_filenames = datapoint.get("modified", [])
            fallback_file = None
            # Try to find a recent file with at least 10 lines
            for filename in recent_filenames:
                if filename.endswith(extension):
                    file_path = os.path.join(root_directory, filename)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            lines = f.readlines()
                            if len(lines) >= 10:
                                fallback_file = file_path
                                break
                    except Exception:
                        pass
            # If not found, pick a random .py file with at least 10 lines
            if fallback_file is None:
                for dirpath, _, filenames in os.walk(root_directory):
                    for filename in filenames:
                        if filename.endswith(extension):
                            file_path = os.path.join(dirpath, filename)
                            try:
                                with open(file_path, "r", encoding="utf-8") as f:
                                    lines = f.readlines()
                                    if len(lines) >= 10:
                                        fallback_file = file_path
                                        break
                            except Exception:
                                pass
                    if fallback_file:
                        break
            if fallback_file is not None:
                with open(fallback_file, "r", encoding="utf-8") as f:
                    fallback_content = f.read()
                clean_file_name = os.path.relpath(
                    fallback_file, root_directory)
                log_message = "[Fallback recent file]"
                tqdm.write(f"Fallback file used: {clean_file_name}")
                fallback_files.append(
                    (f"{clean_file_name} {log_message}", fallback_content)
                )
            else:
                tqdm.write("No suitable fallback file found.")

        # 5. Compose context in order: README, function/class, import, fallback
        context = ""
        for files in [readme_files, func_class_files, import_files, fallback_files]:
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
