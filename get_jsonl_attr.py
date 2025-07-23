import argparse
import json


def main():
    parser = argparse.ArgumentParser(
        description="Get the value of the N-th attribute of the M-th JSON object in a JSONL file."
    )
    parser.add_argument(
        "--file", type=str, required=True, help="Path to the input JSONL file."
    )
    parser.add_argument(
        "--json_index",
        type=int,
        required=True,
        help="Zero-based index of the JSON object (line number).",
    )
    parser.add_argument(
        "--attr_name",
        type=str,
        default="context",
        help="Name of the attribute in the JSON object (default: 'context').",
    )
    args = parser.parse_args()

    # Read the specified line (JSON object)
    with open(args.file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == args.json_index:
                obj = json.loads(line)
                break
        else:
            print("Error: JSON index out of range.")
            return

    # Get the specified attribute by name
    if args.attr_name not in obj:
        print(f"Error: Attribute '{args.attr_name}' not found in the JSON object.")
        return

    value = obj[args.attr_name]
    print(value)


if __name__ == "__main__":
    main()
