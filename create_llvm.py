from email.mime import base
import json
import os
import pathlib
from pprint import pprint
import subprocess

if __name__ == "__main__":
    INPUT_DIR = "inputs"
    OUTPUT_DIR = "outputs_llvm"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    args = {}

    for l in open("args.txt", "r"):
        l = l.strip()
        if not l:
            continue
        colon_pos = l.index(":")
        key = l[:colon_pos].strip()
        last_colon_pos = l.rindex(":")
        values = l[last_colon_pos + 1 :].strip().split()
        args[key] = values

    pprint(args)

    for file_path in pathlib.Path(INPUT_DIR).glob("*.json"):
        print("Processing", file_path)
        base_name = file_path.stem
        output_file = pathlib.Path(OUTPUT_DIR) / f"{base_name}.cpp"

        bril_program = json.load(open(file_path, "r"))
        main_fn_args = [f for f in bril_program["functions"] if f["name"] == "main"][0].get("args", [])
        main_fn_ret = [f for f in bril_program["functions"] if f["name"] == "main"][
            0
        ].get("type", "void")

        bril_main_sig = "extern \"C\" {} bril_main({});".format(
            main_fn_ret, ", ".join([e["type"] for e in main_fn_args])
        )
        main_fn = "int main() {bril_main(" + ', '.join(args.get(base_name + ".bril", [])) + "); return 0;}"
        with open(output_file, "w") as f:
            f.write(bril_main_sig + "\n" + main_fn)

        subprocess.run(['clang++', pathlib.Path(OUTPUT_DIR) / f'{base_name}.ll', str(output_file), '-o', pathlib.Path(OUTPUT_DIR) / f'{base_name}.exe'])
        # break
