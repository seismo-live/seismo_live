import argparse
import pathlib
import shutil
import subprocess
import typing


# subprocess.run(["jupytext", "--to", "py", str(f), "-o", str(output_file)], check=True)
# f.unlink()


def convert_jupytext_to_ipynb(input_file: pathlib.Path):
    pass


# subprocess.run(["jupytext", "--to", "py", str(f), "-o", str(output_file)], check=True)


def find_jupytext_files(folder: pathlib.Path) -> typing.List[pathlib.Path]:
    jupytext_files = []
    for f in folder.glob("**/*.py"):
        with open(f, "r") as fh:
            header = fh.read(1000)
            if (
                "# ---" in header
                and "jupyter:" in header
                and "jupytext:" in header
            ):
                jupytext_files.append(f)
    return jupytext_files


def convert_folder(input_folder: pathlib.Path, output_folder: pathlib.Path):
    shutil.copytree(input_folder, output_folder)

    jupytext_files = find_jupytext_files(folder=output_folder)

    if jupytext_files < 134:
        raise ValueError("Not enough jupytext files found!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converts all Jupytext Python files to jupyter notebook "
        "files"
    )
    parser.add_argument("input_folder", type=str, help="Input folder")
    parser.add_argument("output_folder", type=str, help="Output folder")

    args = parser.parse_args()

    input_folder = pathlib.Path(args.input_folder)
    output_folder = pathlib.Path(args.output_folder)

    if not input_folder.exists():
        raise ValueError("Input folder does not exist.")
    if output_folder.exists():
        raise ValueError("Output folder must not yet exist.")

    convert_folder(input_folder=input_folder, output_folder=output_folder)
