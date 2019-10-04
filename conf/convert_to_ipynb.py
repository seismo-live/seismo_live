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


def check_for_duplicate_solution_files(files: typing.List[pathlib.Path]):
    """
    Makes sure that every file that ends with _solution.py does not have a
    corresponding no solution file.
    """
    no_solution_names = []

    for file in files:
        if file.stem.endswith("_solution"):
            no_solution_name = file.parent / (
                file.stem[: -(len("_solution"))] + ".py"
            )
            no_solution_names.append(no_solution_name)

    if no_solution_names:
        raise ValueError(
            "Solution files exist for the following files (these are thus not "
            "needed):\n\n" + "\n".join(str(i) for i in no_solution_names)
        )


def convert_folder(input_folder: pathlib.Path, output_folder: pathlib.Path):
    shutil.copytree(input_folder, output_folder)

    jupytext_files = find_jupytext_files(folder=output_folder)

    if len(jupytext_files) < 134:
        raise ValueError("Not enough jupytext files found!")

    check_for_duplicate_solution_files(jupytext_files)

    # Only use the last 5 for testing purposes for now.
    jupytext_files = jupytext_files[-5:]
    print(jupytext_files)


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
