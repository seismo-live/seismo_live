import argparse
import json
import pathlib
import shutil
import subprocess
import typing


def convert_jupytext_to_ipynb(input_file: pathlib.Path):
    pass


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
            if no_solution_name.exists():
                no_solution_names.append(no_solution_name)

    if no_solution_names:
        raise ValueError(
            "Solution files exist for the following files (these are thus not "
            "needed):\n\n" + "\n".join(str(i) for i in no_solution_names)
        )


def strip_solution_content(
    ipynb_filename: pathlib.Path, no_solution_filename: pathlib.Path
):
    with open(ipynb_filename, "r") as fh:
        nb = json.load(fh)
    stripped_cell_count = 0
    for cell in nb["cells"]:
        if (
            not "metadata" in cell
            or not "tags" in cell["metadata"]
            or not "solution" in cell["metadata"]["tags"]
        ):
            continue
        cell["source"] = []
        stripped_cell_count += 1

    print(f"Stripped the solution in {stripped_cell_count} cells.")

    with open(no_solution_filename, "w") as fh:
        json.dump(nb, fh)
        
def strip_exercise_content(
    ipynb_filename: pathlib.Path, no_exercise_filename: pathlib.Path
):
    with open(ipynb_filename, "r") as fh:
        nb = json.load(fh)
    stripped_cell_count = 0
    for cell in nb["cells"]:
        if (
            not "metadata" in cell
            or not "tags" in cell["metadata"]
            or not "exercise" in cell["metadata"]["tags"]
        ):
            continue
        cell["source"] = []
        stripped_cell_count += 1

    print(f"Stripped the exercise in {stripped_cell_count} cells.")

    with open(no_exercise_filename, "w") as fh:
        json.dump(nb, fh)
        


def get_html_folder(
    ipynb_filename: pathlib.Path,
    notebook_folder: pathlib.Path,
    html_folder: pathlib.Path,
):
    return (html_folder / ipynb_filename.relative_to(notebook_folder)).parent


def convert_file(
    filename: pathlib.Path,
    notebook_folder: pathlib.Path,
    html_folder: pathlib.Path,
):
    """
    Does a few things:

    (1) Converts the jupytext file to a ipynb file.
    (2) Deletes the jupytext file.
    (3) If the file ends with "_solution":
        * Creates a version of that file with stripped "solution" tags.
        * Creates a version of that file with stripped "exercise" tags.
    (4) Executes the file and saves the output
    (5) Renders HTML version of all files.
    """
    jupytext_filename = filename
    ipynb_filename = jupytext_filename.parent / (
        jupytext_filename.stem + ".ipynb"
    )

    print("Convert to .ipynb file.")
    subprocess.run(
        [
            "jupytext",
            "--to",
            "notebook",
            str(jupytext_filename),
            "-o",
            str(ipynb_filename),
        ],
        check=True,
    )

    if ipynb_filename.stem.endswith("_solution"):
        no_solution_filename = ipynb_filename.parent / (
            ipynb_filename.stem[: -len("_solution")] + ".ipynb"
        )
        no_exercise_filename = ipynb_filename.parent / (
            ipynb_filename.stem[: -len("_solution")] + "_exercise.ipynb"
        print(f"Creating no-solution file: {no_solution_filename}")
        print(f"Creating no-exercise file: {no_exercise_filename}")    
        strip_solution_content(ipynb_filename, no_solution_filename)
        strip_exercise_content(ipynb_filename, no_exercise_filename)
        print(f"Converting to HTML: {no_solution_filename}")
        subprocess.run(
            [
                "jupyter",
                "nbconvert",
                "--to",
                "html",
                str(no_solution_filename),
                "--output-dir",
                str(
                    get_html_folder(
                        no_solution_filename, notebook_folder, html_folder
                    )
                ),
            ],
            check=True,
        )

    print(f"Running .ipynb file: {ipynb_filename}")
    subprocess.run(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            str(ipynb_filename),
        ],
        check=True,
    )

    print(f"Converting to HTML: {ipynb_filename}")
    subprocess.run(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "html",
            str(ipynb_filename),
            "--output-dir",
            str(get_html_folder(ipynb_filename, notebook_folder, html_folder)),
        ],
        check=True,
    )


def convert_folder(
    input_folder: pathlib.Path,
    notebook_folder: pathlib.Path,
    html_folder: pathlib.Path,
):
    shutil.copytree(input_folder, notebook_folder)

    jupytext_files = find_jupytext_files(folder=notebook_folder)

    if len(jupytext_files) < 92:
        raise ValueError("Not enough jupytext files found!")

    check_for_duplicate_solution_files(jupytext_files)

    for filename in jupytext_files:
        convert_file(
            filename=filename,
            notebook_folder=notebook_folder,
            html_folder=html_folder,
        )


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

    notebook_folder = output_folder / "notebooks"
    html_folder = output_folder / "html"

    convert_folder(
        input_folder=input_folder,
        notebook_folder=notebook_folder,
        html_folder=html_folder,
    )
