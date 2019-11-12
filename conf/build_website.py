import argparse
import pathlib
import shutil

website_path = pathlib.Path(__file__).parent.parent / "website"
assert website_path.exists()

tree_template_path = pathlib.Path("tree") / "index.html"

assert (website_path / tree_template_path).exists()


def slugify(name: str):
    return name.lower().replace(" ", "-").replace("&", "and")


def _parse_files(
    files, notebook_folder: pathlib.Path, html_folder: pathlib.Path
):
    c = {}
    for f in files:
        name = f.stem
        if name.endswith("_solution"):
            name = name[: -len("_solution")]

        # Already got that one.
        if name in c:
            continue

        possible_files = {
            "html_file": f.parent / (name + ".html"),
            "solution_html_file": f.parent / (name + "_solution.html"),
            "ipynb_file": notebook_folder
            / f.parent.relative_to(html_folder)
            / (name + ".ipynb"),
            "ipynb_html_file": notebook_folder
            / f.parent.relative_to(html_folder)
            / (name + "_solution.ipynb"),
        }

        for k, v in list(possible_files.items()):
            if not v.exists():
                del possible_files[k]
        c[name] = possible_files

    return c


def _parse_recursively(
    folder: pathlib.Path,
    parent: str,
    notebook_folder: pathlib.Path,
    html_folder: pathlib.Path,
):
    contents = {
        "children": {},
        "files": [],
        "name": folder.name,
        "slug": f"{parent}-{slugify(folder.name)}",
        "parent": parent,
    }
    for f in folder.glob("*"):
        # Ignore all hidden files.
        if f.name.startswith("."):
            continue
        if f.is_dir():
            name = slugify(f.name)
            contents["children"][name] = _parse_recursively(
                folder=f,
                parent=contents["slug"],
                notebook_folder=notebook_folder,
                html_folder=html_folder,
            )
        elif f.is_file():
            contents["files"].append(f)
        else:  # pragma: no cover
            raise NotImplementedError

    if contents["files"]:
        contents["notebooks"] = _parse_files(
            contents["files"],
            notebook_folder=notebook_folder,
            html_folder=html_folder,
        )
    else:
        contents["notebooks"] = []

    return contents


def build_website(
    notebook_folder: pathlib.Path,
    html_folder: pathlib.Path,
    output_folder: pathlib.Path,
):

    if not output_folder.exists():
        shutil.copytree(src=website_path, dst=output_folder)

    with open(website_path / tree_template_path, "r") as fh:
        template = fh.read()

    contents = _parse_recursively(
        html_folder,
        parent="root",
        notebook_folder=notebook_folder,
        html_folder=html_folder,
    )

    # Cheap templating for the masses...
    tr = """
    <tr class="treegrid-{node_name} treegrid-parent-{parent_node_name}">
        <td>{title}</td>
        <td>{subtitle}</td>
        <td>{buttons}</td>
    </tr>
    """

    all_table_rows = []

    def _r(s):
        return s.replace("root-html", "root")

    def parse_contents(c):
        if c["name"] != "html":
            all_table_rows.append(
                tr.format(
                    node_name=_r(c["slug"]),
                    parent_node_name=_r(c["parent"]),
                    title=c["name"],
                    subtitle="",
                    buttons="",
                )
            )

        if c["notebooks"]:
            for k in sorted(c["notebooks"].keys()):
                v = c["notebooks"][k]
                all_table_rows.append(
                    tr.format(
                        node_name=f"{c['slug']}-{slugify(k)}",
                        parent_node_name=_r(c["slug"]),
                        title=k,
                        subtitle="",
                        buttons="BUTTON SHOULD BE HERE",
                    )
                )

        if c["children"]:
            for k in sorted(c["children"].keys()):
                v = c["children"][k]
                parse_contents(v)

    parse_contents(contents)

    with open(output_folder / tree_template_path, "w") as fh:
        fh.write(template.format(all_table_rows="\n".join(all_table_rows)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Takes the output of convert_to_ipynb.py and builds the "
        "final seimo-live website for it."
    )
    parser.add_argument("input_folder", type=str, help="Input folder")
    parser.add_argument("output_folder", type=str, help="Output folder")

    args = parser.parse_args()

    input_folder = pathlib.Path(args.input_folder)
    output_folder = pathlib.Path(args.output_folder)

    if not input_folder.exists():
        raise ValueError("Input folder does not exist.")

    notebook_folder = input_folder / "notebooks"
    html_folder = input_folder / "html"

    if not notebook_folder.exists():
        raise ValueError(f"'{notebook_folder}' does not exist.")
    if not html_folder.exists():
        raise ValueError(f"'{html_folder}' does not exist.")

    build_website(
        notebook_folder=notebook_folder,
        html_folder=html_folder,
        output_folder=output_folder,
    )
