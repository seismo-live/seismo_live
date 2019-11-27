import argparse
import pathlib
import shutil
import urllib

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
            "ipynb_solution_file": notebook_folder
            / f.parent.relative_to(html_folder)
            / (name + "_solution.ipynb"),
        }

        for k, v in list(possible_files.items()):
            if not v.exists():
                del possible_files[k]
        if "ipynb_file" not in possible_files:
            continue
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

    # A bit dirty here but we just copy over the html and notebook directories
    # to the website directory.
    new_html_folder = output_folder / "html"
    new_notebook_folder = output_folder / "notebooks"

    if new_html_folder.exists():
        shutil.rmtree(new_html_folder)
    if new_notebook_folder.exists():
        shutil.rmtree(new_notebook_folder)

    shutil.copytree(html_folder, new_html_folder)
    shutil.copytree(notebook_folder, new_notebook_folder)

    # Also copy over the shared folder.
    share_folder = output_folder / "html" / "share"
    shutil.copytree(src=notebook_folder / "share", dst=share_folder)

    html_folder = new_html_folder
    notebook_folder = new_notebook_folder

    with open(website_path / tree_template_path, "r") as fh:
        template = fh.read()

    tree_output_path = output_folder / tree_template_path

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

    def create_iframe_html(
        target_link, notebook_link, path_to_html_root, notebook_path
    ):
        target_link = str(target_link)
        notebook_link = str(notebook_link)
        path_to_html_root = str(path_to_html_root)
        notebook_path = urllib.parse.quote(str(notebook_path), safe="")
        return (
            """
<html>
<body>
  <head>
    <link href="%%PATH_TO_HTML_ROOT%%/tree/bootstrap.min.css" rel="stylesheet">
    <style>
      * {
        box-sizing: border-box;
        padding: 0;
        margin: 0;
      }

      html {
        height: 100%;
        width: 100%;
      }

      body {
        font-family: "Josefin Sans", sans-serif;
        display: flex;
        flex-direction: column;
        height: 100%;
        width: 100%;
      }

      .navbar {
        min-height: 60px;
        max-height: 60px;
        font-size: 18px;
        background-image: linear-gradient(260deg, #2376ae 0%, #c16ecf 100%);
        border: 1px solid rgba(0, 0, 0, 0.2);
        padding-bottom: 10px;
      }

      .logo {
        text-decoration: none;
        color: rgba(255, 255, 255, 0.7);
        display: inline-block;
        font-size: 22px;
        margin-top: 10px;
        margin-left: 20px;
      }

      #header_content {
          color: rgba(255, 255, 255, 0.7);
        margin-right: 20px;
      }

      .navbar {
        border-width: 0px !important;
        border-radius: 0px !important;
            margin-bottom: 0px !important;
        display: flex;
        justify-content: space-between;
        padding-bottom: 0;
        height: 70px;
        align-items: center;
      }

      .logo {
        margin-top: 0;
      }


      .logo:hover {
        color: rgba(255, 255, 255, 1);
      }

      #iframe {
        width: 100%;
        height: 100%;
        border: none;
      }
    </style>
  </head>
  <div class="navbar">
<a href="http://seismo-live.org" class="logo">Seismo-Live</a>
    <div id="header_content">
        This is a static preview.
      <a class="btn btn-info btn-sm" target="_blank" href="https://mybinder.org/v2/gh/krischer/seismo_live_build/master">
        Open All on Binder
      </a>
      <a class="btn btn-success btn-sm" target="_blank" href="https://mybinder.org/v2/gh/krischer/seismo_live_build/master?filepath=%%NOTEBOOK_PATH%%">
        Open Live on Binder
      </a>
      <a class="btn btn-warning btn-sm" target="_blank" href="%%NOTEBOOK_LINK%%" download>
        Download Notebook
      </a>
    </div>
  </div>
  <iframe id="iframe" src="%%TARGET_LINK%%"></iframe>
  </iframe>
</body>
</html>
        """.replace(
                "%%TARGET_LINK%%", target_link
            )
            .replace("%%NOTEBOOK_LINK%%", notebook_link)
            .replace("%%PATH_TO_HTML_ROOT%%", path_to_html_root)
            .replace("%%NOTEBOOK_PATH%%", notebook_path)
        )

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
                buttons = []
                if "html_file" in v:
                    # The generated wrapper file.
                    wrapper_file = v["html_file"].parent / (
                        v["html_file"].stem + "_wrapper.html"
                    )
                    path_to_html_root = pathlib.Path(
                        "/".join([".."] * (len(wrapper_file.parent.parts) - 1))
                    )

                    # Link to the rendered HTML in the wrapper file. Always
                    # lives in the same folder - thus a relative file is okay.
                    link = pathlib.Path(".") / v["html_file"].name
                    # Link to the notebook. Also must be a relative path. In
                    # this case we have to count how many levels we have to go
                    # down.
                    notebook_link = pathlib.Path(
                        "/".join([".."] * (len(wrapper_file.parent.parts) - 1))
                    ) / v["ipynb_file"].relative_to(output_folder)

                    html_content = create_iframe_html(
                        link,
                        notebook_link,
                        path_to_html_root,
                        v["ipynb_file"].relative_to(output_folder / "notebooks"),
                    )
                    with open(wrapper_file, "w") as fh:
                        fh.write(html_content)

                    wrapper_link = pathlib.Path(
                        ".."
                    ) / wrapper_file.relative_to(output_folder)

                    buttons.append(
                        f'<a class="btn btn-success btn-sm" target="_blank" href="{wrapper_link}">OPEN</a>'
                    )

                if "solution_html_file" in v:
                    link = pathlib.Path(".") / v["solution_html_file"].name

                    wrapper_file = v["solution_html_file"].parent / (
                        v["solution_html_file"].stem + "_wrapper.html"
                    )
                    path_to_html_root = pathlib.Path(
                        "/".join([".."] * (len(wrapper_file.parent.parts) - 1))
                    )

                    notebook_link = pathlib.Path(
                        "/".join([".."] * (len(wrapper_file.parent.parts) - 1))
                    ) / v["ipynb_solution_file"].relative_to(output_folder)
                    html_content = create_iframe_html(
                        link,
                        notebook_link,
                        path_to_html_root,
                        v["ipynb_solution_file"].relative_to(output_folder / "notebooks"),
                    )
                    with open(wrapper_file, "w") as fh:
                        fh.write(html_content)

                    wrapper_link = pathlib.Path(
                        ".."
                    ) / wrapper_file.relative_to(output_folder)

                    buttons.append(
                        f'<a class="btn btn-warning btn-sm" target="_blank" href="{wrapper_link}">SOLUTION</a>'
                    )

                all_table_rows.append(
                    tr.format(
                        node_name=f"{c['slug']}-{slugify(k)}",
                        parent_node_name=_r(c["slug"]),
                        title=k,
                        subtitle="",
                        buttons=" ".join(buttons),
                    )
                )

        if c["children"]:
            for k in sorted(c["children"].keys()):
                v = c["children"][k]
                parse_contents(v)

    parse_contents(contents)

    with open(tree_output_path, "w") as fh:
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
