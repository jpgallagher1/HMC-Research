## John Gallagher
## Augus 19, 2026
## Used for writing out the files to the pinnacles scratch folder for some parameter sweeps

from pathlib import Path

def write_tree(root: Path, outfile="file_tree.txt"):
    """
    Write an ASCII tree of the directory rooted at `root`.
    made this using chatgpt, chat title, "guardar arreglos con indices"
    """

    root = Path(root)

    def tree(path: Path, prefix=""):
        entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))

        for i, entry in enumerate(entries):
            last = i == len(entries) - 1
            connector = "└── " if last else "├── "

            yield prefix + connector + entry.name

            if entry.is_dir():
                extension = "    " if last else "│   "
                yield from tree(entry, prefix + extension)

    with open(outfile, "w") as f:
        f.write(root.name + "/\n")
        for line in tree(root):
            f.write(line + "\n")

    print(f"Saved tree to {outfile}")

def load_result(base, method, tau, T, length, run):
    """
    navigating the file path generated from the forloops. 
    result = load_result(
        base,
        method = "AA",
        tau = 2**-1,
        T = 1.0,
        length = 1000,
        run = 3,
    )

    q = result["q"]
    deltaHs = result["deltaHs"]
    accepted = result["accepted"]
    runtime = result["runtime"]

    """
    path = (
        Path(base)
        / method
        / f"tau_{float(tau):.12g}"
        / f"T_{float(T):.1f}"
        / f"len_{int(length)}"
        / f"run_{run}.npz"
    )
    return jnp.load(path)