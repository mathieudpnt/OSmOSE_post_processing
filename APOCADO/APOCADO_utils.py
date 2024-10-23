from pathlib import Path
import os
from tqdm import tqdm
from shutil import copytree


def sort_ST_file(master_folder: Path):
    """Sort SoundTrap data into folders corresponding to the file extensions.

    Parameters
    ----------
    master_folder : Path
        Path to the main folder containing the SoundTrap files.
    """
    folders = []
    for root, dirs, files in os.walk(master_folder):
        if not dirs:
            folders.append(Path(root))

    for folder in folders:

        files = list(folder.glob("**/*.*"))

        sorted_files = {}
        extensions = []
        for file in files:
            ext = file.suffix[1:]
            if ext not in sorted_files:
                extensions.append(ext)
                sorted_files[ext] = []
            sorted_files[ext].append(file)

        if len(extensions) > 1:

            for ext in extensions:
                Path.mkdir(folder / ext, parents=True, exist_ok=True)

                for i in range(len(sorted_files[ext])):
                    os.replace(
                        sorted_files[ext][i], folder / ext / sorted_files[ext][i].name
                    )

                print(
                    f"-{ext}: {len(sorted_files[ext])} files",
                )
    print(f"Files sorted in '{master_folder}'")
    return


def copy_sud_files(folder: Path, destination_folder: Path):
    """Copy all sud files and associated log files from base folder to a
    specified folder and preserving the folder and subfolders architecture

    Parameters
    ----------
    folder: Path to base folder containing all data
    destination_folder: Path to destination folder

    Examples
    --------
    base_folder = Path(r"L:\acoustock3\Bioacoustique\DATASETS\APOCADO3")
    destination_folder = Path(r"D:\test_copy")
    copy_sud_files(base_folder, destination_folder)
    """
    directories = list(set([p.parents[1] for p in list(folder.glob("**/*.sud"))]))

    for d in tqdm(directories, total=len(directories)):
        for ext in ["sud", "csv", "xml", "log"]:
            ext_folder = d / ext
            if ext_folder.exists():
                dest_path = destination_folder / ext_folder.relative_to(folder)
                copytree(ext_folder, dest_path)

    print(f"Data copied successfully to '{destination_folder}'")

    return
