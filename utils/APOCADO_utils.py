from pathlib import Path
import os

def sort_ST_file(master_folder: Path):
    """Sort SoundTrap data into folders corresponding to the file extensions.
    
    Parameters
    ----------
    master_folder : Path
        Path to the main folder containing the SoundTrap files.

    Returns
    -------
    None.

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
                    os.replace(sorted_files[ext][i], folder / ext / sorted_files[ext][i].name)

                print(
                    f"-{ext}: {len(sorted_files[ext])} files" ,
                )
    print(f"Files sorted in '{master_folder}'")
    return
