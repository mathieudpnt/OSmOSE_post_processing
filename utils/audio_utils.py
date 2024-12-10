import soundfile as sf
import numpy as np
from pathlib import Path


def normalize_audio(file: Path, output_folder: Path = None):
    """Normalize the audio data of a file

    Parameters
    ----------
    file : Path
        Path of the audio file to normalize
    output_folder : Path
        Path to output destination
    """
    try:
        data, fs = sf.read(file)
    except Exception as e:
        print(f"An error occurred while reading file {file}: {e}")

    if not output_folder:
        output_folder = file.parent

    format_file = sf.info(file).format
    subtype = sf.info(file).subtype

    data_norm = np.transpose(np.array([(data / np.max(np.abs(data)))]))[:, 0]

    new_fn = (
        file.stem + ".wav"
        if output_folder != file.parent
        else file.stem + "_normalized.wav"
    )
    new_path = output_folder / new_fn
    sf.write(
        file=new_path,
        data=data_norm,
        samplerate=fs,
        subtype=subtype,
        format=format_file,
    )

    print(f"File '{new_fn}' exported in '{file.parent}'")

    return


def create_raven_file_list(directory: Path):
    """Creates a text file with the paths of the audio files contained in all the subfolders
    of a given directory. This is useful to open several audio located in different subfolders in Raven.

    Parameters
    ----------
    directory: Path to the folder containing all audio data
    """
    # get file list
    files = list(directory.glob(r"*\*.wav"))

    # save file list as a txt file
    filename = directory / "Raven_file_list.txt"
    with open(filename, "w") as f:
        for item in files:
            f.write(f"{item}\n")

    print(f"File list saved in '{directory}'")

    return
