from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from post_processing.utils.audio_utils import create_raven_file_list, normalize_audio


@pytest.fixture
def dummy_wav(tmp_path: Path) -> Path:
    file_path = tmp_path / "test.wav"
    data = np.array([0.0, 0.24, -0.5, 0.78, -0.62], dtype=np.float32)
    sf.write(file_path, data, samplerate=44100)
    return file_path


def test_normalize_audio_default_folder(dummy_wav: Path, tmp_path: Path) -> None:
    normalize_audio(dummy_wav)

    normalized_file = tmp_path / "test_normalized.wav"
    assert normalized_file.exists()

    data, _ = sf.read(normalized_file)
    assert np.isclose(np.max(np.abs(data)), 1.0, rtol=1e-4)


def test_normalize_audio_custom_folder(dummy_wav: Path, tmp_path: Path) -> None:
    out_folder = tmp_path / "output"
    out_folder.mkdir()

    normalize_audio(dummy_wav, output_folder=out_folder)

    normalized_file = out_folder / "test.wav"
    assert normalized_file.exists()

    data, _ = sf.read(normalized_file)
    assert np.isclose(np.max(np.abs(data)), 1.0, rtol=1e-4)


def test_normalize_audio_invalid_file(tmp_path: Path) -> None:
    invalid_file = tmp_path / "invalid.wav"
    invalid_file.write_text("not audio data")

    with pytest.raises(ValueError, match="An error occurred while reading file"):
        normalize_audio(invalid_file)


@pytest.fixture
def tmp_audio_dir(tmp_path: Path) -> Path:
    (tmp_path / "file1.wav").write_text("dummy")
    (tmp_path / "file2.wav").write_text("dummy")

    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "file3.wav").write_text("dummy")
    (nested / "file4.wav").write_text("dummy")

    (tmp_path / "ignore.txt").write_text("not audio")

    return tmp_path

def test_create_raven_file_list(tmp_audio_dir: Path) -> None:
    create_raven_file_list(tmp_audio_dir)

    file_list_path = tmp_audio_dir / "Raven_file_list.txt"
    assert file_list_path.exists(), "Raven_file_list.txt should exist"

    with file_list_path.open() as f:
        lines = [line.strip() for line in f.readlines()]

    expected_files = [
        tmp_audio_dir / "file1.wav",
        tmp_audio_dir / "file2.wav",
        tmp_audio_dir / "nested" / "file3.wav",
        tmp_audio_dir / "nested" / "file4.wav",
    ]
    expected_files_str = [str(f) for f in expected_files]

    assert set(lines) == set(expected_files_str), "All WAV files should be listed"
    assert all(line.endswith(".wav") for line in lines), "Only WAV files should be listed"

def test_create_raven_file_list_empty_dir(tmp_path: Path) -> None:
    create_raven_file_list(tmp_path)

    file_list_path = tmp_path / "Raven_file_list.txt"
    assert file_list_path.exists(), "Raven_file_list.txt should exist"
    with file_list_path.open() as f:
        lines = f.readlines()
    assert lines == [], "Empty directory should produce empty file list"