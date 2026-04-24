from pathlib import Path


def test_scaffold_directories_exist() -> None:
    expected = ["src", "scripts", "tests", "data", "results", "configs", "docs"]
    for directory in expected:
        assert Path(directory).exists(), f"Missing expected directory: {directory}"
