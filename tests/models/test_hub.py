import pytest

from rosellm.models.hub import extract_commit_hash, resolve_file


def test_extract_commit_hash_valid():
    # Test with a valid commit hash.
    resolved_file = (
        "/path/to/snapshots/0123456789abcdef0123456789abcdef01234567/file.txt"
    )
    assert (
        extract_commit_hash(resolved_file)
        == "0123456789abcdef0123456789abcdef01234567"
    )


def test_extract_commit_hash_invalid():
    # Test with an invalid commit hash.
    resolved_file = "/path/to/snapshots/invalid_hash/file.txt"
    assert extract_commit_hash(resolved_file) is None


def test_extract_commit_hash_no_hash():
    # Test with a path that doesn't contain a commit hash.
    resolved_file = "/path/to/file.txt"
    assert extract_commit_hash(resolved_file) is None


def test_resolve_file_and_extract_commit_hash():
    # Test resolving a file from the hub and extracting its commit hash.
    model_path = "Qwen/Qwen2.5-0.5B"
    filename = "config.json"
    
    # Test resolving file without commit hash
    resolved_file = resolve_file(model_path, filename)
    assert isinstance(resolved_file, str)
    assert resolved_file.endswith(filename)
    
    # Test file content is valid JSON
    with open(resolved_file, "r") as f:
        content = f.read()
        assert "architectures" in content
        assert "Qwen2ForCausalLM" in content
    
    # Test extracting commit hash
    commit_hash = extract_commit_hash(resolved_file)
    assert commit_hash is not None
    assert len(commit_hash) == 40
    
    # Test resolving file with commit hash
    resolved_file_with_hash = resolve_file(model_path, filename, _commit_hash=commit_hash)
    assert isinstance(resolved_file_with_hash, str)
    assert resolved_file_with_hash.endswith(filename)
