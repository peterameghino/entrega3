"""RAG, paso 1: descubrimiento de documentos en data/, limpieza y chunking por tokens. Sin red."""

import itertools
from pathlib import Path

import pytest

from errors import KnowledgeBaseError
from rag.documents import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    PROJECT_ROOT,
    clean_text,
    discover_documents,
    extract_title,
    load_chunks,
    split_tokens,
)

from .rag_fakes import WordEncoding


def words(n: int, start: int = 0) -> str:
    return " ".join(f"w{i}" for i in range(start, start + n))


def test_data_dir_is_the_project_data_folder_regardless_of_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    assert DATA_DIR == PROJECT_ROOT / "data"
    assert DATA_DIR.is_absolute()
    assert [p.name for p in discover_documents()] == [p.name for p in discover_documents(DATA_DIR)]


def test_real_data_folder_has_three_or_four_md_or_txt_documents():
    paths = discover_documents()
    assert 3 <= len(paths) <= 4
    assert {p.suffix for p in paths} <= {".md", ".txt"}
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert len(text) > 1000, f"{path.name} parece un placeholder"


def test_discovery_ignores_other_files_and_subfolders(tmp_path: Path):
    (tmp_path / "b.md").write_text("# B", encoding="utf-8")
    (tmp_path / "a.TXT").write_text("A", encoding="utf-8")
    (tmp_path / "imagen.png").write_bytes(b"\x89PNG")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.md").write_text("C", encoding="utf-8")
    assert [p.name for p in discover_documents(tmp_path)] == ["a.TXT", "b.md"]


@pytest.mark.parametrize("make", [lambda p: p / "no-existe", lambda p: p])
def test_missing_or_empty_data_folder_is_a_knowledge_base_error(tmp_path: Path, make):
    with pytest.raises(KnowledgeBaseError):
        discover_documents(make(tmp_path))


def test_clean_text_normalizes_without_changing_content():
    raw = "﻿# Título\r\n\r\n\r\n\r\nLínea  con   espacios\t ​\nCafé\x07\n\n  indentado"
    assert clean_text(raw) == "# Título\n\nLínea con espacios\nCafé\n\n  indentado"


def test_extract_title_uses_first_heading_or_first_line():
    assert extract_title("# Runbook: AKS\n\ntexto", "f") == "Runbook: AKS"
    assert extract_title("PROCEDIMIENTO\n\n1. x", "f") == "PROCEDIMIENTO"
    assert extract_title("", "fallback") == "fallback"


def test_short_text_is_a_single_chunk():
    assert split_tokens(words(10), WordEncoding()) == [(words(10), 10)]
    assert split_tokens("", WordEncoding()) == []


def test_chunks_have_the_configured_size_and_overlap():
    chunks = split_tokens(words(1200), WordEncoding(), chunk_size=500, overlap=50)
    assert [n for _, n in chunks] == [500, 500, 300]
    assert chunks[0][0] == words(500, 0)
    assert chunks[1][0] == words(500, 450)  # empieza 50 tokens antes del final del anterior
    assert chunks[2][0] == words(300, 900)
    for (prev, _), (curr, _) in itertools.pairwise(chunks):
        assert prev.split()[-CHUNK_OVERLAP:] == curr.split()[:CHUNK_OVERLAP]


def test_chunking_covers_the_whole_text_without_a_pure_overlap_tail():
    chunks = split_tokens(words(950), WordEncoding(), chunk_size=500, overlap=50)
    assert [n for _, n in chunks] == [500, 500]
    covered = set(" ".join(t for t, _ in chunks).split())
    assert covered == set(words(950).split())


def test_chunk_boundaries_never_split_characters():
    text = "ñandú camión acción " * 400
    for chunk, _ in split_tokens(text, WordEncoding(), chunk_size=37, overlap=5):
        assert "�" not in chunk
        assert chunk in text


@pytest.mark.parametrize(("size", "overlap"), [(500, 500), (500, -1), (10, 20)])
def test_invalid_chunk_parameters_are_rejected(size: int, overlap: int):
    with pytest.raises(ValueError):
        split_tokens("a b c", WordEncoding(), chunk_size=size, overlap=overlap)


def test_default_parameters_follow_the_course_instructions():
    assert (CHUNK_SIZE, CHUNK_OVERLAP) == (500, 50)


def test_load_chunks_keeps_source_metadata_and_stable_ids(tmp_path: Path):
    (tmp_path / "largo.md").write_text("# Largo\n\n" + words(1100), encoding="utf-8")
    (tmp_path / "corto.txt").write_text("CORTO\n" + words(20), encoding="utf-8")
    chunks = load_chunks(tmp_path, WordEncoding())
    assert [c.id for c in chunks] == ["corto.txt#0", "largo.md#0", "largo.md#1", "largo.md#2"]
    largo = [c for c in chunks if c.documento == "largo.md"]
    assert {c.titulo for c in largo} == {"Largo"}
    assert {c.total for c in largo} == {3}
    meta = largo[1].metadata()
    assert meta["documento"] == "largo.md"
    assert meta["fragmento"] == 1
    assert meta["total_fragmentos"] == 3
    assert len(str(meta["sha256"])) == 64
    assert load_chunks(tmp_path, WordEncoding()) == chunks  # determinista


def test_real_documents_produce_traceable_chunks():
    chunks = load_chunks(DATA_DIR, WordEncoding())
    assert {c.documento for c in chunks} == {p.name for p in discover_documents()}
    assert all(c.texto and c.tokens <= CHUNK_SIZE for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)
