from __future__ import annotations

import re

from _pytest.monkeypatch import MonkeyPatch

from radar_core.common import korean_analyzer


def test_is_kiwi_available_returns_bool() -> None:
    available = korean_analyzer.is_kiwi_available()
    assert isinstance(available, bool)


def test_tokenize_korean_fallback_without_kiwi(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(korean_analyzer, "_KIWI_AVAILABLE", False)
    monkeypatch.setattr(korean_analyzer, "_kiwi_class", None)
    monkeypatch.setattr(korean_analyzer, "_kiwi_instance", None)

    assert korean_analyzer.tokenize_korean("안녕하세요") == []


def test_extract_stems_fallback_without_kiwi(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(korean_analyzer, "_KIWI_AVAILABLE", False)
    monkeypatch.setattr(korean_analyzer, "_kiwi_class", None)
    monkeypatch.setattr(korean_analyzer, "_kiwi_instance", None)

    assert korean_analyzer.extract_stems("인공지능") == []


def test_build_korean_pattern_fallback_uses_escaped_keyword(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(korean_analyzer, "_KIWI_AVAILABLE", False)
    monkeypatch.setattr(korean_analyzer, "_kiwi_class", None)
    monkeypatch.setattr(korean_analyzer, "_kiwi_instance", None)

    pattern = korean_analyzer.build_korean_pattern("인공지능")
    assert pattern == re.escape("인공지능")


def test_tokenize_and_extract_stems_with_kiwi(monkeypatch: MonkeyPatch) -> None:
    class _Token:
        form: str
        tag: str
        lemma: str

        def __init__(self, form: str, tag: str, lemma: str) -> None:
            self.form = form
            self.tag = tag
            self.lemma = lemma

    class _FakeKiwi:
        def tokenize(self, _text: str) -> list[_Token]:
            return [
                _Token("인공", "NNG", "인공"),
                _Token("지능", "NNG", "지능"),
                _Token("은", "JX", "은"),
            ]

    monkeypatch.setattr(korean_analyzer, "_KIWI_AVAILABLE", True)
    monkeypatch.setattr(korean_analyzer, "_kiwi_class", _FakeKiwi)
    monkeypatch.setattr(korean_analyzer, "_kiwi_instance", None)

    assert korean_analyzer.tokenize_korean("인공지능은") == ["인공", "지능", "은"]
    assert korean_analyzer.extract_stems("인공지능은") == ["인공", "지능"]


def test_build_korean_pattern_with_kiwi_stems(monkeypatch: MonkeyPatch) -> None:
    class _Token:
        form: str
        tag: str
        lemma: str

        def __init__(self, form: str, tag: str, lemma: str) -> None:
            self.form = form
            self.tag = tag
            self.lemma = lemma

    class _FakeKiwi:
        def tokenize(self, _text: str) -> list[_Token]:
            return [
                _Token("인공", "NNG", "인공"),
                _Token("지능", "NNG", "지능"),
            ]

    monkeypatch.setattr(korean_analyzer, "_KIWI_AVAILABLE", True)
    monkeypatch.setattr(korean_analyzer, "_kiwi_class", _FakeKiwi)
    monkeypatch.setattr(korean_analyzer, "_kiwi_instance", None)

    assert korean_analyzer.build_korean_pattern("인공지능") == "(인공|지능)"
