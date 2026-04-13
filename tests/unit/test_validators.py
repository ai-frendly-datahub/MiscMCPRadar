"""
Unit tests for radar.common.validators module.

Tests cover:
- Title normalization
- URL similarity detection
- URL format validation
- Duplicate article detection
- Article validation
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from radar.common.validators import (
    detect_duplicate_articles,
    is_similar_url,
    normalize_title,
    validate_article,
    validate_url_format,
)
from radar.models import Article


class TestNormalizeTitle:
    """Tests for normalize_title function."""

    def test_normalize_basic_title(self) -> None:
        result = normalize_title("Breaking News")
        assert result == "breaking news"

    def test_normalize_extra_whitespace(self) -> None:
        result = normalize_title("  Breaking  News  ")
        assert result == "breaking news"

    def test_normalize_with_parentheses(self) -> None:
        result = normalize_title("Title (Updated)")
        assert result == "title updated"

    def test_normalize_with_brackets(self) -> None:
        result = normalize_title("Article [New]")
        assert result == "article new"

    def test_normalize_special_characters(self) -> None:
        result = normalize_title("Title@#$%Name")
        assert result == "titlename"

    def test_normalize_with_hyphens(self) -> None:
        result = normalize_title("Title-With-Hyphens")
        assert result == "title-with-hyphens"

    def test_normalize_empty_string(self) -> None:
        result = normalize_title("")
        assert result == ""

    def test_normalize_only_whitespace(self) -> None:
        result = normalize_title("   ")
        assert result == ""

    def test_normalize_complex_title(self) -> None:
        result = normalize_title("  Breaking News (Updated) [2024]  ")
        assert result == "breaking news updated 2024"

    def test_normalize_korean_characters(self) -> None:
        result = normalize_title("속보 뉴스")
        assert result == "속보 뉴스"


class TestValidateUrlFormat:
    """Tests for validate_url_format function."""

    def test_valid_https_url(self) -> None:
        assert validate_url_format("https://example.com/article") is True

    def test_valid_http_url(self) -> None:
        assert validate_url_format("http://example.com") is True

    def test_invalid_url_no_scheme(self) -> None:
        assert validate_url_format("example.com/article") is False

    def test_invalid_url_no_domain(self) -> None:
        assert validate_url_format("https://") is False

    def test_invalid_url_empty_string(self) -> None:
        assert validate_url_format("") is False

    def test_invalid_url_not_a_url(self) -> None:
        assert validate_url_format("not-a-url") is False

    def test_invalid_url_none_type(self) -> None:
        assert validate_url_format(None) is False  # type: ignore

    def test_valid_url_with_query_params(self) -> None:
        assert validate_url_format("https://example.com/article?id=123") is True

    def test_valid_url_with_fragment(self) -> None:
        assert validate_url_format("https://example.com/article#section") is True

    def test_valid_url_with_port(self) -> None:
        assert validate_url_format("https://example.com:8080/article") is True


class TestIsSimilarUrl:
    """Tests for is_similar_url function."""

    def test_identical_urls(self) -> None:
        url = "https://example.com/article/123"
        assert is_similar_url(url, url) is True

    def test_same_domain_same_path(self) -> None:
        url1 = "https://example.com/article/123"
        url2 = "https://example.com/article/123?ref=abc"
        assert is_similar_url(url1, url2) is True

    def test_different_domains(self) -> None:
        url1 = "https://example.com/article/123"
        url2 = "https://other.com/article/123"
        assert is_similar_url(url1, url2) is False

    def test_different_paths_below_threshold(self) -> None:
        url1 = "https://example.com/article/123"
        url2 = "https://example.com/article/456"
        # Paths are NOT similar enough (0.75 < 0.8 threshold)
        assert is_similar_url(url1, url2) is False

    def test_similar_paths_high_threshold(self) -> None:
        url1 = "https://example.com/article/123"
        url2 = "https://example.com/article/124"
        assert is_similar_url(url1, url2, threshold=0.95) is False

    def test_similar_paths_low_threshold(self) -> None:
        url1 = "https://example.com/article/123"
        url2 = "https://example.com/article/124"
        assert is_similar_url(url1, url2, threshold=0.5) is True

    def test_invalid_url_format(self) -> None:
        url1 = "not-a-url"
        url2 = "https://example.com/article/123"
        assert is_similar_url(url1, url2) is False

    def test_empty_urls(self) -> None:
        assert is_similar_url("", "") is True

    def test_url_with_fragments(self) -> None:
        url1 = "https://example.com/article/123"
        url2 = "https://example.com/article/123#section"
        assert is_similar_url(url1, url2) is True


class TestDetectDuplicateArticles:
    """Tests for detect_duplicate_articles function."""

    def test_identical_articles(self) -> None:
        assert (
            detect_duplicate_articles(
                "Breaking News",
                "https://example.com/article/123",
                "Breaking News",
                "https://example.com/article/123",
            )
            is True
        )

    def test_same_title_same_url_with_params(self) -> None:
        assert (
            detect_duplicate_articles(
                "Breaking News",
                "https://example.com/article/123",
                "Breaking News",
                "https://example.com/article/123?ref=abc",
            )
            is True
        )

    def test_different_titles_different_urls(self) -> None:
        assert (
            detect_duplicate_articles(
                "Breaking News",
                "https://example.com/article/123",
                "Other News",
                "https://example.com/article/456",
            )
            is False
        )

    def test_similar_titles_different_domains(self) -> None:
        assert (
            detect_duplicate_articles(
                "Breaking News",
                "https://example.com/article/123",
                "Breaking News",
                "https://other.com/article/456",
            )
            is False
        )

    def test_similar_titles_similar_urls(self) -> None:
        assert (
            detect_duplicate_articles(
                "  Breaking News  ",
                "https://example.com/article/123",
                "Breaking News",
                "https://example.com/article/123?ref=abc",
            )
            is True
        )

    def test_titles_with_special_chars(self) -> None:
        assert (
            detect_duplicate_articles(
                "Title (Updated)",
                "https://example.com/article/123",
                "Title Updated",
                "https://example.com/article/123",
            )
            is True
        )

    def test_custom_thresholds_strict(self) -> None:
        assert (
            detect_duplicate_articles(
                "Breaking News",
                "https://example.com/article/123",
                "Breaking News Update",
                "https://example.com/article/123",
                title_threshold=0.99,
            )
            is False
        )

    def test_custom_thresholds_lenient(self) -> None:
        assert (
            detect_duplicate_articles(
                "Breaking News",
                "https://example.com/article/123",
                "Breaking News Update",
                "https://example.com/article/123",
                title_threshold=0.7,
            )
            is True
        )

    def test_empty_titles(self) -> None:
        assert (
            detect_duplicate_articles(
                "",
                "https://example.com/article/123",
                "",
                "https://example.com/article/123",
            )
            is True
        )


class TestValidateArticle:
    """Tests for validate_article function."""

    def test_valid_article(self) -> None:
        article = Article(
            title="Valid Article",
            link="https://example.com/article",
            summary="This is a summary",
            published=datetime.now(UTC),
            source="Example Source",
            category="news",
        )
        is_valid, errors = validate_article(article)
        assert is_valid is True
        assert errors == []

    def test_article_missing_title(self) -> None:
        article = Article(
            title="",
            link="https://example.com/article",
            summary="Summary",
            published=datetime.now(UTC),
            source="Source",
            category="news",
        )
        is_valid, errors = validate_article(article)
        assert is_valid is False
        assert any("title" in error for error in errors)

    def test_article_invalid_url(self) -> None:
        article = Article(
            title="Title",
            link="not-a-url",
            summary="Summary",
            published=datetime.now(UTC),
            source="Source",
            category="news",
        )
        is_valid, errors = validate_article(article)
        assert is_valid is False
        assert any("link" in error for error in errors)

    def test_article_missing_summary(self) -> None:
        article = Article(
            title="Title",
            link="https://example.com/article",
            summary="",
            published=datetime.now(UTC),
            source="Source",
            category="news",
        )
        is_valid, errors = validate_article(article)
        assert is_valid is False
        assert any("summary" in error for error in errors)

    def test_article_missing_source(self) -> None:
        article = Article(
            title="Title",
            link="https://example.com/article",
            summary="Summary",
            published=datetime.now(UTC),
            source="",
            category="news",
        )
        is_valid, errors = validate_article(article)
        assert is_valid is False
        assert any("source" in error for error in errors)

    def test_article_missing_category(self) -> None:
        article = Article(
            title="Title",
            link="https://example.com/article",
            summary="Summary",
            published=datetime.now(UTC),
            source="Source",
            category="",
        )
        is_valid, errors = validate_article(article)
        assert is_valid is False
        assert any("category" in error for error in errors)

    def test_article_multiple_errors(self) -> None:
        article = Article(
            title="",
            link="invalid",
            summary="",
            published=datetime.now(UTC),
            source="",
            category="",
        )
        is_valid, errors = validate_article(article)
        assert is_valid is False
        assert len(errors) >= 4

    def test_article_with_none_published(self) -> None:
        article = Article(
            title="Title",
            link="https://example.com/article",
            summary="Summary",
            published=None,
            source="Source",
            category="news",
        )
        is_valid, errors = validate_article(article)
        assert is_valid is True
        assert errors == []

    def test_article_with_matched_entities(self) -> None:
        article = Article(
            title="Title",
            link="https://example.com/article",
            summary="Summary",
            published=datetime.now(UTC),
            source="Source",
            category="news",
            matched_entities={"entity1": ["keyword1", "keyword2"]},
        )
        is_valid, errors = validate_article(article)
        assert is_valid is True
        assert errors == []


@pytest.mark.unit
class TestValidatorsIntegration:
    """Integration tests for validators."""

    def test_full_article_validation_flow(self) -> None:
        title1 = "  Breaking News  "
        url1 = "https://example.com/article/123"

        title2 = "Breaking News"
        url2 = "https://example.com/article/123?ref=abc"

        assert normalize_title(title1) == normalize_title(title2)
        assert validate_url_format(url1) is True
        assert validate_url_format(url2) is True
        assert detect_duplicate_articles(title1, url1, title2, url2) is True

    def test_invalid_article_validation_flow(self) -> None:
        title = "Article"
        url = "not-a-url"

        assert validate_url_format(url) is False
        assert normalize_title(title) == "article"

    def test_article_object_validation(self) -> None:
        article = Article(
            title="Valid Article Title",
            link="https://example.com/news/article-123",
            summary="This is a comprehensive summary of the article content.",
            published=datetime.now(UTC),
            source="News Source",
            category="technology",
        )
        is_valid, errors = validate_article(article)
        assert is_valid is True
        assert len(errors) == 0
