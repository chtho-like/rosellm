import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "literature_discovery.py"
SPEC = importlib.util.spec_from_file_location("literature_discovery", MODULE_PATH)
assert SPEC and SPEC.loader
literature_discovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(literature_discovery)


def test_parse_sitemap_urlset():
    payload = b"""<?xml version='1.0'?>
    <urlset xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'>
      <url><loc>https://example.com/a</loc><lastmod>2026-07-19</lastmod></url>
      <url><loc>https://example.com/b</loc></url>
    </urlset>"""
    assert literature_discovery.parse_sitemap(payload, "fixture") == [
        ("https://example.com/a", "2026-07-19"),
        ("https://example.com/b", None),
    ]


def test_merge_preserves_all_discovery_feeds():
    rows = [
        literature_discovery.record(
            "Lab", "https://example.com/paper", "feed-a", "publication", "2026-01-01", "2026-07-19"
        ),
        literature_discovery.record(
            "Lab", "https://example.com/paper", "feed-b", "safety", "2026-02-01", "2026-07-19"
        ),
    ]
    merged = literature_discovery._merge(rows)
    assert len(merged) == 1
    assert merged[0]["source_kinds"] == ["publication", "safety"]
    assert merged[0]["source_urls"] == ["feed-a", "feed-b"]
    assert merged[0]["lastmod"] == "2026-02-01"
    assert rows[0]["source_kind"] == "publication"


def test_partial_refresh_preserves_other_organizations_only():
    current = [{"org": "OpenAI", "url": "https://openai.com/new"}]
    existing = [
        {"org": "OpenAI", "url": "https://openai.com/old"},
        {"org": "Anthropic", "url": "https://anthropic.com/keep"},
    ]
    assert literature_discovery._preserve_unselected(
        current, existing, {"OpenAI"}
    ) == [
        {"org": "Anthropic", "url": "https://anthropic.com/keep"},
        {"org": "OpenAI", "url": "https://openai.com/new"},
    ]


def test_parse_openai_rss(monkeypatch):
    payload = b"""<?xml version='1.0'?>
    <rss version='2.0'><channel><item>
      <title>Example research</title>
      <description>A concise abstract.</description>
      <link>https://openai.com/index/example</link>
      <category>Research</category>
      <pubDate>Sun, 19 Jul 2026 00:00:00 GMT</pubDate>
    </item></channel></rss>"""
    monkeypatch.setattr(literature_discovery, "fetch", lambda *_args: payload)
    rows = literature_discovery.discover_openai_rss(1, 0, "2026-07-19")
    assert rows == [
        {
            "org": "OpenAI",
            "url": "https://openai.com/index/example",
            "title": "Example research",
            "description": "A concise abstract.",
            "category": "Research",
            "published_at": "2026-07-19",
            "source_url": "https://openai.com/news/rss.xml",
            "discovered_at": "2026-07-19",
        }
    ]
