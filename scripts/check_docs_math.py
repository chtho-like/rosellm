#!/usr/bin/env python3
"""Validate Markdown mathematics from source text through browser rendering."""

from __future__ import annotations

import argparse
import contextlib
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit


BACKTICK = chr(96)
TEXT_COMMANDS = (
    "text",
    "operatorname",
    "mathrm",
    "mathbf",
    "mathit",
    "mathtt",
)
BARE_COMMANDS = (
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "varepsilon",
    "theta",
    "lambda",
    "mu",
    "pi",
    "rho",
    "sigma",
    "tau",
    "phi",
    "psi",
    "omega",
    "widehat",
    "hat",
    "tilde",
    "bar",
    "overline",
    "underline",
    "underbrace",
    "overbrace",
    "frac",
    "dfrac",
    "sqrt",
    "sum",
    "prod",
    "nabla",
    "partial",
    "left",
    "right",
    "operatorname",
    "mathrm",
    "mathbf",
    "mathbb",
    "mathcal",
    "mathsf",
    "mathit",
    "mathtt",
    "text",
)
BARE_FUNCTIONS = ("log", "exp", "sin", "cos", "tan", "tanh", "softmax")
KNOWN_TEX_COMMANDS = frozenset(
    """
    Delta Phi Pr Vert alpha approx bar begin beta binom boxed cdot delta dfrac
    ell end epsilon eta exp frac gamma ge hat in infty int lambda langle ldots
    le left leftarrow leq lesssim lfloor log longleftrightarrow longrightarrow
    lvert mathbb mathbf mathcal mathrm max mid min mu nabla ne neq notin omega
    operatorname overline partial phi pi pm prod propto psi qquad quad rangle
    rfloor rho right rightarrow rvert sigma sim sqrt succ sum tau text tfrac
    theta times to top underbrace varepsilon widehat widetilde
    """.split()
)
EXPECTED_MATHJAX_VERSION = "3.2.2"


@dataclass(frozen=True)
class Expression:
    """One source or generated mathematical expression."""

    path: Path
    line: int
    kind: str
    tex: str


def line_number(text: str, offset: int) -> int:
    """Return the one-based line containing an offset."""

    return text.count("\n", 0, offset) + 1


def mask_range(chars: List[str], start: int, end: int) -> None:
    """Replace a range with spaces while preserving newlines and offsets."""

    for index in range(start, end):
        if chars[index] != "\n":
            chars[index] = " "


def fence_marker(line: str) -> Optional[str]:
    """Return a Markdown fence marker when a line opens or closes one."""

    stripped = line.lstrip(" ")
    indent = len(line) - len(stripped)
    if indent > 3 or not stripped:
        return None
    char = stripped[0]
    if char not in (BACKTICK, "~"):
        return None
    length = 0
    while length < len(stripped) and stripped[length] == char:
        length += 1
    return char * length if length >= 3 else None


def mask_markdown_code(text: str) -> List[str]:
    """Mask fenced and inline code without changing source offsets."""

    chars = list(text)
    offset = 0
    active_char: Optional[str] = None
    active_length = 0
    for line in text.splitlines(keepends=True):
        marker = fence_marker(line)
        if active_char is not None:
            mask_range(chars, offset, offset + len(line))
            if marker and marker[0] == active_char and len(marker) >= active_length:
                active_char = None
                active_length = 0
            offset += len(line)
            continue
        if marker:
            active_char = marker[0]
            active_length = len(marker)
            mask_range(chars, offset, offset + len(line))
            offset += len(line)
            continue

        cursor = 0
        while cursor < len(line):
            if line[cursor] != BACKTICK:
                cursor += 1
                continue
            run = 1
            while cursor + run < len(line) and line[cursor + run] == BACKTICK:
                run += 1
            marker = BACKTICK * run
            end = line.find(marker, cursor + run)
            if end < 0:
                cursor += run
                continue
            mask_range(chars, offset + cursor, offset + end + run)
            cursor = end + run
        offset += len(line)
    return chars


def unescaped_dollars(line: str) -> List[int]:
    """Return dollar-sign offsets that are not escaped by a backslash."""

    result = []
    for index, char in enumerate(line):
        if char != "$":
            continue
        slashes = 0
        cursor = index - 1
        while cursor >= 0 and line[cursor] == "\\":
            slashes += 1
            cursor -= 1
        if slashes % 2 == 0:
            result.append(index)
    return result


def mask_text_commands(tex: str) -> str:
    """Mask simple text-like command arguments before typo heuristics."""

    masked = tex
    command_group = "|".join(TEXT_COMMANDS)
    pattern = re.compile(r"\\(?:" + command_group + r")\{[^{}]*\}")
    while True:
        updated = pattern.sub("", masked)
        if updated == masked:
            return masked
        masked = updated


def validate_tex(expression: Expression, errors: List[str]) -> None:
    """Check structural TeX invariants and common silent command typos."""

    tex = expression.tex
    label = f"{expression.path}:{expression.line}"
    if not tex.strip():
        errors.append(f"{label}: empty {expression.kind} expression")
        return

    braces: List[int] = []
    for index, char in enumerate(tex):
        if char not in "{}":
            continue
        slashes = 0
        cursor = index - 1
        while cursor >= 0 and tex[cursor] == "\\":
            slashes += 1
            cursor -= 1
        if slashes % 2:
            continue
        if char == "{":
            braces.append(index)
        elif braces:
            braces.pop()
        else:
            errors.append(f"{label}: unmatched closing brace in TeX")
            break
    if braces:
        errors.append(f"{label}: unmatched opening brace in TeX")

    left_count = len(re.findall(r"\\left\b", tex))
    right_count = len(re.findall(r"\\right\b", tex))
    if left_count != right_count:
        errors.append(
            f"{label}: \\left/\\right count differs "
            f"({left_count} versus {right_count})"
        )

    environments: List[str] = []
    for match in re.finditer(r"\\(begin|end)\{([^{}]+)\}", tex):
        operation, name = match.groups()
        if operation == "begin":
            environments.append(name)
        elif not environments or environments[-1] != name:
            errors.append(f"{label}: unmatched \\end{{{name}}}")
            break
        else:
            environments.pop()
    if environments:
        errors.append(f"{label}: unclosed environment {environments[-1]}")

    heuristic_text = mask_text_commands(tex)
    bare_pattern = re.compile(
        r"(?<![\\A-Za-z])(" + "|".join(BARE_COMMANDS) + r")(?![A-Za-z])"
    )
    match = bare_pattern.search(heuristic_text)
    if match:
        errors.append(f"{label}: probable missing backslash before {match.group(1)!r}")

    function_pattern = re.compile(
        r"(?<![\\A-Za-z])(" + "|".join(BARE_FUNCTIONS) + r")(?=\\|\s*\()"
    )
    match = function_pattern.search(heuristic_text)
    if match:
        errors.append(
            f"{label}: probable missing backslash before function "
            f"{match.group(1)!r}"
        )

    for match in re.finditer(r"\\([A-Za-z]+)", tex):
        command = match.group(1)
        if command not in KNOWN_TEX_COMMANDS:
            errors.append(
                f"{label}: TeX command \\{command} is not in the reviewed "
                "MathJax command set"
            )


def mask_link_destinations(text: str, chars: List[str]) -> None:
    """Mask ordinary Markdown link destinations for prose heuristics."""

    pattern = re.compile(r"\]\((?:[^()\\]|\\.|\([^()]*\))*\)")
    for match in pattern.finditer(text):
        mask_range(chars, match.start() + 1, match.end())


def validate_plain_prose(
    path: Path,
    text: str,
    outside_math: List[str],
    errors: List[str],
    check_parentheses: bool,
) -> None:
    """Find TeX-looking expressions accidentally left as ordinary prose."""

    mask_link_destinations(text, outside_math)
    plain = "".join(outside_math)
    for match in re.finditer(r"\\[A-Za-z]+", plain):
        errors.append(
            f"{path}:{line_number(text, match.start())}: "
            "TeX command appears outside math or code delimiters"
        )
    if not check_parentheses:
        return

    tex_words = re.compile(
        r"\b(?:widehat|operatorname|mathbb|mathcal|mathrm|theta|tau|lambda|"
        r"epsilon|rho|sigma|gamma|mu|pi)\b"
    )
    stack: List[int] = []
    candidates: List[Tuple[int, int]] = []
    for index, char in enumerate(plain):
        if char == "(":
            stack.append(index)
        elif char == ")" and stack:
            candidates.append((stack.pop(), index))
    for start, end in candidates:
        content = plain[start + 1 : end]
        if "\n" in content or len(content) > 180:
            continue
        suspicious = (
            "\\" in content
            or bool(re.search(r"[A-Za-z0-9}]_[{A-Za-z0-9]", content))
            or bool(re.search(r"[A-Za-z0-9}]\^[{A-Za-z0-9]", content))
            or bool(re.fullmatch(r"[A-Za-z]", content))
            or bool(re.search(r"(?<![<>])=(?!=)", content))
            or bool(re.search(r"\|[^|]+\|", content))
            or bool(tex_words.search(content))
        )
        if suspicious:
            errors.append(
                f"{path}:{line_number(text, start)}: "
                "math-like parenthesized prose must use math delimiters"
            )


def scan_markdown(path: Path, text: str) -> Tuple[List[Expression], List[str]]:
    """Extract canonical dollar math and report source-level failures."""

    errors: List[str] = []
    code_mask = mask_markdown_code(text)
    outside_math = code_mask.copy()
    expressions: List[Expression] = []

    for match in re.finditer("\t", text):
        errors.append(f"{path}:{line_number(text, match.start())}: tab character")

    visible = "".join(code_mask)
    for match in re.finditer(r"\\[()]", visible):
        errors.append(
            f"{path}:{line_number(text, match.start())}: "
            "legacy \\(...\\) delimiter; use $...$ for GitHub and MkDocs"
        )
    for match in re.finditer(r"(?m)^\s*\\[\[\]]\s*$", visible):
        errors.append(
            f"{path}:{line_number(text, match.start())}: "
            "legacy display delimiter; use a standalone $$ line"
        )

    offset = 0
    display_start: Optional[int] = None
    display_line = 0
    display_tex: List[str] = []
    for line in visible.splitlines(keepends=True):
        body = line[:-1] if line.endswith("\n") else line
        dollars = unescaped_dollars(body)
        standalone = body.strip() == "$$"
        if standalone:
            delimiter_at = body.index("$$")
            absolute = offset + delimiter_at
            if delimiter_at != 0:
                errors.append(
                    f"{path}:{line_number(text, absolute)}: "
                    "display $$ delimiters must start at column 1"
                )
            if display_start is None:
                display_start = absolute
                display_line = line_number(text, absolute)
                display_tex = []
            else:
                expression = Expression(
                    path=path,
                    line=display_line,
                    kind="display",
                    tex="".join(display_tex).strip(),
                )
                expressions.append(expression)
                mask_range(outside_math, display_start, offset + len(body))
                display_start = None
                display_tex = []
            offset += len(line)
            continue

        if display_start is not None:
            if dollars:
                errors.append(
                    f"{path}:{line_number(text, offset + dollars[0])}: "
                    "nested dollar delimiter inside display math"
                )
            display_tex.append(line)
            offset += len(line)
            continue

        adjacent = [
            index
            for index in dollars
            if (index + 1 in dollars) or (index - 1 in dollars)
        ]
        if adjacent:
            errors.append(
                f"{path}:{line_number(text, offset + adjacent[0])}: "
                "display $$ delimiters must occupy their own lines"
            )
            offset += len(line)
            continue
        if len(dollars) % 2:
            errors.append(
                f"{path}:{line_number(text, offset + dollars[-1])}: "
                "unclosed inline dollar delimiter or unescaped currency sign"
            )
            offset += len(line)
            continue
        for index in range(0, len(dollars), 2):
            start, end = dollars[index], dollars[index + 1]
            if (
                start + 1 < len(body)
                and body[start + 1].isdigit()
                and end + 1 < len(body)
                and body[end + 1].isdigit()
            ):
                errors.append(
                    f"{path}:{line_number(text, offset + start)}: "
                    "probable paired currency signs; escape each dollar as \\$"
                )
            expression = Expression(
                path=path,
                line=line_number(text, offset + start),
                kind="inline",
                tex=body[start + 1 : end].strip(),
            )
            expressions.append(expression)
            mask_range(outside_math, offset + start, offset + end + 1)
        offset += len(line)

    if display_start is not None:
        errors.append(f"{path}:{display_line}: unclosed display math delimiter")

    for expression in expressions:
        validate_tex(expression, errors)
    validate_plain_prose(
        path,
        text,
        outside_math,
        errors,
        check_parentheses=bool(expressions)
        or (bool(path.parts) and path.parts[0] == "docs"),
    )
    return expressions, errors


class GeneratedMathParser(HTMLParser):
    """Collect Arithmatex payloads from one generated documentation page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_article = 0
        self.math_tag: Optional[str] = None
        self.math_kind: Optional[str] = None
        self.math_depth = 0
        self.buffer: List[str] = []
        self.expressions: List[str] = []
        self.kinds: List[str] = []

    @staticmethod
    def classes(attrs: Sequence[Tuple[str, Optional[str]]]) -> List[str]:
        """Return an element's class tokens."""

        value = dict(attrs).get("class") or ""
        return value.split()

    def handle_starttag(
        self, tag: str, attrs: Sequence[Tuple[str, Optional[str]]]
    ) -> None:
        classes = self.classes(attrs)
        if tag == "article" and "md-content__inner" in classes:
            self.in_article += 1
        if self.math_tag is not None:
            self.math_depth += 1
        elif self.in_article and "arithmatex" in classes:
            self.math_tag = tag
            self.math_kind = "display" if tag == "div" else "inline"
            self.math_depth = 1
            self.buffer = []

    def handle_endtag(self, tag: str) -> None:
        if self.math_tag is not None:
            self.math_depth -= 1
            if self.math_depth == 0:
                payload = "".join(self.buffer).strip()
                if payload.startswith(r"\(") and payload.endswith(r"\)"):
                    payload = payload[2:-2].strip()
                elif payload.startswith(r"\[") and payload.endswith(r"\]"):
                    payload = payload[2:-2].strip()
                self.expressions.append(payload)
                self.kinds.append(self.math_kind or "unknown")
                self.math_tag = None
                self.math_kind = None
                self.buffer = []
        if tag == "article" and self.in_article:
            self.in_article -= 1

    def handle_data(self, data: str) -> None:
        if self.math_tag is not None:
            self.buffer.append(data)


def generated_path(site_dir: Path, relative_source: Path) -> Path:
    """Map a docs-relative Markdown path to MkDocs directory-style output."""

    if relative_source == Path("index.md"):
        return site_dir / "index.html"
    if relative_source.name == "index.md":
        return site_dir / relative_source.parent / "index.html"
    return site_dir / relative_source.with_suffix("") / "index.html"


def compare_generated_site(
    root: Path,
    site_dir: Path,
    by_path: Dict[Path, List[Expression]],
    errors: List[str],
) -> int:
    """Compare ordered TeX payloads in source and generated HTML."""

    generated_total = 0
    docs_root = root / "docs"
    for source_path, source_expressions in sorted(by_path.items()):
        try:
            relative = source_path.relative_to(docs_root)
        except ValueError:
            continue
        output_path = generated_path(site_dir, relative)
        if not output_path.is_file():
            errors.append(f"{relative}: generated page is missing at {output_path}")
            continue
        parser = GeneratedMathParser()
        parser.feed(output_path.read_text(encoding="utf-8"))
        generated = parser.expressions
        generated_kinds = parser.kinds
        generated_total += len(generated)
        expected = [expression.tex.strip() for expression in source_expressions]
        expected_kinds = [expression.kind for expression in source_expressions]
        if len(expected) != len(generated):
            errors.append(
                f"{relative}: source has {len(expected)} math expressions but "
                f"generated HTML has {len(generated)}"
            )
        for index, (
            source_kind,
            source_tex,
            generated_kind,
            generated_tex,
        ) in enumerate(
            zip(expected_kinds, expected, generated_kinds, generated), start=1
        ):
            if source_kind != generated_kind or source_tex != generated_tex:
                errors.append(
                    f"{relative}: generated expression {index} differs from source "
                    f"({source_kind} versus {generated_kind})"
                )
                break
    return generated_total


class QuietHandler(SimpleHTTPRequestHandler):
    """Serve generated docs without request logging."""

    def log_message(self, format_string: str, *args: object) -> None:
        """Suppress per-request output."""


@contextlib.contextmanager
def serve_directory(directory: Path) -> Iterable[str]:
    """Serve docs and rewrite the production sitemap to the local origin."""

    class AuditHandler(QuietHandler):
        """Serve the site with instant-navigation-compatible sitemap URLs."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, directory=str(directory), **kwargs)

        def do_GET(self) -> None:
            """Rewrite sitemap locations to this server's ephemeral origin."""

            request_path = urlsplit(self.path).path
            if request_path.endswith("/sitemap.xml"):
                sitemap = (directory / "sitemap.xml").read_text(encoding="utf-8")
                origin = "http://" + (self.headers.get("Host") or "")

                def local_location(match: re.Match) -> str:
                    parsed = urlsplit(match.group(1))
                    path = parsed.path
                    if path == "/rosellm":
                        path = "/"
                    elif path.startswith("/rosellm/"):
                        path = path[len("/rosellm") :]
                    return f"<loc>{origin}{path}</loc>"

                payload = re.sub(
                    r"<loc>(https?://[^<]+)</loc>",
                    local_location,
                    sitemap,
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/xml")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            super().do_GET()

    server = ThreadingHTTPServer(("127.0.0.1", 0), AuditHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def chrome_binary(explicit: Optional[str]) -> Optional[str]:
    """Find a Chrome or Chromium executable."""

    candidates = [
        explicit,
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    return None


def browser_audit(
    site_dir: Path, explicit_chrome: Optional[str], errors: List[str]
) -> Tuple[int, str]:
    """Render every page with MathJax at desktop and mobile widths."""

    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.support.ui import WebDriverWait
    except ImportError:
        errors.append(
            "browser audit requires Selenium; install the repository dev extra"
        )
        return 0, "unknown"

    binary = chrome_binary(explicit_chrome)
    if binary is None:
        errors.append("browser audit could not find Chrome or Chromium")
        return 0, "unknown"

    pages = sorted(site_dir.glob("**/index.html"))
    options = Options()
    options.binary_location = binary
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.set_capability("goog:loggingPrefs", {"browser": "ALL"})
    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(45)
    driver.set_script_timeout(30)
    wait = WebDriverWait(driver, 30)
    rendered = 0
    version = "unknown"

    with serve_directory(site_dir) as base_url:
        try:
            for width, height, mobile in (
                (1440, 1000, False),
                (390, 844, True),
            ):
                driver.execute_cdp_cmd(
                    "Emulation.setDeviceMetricsOverride",
                    {
                        "width": width,
                        "height": height,
                        "deviceScaleFactor": 1,
                        "mobile": mobile,
                    },
                )
                for page in pages:
                    relative = page.relative_to(site_dir)
                    route = (
                        "/"
                        if relative == Path("index.html")
                        else "/" + relative.parent.as_posix() + "/"
                    )
                    try:
                        driver.get(base_url + route)
                        wait.until(
                            lambda current: current.execute_script(
                                "return !!(window.MathJax && " "window.MathJax.version)"
                            )
                        )
                        wait.until(
                            lambda current: current.execute_script(
                                "const math = [...document.querySelectorAll("
                                "'.arithmatex')];"
                                "return math.every((element) => "
                                "element.querySelectorAll(':scope > mjx-container')"
                                ".length === 1 && "
                                "element.querySelectorAll("
                                "'mjx-container mjx-container').length === 0)"
                            )
                        )
                        browser_ready = driver.execute_async_script(
                            """
                            const done = arguments[0];
                            Promise.resolve(window.MathJax.startup.promise)
                              .then(() => document.fonts
                                ? document.fonts.ready
                                : Promise.resolve())
                              .then(() => new Promise((resolve) =>
                                requestAnimationFrame(() =>
                                  requestAnimationFrame(resolve))))
                              .then(() => done(true))
                              .catch((error) => done(String(error)));
                            """
                        )
                        if browser_ready is not True:
                            raise RuntimeError(str(browser_ready))
                    except Exception as exc:
                        errors.append(
                            f"{route}: MathJax did not finish at {width}px "
                            f"({type(exc).__name__})"
                        )
                        continue

                    stats = driver.execute_script(
                        """
                        const math = [...document.querySelectorAll('.arithmatex')];
                        const badOverflow = math.filter((element) => {
                          if (element.tagName !== 'DIV') return false;
                          if (element.scrollWidth <= element.clientWidth + 1) {
                            return false;
                          }
                          const overflow = getComputedStyle(element).overflowX;
                          return overflow !== 'auto' && overflow !== 'scroll';
                        });
                        return {
                          version: window.MathJax.version,
                          math: math.length,
                          typed: document.querySelectorAll(
                            '.arithmatex > mjx-container'
                          ).length,
                          allTyped: document.querySelectorAll(
                            'mjx-container'
                          ).length,
                          invalidContainers: math.filter((element) =>
                            element.querySelectorAll(
                              ':scope > mjx-container'
                            ).length !== 1
                            || element.querySelectorAll(
                              'mjx-container mjx-container'
                            ).length !== 0
                          ).length,
                          errors: [...document.querySelectorAll('mjx-merror')]
                            .map((element) =>
                              element.getAttribute('data-mjx-error')
                              || element.textContent
                            ),
                          recordedErrors: window.__mathjaxErrors || [],
                          badOverflow: badOverflow.length,
                          documentOverflow:
                            document.documentElement.scrollWidth
                            > document.documentElement.clientWidth + 1,
                        };
                        """
                    )
                    version = stats["version"]
                    rendered += stats["typed"]
                    if version != EXPECTED_MATHJAX_VERSION:
                        errors.append(
                            f"{route}: expected MathJax "
                            f"{EXPECTED_MATHJAX_VERSION}, loaded {version}"
                        )
                    if (
                        stats["typed"] != stats["math"]
                        or stats["allTyped"] != stats["math"]
                        or stats["invalidContainers"]
                    ):
                        errors.append(
                            f"{route}: expected one direct MathJax container per "
                            f"expression at {width}px (wrappers={stats['math']}, "
                            f"direct={stats['typed']}, all={stats['allTyped']})"
                        )
                    if stats["errors"]:
                        errors.append(
                            f"{route}: MathJax errors at {width}px: "
                            + "; ".join(stats["errors"])
                        )
                    if stats["recordedErrors"]:
                        errors.append(
                            f"{route}: recorded MathJax errors at {width}px: "
                            + "; ".join(stats["recordedErrors"])
                        )
                    if stats["badOverflow"]:
                        errors.append(
                            f"{route}: {stats['badOverflow']} display equations "
                            f"can be clipped at {width}px"
                        )
                    if stats["documentOverflow"]:
                        errors.append(
                            f"{route}: page has global horizontal overflow at "
                            f"{width}px"
                        )
                    logs = driver.get_log("browser")
                    math_logs = [
                        entry["message"]
                        for entry in logs
                        if entry["level"] == "SEVERE"
                        and re.search(r"mathjax|tex-mml-chtml", entry["message"], re.I)
                    ]
                    if math_logs:
                        errors.append(
                            f"{route}: browser reported MathJax load errors: "
                            + "; ".join(math_logs)
                        )

            driver.execute_cdp_cmd("Emulation.clearDeviceMetricsOverride", {})
            driver.get(base_url + "/agentic-rl/")
            wait.until(
                lambda current: current.execute_script(
                    "const math = [...document.querySelectorAll('.arithmatex')];"
                    "return math.every((element) => "
                    "element.querySelectorAll(':scope > mjx-container').length "
                    "=== 1)"
                )
            )
            driver.execute_script("window.__rose_math_audit = 'preserved'")
            target_parser = GeneratedMathParser()
            target_parser.feed(
                (
                    site_dir / "agentic-rl" / "mathematical-foundations" / "index.html"
                ).read_text(encoding="utf-8")
            )
            target_count = len(target_parser.expressions)
            target = driver.execute_script(
                """
                const link = [...document.querySelectorAll('.md-content a')]
                  .find((item) =>
                    item.href.includes('mathematical-foundations')
                  );
                if (!link) return null;
                link.click();
                return link.href;
                """
            )
            if target:
                wait.until(
                    lambda current: "mathematical-foundations" in current.current_url
                )
                wait.until(
                    lambda current: current.execute_script(
                        "const heading = document.querySelector('h1');"
                        "return heading?.firstChild?.textContent.trim()"
                    )
                    == "Mathematical Foundations"
                )
                wait.until(
                    lambda current: current.execute_script(
                        "return {"
                        "math: document.querySelectorAll('.arithmatex').length,"
                        "typed: document.querySelectorAll("
                        "'.arithmatex > mjx-container').length"
                        "}"
                    )
                    == {"math": target_count, "typed": target_count}
                )
                instant_errors = driver.execute_script(
                    "return {"
                    "merrors: document.querySelectorAll('mjx-merror').length,"
                    "recorded: (window.__mathjaxErrors || []).length"
                    "}"
                )
                if instant_errors != {"merrors": 0, "recorded": 0}:
                    errors.append(
                        "instant-navigation destination contains MathJax errors"
                    )
                instant = driver.execute_script(
                    "return window.__rose_math_audit || null"
                )
                if instant != "preserved":
                    errors.append(
                        "instant-navigation audit performed a full reload; "
                        "the MathJax document$ hook was not exercised"
                    )
            else:
                errors.append("instant-navigation audit could not find target link")
        finally:
            driver.quit()
    return rendered, version


def tracked_markdown(root: Path) -> List[Path]:
    """Return tracked and unignored untracked Markdown files."""

    result = subprocess.run(
        [
            "git",
            "ls-files",
            "-c",
            "-o",
            "--exclude-standard",
            "-z",
            "--",
            "*.md",
        ],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
    )
    return [root / item.decode("utf-8") for item in result.stdout.split(b"\0") if item]


def validate_checker_fixtures(errors: List[str]) -> None:
    """Protect the validator's most important malformed-input checks."""

    fixture_path = Path("docs/__math_checker_fixture__.md")
    malformed = (
        ("raw TeX prose", r"Bad \alpha prose.", "outside math"),
        (
            "missing command backslash",
            "$operatorname{clip}(x)$",
            "missing backslash",
        ),
        (
            "unknown TeX command",
            r"$\doesnotexist{x}$",
            "not in the reviewed",
        ),
        (
            "counterfeit parenthesized math",
            "Bad (4p(1-p)=1) prose.",
            "math-like parenthesized prose",
        ),
        ("tab character", "Bad\tprose.", "tab character"),
        (
            "paired currency signs",
            "Costs rose from $5 million to $10 million.",
            "paired currency signs",
        ),
        (
            "indented display delimiter",
            "  $$\n  x=1\n  $$\n",
            "start at column 1",
        ),
    )
    for name, source, expected in malformed:
        _, fixture_errors = scan_markdown(fixture_path, source)
        if not any(expected in error for error in fixture_errors):
            errors.append(
                f"validator regression: {name} fixture did not produce " f"{expected!r}"
            )

    clean = """Literal examples stay in code: `\\alpha` and `$bad$`.

```tex
\\doesnotexist{x}
(4p(1-p)=1)
```

Escaped currency \\$5 and valid $\\operatorname{clip}(x)$ are allowed.
"""
    _, fixture_errors = scan_markdown(fixture_path, clean)
    if fixture_errors:
        errors.append(
            "validator regression: clean code/currency fixture failed: "
            + "; ".join(fixture_errors)
        )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--site-dir",
        type=Path,
        help="compare source math with an existing MkDocs output directory",
    )
    parser.add_argument(
        "--browser",
        action="store_true",
        help="render every generated page in headless Chrome",
    )
    parser.add_argument(
        "--chrome-binary",
        help="explicit Chrome or Chromium executable for --browser",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run source, generated-HTML, and optional browser checks."""

    args = parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    paths = tracked_markdown(root)
    by_path: Dict[Path, List[Expression]] = {}
    errors: List[str] = []
    validate_checker_fixtures(errors)
    for path in paths:
        expressions, path_errors = scan_markdown(
            path.relative_to(root), path.read_text(encoding="utf-8")
        )
        by_path[path] = expressions
        errors.extend(path_errors)

    source_total = sum(len(items) for items in by_path.values())
    source_files = sum(bool(items) for items in by_path.values())
    print(
        f"source: {source_total} expressions in {source_files} of "
        f"{len(paths)} repository Markdown files"
    )

    site_dir: Optional[Path] = None
    if args.site_dir:
        site_dir = args.site_dir
        if not site_dir.is_absolute():
            site_dir = root / site_dir
        if not site_dir.is_dir():
            errors.append(f"generated site directory does not exist: {site_dir}")
        else:
            generated_total = compare_generated_site(root, site_dir, by_path, errors)
            print(f"generated HTML: {generated_total} Arithmatex expressions")

    if args.browser:
        if site_dir is None:
            site_dir = root / "site"
        if not site_dir.is_dir():
            errors.append(f"browser site directory does not exist: {site_dir}")
        else:
            rendered, version = browser_audit(site_dir, args.chrome_binary, errors)
            print(
                f"browser: {rendered} rendered expressions across two "
                f"viewports with MathJax {version}"
            )

    if errors:
        print("\nMath documentation check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Math documentation check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
