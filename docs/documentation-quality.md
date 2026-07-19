# Documentation and Mathematical Rendering Standard

This page is the executable quality contract for RoseLLM documentation. It
exists because a successful Markdown build does not prove that mathematics is
visible, semantically correct, responsive, or re-rendered after client-side
navigation.

## The rendering contract

One expression passes only when every layer agrees:

1. the Markdown source contains an unambiguous mathematical span;
2. Python-Markdown and Pymdownx Arithmatex convert that span into an
   `arithmatex` element;
3. the strict MkDocs build emits the expected expression in the expected page;
4. MathJax parses the TeX without an unknown command or structural error;
5. the browser emits one CHTML container for the expression;
6. a wide display expression scrolls horizontally instead of being clipped on
   a narrow screen;
7. instant navigation typesets the newly loaded page; and
8. the rendered notation expresses the intended mathematics.

The final condition requires technical review. Automation can prove that
`mu` is valid TeX text; it cannot infer that the author intended the Greek
letter $\mu$.

## Canonical source syntax

Use dollar delimiters in source Markdown. This is the repository's single
authoring convention, and both GitHub's repository renderer and the MkDocs
pipeline support it. GitHub documents dollar-delimited LaTeX in
[Writing mathematical expressions](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions),
while the
[Arithmatex reference](https://facelessuser.github.io/pymdown-extensions/extensions/arithmatex/)
specifies how dollar input becomes `arithmatex` inline and block elements.

```markdown
Inline probability: $p_\theta(y\mid x)$.

$$
\nabla_\theta J(\theta)
=\mathbb E_{\tau\sim p_\theta}
\left[R(\tau)\nabla_\theta\log p_\theta(\tau)\right].
$$
```

The rules are strict:

- use `$...$` for inline mathematics;
- put each opening and closing `$$` for display mathematics on its own line at
  column 1;
- do not use `\(...\)` or `\[...\]` in Markdown source;
- escape a currency marker as `\$`, for example `\$5.6M`;
- keep inline mathematics on one physical source line;
- put prose outside the delimiters and use `\text{...}` only when words are
  genuinely part of a formula; and
- use fenced code for literal TeX or delimiter examples that must not render.

Arithmatex rewrites the canonical source into `\(...\)` or `\[...\]`
inside generated HTML. Those wrappers are an implementation detail, not the
authoring syntax.

## Command and typography rules

A TeX command always keeps its backslash. Silent omissions are dangerous
because MathJax often renders the remaining letters instead of raising an
error:

| Wrong source | Correct source | Failure |
|---|---|---|
| `$mu(a)$` | `$\mu(a)$` | italic `m u` replaces the Greek symbol |
| `$exp(log p)$` | `$\exp(\log p)$` | function names become products of variables |
| `$widehat A` | `$\widehat A` | the accent disappears |
| `$D_{KL}$` | `$D_{\mathrm{KL}}$` | acronym letters become multiplied variables |

Use upright text for labels and acronyms, such as
`\mathrm{KL}`, `\text{old}`, and `\operatorname{clip}`. Use
`\lvert x\rvert` or `\Vert x\Vert` instead of raw vertical bars,
especially in Markdown tables.

Every `\left` needs a matching `\right`. Every
`\begin{environment}` needs the corresponding
`\end{environment}`. Braces must balance after escaped literal braces are
excluded.

## Markdown context rules

### Lists

Do not indent display-math delimiters under a numbered list item. The Markdown
list parser can consume a three-space-indented display block before Arithmatex
sees it, leaving apparently balanced TeX as plain text.

Use inline math for a short list formula. For a long formula, finish the list
sentence, return to the top level, introduce the equation in a normal paragraph,
and place the display block there.

### Tables

Prefer TeX commands that do not contain a literal Markdown column separator:

- conditional probability: `p(y\mid x)`;
- norms: `\lVert x\rVert_2`;
- set conditions: `\{x:\;x>0\}`.

If a literal vertical bar is unavoidable, escape it for the Markdown table and
verify the generated cell. A table that builds is not necessarily a table whose
formula stayed in the intended column.

### Code and logs

Put Python, shell, configuration, tensor shapes, and literal TeX in fenced code
blocks. Inline identifiers belong in inline code. Do not use ordinary
parentheses as counterfeit math, such as `(r=1)` or `(L_i)`; use
real math delimiters so the notation is searchable and rendered consistently.

Tabs are forbidden in Markdown. A missing backslash before `\tau` can
silently become a tab followed by `au` when text passes through an
incorrect string-escaping layer.

## MathJax configuration

The site loads `docs/javascripts/mathjax.js` before the pinned MathJax
3.2.2 combined TeX-to-CHTML component. The configuration:

- processes only `arithmatex` elements;
- declares the wrappers emitted by Arithmatex;
- removes the `noundefined` package so unknown commands become detectable
  errors instead of red but otherwise valid output;
- records parser errors in `window.__mathjaxErrors`; and
- subscribes to Material for MkDocs' `document$` stream so instant
  navigation clears stale state, resets TeX numbering, and typesets the new
  page.

This lifecycle follows Material for MkDocs'
[MathJax integration](https://squidfunk.github.io/mkdocs-material/reference/math/).
Material's
[instant-navigation documentation](https://squidfunk.github.io/mkdocs-material/setup/setting-up-navigation/#instant-loading)
also explains why `site_url` and a valid sitemap are required. MathJax 3.2's
[TeX input options](https://docs.mathjax.org/en/v3.2/options/input/tex.html) and
[`noundefined` reference](https://docs.mathjax.org/en/v3.2/input/tex/extensions/noundefined.html)
document both the error callback and removal syntax used here.

Keep the local configuration before the remote MathJax script in
`mkdocs.yml`. Pin the exact MathJax version so a documentation build does
not silently change parser behavior.

## Automated acceptance gates

Create an isolated documentation environment with the tested dependency set:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-docs.txt
```

Run the complete non-browser gate:

```bash
make docs
```

It performs three checks in order:

1. `scripts/check_docs_math.py` scans every tracked or unignored new Markdown
   file, ignores fenced and inline code, checks canonical delimiters, rejects
   raw currency dollars, validates TeX structure, and detects common
   missing-command and plain-prose failures.
2. `python3 -m mkdocs build --strict --clean` validates navigation, Markdown,
   links handled by MkDocs, and page generation.
3. the checker compares every ordered source expression with the corresponding
   generated `arithmatex` payload. A count-only comparison is insufficient:
   one lost expression and one duplicated expression could cancel.

Run the browser gate before merging a math-heavy change:

```bash
make docs-render
```

The browser gate serves the generated site locally and uses headless Chrome to
load every page at desktop and mobile viewport sizes. It requires:

- every `arithmatex` element to contain a MathJax CHTML container;
- exactly one direct CHTML container per expression, with no nested or duplicate
  typesetting;
- the configured MathJax version to be exactly 3.2.2;
- no `mjx-merror` element and no recorded MathJax parser error;
- no MathJax script-load failure in the browser console;
- every over-wide display equation to have horizontal scrolling and the full
  page to have no global horizontal overflow; and
- a client-side navigation from the curriculum to the mathematics chapter to
  preserve the page runtime and type-set the new content.

For an interactive visual review:

```bash
python3 -m mkdocs serve
```

Inspect at least the longest aligned derivation, one cases environment, a
formula-heavy table, and a wide equation at a narrow viewport. Automated width
checks establish scrollability, not visual elegance.

## Failure signatures

| Symptom | Likely layer | First check |
|---|---|---|
| literal dollars or TeX appear on the page | Markdown/Arithmatex | source delimiters and list indentation |
| `Missing \left or extra \right` | TeX structure | paired scalable delimiters |
| red command text without `mjx-merror` | undefined-command handling | MathJax package configuration |
| first page works, next page shows raw TeX | instant navigation | `document$` subscription |
| equation disappears on mobile | responsive CHTML | display-block overflow style |
| formula looks plausible but uses letter products | semantic typography | missing command backslashes |
| dollar amounts become italic math | source ambiguity | escape currency with `\$` |

## Clean acceptance checklist

A documentation change is complete only when:

- all prose and code-facing text remain English;
- source validation passes over every tracked or unignored new Markdown file;
- strict MkDocs build passes;
- source and generated expression sequences match;
- every generated page passes desktop and mobile browser rendering;
- instant navigation re-typesets a formula-heavy destination;
- a reviewer checks the mathematics, not only the absence of parser errors;
- `git diff --check` is clean; and
- the generated `site/` directory remains untracked.

The pre-commit hook runs the source layer automatically. The full generated and
browser gates remain explicit because they require MkDocs, Chrome, and network
access to the pinned MathJax asset.
