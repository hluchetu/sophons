from __future__ import annotations

import argparse
import html
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


NAV_ITEMS = [
    ("Overview", "/"),
    ("Agents", "/agents/"),
    ("Tools", "/tools/"),
    ("LLM Architecture", "/llm_architecture"),
    ("Loaders", "/loaders_architecture"),
    ("Splitters", "/splitters_architecture"),
    ("Retrievers", "/retrievers_architecture"),
    ("Storage", "/storage_architecture"),
    ("Memory", "/memory_architecture"),
    ("RAG", "/rag_architecture"),
    ("Integrations", "/integrations_architecture"),
]


CSS = """
:root {
  color-scheme: light;
  --bg: #fbfaf7;
  --panel: #ffffff;
  --text: #1f2933;
  --muted: #687382;
  --line: #e5e0d8;
  --accent: #256f68;
  --accent-soft: #e8f3f1;
  --code-bg: #10201f;
  --code-text: #eaf5f2;
  --shadow: 0 18px 45px rgba(50, 43, 32, 0.08);
}

* { box-sizing: border-box; }

body {
  margin: 0;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--text);
  background:
    linear-gradient(180deg, rgba(37, 111, 104, 0.08), transparent 360px),
    var(--bg);
}

a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }

.shell {
  min-height: 100vh;
  display: grid;
  grid-template-columns: 280px minmax(0, 1fr);
}

.sidebar {
  position: sticky;
  top: 0;
  height: 100vh;
  border-right: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.72);
  backdrop-filter: blur(18px);
  padding: 28px 18px;
  overflow: auto;
}

.brand {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 0 10px 24px;
}

.mark {
  width: 34px;
  height: 34px;
  border-radius: 9px;
  background: var(--accent);
  color: white;
  display: grid;
  place-items: center;
  font-weight: 800;
}

.brand-title { font-size: 16px; font-weight: 760; }
.brand-subtitle { font-size: 12px; color: var(--muted); margin-top: 2px; }

.nav {
  display: grid;
  gap: 4px;
}

.nav a {
  color: #34404d;
  padding: 9px 10px;
  border-radius: 8px;
  font-size: 14px;
}

.nav a.active {
  background: var(--accent-soft);
  color: var(--accent);
  font-weight: 700;
}

.main {
  min-width: 0;
  padding: 42px 48px 80px;
}

.content {
  max-width: 920px;
  background: rgba(255, 255, 255, 0.84);
  border: 1px solid var(--line);
  border-radius: 16px;
  box-shadow: var(--shadow);
  padding: 46px 54px;
}

.eyebrow {
  display: inline-flex;
  align-items: center;
  height: 26px;
  padding: 0 10px;
  border-radius: 999px;
  color: var(--accent);
  background: var(--accent-soft);
  font-size: 12px;
  font-weight: 700;
  margin-bottom: 16px;
}

h1 {
  margin: 0 0 14px;
  font-size: 44px;
  line-height: 1.05;
  letter-spacing: 0;
}

h2 {
  margin: 42px 0 14px;
  padding-top: 10px;
  font-size: 25px;
  letter-spacing: 0;
}

h3 {
  margin: 28px 0 10px;
  font-size: 18px;
  letter-spacing: 0;
}

p, li {
  font-size: 16px;
  line-height: 1.72;
}

p { margin: 12px 0; }
ul, ol { padding-left: 24px; }
li + li { margin-top: 6px; }

hr {
  border: 0;
  border-top: 1px solid var(--line);
  margin: 34px 0;
}

code {
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
  font-size: 0.92em;
}

:not(pre) > code {
  color: #115e59;
  background: var(--accent-soft);
  padding: 2px 6px;
  border-radius: 6px;
}

pre {
  overflow-x: auto;
  border-radius: 12px;
  padding: 18px 20px;
  background: var(--code-bg);
  color: var(--code-text);
  border: 1px solid rgba(255, 255, 255, 0.08);
}

pre code {
  color: inherit;
  background: transparent;
  padding: 0;
}

table {
  width: 100%;
  border-collapse: collapse;
  margin: 18px 0 26px;
  font-size: 14px;
  overflow: hidden;
  border: 1px solid var(--line);
  border-radius: 10px;
}

th, td {
  text-align: left;
  padding: 12px 14px;
  border-bottom: 1px solid var(--line);
  vertical-align: top;
}

th {
  background: #f4f1eb;
  font-size: 13px;
  color: #42505d;
}

tr:last-child td { border-bottom: 0; }

.frontmatter-description {
  color: var(--muted);
  font-size: 18px;
  line-height: 1.6;
  margin-bottom: 30px;
}

@media (max-width: 860px) {
  .shell { display: block; }
  .sidebar {
    position: relative;
    height: auto;
    border-right: 0;
    border-bottom: 1px solid var(--line);
  }
  .main { padding: 24px 16px 48px; }
  .content { padding: 30px 22px; border-radius: 12px; }
  h1 { font-size: 34px; }
}
"""


def read_page(path: str) -> tuple[Path, str]:
    normalized = path.strip("/")
    if normalized == "":
        file_path = DOCS / "index.md"
    else:
        candidate = DOCS / normalized
        if candidate.is_dir():
            file_path = candidate / "index.md"
        elif candidate.suffix:
            file_path = candidate
        else:
            file_path = candidate.with_suffix(".md")

    if not file_path.resolve().is_relative_to(DOCS.resolve()):
        raise FileNotFoundError
    if not file_path.exists():
        raise FileNotFoundError
    return file_path, file_path.read_text(encoding="utf-8")


def split_frontmatter(markdown: str) -> tuple[dict[str, str], str]:
    if not markdown.startswith("---\n"):
        return {}, markdown

    end = markdown.find("\n---\n", 4)
    if end == -1:
        return {}, markdown

    metadata: dict[str, str] = {}
    frontmatter = markdown[4:end]
    for line in frontmatter.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()

    return metadata, markdown[end + 5 :]


def render_markdown(markdown: str) -> str:
    blocks: list[str] = []
    lines = markdown.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped.startswith("```"):
            language = stripped[3:].strip()
            code_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1
            code = html.escape("\n".join(code_lines))
            blocks.append(f'<pre><code class="language-{html.escape(language)}">{code}</code></pre>')
            continue

        if stripped == "---":
            blocks.append("<hr>")
            i += 1
            continue

        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            text = stripped[level:].strip()
            if 1 <= level <= 3:
                blocks.append(f"<h{level}>{render_inline(text)}</h{level}>")
            else:
                blocks.append(f"<p>{render_inline(text)}</p>")
            i += 1
            continue

        if stripped.startswith("|") and "|" in stripped[1:]:
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            blocks.append(render_table(table_lines))
            continue

        if stripped.startswith("- "):
            items: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("- "):
                items.append(lines[i].strip()[2:])
                i += 1
            blocks.append("<ul>" + "".join(f"<li>{render_inline(item)}</li>" for item in items) + "</ul>")
            continue

        paragraph = [stripped]
        i += 1
        while i < len(lines):
            next_line = lines[i].strip()
            if (
                not next_line
                or next_line.startswith("#")
                or next_line.startswith("```")
                or next_line.startswith("- ")
                or next_line.startswith("|")
                or next_line == "---"
            ):
                break
            paragraph.append(next_line)
            i += 1
        blocks.append(f"<p>{render_inline(' '.join(paragraph))}</p>")

    return "\n".join(blocks)


def render_inline(text: str) -> str:
    rendered = html.escape(text)
    rendered = re.sub(r"`([^`]+)`", r"<code>\1</code>", rendered)
    rendered = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", rendered)
    rendered = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', rendered)
    return rendered


def render_table(lines: list[str]) -> str:
    rows = [[cell.strip() for cell in line.strip("|").split("|")] for line in lines]
    if len(rows) >= 2 and all(set(cell) <= {"-", ":"} for cell in rows[1]):
        header = rows[0]
        body = rows[2:]
    else:
        header = rows[0]
        body = rows[1:]

    head_html = "<thead><tr>" + "".join(f"<th>{render_inline(cell)}</th>" for cell in header) + "</tr></thead>"
    body_html = "<tbody>"
    for row in body:
        body_html += "<tr>" + "".join(f"<td>{render_inline(cell)}</td>" for cell in row) + "</tr>"
    body_html += "</tbody>"
    return f"<table>{head_html}{body_html}</table>"


def render_nav(path: str) -> str:
    normalized = "/" + path.strip("/")
    if normalized != "/":
        normalized += "/"

    links = []
    for label, href in NAV_ITEMS:
        active = normalized == href or (href != "/" and normalized.startswith(href))
        class_name = "active" if active else ""
        links.append(f'<a class="{class_name}" href="{href}">{html.escape(label)}</a>')
    return "\n".join(links)


def render_page(path: str) -> bytes:
    _, markdown = read_page(path)
    metadata, body = split_frontmatter(markdown)
    title = metadata.get("title", "Sophons")
    description = metadata.get("description", "")
    description_html = (
        f'<div class="frontmatter-description">{html.escape(description)}</div>'
        if description
        else ""
    )
    content = render_markdown(body)
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)} - Sophons</title>
  <style>{CSS}</style>
</head>
<body>
  <div class="shell">
    <aside class="sidebar">
      <div class="brand">
        <div class="mark">S</div>
        <div>
          <div class="brand-title">Sophons</div>
          <div class="brand-subtitle">Agent SDK docs</div>
        </div>
      </div>
      <nav class="nav">{render_nav(path)}</nav>
    </aside>
    <main class="main">
      <article class="content">
        <div class="eyebrow">Documentation</div>
        {description_html}
        {content}
      </article>
    </main>
  </div>
</body>
</html>"""
    return document.encode("utf-8")


class DocsHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return

        try:
            body = render_page(path)
        except FileNotFoundError:
            self.send_error(404, "Page not found")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=3001)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), DocsHandler)
    print(f"Serving Sophons docs at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
