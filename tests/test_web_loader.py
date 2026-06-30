from __future__ import annotations

from sophons.loaders.web import html_to_text


def test_html_to_text_extracts_static_job_post_content() -> None:
    html = """
    <html>
      <head><style>.hidden { display: none; }</style></head>
      <body>
        <h1>Backend Engineer</h1>
        <div>Greenhouse job post</div>
        <script>alert("ignore me")</script>
        <section>
          <p>We need Python, RAG, and browser automation experience.</p>
        </section>
      </body>
    </html>
    """

    assert html_to_text(html) == (
        "Backend Engineer\n"
        "Greenhouse job post\n"
        "We need Python, RAG, and browser automation experience."
    )
