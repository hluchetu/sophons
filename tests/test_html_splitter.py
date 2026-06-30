from __future__ import annotations

from sophons.documents import Document
from sophons.splitters import HTMLSplitter


def test_html_splitter_splits_job_post_sections() -> None:
    splitter = HTMLSplitter()
    document = Document(
        id="greenhouse_job",
        content="""
        <html>
          <body>
            <h1>Backend Engineer</h1>
            <section>
              <h2>About the role</h2>
              <p>Build Python services and RAG systems.</p>
            </section>
            <section>
              <h2>Requirements</h2>
              <ul>
                <li>Python</li>
                <li>Browser automation</li>
              </ul>
            </section>
            <script>ignore me</script>
          </body>
        </html>
        """,
        metadata={"source": "greenhouse"},
    )

    chunks = splitter.split_document(document)

    assert [
        (chunk.content, chunk.metadata["tag"], chunk.metadata["heading_path"])
        for chunk in chunks
    ] == [
        ("Backend Engineer", "h1", ["Backend Engineer"]),
        ("About the role", "h2", ["Backend Engineer", "About the role"]),
        (
            "Build Python services and RAG systems.",
            "p",
            ["Backend Engineer", "About the role"],
        ),
        (
            "About the role Build Python services and RAG systems.",
            "section",
            ["Backend Engineer", "About the role"],
        ),
        ("Requirements", "h2", ["Backend Engineer", "Requirements"]),
        ("Python", "li", ["Backend Engineer", "Requirements"]),
        ("Browser automation", "li", ["Backend Engineer", "Requirements"]),
        (
            "Requirements Python Browser automation",
            "section",
            ["Backend Engineer", "Requirements"],
        ),
    ]
    assert chunks[0].id == "greenhouse_job#html_0"
    assert chunks[0].metadata["parent_id"] == "greenhouse_job"


def test_html_splitter_ignores_scripts_and_styles() -> None:
    splitter = HTMLSplitter()

    chunks = splitter.split_text(
        "<style>hidden css</style><p>Visible job text.</p><script>hidden js</script>"
    )

    assert [chunk.content for chunk in chunks] == ["Visible job text."]
