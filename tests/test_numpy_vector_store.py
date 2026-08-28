from __future__ import annotations

import numpy as np
import pytest

from sophons import Document
from sophons.stores import NumPyVectorStore, VectorStore


def documents() -> list[Document]:
    return [
        Document(id="x", content="x axis"),
        Document(id="y", content="y axis"),
        Document(id="zero", content="no signal"),
    ]


def test_it_satisfies_the_vector_store_protocol() -> None:
    assert isinstance(NumPyVectorStore(), VectorStore)


def test_add_matrix_keeps_an_existing_float32_matrix() -> None:
    matrix = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    store = NumPyVectorStore()

    store.add_matrix(documents()[:2], matrix)

    assert store._vectors is matrix


def test_raw_vectors_are_ranked_by_cosine_similarity() -> None:
    store = NumPyVectorStore()
    store.add_matrix(
        documents(),
        np.asarray([[10.0, 0.0], [0.0, 2.0], [0.0, 0.0]], dtype=np.float32),
    )

    results = store.search([3.0, 0.0])

    assert [result.id for result in results] == ["x", "y", "zero"]
    assert [result.score for result in results] == pytest.approx([1.0, 0.0, 0.0])


def test_equal_scores_keep_insertion_order() -> None:
    store = NumPyVectorStore()
    store.add_matrix(
        documents()[:2],
        np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
    )
    assert [result.id for result in store.search([1.0, 0.0])] == ["x", "y"]


def test_limit_and_non_positive_limit() -> None:
    store = NumPyVectorStore()
    store.add_matrix(
        documents()[:2],
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    assert len(store.search([1.0, 0.0], limit=1)) == 1
    assert store.search([1.0, 0.0], limit=0) == []


def test_delete_keeps_document_rows_aligned() -> None:
    store = NumPyVectorStore()
    store.add_matrix(
        documents(),
        np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float32),
    )

    store.delete(["x"])

    assert len(store) == 2
    assert [result.id for result in store.search([1.0, 0.0])] == ["y", "zero"]


def test_wrong_row_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="1 vectors for 2 documents"):
        NumPyVectorStore().add_matrix(
            documents()[:2],
            np.asarray([[1.0, 0.0]], dtype=np.float32),
        )


def test_wrong_dimension_on_second_add_is_rejected() -> None:
    store = NumPyVectorStore()
    store.add_matrix(documents()[:1], np.asarray([[1.0, 0.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="does not match existing dimension"):
        store.add_matrix(
            documents()[1:2],
            np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        )


def test_wrong_query_dimension_is_rejected() -> None:
    store = NumPyVectorStore()
    store.add_matrix(documents()[:1], np.asarray([[1.0, 0.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="query dimension 3"):
        store.search([1.0, 0.0, 0.0])


@pytest.mark.parametrize(
    "matrix",
    [
        np.asarray([[np.nan, 0.0]], dtype=np.float32),
        np.asarray([[np.inf, 0.0]], dtype=np.float32),
    ],
)
def test_non_finite_document_vectors_are_rejected(matrix: np.ndarray) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        NumPyVectorStore().add_matrix(documents()[:1], matrix)


def test_non_finite_query_is_rejected() -> None:
    store = NumPyVectorStore()
    store.add_matrix(documents()[:1], np.asarray([[1.0, 0.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="non-finite"):
        store.search([np.nan, 0.0])
