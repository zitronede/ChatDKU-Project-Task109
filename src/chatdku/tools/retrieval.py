"""Shared result types and Qdrant payload helpers for retrieval tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llama_index.core.vector_stores.utils import metadata_dict_to_node


@dataclass(frozen=True, slots=True)
class RetrievedDocument:
    node_id: str
    text: str
    metadata: dict[str, Any]
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "text": self.text,
            "metadata": self.metadata,
            "score": self.score,
        }


def document_from_payload(node_id: str, payload: dict[str, Any], score: float) -> RetrievedDocument:
    """Convert the Qdrant payload produced by LlamaIndex back into a document."""
    node = metadata_dict_to_node(payload)
    return RetrievedDocument(
        node_id=node_id,
        text=node.get_content(),
        metadata=dict(node.metadata),
        score=score,
    )


def format_documents(documents: list[RetrievedDocument]) -> str:
    """Format retrieved results with explicit source metadata for downstream citation."""
    if not documents:
        return "No matching documents found."

    blocks = []

    for index, document in enumerate(documents, start=1):
        source = (
            document.metadata.get("source_path")
            or document.metadata.get("file_name")
            or "Unknown source"
        )

        url = document.metadata.get("url")

        page = (
            document.metadata.get("page_label")
            or document.metadata.get("page_number")
            or document.metadata.get("page")
        )

        source_line = f"Source: {source}"

        if page is not None:
            source_line += f", Page: {page}"

        if url:
            source_line += f"\nURL: {url}"

        blocks.append(
            f"[Evidence {index}]\n"
            f"{source_line}\n"
            f"Retrieval score: {document.score:.4f}\n\n"
            f"{document.text}"
        )

    return "\n\n---\n\n".join(blocks)