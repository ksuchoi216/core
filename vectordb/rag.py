from __future__ import annotations

from pathlib import Path
from typing import Any

from .vectordb import StudentRecordVectorDB


class SimpleRAG:
    def __init__(
        self,
        vdb_dir: str | Path,
        topk: int = 3,
    ) -> None:
        self.vector_store = StudentRecordVectorDB(
            vdb_dir=vdb_dir,
        ).load()
        self.k = topk

    def invoke(self, query: str, with_score: bool = False):
        if with_score:
            return self.vector_store.similarity_search_with_score(query, k=self.k)
        return self.vector_store.similarity_search(query, k=self.k)


class MetadataHeaderRAG(SimpleRAG):
    def invoke(
        self,
        query: str,
        with_score: bool = False,
        metadict: dict[str, Any] | None = None,
    ):
        if metadict:
            query = f"{self._format_metadata_header(metadict)}\n\n{query}"

        return super().invoke(query, with_score=with_score)

    @staticmethod
    def _format_metadata_header(metadict: dict[str, Any]) -> str:
        return "\n".join(
            [
                f"section: {metadict.get('section', '')}",
                f"grade: {metadict.get('grade', '')}",
                f"subsection: {metadict.get('subsection', '')}",
                f"subject: {metadict.get('subject', '')}",
            ]
        )


# class HyDERAG:
