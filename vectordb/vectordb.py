from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextFileVectorDB:
    def __init__(
        self,
        vdb_dir: str | Path,
        source_path: str | Path | None = None,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
    ) -> None:
        self.source_path = Path(source_path) if source_path is not None else None
        self.store_dir = Path(vdb_dir) / "vectordb"
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def load(self) -> FAISS:
        if not self._has_saved_index():
            raise FileNotFoundError(
                f"Vector DB not found in {self.store_dir}. "
                "Create it first or provide source_path to build a new vector database."
            )
        return FAISS.load_local(
            str(self.store_dir),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def create(self) -> FAISS:
        vector_store = FAISS.from_documents(self.build_documents(), self.embeddings)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(self.store_dir))
        return vector_store

    def load_or_create(self, overwrite: bool = False) -> FAISS:
        if not overwrite and self._has_saved_index():
            vector_store = self.load()
            if self._is_current_index(vector_store):
                return vector_store

        return self.create()

    def build_documents(self) -> list[Document]:
        if self.source_path is None:
            raise ValueError("source_path is required to build a new vector database")
        text = self.source_path.read_text(encoding="utf-8").strip()
        if not text:
            return []
        return self.split_text(text, {})

    def split_text(
        self,
        text: str,
        metadata: dict[str, str],
    ) -> list[Document]:
        return [
            Document(
                page_content=chunk_text,
                metadata={**metadata, "chunk_index": str(chunk_index)},
            )
            for chunk_index, chunk_text in enumerate(
                self.text_splitter.split_text(text)
            )
        ]

    @staticmethod
    def _is_current_index(vector_store: FAISS) -> bool:
        for document in vector_store.docstore._dict.values():
            if "chunk_index" in document.metadata:
                return True
        return False

    def _has_saved_index(self) -> bool:
        return (self.store_dir / "index.faiss").exists() and (
            self.store_dir / "index.pkl"
        ).exists()


class StudentRecordVectorDB(TextFileVectorDB):
    def __init__(
        self,
        vdb_dir: str | Path,
        student_record_yaml_path: str | Path | None = None,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        metadata_header: bool = False,
    ) -> None:
        super().__init__(
            vdb_dir=vdb_dir,
            source_path=student_record_yaml_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.metadata_header = metadata_header

    def build_documents(self) -> list[Document]:
        if self.source_path is None:
            raise ValueError(
                "student_record_yaml_path is required to build a new vector database"
            )
        with self.source_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}

        def walk(value: Any, path: tuple[str, ...] = ()) -> list[Document]:
            if isinstance(value, dict):
                if "내용" in value:
                    return self._build_record_documents(
                        text=str(value.get("내용", "")).strip(),
                        path=path,
                        subject=str(value.get("과목", "")).strip(),
                    )

                documents: list[Document] = []
                for key, item in value.items():
                    documents.extend(walk(item, path + (str(key),)))
                return documents

            if isinstance(value, list):
                documents: list[Document] = []
                for item in value:
                    documents.extend(walk(item, path))
                return documents

            text = str(value).strip()
            if not text:
                return []
            return self._build_record_documents(text=text, path=path)

        return walk(data)

    def _build_record_documents(
        self,
        text: str,
        path: tuple[str, ...],
        subject: str = "",
    ) -> list[Document]:
        if not text:
            return []

        metadata = self._build_metadata(path, subject=subject)
        documents = self.split_text(text, metadata)
        if not self.metadata_header:
            return documents

        prefix = self._format_metadata_prefix(metadata)
        return [
            Document(
                page_content=f"{prefix}\n\n{document.page_content}",
                metadata=document.metadata,
            )
            for document in documents
        ]

    @staticmethod
    def _build_metadata(
        path: tuple[str, ...],
        subject: str = "",
    ) -> dict[str, str]:
        metadata: dict[str, str] = {"path": " > ".join(path), "subject": subject}
        semantic_path = path[1:] if path and path[0] == "학교생활기록부(기본)" else path

        metadata["section"] = semantic_path[0] if len(semantic_path) > 0 else ""
        metadata["grade"] = semantic_path[1] if len(semantic_path) > 1 else ""
        metadata["subsection"] = semantic_path[2] if len(semantic_path) > 2 else ""

        level_values = [
            metadata["section"],
            metadata["grade"],
            metadata["subsection"],
            subject,
        ]
        for index, value in enumerate(level_values, start=1):
            if value:
                metadata[f"level{index}"] = value

        return metadata

    @staticmethod
    def _format_metadata_prefix(metadata: dict[str, str]) -> str:
        return "\n".join(
            [
                f"section: {metadata.get('section', '')}",
                f"grade: {metadata.get('grade', '')}",
                f"subsection: {metadata.get('subsection', '')}",
                f"subject: {metadata.get('subject', '')}",
            ]
        )

    def _format_page_content(self, chunk_text: str, metadata: dict[str, str]) -> str:
        if not self.metadata_header:
            return chunk_text
        return f"{self._format_metadata_prefix(metadata)}\n\n{chunk_text}"

    @staticmethod
    def _is_current_index(vector_store: FAISS) -> bool:
        for document in vector_store.docstore._dict.values():
            if (
                document.metadata.get("level1")
                and "chunk_index" in document.metadata
                and document.page_content.startswith("section: ")
            ):
                return True
        return False
