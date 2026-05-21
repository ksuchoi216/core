from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any, Generic, TypeVar

from core.aws.s3_file_transfer import (
    check_file_in_s3,
    download_from_s3,
    upload_to_s3,
)
from core.aws.s3_url import S3Location
from core.support.file import load_file, save_file
import inspect

ArtifactT = TypeVar("ArtifactT")


class S3Artifact(Generic[ArtifactT]):
    """Fundamental S3 helpers and managers representing an S3-backed artifact."""

    def __init__(
        self,
        s3_url: str,
        artifact_foldername: str = "artifacts",
        download_dir: str = "./temp",
    ) -> None:
        self._s3 = S3Location(
            s3_url=s3_url,
            artifact_foldername=artifact_foldername,
            download_dir=download_dir,
        )
        self.__post_init__()

    def __post_init__(self):
        missing = [
            name
            for name in ("artifact_filenames", "required_artifacts")
            if not hasattr(self, name)
        ]
        if missing:
            raise AttributeError(f"Missing required attribute: {', '.join(missing)}")

    @property
    def bucket(self) -> str:
        return self._s3.bucket

    @property
    def local_artifact_dir(self) -> Path:
        return (
            Path(self._s3.download_dir)
            / self._s3.artifact_foldername
            / self._s3.artifact_parent_folder
        )

    @property
    def raw_document_filename(self) -> str:
        return self._s3.raw_doc_filename

    @property
    def raw_document_local_path(self) -> Path:
        return self.local_artifact_dir / self.raw_document_filename

    @property
    def raw_document_s3_key(self) -> str:
        return self._s3.raw_doc_s3_key

    def artifact_filename(self, artifact: ArtifactT) -> str:
        return str(getattr(artifact, "value", artifact))

    def artifact_local_path(self, artifact: ArtifactT) -> Path:
        return self.local_artifact_dir / self.artifact_filename(artifact)

    def artifact_s3_key(self, artifact: ArtifactT) -> str:
        return self._s3.artifact_s3_key(self.artifact_filename(artifact))

    def download_raw_document(self) -> Path:
        return download_from_s3(
            bucket=self.bucket,
            prefix=self.raw_document_s3_key,
            local_path=self.raw_document_local_path,
        )

    def _artifact_exists_in_s3(self, artifact: ArtifactT) -> bool:
        return check_file_in_s3(self.bucket, self.artifact_s3_key(artifact))

    def _download_artifact_if_required(self, artifact: ArtifactT) -> None:
        if not self._artifact_exists_in_s3(artifact):
            return

        local_path = self.artifact_local_path(artifact)
        if local_path.exists():
            return

        download_from_s3(
            bucket=self.bucket,
            prefix=self.artifact_s3_key(artifact),
            local_path=local_path,
        )

    def create_artifact(
        self, artifact_name: ArtifactT, function: Callable[..., Any], *args, **kwargs
    ) -> Any:
        """Create an output artifact from its required artifacts and upload it."""
        is_test = kwargs.get("is_test", False)
        loaded_artifacts = []
        for required_artifact in self.required_artifacts.get(artifact_name, []):
            self._download_artifact_if_required(required_artifact)
            required_path = self.artifact_local_path(required_artifact)

            if not required_path.exists():
                raise FileNotFoundError(
                    f"Required artifact not found locally or in S3: "
                    f"{required_path} / "
                    f"s3://{self.bucket}/{self.artifact_s3_key(required_artifact)}"
                )

            loaded_artifacts.append(load_file(required_path))

        sig = inspect.signature(function)
        if "required_artifacts" in sig.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        ):
            kwargs["required_artifacts"] = {
                required_artifact: content
                for required_artifact, content in zip(
                    self.required_artifacts.get(artifact_name, []), loaded_artifacts
                )
            }

        result = function(*loaded_artifacts, *args, **kwargs)
        local_path = self.artifact_local_path(artifact_name)
        s3_key = self.artifact_s3_key(artifact_name)

        save_file(result, local_path)
        if is_test:
            return result

        upload_to_s3(
            bucket=self.bucket,
            prefix=s3_key,
            local_path=local_path,
        )
        return result
