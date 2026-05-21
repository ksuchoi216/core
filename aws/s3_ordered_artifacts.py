from collections.abc import Callable, Mapping, Sequence
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

from .s3_artifact import S3Artifact

ArtifactT = TypeVar("ArtifactT")


class OrderedArtifact(Generic[ArtifactT]):
    """Resolve artifact processing order and dependency downloads."""

    def __init__(
        self,
        artifacts: Sequence[ArtifactT],
        required_artifacts: Mapping[ArtifactT, Sequence[ArtifactT]],
        artifact_exists: Callable[[ArtifactT], bool],
        download_artifact_if_required: Callable[[ArtifactT], None],
    ) -> None:
        self.artifacts = list(artifacts)
        self.required_artifacts = required_artifacts
        self.artifact_exists = artifact_exists
        self.download_artifact_if_required = download_artifact_if_required

    def initialize_artifact_list(self) -> list[ArtifactT]:
        """Return artifacts from the first missing artifact onward."""
        first_missing_index = None

        for index, artifact in enumerate(self.artifacts):
            if not self.artifact_exists(artifact):
                first_missing_index = index
                break

        if first_missing_index is None:
            return []

        first_missing_artifact = self.artifacts[first_missing_index]
        self.download_required_artifacts(first_missing_artifact)

        return self.artifacts[first_missing_index:]

    def download_required_artifacts(self, artifact: ArtifactT) -> None:
        for required_artifact in self.required_artifacts.get(artifact, []):
            self.download_artifact_if_required(required_artifact)


class S3OrderedArtifact(S3Artifact[ArtifactT]):
    """
    Reusable artifact backed by one source S3 object with dependencies.

    Domain-specific documents should provide artifact enums/order/dependencies.
    This class owns S3 paths, local paths, raw-document handling, and artifact
    processing list initialization.
    """

    artifact_filenames: Sequence[ArtifactT] | type[Any] | None = None
    required_artifacts: Mapping[ArtifactT, Sequence[ArtifactT]] | None = None

    def __init__(
        self,
        s3_url: str,
        artifact_filenames: Sequence[ArtifactT] | None = None,
        required_artifacts: Mapping[ArtifactT, Sequence[ArtifactT]] | None = None,
        artifact_foldername: str = "artifacts",
        download_dir: str = "./temp",
    ) -> None:
        super().__init__(
            s3_url=s3_url,
            artifact_foldername=artifact_foldername,
            download_dir=download_dir,
        )

        self.all_artifacts: list[ArtifactT] = []
        self.artifacts_to_be_processed: list[ArtifactT] = []
        self.artifact_list: list[ArtifactT] = self.artifacts_to_be_processed

        if artifact_filenames is not None and required_artifacts is not None:
            self.configure_artifacts(
                artifact_filenames=artifact_filenames,
                required_artifacts=required_artifacts,
            )

    def setup_artifacts(
        self,
        artifact_filenames: Sequence[ArtifactT] | type[Any] | None = None,
        required_artifacts: Mapping[ArtifactT, Sequence[ArtifactT]] | None = None,
    ) -> list[ArtifactT]:
        artifact_filenames = (
            artifact_filenames
            if artifact_filenames is not None
            else self.artifact_filenames
        )
        required_artifacts = (
            required_artifacts
            if required_artifacts is not None
            else self.required_artifacts
        )

        if artifact_filenames is None or required_artifacts is None:
            raise ValueError("artifact_filenames and required_artifacts are required")

        return self.configure_artifacts(
            artifact_filenames=list(artifact_filenames),
            required_artifacts=required_artifacts,
        )

    def configure_artifacts(
        self,
        artifact_filenames: Sequence[ArtifactT],
        required_artifacts: Mapping[ArtifactT, Sequence[ArtifactT]],
    ) -> list[ArtifactT]:
        self.all_artifacts = list(artifact_filenames)
        self._artifact_manager = OrderedArtifact(
            artifacts=self.all_artifacts,
            required_artifacts=required_artifacts,
            artifact_exists=self._artifact_exists_in_s3,
            download_artifact_if_required=self._download_artifact_if_required,
        )
        self.artifacts_to_be_processed = (
            self._artifact_manager.initialize_artifact_list()
        )
        self.artifact_list = self.artifacts_to_be_processed
        return self.artifacts_to_be_processed

    def download_required_artifacts(self, artifact: ArtifactT) -> None:
        self._artifact_manager.download_required_artifacts(artifact)

    def run_artifact(
        self, function: Callable[[ArtifactT], Any]
    ) -> dict[ArtifactT, Any]:
        results = {}

        for artifact in self.artifacts_to_be_processed:
            self.download_required_artifacts(artifact)

            local_path = self.artifact_local_path(artifact)
            s3_key = self.artifact_s3_key(artifact)
            result = function(artifact)
            save_file(result, local_path)
            upload_to_s3(
                bucket=self.bucket,
                prefix=s3_key,
                local_path=local_path,
            )
            results[artifact] = result

        return results
