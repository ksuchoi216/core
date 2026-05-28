import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from core.aws.s3_file_transfer import download_from_s3


def extract_bucket_from_s3_url(s3_url: str) -> str:
    parsed = urlparse(s3_url)
    if parsed.scheme == "s3":
        return parsed.netloc
    raise ValueError(f"Invalid S3 URL: {s3_url}")


def refine_prefix(s3_url: str, bucket: str) -> str:
    parsed = urlparse(s3_url)
    if parsed.scheme == "s3":
        return parsed.path.lstrip("/")
    # Fallback logic for non-s3:// strings
    prefix = s3_url.replace("s3://", "")
    return re.sub(rf"^{re.escape(bucket)}/?", "", prefix)


@dataclass
class S3Location:
    """
    Low-level path helper for one source S3 object.

    This class does not know parser concepts such as ArtifactName or DocumentType.
    It only parses the source S3 URL and builds paths from plain filenames.

    Example:
        s3_url = "s3://my-bucket/915/학생 생기부.pdf"
        loc = S3Location(s3_url)
    """

    s3_url: str
    artifact_foldername: str | None = None
    download_dir: str | None = None
    config: Any = None
    bucket: str = field(init=False)
    raw_doc_s3_key: str = field(init=False)
    raw_doc_filename: str = field(init=False)
    artifact_parent_folder: str = field(init=False)

    def __post_init__(self) -> None:
        # Ensure config is not None by loading default if necessary
        if self.config is None:
            try:
                from core.support.config import load_config

                configs_dir = Path(__file__).parent.parent.parent / "configs"
                self.config = load_config(configs_dir)
            except Exception:
                try:
                    from core.support.config import load_general_config

                    config_path = (
                        Path(__file__).parent.parent.parent / "configs" / "general.yaml"
                    )
                    self.config = load_general_config(config_path)
                except Exception:
                    from types import SimpleNamespace

                    self.config = SimpleNamespace()

        # Resolve config-based defaults
        config_artifact_foldername = getattr(self.config, "artifact_foldername", None)
        config_download_dir = getattr(self.config, "download_dir", None)

        # Assign resolved values or fallback to hardcoded defaults
        if self.artifact_foldername is None:
            self.artifact_foldername = config_artifact_foldername or "artifacts"
        if self.download_dir is None:
            self.download_dir = config_download_dir or "./temp"

        self.bucket = extract_bucket_from_s3_url(self.s3_url)
        self.raw_doc_s3_key = refine_prefix(self.s3_url, self.bucket)
        self.raw_doc_filename = Path(self.raw_doc_s3_key).name
        self.artifact_parent_folder = Path(self.raw_doc_s3_key).parent.name

    @property
    def local_raw_doc_path(self) -> Path:
        """Default local path for the downloaded source document."""
        return self.artifact_local_path(self.raw_doc_filename)

    def download_raw_doc(self, local_path: Path | str | None = None) -> Path:
        """Download the raw document from S3. Returns the local path."""
        target_path = (
            Path(local_path) if local_path is not None else self.local_raw_doc_path
        )
        return download_from_s3(
            bucket=self.bucket,
            prefix=self.raw_doc_s3_key,
            local_path=target_path,
        )

    def artifact_local_path(self, filename: str) -> Path:
        """Local path for a plain artifact filename."""
        return (
            Path(self.download_dir)
            / self.artifact_foldername
            / self.artifact_parent_folder
            / filename
        )

    def artifact_s3_key(self, filename: str) -> str:
        """S3 key for a plain artifact filename under the source document."""
        return f"{self.raw_doc_s3_key}/{self.artifact_foldername}/{filename}"
