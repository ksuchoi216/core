from core.aws.s3_artifacts import S3Artifact
from core.aws.s3_ordered_artifacts import OrderedArtifact, S3OrderedArtifact

# Backward compatibility aliases
S3Manager = S3Artifact
S3ArtifactManager = S3Artifact
OrderedArtifactManager = OrderedArtifact
S3OrderedArtifactManager = S3OrderedArtifact


__all__ = [
    "OrderedArtifact",
    "S3Artifact",
    "S3OrderedArtifact",
    "OrderedArtifactManager",
    "S3ArtifactManager",
    "S3Manager",
    "S3OrderedArtifactManager",
]
