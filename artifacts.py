from core.aws.s3_artifact import Artifact, OrderedArtifact

# Backward compatibility aliases
S3Artifact = Artifact
S3Manager = Artifact
S3ArtifactManager = Artifact

__all__ = [
    "Artifact",
    "OrderedArtifact",
    "S3Artifact",
    "S3ArtifactManager",
    "S3Manager",
]
