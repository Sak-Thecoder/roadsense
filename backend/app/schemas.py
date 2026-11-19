"""Response shapes of the API, defined with Pydantic.

Pydantic validates every response automatically: e.g. a confidence must be
between 0 and 1, and a class name must be one of Good / Moderate / Poor. FastAPI
also uses these classes to generate the interactive docs at /docs.
"""
from typing import Annotated, Literal

from pydantic import BaseModel, Field

# Literal = only these exact strings are allowed.
ModelName = Literal["mobilenetv2", "resnet50", "efficientnetb0"]
Quality = Literal["Good", "Moderate", "Poor"]
# A float that must lie between 0 and 1 (ge = greater or equal, le = less or equal).
Score = Annotated[float, Field(ge=0, le=1)]


class ModelInfo(BaseModel):
    """One entry of GET /models."""
    model: ModelName
    name: str
    available: bool
    size_mb: float | None = None
    version: str | None = None
    use_scope: str | None = None


class Prediction(BaseModel):
    """Result of classifying one photo with one model."""
    model: ModelName
    prediction: Quality
    confidence: float = Field(ge=0, le=1)  # probability of the predicted class
    scores: dict[Quality, Score]           # probability of every class
    inference_ms: float = Field(ge=0)


class Comparison(BaseModel):
    """Result of POST /compare: successful predictions plus per-model error messages."""
    predictions: list[Prediction]
    errors: dict[ModelName, str]


class PerClassMetrics(BaseModel):
    precision: Score  # of the photos predicted as this class, how many really were
    recall: Score     # of the photos that really are this class, how many were found
    f1: Score         # harmonic mean of precision and recall
    support: int = Field(ge=0)  # number of test photos of this class


class Benchmark(BaseModel):
    """Test-set evaluation saved by training/evaluate.py and shown on the Benchmark page."""
    evaluation_schema: Literal[1]
    model: ModelName
    model_sha256: str      # ties the result to one exact model file
    model_version: str
    classes: list[Quality] = Field(min_length=3, max_length=3)
    accuracy: float = Field(ge=0, le=1)
    # "Macro" = plain average over the three classes, so the small Moderate and
    # Poor classes count as much as the large Good class.
    macro_precision: float = Field(ge=0, le=1)
    macro_recall: float = Field(ge=0, le=1)
    macro_f1: float = Field(ge=0, le=1)
    per_class: dict[Quality, PerClassMetrics]
    # Rows = true class, columns = predicted class (Good, Moderate, Poor).
    confusion_matrix: list[list[int]] = Field(min_length=3, max_length=3)
    size_mb: float = Field(gt=0)
    parameters: int = Field(gt=0)
    latency_ms: float = Field(ge=0)
    latency_method: str
    hardware: str
    dataset_size: int = Field(gt=0)
    dataset_sha256: str
    split: Literal["test", "cambodia_holdout"]
    use_scope: str | None = None
