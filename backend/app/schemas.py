from typing import Annotated, Literal

from pydantic import BaseModel, Field

ModelName = Literal["mobilenetv2", "resnet50", "efficientnetb0"]
Quality = Literal["Good", "Moderate", "Poor"]
Score = Annotated[float, Field(ge=0, le=1)]


class ModelInfo(BaseModel):
    model: ModelName
    name: str
    available: bool
    size_mb: float | None = None
    version: str | None = None
    use_scope: str | None = None


class Prediction(BaseModel):
    model: ModelName
    prediction: Quality
    confidence: float = Field(ge=0, le=1)
    scores: dict[Quality, Score]
    inference_ms: float = Field(ge=0)


class Comparison(BaseModel):
    predictions: list[Prediction]
    errors: dict[ModelName, str]


class PerClassMetrics(BaseModel):
    precision: Score
    recall: Score
    f1: Score
    support: int = Field(ge=0)


class Benchmark(BaseModel):
    evaluation_schema: Literal[1]
    model: ModelName
    model_sha256: str
    model_version: str
    classes: list[Quality] = Field(min_length=3, max_length=3)
    accuracy: float = Field(ge=0, le=1)
    macro_precision: float = Field(ge=0, le=1)
    macro_recall: float = Field(ge=0, le=1)
    macro_f1: float = Field(ge=0, le=1)
    per_class: dict[Quality, PerClassMetrics]
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
