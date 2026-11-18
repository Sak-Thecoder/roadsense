import logging
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from .preprocess import CLASSES, MODELS, InvalidImage, decode_image
from .registry import Registry, Unavailable
from .road_photo import RoadPhotoCheckUnavailable, RoadPhotoValidator, model_path
from .schemas import Benchmark, Comparison, ModelInfo, Prediction

BACKEND = Path(__file__).resolve().parents[1]
log = logging.getLogger(__name__)


class UploadLimit:
    """Bound the full multipart body before the framework parses it."""
    def __init__(self, app, limit):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > self.limit:
                return await JSONResponse({"detail": "Image upload is too large."}, status_code=413)(
                    scope, receive, send)
            body.extend(chunk)
            if not message.get("more_body"):
                break
        first = True

        async def replay():
            nonlocal first
            if first:
                first = False
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()
        return await self.app(scope, replay, send)


def create_app(registry=None, max_upload=None, photo_validator=None):
    registry = registry if registry is not None else Registry(
        Path(os.getenv("ROADSENSE_MODELS_DIR", BACKEND / "models")),
        Path(os.getenv("ROADSENSE_RESULTS_DIR", BACKEND / "results")),
        enabled=os.getenv("ROADSENSE_ENABLED_MODELS", ",".join(MODELS)).split(","),
        cache_size=int(os.getenv("ROADSENSE_MODEL_CACHE", "1")),
    )
    max_upload = max_upload or int(os.getenv("ROADSENSE_MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))
    max_pixels = int(os.getenv("ROADSENSE_MAX_IMAGE_PIXELS", "24000000"))
    photo_validator = photo_validator if photo_validator is not None else RoadPhotoValidator(model_path())
    app = FastAPI(title="RoadSense", version="0.1.0", description="Road surface quality classification.")
    app.add_middleware(UploadLimit, limit=max_upload + 64 * 1024)
    app.add_middleware(CORSMiddleware,
        allow_origins=[x.strip() for x in os.getenv(
            "ROADSENSE_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if x.strip()],
        allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

    def check_model(name):
        if name not in MODELS:
            raise HTTPException(404, "Unknown model.")

    def validated_image(data):
        # Each model resizes (and optionally crops) the full-resolution image itself.
        image = decode_image(data, max_pixels)
        photo_validator(image)
        return image

    async def pixels_from(image):
        try:
            data = await image.read(max_upload + 1)
        finally:
            await image.close()
        if len(data) > max_upload:
            raise HTTPException(413, "Image upload is too large.")
        try:
            return await run_in_threadpool(validated_image, data)
        except InvalidImage as exc:
            raise HTTPException(422, str(exc)) from exc
        except RoadPhotoCheckUnavailable as exc:
            log.exception("Road-photo check failed")
            raise HTTPException(503, str(exc)) from exc

    @app.get("/health")
    def health():
        return {"status": "ok", "available_models": [name for name in MODELS if registry.info(name).available],
                "classes": list(CLASSES), "max_upload_bytes": max_upload}

    @app.get("/models", response_model=list[ModelInfo])
    def models():
        return [registry.info(name) for name in MODELS]

    @app.post("/predict/{model_name}", response_model=Prediction)
    async def predict(model_name: str, image: UploadFile = File(...)):
        check_model(model_name)
        pixels = await pixels_from(image)
        try:
            return await run_in_threadpool(registry.predict, model_name, pixels)
        except Unavailable as exc:
            raise HTTPException(503, str(exc)) from exc
        except Exception as exc:
            log.exception("Prediction failed for %s", model_name)
            raise HTTPException(500, "The model could not complete the prediction.") from exc

    @app.post("/compare", response_model=Comparison)
    async def compare(image: UploadFile = File(...)):
        pixels = await pixels_from(image)
        predictions, errors = [], {}
        for name in MODELS:
            try:
                predictions.append(await run_in_threadpool(registry.predict, name, pixels))
            except Unavailable as exc:
                errors[name] = str(exc)
            except Exception:
                log.exception("Comparison failed for %s", name)
                errors[name] = "The model could not complete the prediction."
        return Comparison(predictions=predictions, errors=errors)

    @app.get("/benchmark", response_model=list[Benchmark])
    def benchmarks():
        return [result for name in MODELS if (result := registry.benchmark(name)) is not None]

    @app.get("/benchmark/{model_name}", response_model=Benchmark)
    def benchmark(model_name: str):
        check_model(model_name)
        result = registry.benchmark(model_name)
        if result is None:
            raise HTTPException(404, "Benchmark not available yet.")
        return result

    return app


app = create_app()
