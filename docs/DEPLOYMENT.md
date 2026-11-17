# Deploy RoadSense

Build and test the project locally first. Provision the backend with space for TensorFlow and all enabled artifact files. Start with a single process and a one-model cache, then measure memory, response time, and cold-start behavior on the selected host.

## Vercel frontend

1. Import the repository and select `frontend` as the project root. The checked-in Vercel configuration runs `npm ci` and `npm run build`, serves `dist`, and supports direct links to React Router routes.
2. Set `VITE_API_BASE_URL` to the HTTPS origin of the FastAPI service before building. Vite embeds this variable in the browser bundle.
3. Add the resulting Vercel origin to `ROADSENSE_CORS_ORIGINS` on the backend. Use exact origins for any approved preview deployments.
4. Open the hosted app over HTTPS and verify camera availability, upload, and API responses.

## FastAPI hosting

Use the backend Dockerfile with build context `backend`, or choose a Python 3.11 runtime, install `backend/requirements.txt`, and use this command from the repository root.

```sh
uvicorn backend.app.main:app --host 0.0.0.0 --port "$PORT" --workers 1
```

Set the host's health check to `/health`. Configure the variables in the root environment example and mount real `.keras` files and their versioned `.json` sidecars under `ROADSENSE_MODELS_DIR`. Mount evaluation outputs under `ROADSENSE_RESULTS_DIR`. A health response can succeed with zero installed models, which is useful for diagnosing storage and artifact configuration. The application uses HTTPS at the hosting proxy or load balancer. Match the proxy body limit to the API upload limit plus multipart framing.

For a local container run:

```sh
docker compose up --build
```

Open `http://localhost:8080`. The frontend container forwards `/api` to FastAPI. The backend mounts the artifact directories read-only. OpenAPI is at `http://localhost:8000/docs`.

The benchmark compares only results generated from the same manifest and partition. Run all model latency measurements on the same hardware with identical measurement settings before comparing timings. Server latency excludes image decoding, network time, and initial model loading.
