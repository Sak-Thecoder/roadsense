import type {Benchmark, Comparison, Health, Model, ModelInfo, Prediction} from '../types/api'

// API address. In development Vite forwards "/api/..." to the FastAPI server (see vite.config.ts).
// The regex removes a trailing "/" so paths can be joined safely.
const base = (import.meta.env.VITE_API_BASE_URL || '/api').replace(/\/$/, '')

/** Error with the HTTP status code, so pages can react differently (e.g. 422 = not a road photo). */
export class ApiError extends Error {
  constructor(public readonly status: number, message: string) {
    super(message)
  }
}

/**
 * Shared helper for every API call.
 * - GET when no file is given, otherwise POST the file as multipart form field "image".
 * - Cancels after 60 seconds, or when the caller's `signal` is aborted.
 * - Turns server and network failures into readable error messages.
 */
async function request<T>(path: string, signal?: AbortSignal, file?: File): Promise<T> {
  const controller = new AbortController()
  // Forward the caller's cancel signal to our own controller (which also handles the timeout).
  const abort = () => controller.abort(signal?.reason)
  signal?.addEventListener('abort', abort, {once:true})
  if (signal?.aborted) abort()
  let timedOut = false
  const timer = setTimeout(() => {timedOut = true; controller.abort()}, 60_000)
  try {
    const body = file ? new FormData() : undefined
    if (body && file) body.append('image', file)
    const response = await fetch(base + path, {method: file ? 'POST' : 'GET', body, signal: controller.signal})
    if (!response.ok) {
      // FastAPI returns errors as {"detail": "message"}; show that message if there is one.
      const error = await response.json().catch(() => null)
      throw new ApiError(response.status, typeof error?.detail === 'string' ? error.detail : `Request failed (${response.status}).`)
    }
    return await response.json() as T
  } catch (error) {
    if (timedOut) throw new Error('The server took too long to respond. Please try again.')
    // fetch throws a TypeError when the server can't be reached at all.
    if (error instanceof TypeError) throw new Error('Could not connect to the RoadSense API. Check your connection and try again.')
    throw error
  } finally {
    clearTimeout(timer)
    signal?.removeEventListener('abort', abort)
  }
}

// One function per backend endpoint.
export const api = {
  health: (signal?: AbortSignal) => request<Health>('/health', signal),
  models: (signal?: AbortSignal) => request<ModelInfo[]>('/models', signal),
  predict: (model: Model, file: File, signal?: AbortSignal) => request<Prediction>(`/predict/${model}`, signal, file),
  compare: (file: File, signal?: AbortSignal) => request<Comparison>('/compare', signal, file),
  benchmarks: (signal?: AbortSignal) => request<Benchmark[]>('/benchmark', signal),
}
