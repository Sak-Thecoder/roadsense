import type {Benchmark, Comparison, Health, Model, ModelInfo, Prediction} from '../types/api'

const base = (import.meta.env.VITE_API_BASE_URL || '/api').replace(/\/$/, '')

async function request<T>(path: string, signal?: AbortSignal, file?: File): Promise<T> {
  const controller = new AbortController()
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
      const error = await response.json().catch(() => null)
      throw new Error(typeof error?.detail === 'string' ? error.detail : `Request failed (${response.status}).`)
    }
    return await response.json() as T
  } catch (error) {
    if (timedOut) throw new Error('The server took too long to respond. Please try again.')
    if (error instanceof TypeError) throw new Error('Could not connect to the RoadSense API. Check your connection and try again.')
    throw error
  } finally {
    clearTimeout(timer)
    signal?.removeEventListener('abort', abort)
  }
}

export const api = {
  health: (signal?: AbortSignal) => request<Health>('/health', signal),
  models: (signal?: AbortSignal) => request<ModelInfo[]>('/models', signal),
  predict: (model: Model, file: File, signal?: AbortSignal) => request<Prediction>(`/predict/${model}`, signal, file),
  compare: (file: File, signal?: AbortSignal) => request<Comparison>('/compare', signal, file),
  benchmarks: (signal?: AbortSignal) => request<Benchmark[]>('/benchmark', signal),
}
