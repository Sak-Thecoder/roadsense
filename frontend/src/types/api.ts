export const classes = ['Good', 'Moderate', 'Poor'] as const
export type Quality = typeof classes[number]
export type Model = 'mobilenetv2' | 'resnet50' | 'efficientnetb0'
export interface ModelInfo {model: Model; name: string; available: boolean; size_mb: number | null; version: string | null; use_scope?: string | null}
export interface Prediction {model: Model; prediction: Quality; confidence: number; scores: Record<Quality, number>; inference_ms: number}
export interface Comparison {predictions: Prediction[]; errors: Partial<Record<Model, string>>}
export interface Benchmark {
  model: Model; model_version: string; classes: Quality[]; accuracy: number; macro_precision: number
  macro_recall: number; macro_f1: number; per_class: Record<Quality, Record<string, number>>
  confusion_matrix: number[][]; size_mb: number; parameters: number; latency_ms: number; hardware: string
  use_scope?: string | null; latency_method: string; dataset_size: number; dataset_sha256: string; split: 'test' | 'cambodia_holdout'
}
export interface Health {status: string; available_models: Model[]; classes: Quality[]; max_upload_bytes: number}
export const names: Record<Model, string> = {mobilenetv2:'MobileNetV2', resnet50:'ResNet50', efficientnetb0:'EfficientNetB0'}
