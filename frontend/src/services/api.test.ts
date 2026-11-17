import {afterEach, expect, test, vi} from 'vitest'
import {api} from './api'

afterEach(() => vi.unstubAllGlobals())
test('prediction sends a multipart image under the server field name', async () => {
  const fetcher = vi.fn(async (_path: string, options: RequestInit) => {
    expect(options.method).toEqual('POST')
    expect((options.body as FormData).get('image')).toBeInstanceOf(File)
    return Response.json({model:'mobilenetv2'})
  })
  vi.stubGlobal('fetch', fetcher)
  await expect(api.predict('mobilenetv2',new File(['x'],'road.jpg'))).resolves.toMatchObject({model:'mobilenetv2'})
  expect(fetcher.mock.calls[0][0]).toMatch(/\/predict\/mobilenetv2$/)
})
test('server errors and cancellation propagate', async () => {
  vi.stubGlobal('fetch', vi.fn(async () => Response.json({detail:'Benchmark unavailable.'},{status:503})))
  await expect(api.benchmarks()).rejects.toThrow('Benchmark unavailable.')
  const controller = new AbortController()
  controller.abort()
  vi.stubGlobal('fetch', vi.fn(async (_path, options: RequestInit) => {
    options.signal?.throwIfAborted()
    return Response.json([])
  }))
  await expect(api.models(controller.signal)).rejects.toMatchObject({name:'AbortError'})
})
