import {useEffect, useState} from 'react'
import {ChartNoAxesColumn} from 'lucide-react'
import {Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis} from 'recharts'
import {Empty, ErrorNotice, Loading, PageHeading, percent} from '../components/UI'
import {api} from '../services/api'
import {names, type Benchmark as Metrics, type Model} from '../types/api'

export default function Benchmark() {
  const [data, setData] = useState<Metrics[]>()
  const [error, setError] = useState('')
  const [selected, select] = useState<Model>('mobilenetv2')
  const [attempt, retry] = useState(0)
  useEffect(() => {
    const controller = new AbortController()
    setError('');setData(undefined)
    api.benchmarks(controller.signal).then(setData)
      .catch(e => {if (!controller.signal.aborted) setError(e.message)})
    return () => controller.abort()
  }, [attempt])
  const model = data?.find(m => m.model === selected) || data?.[0]
  const comparable = data?.filter(m => model && m.dataset_sha256 === model.dataset_sha256 && m.split === model.split && m.use_scope === model.use_scope)
  const matchingLatency = comparable?.filter(m => m.hardware === model?.hardware && m.latency_method === model?.latency_method)
  return <>
    <PageHeading title="Benchmark">Road quality performance measured on held-out data.</PageHeading>
    <div className="model-tabs" aria-label="Benchmark model">{(Object.keys(names) as Model[]).map(name =>
      <button key={name} className={model?.model === name ? 'selected' : ''} aria-pressed={model?.model === name}
        disabled={!data?.some(m => m.model === name)} onClick={() => select(name)}>{names[name]}</button>)}</div>
    {error ? <><ErrorNotice>{error}</ErrorNotice><button className="button secondary" onClick={() => retry(n=>n+1)}>Retry</button></>
      : !data ? <Loading>Loading saved evaluation results…</Loading>
      : !model ? <div className="card"><Empty icon={ChartNoAxesColumn} title="Benchmark not available yet">Measured scores will appear here after a trained model is evaluated on held-out road images.</Empty></div>
      : <>
        {model.use_scope === 'cambodian_capture_date_pilot' && <p className="notice">
          Local pilot evaluated on {model.dataset_size} photos from one separate capture date. Geographic independence is unverified.
        </p>}
        <div className="metrics">{[['Accuracy',model.accuracy],['Macro precision',model.macro_precision],['Macro recall',model.macro_recall],['Macro F1',model.macro_f1]].map(([title,value]) =>
          <section className="card metric" key={title}><span>{title}</span><strong>{percent(Number(value))}</strong></section>)}</div>
        <div className="benchmark-grid">
          <section className="card"><h2>Confusion matrix</h2><p className="subtle">Rows are actual classes. Columns are predicted classes.</p>
            <table className="matrix"><thead><tr><th scope="col">Actual</th>{model.classes.map(c=><th scope="col" key={c}>{c}</th>)}</tr></thead>
              <tbody>{model.confusion_matrix.map((row,i)=><tr key={i}><th scope="row">{model.classes[i]}</th>{row.map((n,j)=><td className={i===j ? 'diagonal' : ''} key={j}>{n}</td>)}</tr>)}</tbody></table>
          </section>
          <section className="card"><h2>Accuracy by model</h2>
            <div className="chart"><ResponsiveContainer width="100%" height="100%"><BarChart accessibilityLayer layout="vertical" data={comparable?.map(m => ({name:names[m.model],accuracy:Number((m.accuracy*100).toFixed(1))}))} margin={{left:15,right:12,top:6,bottom:6}}>
              <XAxis type="number" domain={[0,100]} unit="%"/><YAxis type="category" dataKey="name" width={120}/><Tooltip formatter={value=>`${value}%`}/><Bar dataKey="accuracy" name="Accuracy" fill="#245ced" radius={[0,5,5,0]}/>
            </BarChart></ResponsiveContainer></div>
          </section>
        </div>
        <section className="card table-wrap"><h2>Speed and size</h2><table><thead><tr><th>Model</th><th>Latency</th><th>Size</th><th>Parameters</th></tr></thead><tbody>
          {matchingLatency?.map(m=><tr key={m.model}><th scope="row">{names[m.model]}</th><td>{m.latency_ms.toFixed(1)} ms</td><td>{m.size_mb.toFixed(2)} MB</td><td>{m.parameters.toLocaleString()}</td></tr>)}
        </tbody></table><p className="subtle">{model.dataset_size.toLocaleString()} images · {model.split.replaceAll('_',' ')} · version {model.model_version}</p>
        <p className="subtle">{model.hardware}. {model.latency_method}</p>
        {comparable?.length !== matchingLatency?.length && <p className="subtle">Latency comparison includes evaluations with matching hardware and timing methods.</p>}
        </section>
        <details className="card">
          <summary>Per-class evaluation</summary>
          <table><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Images</th></tr></thead><tbody>
            {model.classes.map(label => {
              const metrics = model.per_class[label]
              return <tr key={label}><th scope="row">{label}</th><td>{percent(metrics.precision)}</td>
                <td>{percent(metrics.recall)}</td><td>{percent(metrics.f1)}</td><td>{metrics.support}</td></tr>
            })}
          </tbody></table>
        </details>
      </>}
  </>
}
