import {Camera, Check, ChevronRight, CircleAlert, LoaderCircle, TriangleAlert, X} from 'lucide-react'
import {Link} from 'react-router-dom'
import type {LucideIcon} from 'lucide-react'
import type {ReactNode} from 'react'
import {classes, names, type Prediction, type Quality} from '../types/api'

export const percent = (value: number) => `${Math.round(value * 100)}%`
const symbols: Record<Quality, LucideIcon> = {Good:Check, Moderate:TriangleAlert, Poor:X}

export function QualityTag({quality}: {quality: Quality}) {
  const Icon = symbols[quality]
  return <span className={`quality ${quality.toLowerCase()}`}><Icon size={15} aria-hidden/>{quality}</span>
}
export function Scores({prediction}: {prediction: Prediction}) {
  return <section className="card"><h2>Class scores</h2>
    <div className="score-list">{classes.map(label => <div key={label}>
      <div className="row spread"><span>{label}</span><strong>{percent(prediction.scores[label])}</strong></div>
      <div className="track"><div className={`bar ${label.toLowerCase()}`} style={{width:percent(prediction.scores[label])}}/></div>
    </div>)}</div>
  </section>
}
export function Outcome({prediction}: {prediction: Prediction}) {
  return <section className={`outcome ${prediction.prediction.toLowerCase()}`}>
    <div><QualityTag quality={prediction.prediction}/><h2>{prediction.prediction} road quality</h2>
      <p>{names[prediction.model]} · {prediction.inference_ms.toFixed(1)} ms server inference</p></div>
    <div className="confidence"><strong>{percent(prediction.confidence)}</strong><span>confidence</span></div>
  </section>
}
export function ErrorNotice({children}: {children: ReactNode}) {
  return <p role="alert" className="notice error"><CircleAlert size={18} aria-hidden/>{children}</p>
}
export function Loading({children}: {children: ReactNode}) {
  return <p className="notice" role="status"><LoaderCircle className="spin" size={18} aria-hidden/>{children}</p>
}
export function Empty({icon: Icon = Camera, title, children}: {icon?: LucideIcon; title: string; children: ReactNode}) {
  return <div className="empty"><span className="empty-icon"><Icon aria-hidden size={25}/></span>
    <h2>{title}</h2><p>{children}</p></div>
}
export function PageHeading({title, children}: {title:string; children:ReactNode}) {
  return <header className="page-title"><h1>{title}</h1><p>{children}</p></header>
}
export function ActionLink({to, children}: {to:string; children:ReactNode}) {
  return <Link className="action-link" to={to}>{children}<ChevronRight size={16} aria-hidden/></Link>
}
