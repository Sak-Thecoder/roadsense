import {createRoot} from 'react-dom/client'
import {BrowserRouter} from 'react-router-dom'
import App from './App'
import {ScanProvider} from './state'
import './styles/app.css'

createRoot(document.getElementById('root')!).render(<BrowserRouter><ScanProvider><App/></ScanProvider></BrowserRouter>)
