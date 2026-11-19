import {createRoot} from 'react-dom/client'
import {BrowserRouter} from 'react-router-dom'
import App from './App'
import {ScanProvider} from './state'
import './styles/app.css'

// Entry point. The app is wrapped in:
//  - BrowserRouter: enables page URLs such as /compare and /benchmark
//  - ScanProvider:  shared state (models, chosen photo, scan history) for every page
createRoot(document.getElementById('root')!).render(<BrowserRouter><ScanProvider><App/></ScanProvider></BrowserRouter>)
