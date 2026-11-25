// Welcome page entry point.
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import Welcome from './welcome/Welcome.jsx'
import './styles/tokens.css'
import './styles/welcome.css'

createRoot(document.getElementById('root')).render(<StrictMode><Welcome /></StrictMode>)