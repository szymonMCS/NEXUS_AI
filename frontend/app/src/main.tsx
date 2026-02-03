import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { ClerkProvider } from '@clerk/clerk-react'
import './index.css'
import App from './App.tsx'

// Get Clerk publishable key from environment
const PUBLISHABLE_KEY = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY

// Check if key is valid (not placeholder)
const isValidClerkKey = PUBLISHABLE_KEY && 
  PUBLISHABLE_KEY.startsWith('pk_') && 
  !PUBLISHABLE_KEY.includes('your_publishable_key_here') &&
  PUBLISHABLE_KEY.length > 20

// Create the app with or without Clerk
const AppWithOptionalAuth = () => {
  // If no valid Clerk key, just render without auth
  if (!isValidClerkKey) {
    return (
      <BrowserRouter>
        <App />
      </BrowserRouter>
    )
  }

  // With Clerk - only protects specific routes if needed
  return (
    <ClerkProvider 
      publishableKey={PUBLISHABLE_KEY}
      appearance={{
        baseTheme: undefined,
        variables: {
          colorPrimary: '#00d4ff',
          colorBackground: '#0a0e1a',
          colorText: '#ffffff',
          colorInputBackground: '#141b2d',
          colorInputText: '#ffffff',
        },
        elements: {
          card: 'bg-[#141b2d] border border-white/10',
          headerTitle: 'text-white',
          headerSubtitle: 'text-gray-400',
          formButtonPrimary: 'bg-gradient-to-r from-[#00d4ff] to-[#00ff88] text-black font-bold',
          footerActionLink: 'text-[#00d4ff]',
        }
      }}
    >
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </ClerkProvider>
  )
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <AppWithOptionalAuth />
  </StrictMode>,
)
