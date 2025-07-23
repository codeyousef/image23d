import { Routes, Route, Navigate } from 'react-router-dom'
import { Toaster } from '@/components/ui/toaster'
import { ThemeProvider } from '@/components/theme-provider'
import Layout from '@/components/Layout'
import { useAuthStore } from '@/store/authStore'

// Pages
import LoginPage from '@/pages/Login'
import RegisterPage from '@/pages/Register'
import CreatePage from '@/pages/Create'
import LibraryPage from '@/pages/Library'
import ModelsPage from '@/pages/Models'
import BillingPage from '@/pages/Billing'
import SettingsPage from '@/pages/Settings'

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const isAuthenticated = useAuthStore((state) => state.isAuthenticated)
  return isAuthenticated ? <>{children}</> : <Navigate to="/login" />
}

function App() {
  return (
    <ThemeProvider defaultTheme="dark" storageKey="neuralforge-theme">
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route path="/register" element={<RegisterPage />} />
        
        <Route
          path="/"
          element={
            <ProtectedRoute>
              <Layout />
            </ProtectedRoute>
          }
        >
          <Route index element={<Navigate to="/create" />} />
          <Route path="create" element={<CreatePage />} />
          <Route path="library" element={<LibraryPage />} />
          <Route path="models" element={<ModelsPage />} />
          <Route path="billing" element={<BillingPage />} />
          <Route path="settings" element={<SettingsPage />} />
        </Route>
      </Routes>
      <Toaster />
    </ThemeProvider>
  )
}