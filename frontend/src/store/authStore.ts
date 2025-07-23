import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { authApi, type User } from '@/api/auth'

interface AuthState {
  user: User | null
  accessToken: string | null
  refreshToken: string | null
  isAuthenticated: boolean
  isLoading: boolean
  
  setUser: (user: User) => void
  setTokens: (accessToken: string, refreshToken: string) => void
  login: (username: string, password: string) => Promise<void>
  register: (email: string, username: string, password: string) => Promise<void>
  logout: () => void
  checkAuth: () => Promise<void>
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      accessToken: null,
      refreshToken: null,
      isAuthenticated: false,
      isLoading: false,

      setUser: (user) => set({ user, isAuthenticated: true }),
      
      setTokens: (accessToken, refreshToken) => 
        set({ accessToken, refreshToken, isAuthenticated: true }),

      login: async (username, password) => {
        set({ isLoading: true })
        try {
          const response = await authApi.login({ username, password })
          set({ 
            accessToken: response.access_token,
            refreshToken: response.refresh_token,
            isAuthenticated: true,
          })
          
          // Fetch user data
          const user = await authApi.getCurrentUser()
          set({ user })
        } finally {
          set({ isLoading: false })
        }
      },

      register: async (email, username, password) => {
        set({ isLoading: true })
        try {
          const user = await authApi.register({ email, username, password })
          set({ user })
          
          // Auto-login after registration
          await get().login(username, password)
        } finally {
          set({ isLoading: false })
        }
      },

      logout: () => {
        authApi.logout().catch(console.error)
        set({
          user: null,
          accessToken: null,
          refreshToken: null,
          isAuthenticated: false,
        })
      },

      checkAuth: async () => {
        const token = get().accessToken
        if (!token) {
          set({ isAuthenticated: false })
          return
        }

        try {
          const user = await authApi.getCurrentUser()
          set({ user, isAuthenticated: true })
        } catch {
          get().logout()
        }
      },
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({
        accessToken: state.accessToken,
        refreshToken: state.refreshToken,
      }),
    }
  )
)