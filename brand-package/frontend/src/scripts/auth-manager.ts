/**
 * Auth Manager - Supabase Auth Integration
 */

import { createClient, SupabaseClient, User, Session } from '@supabase/supabase-js'

// @ts-ignore - Vite provides import.meta.env at runtime
const supabaseUrl = import.meta.env.VITE_SUPABASE_URL || 'https://pmzxlnpmapqlwsphiath.supabase.co'
// @ts-ignore - Vite provides import.meta.env at runtime
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY || 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBtenhsbnBtYXBxbHdzcGhpYXRoIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTkzOTYwMDUsImV4cCI6MjA3NDk3MjAwNX0.lmLmBgEz0O5dfKeu_4HwOxNMJq0IrXVFiSAKKCs2OCM'

class AuthManager {
  private supabase: SupabaseClient
  private currentUser: User | null = null
  private currentSession: Session | null = null
  
  constructor() {
    // ✅ Initialize Supabase client
    this.supabase = createClient(supabaseUrl, supabaseAnonKey)
    
    // Load session from localStorage
    this.loadSession()
    
    // Listen for auth changes
    this.supabase.auth.onAuthStateChange((event, session) => {
      console.log('Auth state changed:', event)
      this.currentSession = session
      this.currentUser = session?.user || null
      
      window.dispatchEvent(new CustomEvent('auth-state-changed', {
        detail: { user: this.currentUser, session }
      }))
    })
  }
  
  /**
   * Load session from localStorage
   */
  private async loadSession() {
    const { data: { session } } = await this.supabase.auth.getSession()
    this.currentSession = session
    this.currentUser = session?.user || null
  }
  
  /**
   * Sign up with email/password
   */
  async signUp(email: string, password: string): Promise<boolean> {
    try {
      const { data, error } = await this.supabase.auth.signUp({
        email,
        password
      })
      
      if (error) throw error
      
      this.currentUser = data.user
      this.currentSession = data.session
      
      return true
    } catch (error) {
      console.error('Sign up failed:', error)
      return false
    }
  }
  
  /**
   * Sign in with email/password
   */
  async signIn(email: string, password: string): Promise<boolean> {
    try {
      const { data, error } = await this.supabase.auth.signInWithPassword({
        email,
        password
      })
      
      if (error) throw error
      
      this.currentUser = data.user
      this.currentSession = data.session
      
      return true
    } catch (error) {
      console.error('Sign in failed:', error)
      return false
    }
  }
  
  /**
   * Sign in with Google OAuth
   */
  async signInWithGoogle(): Promise<void> {
    await this.supabase.auth.signInWithOAuth({
      provider: 'google',
      options: {
        redirectTo: window.location.origin + '/studio.html'
      }
    })
  }
  
  /**
   * Sign out
   */
  async signOut(): Promise<void> {
    await this.supabase.auth.signOut()
    this.currentUser = null
    this.currentSession = null
  }

    /**
   * Alias for signOut (backwards compatibility)
   */
  async logout(): Promise<void> {
    await this.signOut()
  }
  
  /**
   * Get current user
   */
  getCurrentUser(): User | null {
    return this.currentUser
  }
  
  /**
   * Get current user ID
   */
  getCurrentUserId(): string | null {
    return this.currentUser?.id || null
  }
  
  /**
   * Get access token
   */
  getToken(): string | null {
    return this.currentSession?.access_token || null
  }
  
  /**
   * Is authenticated?
   */
  isAuthenticated(): boolean {
    return !!this.currentUser && !!this.currentSession
  }
  
  /**
   * Get Supabase client (for direct use)
   */
  getSupabase(): SupabaseClient {
    return this.supabase
  }
}

// Singleton instance
export const authManager = new AuthManager()