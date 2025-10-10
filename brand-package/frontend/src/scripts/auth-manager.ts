/**
 * Auth Manager - Handles authentication state
 */

interface User {
  id: string;
  email: string;
  created_at: string;
}

interface AuthState {
  isAuthenticated: boolean;
  user: User | null;
  token: string | null;
}

export class AuthManager {
  private state: AuthState;
  
  constructor() {
    this.state = this.loadAuthState();
  }
  
  /**
   * Load auth state from localStorage
   */
  private loadAuthState(): AuthState {
    const token = localStorage.getItem('auth_token');
    const userStr = localStorage.getItem('user');
    
    if (token && userStr) {
      try {
        const user = JSON.parse(userStr);
        return {
          isAuthenticated: true,
          user,
          token
        };
      } catch (e) {
        return this.getEmptyState();
      }
    }
    
    return this.getEmptyState();
  }
  
  /**
   * Get empty auth state
   */
  private getEmptyState(): AuthState {
    return {
      isAuthenticated: false,
      user: null,
      token: null
    };
  }
  
  /**
   * Login user
   */
  async login(email: string, password: string): Promise<boolean> {
    try {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });
      
      if (!response.ok) {
        throw new Error('Login failed');
      }
      
      const data = await response.json();
      
      // Save to state
      this.state = {
        isAuthenticated: true,
        user: data.user,
        token: data.access_token
      };
      
      // Save to localStorage
      localStorage.setItem('auth_token', data.access_token);
      localStorage.setItem('user', JSON.stringify(data.user));
      
      // Dispatch event
      window.dispatchEvent(new CustomEvent('auth-state-changed', {
        detail: { isAuthenticated: true, user: data.user }
      }));
      
      return true;
    } catch (error) {
      console.error('Login failed:', error);
      return false;
    }
  }
  
  /**
   * Register user
   */
  async register(email: string, password: string): Promise<boolean> {
    try {
      const response = await fetch('/api/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });
      
      if (!response.ok) {
        throw new Error('Registration failed');
      }
      
      const data = await response.json();
      
      // Same as login
      this.state = {
        isAuthenticated: true,
        user: data.user,
        token: data.access_token
      };
      
      localStorage.setItem('auth_token', data.access_token);
      localStorage.setItem('user', JSON.stringify(data.user));
      
      window.dispatchEvent(new CustomEvent('auth-state-changed', {
        detail: { isAuthenticated: true, user: data.user }
      }));
      
      return true;
    } catch (error) {
      console.error('Registration failed:', error);
      return false;
    }
  }
  
  /**
   * Logout
   */
  logout(): void {
    this.state = this.getEmptyState();
    localStorage.removeItem('auth_token');
    localStorage.removeItem('user');
    
    window.dispatchEvent(new CustomEvent('auth-state-changed', {
      detail: { isAuthenticated: false, user: null }
    }));
  }
  
  /**
   * Get current user
   */
  getCurrentUser(): User | null {
    return this.state.user;
  }
  
  /**
   * Get current user ID
   */
  getCurrentUserId(): string | null {
    return this.state.user?.id || null;
  }
  
  /**
   * Get auth token
   */
  getToken(): string | null {
    return this.state.token;
  }
  
  /**
   * Is authenticated?
   */
  isAuthenticated(): boolean {
    return this.state.isAuthenticated;
  }
}

// Singleton instance
export const authManager = new AuthManager();