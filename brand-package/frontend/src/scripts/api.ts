import type {
  AuthResponse,
  GeneratePackageRequest,
  GeneratePackageResponse,
  ApiError,
  User
} from '@/types/api.types';

class API {
  private baseURL: string;
  private token: string | null = null;

  constructor() {
    this.baseURL = '/api';
    this.token = localStorage.getItem('auth_token');
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    // Fix: Use Record type for headers
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...(options.headers as Record<string, string> || {})
    };

    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`;
    }

    const url = `${this.baseURL}${endpoint}`;
    
    try {
      const response = await fetch(url, {
        ...options,
        headers
      });

      if (!response.ok) {
        let error: ApiError;
        try {
          error = await response.json();
        } catch {
          error = {
            error: 'Request failed',
            message: `HTTP ${response.status}: ${response.statusText}`
          };
        }
        throw new Error(error.message || error.error || 'Request failed');
      }

      return response.json();
    } catch (error) {
      console.error('API request failed:', error);
      throw error;
    }
  }

  /* ========================================
     AUTHENTICATION
     ======================================== */

  async login(email: string): Promise<AuthResponse> {
    const response = await this.request<AuthResponse>('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email })
    });

    if (response.access_token) {
      this.token = response.access_token;
      localStorage.setItem('auth_token', response.access_token);
      localStorage.setItem('user', JSON.stringify(response.user));
    }

    return response;
  }

  async register(email: string): Promise<AuthResponse> {
    const response = await this.request<AuthResponse>('/auth/register', {
      method: 'POST',
      body: JSON.stringify({ email })
    });

    if (response.access_token) {
      this.token = response.access_token;
      localStorage.setItem('auth_token', response.access_token);
      localStorage.setItem('user', JSON.stringify(response.user));
    }

    return response;
  }

  logout() {
    this.token = null;
    localStorage.removeItem('auth_token');
    localStorage.removeItem('user');
  }

  isAuthenticated(): boolean {
    return !!this.token;
  }

  getUser(): User | null {
    const userStr = localStorage.getItem('user');
    if (!userStr) return null;
    
    try {
      return JSON.parse(userStr);
    } catch {
      return null;
    }
  }

  /* ========================================
     GENERATION
     ======================================== */

  async generatePackage(
    data: GeneratePackageRequest
  ): Promise<GeneratePackageResponse> {
    return this.request<GeneratePackageResponse>('/generate/package', {
      //                                            ^^^^^^^^^ REMOVED "ation"
      method: 'POST',
      body: JSON.stringify(data)
    });
  }

  async generateNames(
    description: string,
    industry?: string
  ): Promise<any> {
    return this.request('/generate/names', {
      method: 'POST',
      body: JSON.stringify({ description, industry })
    });
  }

  async generateLogos(
    businessName: string,
    description: string,
    style?: string
  ): Promise<any> {
    return this.request('/generate/logos', {
      method: 'POST',
      body: JSON.stringify({ 
        business_name: businessName,
        description,
        style 
      })
    });
  }

  async generateColors(
    businessName: string,
    description: string,
    theme?: string
  ): Promise<any> {
    return this.request('/generate/colors', {
      method: 'POST',
      body: JSON.stringify({ 
        business_name: businessName,
        description,
        theme 
      })
    });
  }

  async generateTaglines(
    businessName: string,
    description: string,
    tone?: string
  ): Promise<any> {
    return this.request('/generate/taglines', {
      method: 'POST',
      body: JSON.stringify({ 
        business_name: businessName,
        description,
        tone 
      })
    });
  }

  async checkDomains(domains: string[]): Promise<any> {
    return this.request('/generate/domains/check', {
      method: 'POST',
      body: JSON.stringify({ domains })
    });
  }

  /* ========================================
     STREAMING GENERATION (Optional)
     ======================================== */

  async streamGeneratePackage(
    data: GeneratePackageRequest,
    onUpdate: (chunk: any) => void
  ): Promise<void> {
    // Fix: Use Record type for streaming headers too
    const headers: Record<string, string> = {
      'Content-Type': 'application/json'
    };

    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`;
    }

    const response = await fetch(`${this.baseURL}/generation/package`, {
      method: 'POST',
      headers,
      body: JSON.stringify(data)
    });

    if (!response.ok) {
      throw new Error('Generation failed');
    }

    const reader = response.body?.getReader();
    const decoder = new TextDecoder();

    if (!reader) return;

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n').filter(line => line.trim());
        
        for (const line of lines) {
          try {
            const data = JSON.parse(line);
            onUpdate(data);
          } catch (e) {
            console.warn('Failed to parse chunk:', line);
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  /* ========================================
     USER MANAGEMENT
     ======================================== */

  async getUserProfile(): Promise<User> {
    return this.request<User>('/users/me');
  }

  async updateUserProfile(data: Partial<User>): Promise<User> {
    return this.request<User>('/users/me', {
      method: 'PUT',
      body: JSON.stringify(data)
    });
  }

  async getGenerationHistory(limit: number = 10): Promise<any[]> {
    return this.request(`/users/history?limit=${limit}`);
  }

  /* ========================================
     PROJECTS
     ======================================== */

  async getProjects(): Promise<any[]> {
    return this.request('/projects');
  }

  async getProject(id: string): Promise<any> {
    return this.request(`/projects/${id}`);
  }

  async createProject(data: any): Promise<any> {
    return this.request('/projects', {
      method: 'POST',
      body: JSON.stringify(data)
    });
  }

  async updateProject(id: string, data: any): Promise<any> {
    return this.request(`/projects/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data)
    });
  }

  async deleteProject(id: string): Promise<void> {
    return this.request(`/projects/${id}`, {
      method: 'DELETE'
    });
  }

  /* ========================================
     HEALTH CHECK
     ======================================== */

  async healthCheck(): Promise<{ status: string; timestamp: string }> {
    return this.request('/health');
  }
}

// Export singleton instance
export const api = new API();

// Export class for testing
export { API };