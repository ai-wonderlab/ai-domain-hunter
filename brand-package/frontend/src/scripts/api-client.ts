/**
 * API Client - Handles all backend communication
 */

interface _ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

export class ApiClient {
  private baseUrl: string;
  private token: string | null = null;
  
  constructor() {
    // Use window location to determine API URL
    const isDevelopment = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
    this.baseUrl = isDevelopment ? 'http://localhost:8000' : 'https://your-production-api.com';
    
    // Load token from localStorage if exists
    this.token = localStorage.getItem('auth_token');
  }
  
  /**
   * Set authentication token
   */
  setToken(token: string): void {
    this.token = token;
    localStorage.setItem('auth_token', token);
  }
  
  /**
   * Clear authentication token
   */
  clearToken(): void {
    this.token = null;
    localStorage.removeItem('auth_token');
  }
  
  /**
   * Get headers for requests
   */
  private getHeaders(): HeadersInit {
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
    };
    
    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`;
    }
    
    return headers;
  }
  
  /**
   * Handle API response
   */
  private async handleResponse<T>(response: Response): Promise<T> {
    if (!response.ok) {
      const error = await response.json().catch(() => ({
        error: 'An error occurred'
      }));
      throw new Error(error.error || error.message || 'API request failed');
    }
    
    return response.json();
  }
  
  /**
   * Make GET request
   */
  private async get<T>(endpoint: string): Promise<T> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      method: 'GET',
      headers: this.getHeaders(),
    });
    
    return this.handleResponse<T>(response);
  }
  
  /**
   * Make POST request
   */
  private async post<T>(endpoint: string, data?: any): Promise<T> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify(data),
    });
    
    return this.handleResponse<T>(response);
  }
  
  /**
   * Generate business names
   */
  async generateNames(params: {
    description: string;
    industry?: string;
    style?: string;
  }): Promise<{
    generation_id: string;
    names: Array<{
      name: string;
      reasoning: string;
      score: number;
      style?: string;
      memorability?: number;
      pronounceability?: number;
      uniqueness?: number;
    }>;
  }> {
    return this.post('/api/generate/names', params);
  }
  
  /**
   * Regenerate names with feedback
   */
  async regenerateNames(params: {
    generation_id: string;
    feedback: string;
  }): Promise<{
    generation_id: string;
    names: Array<any>;
  }> {
    return this.post('/api/generate/names/regenerate', params);
  }
  
   /**
   * Find available domains
   */
    async findAvailableDomains(params: {
    domains: string[];
    }): Promise<{
    success: boolean;
    results: Array<{  // ✅ Changed from 'domains' to 'results'
        domain: string;
        available: boolean;
        status: string;
        price: string;
        registrar?: string;
        registrar_link?: string;
        checked_at: string;
    }>;
    generation_id: string;
    total_checked: number;
    available_count: number;
    }> {
    return this.post('/api/generate/domains/find-available', {
        domains: params.domains
    });
    }
  
  /**
   * Check specific domains
   */
  async checkDomains(domains: string[]): Promise<{
    results: Array<{
      domain: string;
      available: boolean;
      status: string;
      price: string;
      registrar_link?: string;
    }>;
  }> {
    return this.post('/api/domains/check', { domains });
  }

  /**
     * Generate domain variations with AI
     */
    async generateDomains(params: {
    business_name: string;
    description?: string;
    }): Promise<{
    success: boolean;
    results: Array<{
        domain: string;
        available: boolean;
        status: string;
        price: string;
        registrar?: string;
        registrar_link?: string;
        checked_at: string;
        method: string;
    }>;
    generation_id: string;
    total_checked: number;
    available_count: number;
    rounds?: number;
    }> {
    return this.post('/api/generate/domains/generate', {
        business_name: params.business_name,
        description: params.description || ''
    });
    }
  
    /**
     * Regenerate domains with feedback
     */
    async regenerateDomains(params: {
        business_name: string;
        description: string;
        feedback: string;
        exclude_domains: string[];
    }): Promise<{
        success: boolean;
        results: Array<any>;
        generation_id: string;
        total_checked: number;
        available_count: number;
        rounds?: number;
        }> {
            return this.post('/api/generate/domains/regenerate', params);
        }

    /**
     * Analyze preferences for logo/tagline
     */
    async analyzePreferences(data: {
        business_name: string;
        description: string;
        for_type: 'logo' | 'tagline';
        industry?: string;
    }): Promise<any> {
        const response = await fetch(`${this.baseUrl}/api/generate/analyze/preferences`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error('Failed to analyze preferences');
        }
        
        return response.json();
    }
  
  /**
   * Generate logos
   */
  async generateLogos(params: {
    description: string;
    business_name: string;
    style?: string;
    colors?: string[];
    industry?: string;
  }): Promise<{
    generation_id: string;
    logos: Array<{
      id: string;
      concept_name: string;
      description: string;
      style: string;
      colors: string[];
      rationale: string;
      urls: { [key: string]: string };
    }>;
  }> {
    return this.post('/api/generate/logos', params);
  }
  
  /**
   * Generate color palettes
   */
  async generateColors(params: {
    description: string;
    business_name?: string;
    style?: string;
  }): Promise<{
    generation_id: string;
    palettes: Array<{
      id: string;
      name: string;
      colors: Array<{
        hex: string;
        rgb: string;
        name: string;
        role: string;
      }>;
      reasoning: string;
    }>;
  }> {
    return this.post('/api/generate/colors', params);
  }
  
  /**
   * Generate taglines
   */
  async generateTaglines(params: {
    description: string;
    business_name: string;
    tone?: string;
  }): Promise<{
    generation_id: string;
    taglines: Array<{
      id: string;
      text: string;
      tone: string;
      reasoning?: string;
    }>;
  }> {
    return this.post('/api/generate/taglines', params);
  }
  
  /**
   * Generate complete package
   */
  async generatePackage(params: {
    description: string;
    business_name?: string;
    industry?: string;
    style_preferences?: any;
    include_services?: string[];
  }): Promise<any> {
    return this.post('/api/generate/package', params);
  }
  
  /**
   * Save session to backend
   */
  async saveSession(sessionId: string, data: any): Promise<void> {
    if (!this.token) return;
    
    return this.post('/api/sessions/save', {
      session_id: sessionId,
      data
    });
  }
  
  /**
   * Load session from backend
   */
  async loadSession(sessionId: string): Promise<any> {
    if (!this.token) return null;
    
    return this.get(`/api/sessions/${sessionId}`);
  }
}

// Export singleton instance
export const apiClient = new ApiClient();