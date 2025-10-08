/**
 * State Manager - Central state management for Brand Studio
 * Handles all session state, persistence, and navigation
 */

export type PhaseType = 
  | 'initial' 
  | 'names' 
  | 'domains' 
  | 'logo_prefs' 
  | 'logos' 
  | 'tagline_prefs' 
  | 'taglines' 
  | 'complete';

export type PhaseStatus = 'not_started' | 'in_progress' | 'completed' | 'skipped';

export interface NameOption {
  name: string;
  reasoning: string;
  score: number;
  style?: string;
  memorability?: number;
  pronounceability?: number;
  uniqueness?: number;
}

export interface DomainOption {
  domain: string;
  available: boolean;
  status: string;
  price: string;
  registrar?: string;
  registrar_link?: string;
  checked_at: string;
}

export interface ColorSuggestion {
  hex: string;
  rgb: string;
  name: string;
  role: string;
  reasoning?: string;
}

export interface LogoOption {
  id: string;
  concept_name: string;
  description: string;
  style: string;
  colors: string[];
  rationale: string;
  urls: { [key: string]: string };
}

export interface TaglineOption {
  id: string;
  text: string;
  tone: string;
  reasoning?: string;
}

export interface StudioSession {
  // Metadata
  sessionId: string;
  userId: string | null;
  createdAt: number;
  lastSaved: number;
  currentPhase: PhaseType;
  
  // User Input
  input: {
    description: string;
    businessName: string | null;
    hasBusinessName: boolean;
    selectedServices: string[];
  };
  
  // Progress through phases
  phases: {
    names: {
      status: PhaseStatus;
      startedAt: number | null;
      completedAt: number | null;
      generatedOptions: NameOption[];
      selectedName: string | null;
      generationId: string | null;
    };
    
    domains: {
      status: PhaseStatus;
      startedAt: number | null;
      completedAt: number | null;
      checkedVariations: string[];
      availableOptions: DomainOption[];
      selectedDomain: string | null;
      checkRounds: number;
    };
    
    logoPreferences: {
      status: PhaseStatus;
      startedAt: number | null;
      completedAt: number | null;
      aiSuggestions: {
        style: string;
        styleReasoning: string;
        colors: ColorSuggestion[];
        colorReasoning: string;
      } | null;
      userChoice: {
        style: string;
        colors: string[];
        customized: boolean;
      } | null;
    };
    
    logos: {
      status: PhaseStatus;
      startedAt: number | null;
      completedAt: number | null;
      generatedOptions: LogoOption[];
      selectedLogo: LogoOption | null;
      generationId: string | null;
    };
    
    taglinePreferences: {
      status: PhaseStatus;
      startedAt: number | null;
      completedAt: number | null;
      aiSuggestions: {
        tone: string;
        toneReasoning: string;
      } | null;
      userChoice: {
        tone: string;
        customized: boolean;
      } | null;
    };
    
    taglines: {
      status: PhaseStatus;
      startedAt: number | null;
      completedAt: number | null;
      generatedOptions: TaglineOption[];
      selectedTagline: string | null;
      generationId: string | null;
    };
  };
  
  // Navigation
  navigation: {
    history: PhaseType[];
    canGoBack: boolean;
    canGoForward: boolean;
    nextPhase: PhaseType | null;
  };
  
  // Final Package
  finalPackage: {
    businessName: string;
    domain: string;
    logo: LogoOption;
    colors: string[];
    tagline: string;
    downloadUrl: string | null;
  } | null;
}

export class StateManager {
  private session: StudioSession;
  private hasUnsavedChanges: boolean = false;
  private autoSaveInterval: number | null = null;
  
  constructor() {
    this.session = this.createNewSession();
    this.setupAutoSave();
    this.setupBeforeUnload();
  }
  
  /**
   * Create a new session with default values
   */
  private createNewSession(): StudioSession {
    return {
      sessionId: this.generateSessionId(),
      userId: null,
      createdAt: Date.now(),
      lastSaved: Date.now(),
      currentPhase: 'initial',
      
      input: {
        description: '',
        businessName: null,
        hasBusinessName: false,
        selectedServices: ['name', 'domain', 'logo', 'tagline']
      },
      
      phases: {
        names: {
          status: 'not_started',
          startedAt: null,
          completedAt: null,
          generatedOptions: [],
          selectedName: null,
          generationId: null
        },
        domains: {
          status: 'not_started',
          startedAt: null,
          completedAt: null,
          checkedVariations: [],
          availableOptions: [],
          selectedDomain: null,
          checkRounds: 0
        },
        logoPreferences: {
          status: 'not_started',
          startedAt: null,
          completedAt: null,
          aiSuggestions: null,
          userChoice: null
        },
        logos: {
          status: 'not_started',
          startedAt: null,
          completedAt: null,
          generatedOptions: [],
          selectedLogo: null,
          generationId: null
        },
        taglinePreferences: {
          status: 'not_started',
          startedAt: null,
          completedAt: null,
          aiSuggestions: null,
          userChoice: null
        },
        taglines: {
          status: 'not_started',
          startedAt: null,
          completedAt: null,
          generatedOptions: [],
          selectedTagline: null,
          generationId: null
        }
      },
      
      navigation: {
        history: ['initial'],
        canGoBack: false,
        canGoForward: false,
        nextPhase: null
      },
      
      finalPackage: null
    };
  }
  
  /**
   * Generate unique session ID
   */
  private generateSessionId(): string {
    return `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }
  
  /**
   * Get current session
   */
  getSession(): StudioSession {
    return this.session;
  }
  
  /**
   * Update session
   */
  updateSession(updates: Partial<StudioSession>): void {
    this.session = { ...this.session, ...updates };
    this.hasUnsavedChanges = true;
  }
  
  /**
   * Update input data
   */
  updateInput(updates: Partial<StudioSession['input']>): void {
    this.session.input = { ...this.session.input, ...updates };
    this.hasUnsavedChanges = true;
  }
  
  /**
   * Update phase data
   */
  updatePhase<K extends keyof StudioSession['phases']>(
    phase: K,
    updates: Partial<StudioSession['phases'][K]>
  ): void {
    this.session.phases[phase] = { 
      ...this.session.phases[phase], 
      ...updates 
    } as StudioSession['phases'][K];
    this.hasUnsavedChanges = true;
  }
  
  /**
   * Mark phase as started
   */
  startPhase(phase: keyof StudioSession['phases']): void {
    this.updatePhase(phase, {
      status: 'in_progress',
      startedAt: Date.now()
    } as any);
  }
  
  /**
   * Mark phase as completed
   */
  completePhase(phase: keyof StudioSession['phases']): void {
    this.updatePhase(phase, {
      status: 'completed',
      completedAt: Date.now()
    } as any);
  }
  
  /**
   * Save to localStorage
   */
  saveToLocalStorage(): void {
    try {
      const key = `studio-session-${this.session.sessionId}`;
      this.session.lastSaved = Date.now();
      localStorage.setItem(key, JSON.stringify(this.session));
      this.hasUnsavedChanges = false;
      console.log('💾 Session saved to localStorage');
    } catch (error) {
      console.error('Failed to save to localStorage:', error);
    }
  }
  
  /**
   * Load from localStorage
   */
  loadFromLocalStorage(sessionId?: string): boolean {
    try {
      // If no sessionId provided, try to find the most recent session
      if (!sessionId) {
        sessionId = this.findMostRecentSession() || undefined;
      }
      
      if (!sessionId) {
        console.log('No previous session found');
        return false;
      }
      
      const key = `studio-session-${sessionId}`;
      const saved = localStorage.getItem(key);
      
      if (saved) {
        this.session = JSON.parse(saved);
        console.log('✅ Session loaded from localStorage');
        return true;
      }
      
      return false;
    } catch (error) {
      console.error('Failed to load from localStorage:', error);
      return false;
    }
  }
  
  /**
   * Find most recent session in localStorage
   */
  private findMostRecentSession(): string | null {
    try {
      let mostRecent: { sessionId: string; timestamp: number } | null = null;
      
      for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (key && key.startsWith('studio-session-')) {
          const data = localStorage.getItem(key);
          if (data) {
            const session = JSON.parse(data);
            if (!mostRecent || session.lastSaved > mostRecent.timestamp) {
              mostRecent = {
                sessionId: session.sessionId,
                timestamp: session.lastSaved
              };
            }
          }
        }
      }
      
      return mostRecent?.sessionId || null;
    } catch (error) {
      console.error('Error finding recent session:', error);
      return null;
    }
  }
  
  /**
   * Clear localStorage
   */
  clearLocalStorage(): void {
    const key = `studio-session-${this.session.sessionId}`;
    localStorage.removeItem(key);
    console.log('🗑️ Session cleared from localStorage');
  }
  
  /**
   * Setup auto-save
   */
  private setupAutoSave(): void {
    // Save every 30 seconds if there are changes
    this.autoSaveInterval = window.setInterval(() => {
      if (this.hasUnsavedChanges) {
        this.saveToLocalStorage();
      }
    }, 30000);
  }
  
  /**
   * Setup beforeunload handler
   */
  private setupBeforeUnload(): void {
    window.addEventListener('beforeunload', () => {
      if (this.hasUnsavedChanges) {
        this.saveToLocalStorage();
      }
    });
  }
  
  /**
   * Cleanup
   */
  destroy(): void {
    if (this.autoSaveInterval) {
      clearInterval(this.autoSaveInterval);
    }
  }
  
  /**
   * Reset to new session
   */
  reset(): void {
    this.clearLocalStorage();
    this.session = this.createNewSession();
    this.hasUnsavedChanges = false;
  }
  
  /**
   * Get business name (from selection or user input)
   */
  getBusinessName(): string | null {
    return this.session.phases.names.selectedName || this.session.input.businessName;
  }
  
  /**
   * Check if service is selected
   */
  isServiceSelected(service: string): boolean {
    return this.session.input.selectedServices.includes(service);
  }
  
  /**
   * Get phase status
   */
  getPhaseStatus(phase: keyof StudioSession['phases']): PhaseStatus {
    return this.session.phases[phase].status;
  }
  
  /**
   * Is phase completed?
   */
  isPhaseCompleted(phase: keyof StudioSession['phases']): boolean {
    return this.session.phases[phase].status === 'completed';
  }
}