/**
 * Studio App - Main application controller
 */

import { StateManager } from './state-manager';
import { NavigationManager } from './navigation-manager';
import { PhaseRenderer } from './phase-renderer';
import { PhaseController } from './phase-controller';
import type { PhaseType } from './state-manager';

class StudioApp {
  private stateManager: StateManager;
  private navigationManager: NavigationManager;
  private phaseController: PhaseController;
  private phaseRenderer: PhaseRenderer;
  
  constructor() {
    console.log('🚀 Studio App initializing...');
    
    // Initialize managers in correct order
    this.stateManager = new StateManager();
    this.navigationManager = new NavigationManager(this.stateManager);
    this.phaseController = new PhaseController(this.stateManager, this.navigationManager);
    this.phaseRenderer = new PhaseRenderer(
      this.stateManager, 
      this.navigationManager,
      this.phaseController
    );
    
    // Setup
    this.init();
  }

  /**
   * Initialize app
   */
  private async init(): Promise<void> {
    // Show session selection modal
    this.showSessionModal();
    
    // Setup global listeners
    this.setupGlobalListeners();
    this.setupPhaseStatusListener();
    
    // Update UI
    this.updateLeftPanel();
  }

  /**
   * Find the most recent session key in localStorage
   */
  private findMostRecentSessionKey(): string | null {
    let mostRecent: { key: string; timestamp: number } | null = null;
    
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key && key.startsWith('studio-session-')) {
        const data = localStorage.getItem(key);
        if (data) {
          try {
            const session = JSON.parse(data);
            if (!mostRecent || session.lastSaved > mostRecent.timestamp) {
              mostRecent = {
                key: key,
                timestamp: session.lastSaved
              };
            }
          } catch (e) {}
        }
      }
    }
    
    return mostRecent?.key || null;
  }

  /**
   * Show session selection modal
   */
  private showSessionModal(): void {
    // Ψάξε για το πιο πρόσφατο session
    const mostRecentKey = this.findMostRecentSessionKey();
    
    if (!mostRecentKey) {
      // No existing session, start fresh
      this.startNewSession();
      return;
    }
    
    // Create modal
    const modal = document.createElement('div');
    modal.className = 'session-modal-overlay';
    modal.innerHTML = `
      <div class="session-modal">
        <h2>Welcome Back! 👋</h2>
        <p>You have an existing session. What would you like to do?</p>
        
        <div class="session-actions">
          <button id="continue-session-btn" class="primary-btn gradient-btn">
            ↩️ Continue Last Session
          </button>
          <button id="new-session-btn" class="secondary-btn">
            ✨ Start Fresh
          </button>
          <button id="view-history-btn" class="secondary-btn">
            📋 View All Sessions
          </button>
        </div>
      </div>
    `;
    
    document.body.appendChild(modal);
    
    // Continue button
    modal.querySelector('#continue-session-btn')?.addEventListener('click', () => {
      modal.remove();
      // Load the most recent session by key
      if (mostRecentKey) {
        const sessionData = localStorage.getItem(mostRecentKey);
        if (sessionData) {
          const session = JSON.parse(sessionData);
          this.stateManager.setSession(session);
          this.resumeSession(session.currentPhase);
        }
      }
    });
    
    // New session button
    modal.querySelector('#new-session-btn')?.addEventListener('click', () => {
      modal.remove();
      // Save current session to history first
      this.saveSessionToHistory();
      // Clear current and start new
      this.stateManager.clearSession();
      this.startNewSession();
    });
    
    // View history button
    modal.querySelector('#view-history-btn')?.addEventListener('click', () => {
      modal.remove();
      this.showSessionHistory();
    });
  }

  /**
   * Save current session to history
   */
  private saveSessionToHistory(): void {
    const session = this.stateManager.getSession();
    const history = JSON.parse(localStorage.getItem('session_history') || '[]');
    
    history.unshift({
      sessionId: session.sessionId,
      createdAt: session.createdAt,
      lastSaved: session.lastSaved,
      currentPhase: session.currentPhase,
      selectedName: session.phases.names.selectedName,
      data: JSON.stringify(session) // Full session data
    });
    
    // Keep last 10 sessions
    if (history.length > 10) history.length = 10;
    
    localStorage.setItem('session_history', JSON.stringify(history));
  }

  /**
   * Show session history modal
   */
  private showSessionHistory(): void {
    const history = JSON.parse(localStorage.getItem('session_history') || '[]');
    
    const modal = document.createElement('div');
    modal.className = 'session-modal-overlay';
    modal.innerHTML = `
      <div class="session-modal large">
        <h2>📋 Your Sessions</h2>
        
        <div class="session-list">
          ${history.length === 0 ? '<p>No saved sessions yet</p>' : 
            history.map((s: any, i: number) => `
              <div class="session-item" data-index="${i}">
                <div class="session-info">
                  <strong>${s.selectedName || 'Unnamed'}</strong>
                  <small>Phase: ${s.currentPhase}</small>
                  <small>${new Date(s.lastSaved).toLocaleString()}</small>
                </div>
                <button class="resume-btn" data-index="${i}">Resume →</button>
              </div>
            `).join('')
          }
        </div>
        
        <button id="close-history-btn" class="secondary-btn">Close</button>
      </div>
    `;
    
    document.body.appendChild(modal);
    
    // Resume buttons
    modal.querySelectorAll('.resume-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const index = parseInt((e.target as HTMLElement).getAttribute('data-index') || '0');
        const sessionData = JSON.parse(history[index].data);
        
        // Load this session
        this.stateManager.setSession(sessionData);
        this.stateManager.saveToLocalStorage();
        
        modal.remove();
        
        // ΜΗΝ κάνεις reload - απλά resume το session
        this.resumeSession(sessionData.currentPhase);
        
        // Trigger phase change για να φορτώσει τα data
        window.dispatchEvent(new CustomEvent('phase-changed', { 
          detail: { phase: sessionData.currentPhase } 
        }));
      });
    });
    
    // Close button
    modal.querySelector('#close-history-btn')?.addEventListener('click', () => {
      modal.remove();
      this.showSessionModal(); // Go back to main modal
    });
  }

  /**
   * Listen for phase status changes
   */
  private setupPhaseStatusListener(): void {
    window.addEventListener('phase-status-changed', () => {
      this.updateBreadcrumb();
      this.updateProgressBar();
    });
  }
  
  /**
   * Start new session
   */
  private startNewSession(): void {
    // Render initial form
    this.phaseRenderer.renderPhase('initial');
    this.updateLeftPanel();
    
    // ✅ Clear preview panel
    setTimeout(() => {
      window.dispatchEvent(new CustomEvent('phase-data-ready'));
    }, 100);
  }
  
  /**
   * Resume existing session
   */
  private resumeSession(phase: any): void {
    console.log(`📍 Resuming session at phase: ${phase}`);
    
    // Render current phase
    this.phaseRenderer.renderPhase(phase);
    this.updateLeftPanel();
    
    // ✅ UPDATE PREVIEW PANEL με τα saved selections
    // Καλούμε το updatePreviewPanel() μέσω event
    setTimeout(() => {
      window.dispatchEvent(new CustomEvent('phase-data-ready'));
    }, 100);
    
    // Show notification
    this.showToast('Session resumed! ✅', 'success');
  }
  
  /**
   * Setup global event listeners
   */
  private setupGlobalListeners(): void {
    // Phase changed event
    window.addEventListener('phase-changed', ((e: CustomEvent) => {
      const session = this.stateManager.getSession();
      const phase = e.detail?.phase || session.currentPhase;
      
      // Check if we need to trigger API call for this phase
      this.handlePhaseChange(phase);
      
      // Render phase
      this.phaseRenderer.renderPhase(phase);
      this.updateLeftPanel();
    }) as EventListener);
    
    // Save session button
    const saveBtn = document.getElementById('save-session-btn');
    saveBtn?.addEventListener('click', () => {
      this.stateManager.saveToLocalStorage();
      this.showToast('Session saved! 💾', 'success');
      this.updateLeftPanel();
    });
    
    // New session button
    const newSessionBtn = document.getElementById('new-session-btn');
    newSessionBtn?.addEventListener('click', () => {
      if (confirm('Start a new session? Current progress will be saved to history.')) {
        this.saveSessionToHistory();
        this.stateManager.clearSession();
        
        // ΜΗΝ κάνεις reload - απλά ξεκίνα νέο session
        this.startNewSession();
        
        // Reset το UI
        this.updateLeftPanel();
        
        // Πήγαινε στο initial phase
        this.navigationManager.goToPhase('initial');
        window.dispatchEvent(new CustomEvent('phase-changed', { 
          detail: { phase: 'initial' } 
        }));
      }
    });

    // View history button  
    const historyBtn = document.getElementById('view-history-btn');
    historyBtn?.addEventListener('click', () => {
      this.showSessionHistory();
    });
    
    // Breadcrumb clicks - allow navigation to any phase in history
    document.addEventListener('click', (e) => {
      const breadcrumb = (e.target as HTMLElement).closest('.breadcrumb-item');
      if (breadcrumb) {
        const phase = breadcrumb.getAttribute('data-phase') as PhaseType;
        if (phase && this.navigationManager.canNavigateTo(phase)) {
          this.navigationManager.goToPhase(phase);
          window.dispatchEvent(new CustomEvent('phase-changed'));
        }
      }
    });
  }
  
  /**
   * Handle phase change - trigger API calls if needed
   */
  private async handlePhaseChange(phase: string): Promise<void> {
    const session = this.stateManager.getSession();
    
    // ✅ SAVE CURRENT PHASE TO HISTORY πριν φύγεις
    const currentPhase = session.currentPhase;
    if (['names', 'domains', 'logos', 'taglines'].includes(currentPhase)) {
      this.phaseController.saveCurrentGenerationToHistory(currentPhase as any);
    }
    
    // Check if this phase needs data
    switch (phase) {
      case 'names':
        if (session.phases.names.generatedOptions.length === 0) {
          await this.phaseController.generateNames();
        }
        break;
        
      case 'domains':
        // ✅ ALWAYS call - it has internal logic to skip if unchanged
        await this.phaseController.generateDomains();
        break;
        
      case 'logo_prefs':
        if (!session.phases.logoPreferences.aiSuggestions) {
          await this.phaseController.prepareLogoPreferences();
        }
        break;
        
      case 'logos':
        // ✅ ALWAYS call - it has internal logic to skip if unchanged
        // This ensures we regenerate when preferences change
        await this.phaseController.generateLogos();
        break;
        
      case 'tagline_prefs':
        if (!session.phases.taglinePreferences.aiSuggestions) {
          await this.phaseController.prepareTaglinePreferences();
        }
        break;
        
      case 'taglines':
        // ✅ ALWAYS call - it has internal logic to skip if unchanged
        await this.phaseController.generateTaglines();
        break;
    }
  }
  
  /**
   * Update left panel UI
   */
  private updateLeftPanel(): void {
    const session = this.stateManager.getSession();
    
    // Update session info
    const startEl = document.getElementById('session-start');
    const savedEl = document.getElementById('session-saved');
    
    if (startEl) {
      startEl.textContent = this.formatTime(session.createdAt);
    }
    
    if (savedEl) {
      const timeSince = Date.now() - session.lastSaved;
      if (timeSince < 5000) {
        savedEl.textContent = 'Just now';
      } else if (timeSince < 60000) {
        savedEl.textContent = `${Math.floor(timeSince / 1000)}s ago`;
      } else {
        savedEl.textContent = this.formatTime(session.lastSaved);
      }
    }
    
    // Update breadcrumb navigation
    this.updateBreadcrumb();
    
    // Update selections
    this.updateSelections();
    
    // Update progress bar
    this.updateProgressBar();
  }
  
  /**
   * Update breadcrumb navigation
   */
  private updateBreadcrumb(): void {
    const session = this.stateManager.getSession();
    const container = document.getElementById('breadcrumb-nav');
    
    if (!container) return;
    
    const phaseNames: { [key: string]: string } = {
      'initial': 'Setup',
      'names': 'Names',
      'domains': 'Domain',
      'logo_prefs': 'Logo Style',
      'logos': 'Logo',
      'tagline_prefs': 'Tagline Tone',
      'taglines': 'Tagline',
      'complete': 'Complete'
    };
    
    const html = session.navigation.history.map((phase, index) => {
      const isActive = phase === session.currentPhase;
      const isCompleted = index < session.navigation.history.length - 1;
      
      return `
        <div class="breadcrumb-item ${isActive ? 'active' : ''} ${isCompleted ? 'completed' : ''}" data-phase="${phase}">
          <span class="breadcrumb-icon">${isCompleted ? '✓' : '○'}</span>
          <span class="breadcrumb-text">${phaseNames[phase] || phase}</span>
        </div>
      `;
    }).join('');
    
    container.innerHTML = html;
  }
  
  /**
   * Update current selections display
   */
  private updateSelections(): void {
    const session = this.stateManager.getSession();
    const container = document.getElementById('selections-list');
    
    if (!container) return;
    
    let html = '';
    
    const businessName = this.stateManager.getBusinessName();
    if (businessName) {
      html += `
        <div class="selection-item">
          <span class="selection-label">Name:</span>
          <span class="selection-value">${this.escapeHtml(businessName)}</span>
        </div>
      `;
    }
    
    if (session.phases.domains.selectedDomain) {
      html += `
        <div class="selection-item">
          <span class="selection-label">Domain:</span>
          <span class="selection-value">${this.escapeHtml(session.phases.domains.selectedDomain)}</span>
        </div>
      `;
    }
    
    if (html === '') {
      html = '<p class="no-selections">No selections yet</p>';
    }
    
    container.innerHTML = html;
  }
  
  /**
   * Update progress bar
   */
  private updateProgressBar(): void {
    const session = this.stateManager.getSession();
    const progressBar = document.querySelector('.progress-bar-fill') as HTMLElement;
    
    if (!progressBar) return;
    
    // Calculate progress based on completed phases
    const totalPhases = session.input.selectedServices.length + 1;
    let completedPhases = 1; // Initial is always complete if we're past it
    
    Object.entries(session.phases).forEach(([, data]) => {
      if (data.status === 'completed') {
        completedPhases++;
      }
    });
    
    const progress = Math.min(100, (completedPhases / totalPhases) * 100);
    progressBar.style.width = `${progress}%`;
  }
  
  /**
   * Format timestamp
   */
  private formatTime(timestamp: number): string {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', { 
      hour: 'numeric', 
      minute: '2-digit',
      hour12: true 
    });
  }
  
  /**
   * Show toast notification
   */
  private showToast(message: string, type: 'success' | 'error' | 'info' = 'info'): void {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    
    const container = document.getElementById('toast-container') || document.body;
    container.appendChild(toast);
    
    setTimeout(() => {
      toast.classList.add('show');
    }, 100);
    
    setTimeout(() => {
      toast.classList.remove('show');
      setTimeout(() => toast.remove(), 300);
    }, 3000);
  }
  
  /**
   * Escape HTML
   */
  private escapeHtml(text: string): string {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
}

// Initialize when DOM is ready
function initStudioApp() {
  console.log('🎬 Initializing StudioApp...');
  new StudioApp();
}

// Check if DOM is already loaded
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initStudioApp);
} else {
  initStudioApp();
}

// Export for debugging
(window as any).StudioApp = StudioApp;