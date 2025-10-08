/**
 * Studio App - Main application controller
 */

import { StateManager } from './state-manager';
import { NavigationManager } from './navigation-manager';
import { PhaseRenderer } from './phase-renderer';
import { PhaseController } from './phase-controller';

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
    // Try to load existing session
    const loaded = this.stateManager.loadFromLocalStorage();
    
    if (loaded) {
      console.log('✅ Loaded existing session');
      const session = this.stateManager.getSession();
      this.resumeSession(session.currentPhase);
    } else {
      console.log('🆕 Starting new session');
      this.startNewSession();
    }
    
    // Setup global listeners
    this.setupGlobalListeners();
    
    // Update UI
    this.updateLeftPanel();
  }
  
  /**
   * Start new session
   */
  private startNewSession(): void {
    // Render initial form
    this.phaseRenderer.renderPhase('initial');
    this.updateLeftPanel();
  }
  
  /**
   * Resume existing session
   */
  private resumeSession(phase: any): void {
    console.log(`📍 Resuming session at phase: ${phase}`);
    
    // Render current phase
    this.phaseRenderer.renderPhase(phase);
    this.updateLeftPanel();
    
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
    
    // Go back button
    const goBackBtn = document.getElementById('go-back-btn');
    goBackBtn?.addEventListener('click', () => {
      const success = this.navigationManager.goBack();
      if (success) {
        window.dispatchEvent(new CustomEvent('phase-changed'));
      }
    });
    
    // Save session button
    const saveBtn = document.getElementById('save-session-btn');
    saveBtn?.addEventListener('click', () => {
      this.stateManager.saveToLocalStorage();
      this.showToast('Session saved! 💾', 'success');
      this.updateLeftPanel();
    });
  }
  
  /**
   * Handle phase change - trigger API calls if needed
   */
  private async handlePhaseChange(phase: string): Promise<void> {
    const session = this.stateManager.getSession();
    
    // Check if this phase needs data and doesn't have it yet
    switch (phase) {
      case 'names':
        if (session.phases.names.generatedOptions.length === 0) {
          await this.phaseController.generateNames();
        }
        break;
        
      case 'domains':
        if (session.phases.domains.availableOptions.length === 0) {
          await this.phaseController.generateDomains();
        }
        break;
        
      case 'logo_prefs':
        if (!session.phases.logoPreferences.aiSuggestions) {
          await this.phaseController.prepareLogoPreferences();
        }
        break;
        
      case 'logos':
        if (session.phases.logos.generatedOptions.length === 0) {
          await this.phaseController.generateLogos();
        }
        break;
        
      case 'tagline_prefs':
        if (!session.phases.taglinePreferences.aiSuggestions) {
          await this.phaseController.prepareTaglinePreferences();
        }
        break;
        
      case 'taglines':
        if (session.phases.taglines.generatedOptions.length === 0) {
          await this.phaseController.generateTaglines();
        }
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
    
    // Update go back button
    const goBackBtn = document.getElementById('go-back-btn');
    if (goBackBtn) {
      if (session.navigation.canGoBack) {
        goBackBtn.removeAttribute('disabled');
      } else {
        goBackBtn.setAttribute('disabled', 'true');
      }
    }
    
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
    
    // Add click listeners to completed breadcrumbs
    const items = container.querySelectorAll('.breadcrumb-item.completed');
    items.forEach(item => {
      item.addEventListener('click', (e) => {
        const phase = (e.currentTarget as HTMLElement).getAttribute('data-phase');
        if (phase) {
          this.navigationManager.goToPhase(phase as any);
          window.dispatchEvent(new CustomEvent('phase-changed'));
        }
      });
    });
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
    
    Object.entries(session.phases).forEach(([phase, data]) => {
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

// ====== FIX: Initialize immediately or when DOM ready ======
function initStudioApp() {
  console.log('🎬 Initializing StudioApp...');
  new StudioApp();
}

// Check if DOM is already loaded
if (document.readyState === 'loading') {
  // DOM still loading, wait for it
  document.addEventListener('DOMContentLoaded', initStudioApp);
} else {
  // DOM already loaded, initialize immediately
  initStudioApp();
}

// Export for debugging
(window as any).StudioApp = StudioApp;