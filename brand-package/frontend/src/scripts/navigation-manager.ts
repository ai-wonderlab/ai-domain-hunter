/**
 * Navigation Manager - Handles phase navigation and flow control
 */

import { StateManager, PhaseType } from './state-manager';

export class NavigationManager {
  private stateManager: StateManager;
  
  constructor(stateManager: StateManager) {
    this.stateManager = stateManager;
  }
  
  /**
   * Go to specific phase
   */
  goToPhase(phase: PhaseType): boolean {
    const session = this.stateManager.getSession();
    
    // Validate can navigate to this phase
    if (!this.canNavigateTo(phase)) {
      console.warn(`Cannot navigate to ${phase} - dependencies not met`);
      return false;
    }
    
    // Remove duplicates in history
    const uniqueHistory = [...new Set([...session.navigation.history, phase])];
    session.navigation.history = uniqueHistory;
    
    // Update current phase
    session.currentPhase = phase;
    
    // Update navigation state
    this.updateNavigationState();
    
    // Save
    this.stateManager.saveToLocalStorage();
    
    console.log(`📍 Navigated to phase: ${phase}`);
    return true;
  }
  
  /**
   * Go back to previous phase
   */
  goBack(): boolean {
  const session = this.stateManager.getSession();
  const history = session.navigation.history;
  
  if (!session.navigation.canGoBack || history.length <= 1) {
    console.warn('Cannot go back - already at first phase');
    return false;
  }
  
  // ΜΗΝ κάνεις pop! Απλά πήγαινε στο προηγούμενο
  const currentIndex = history.indexOf(session.currentPhase);
  if (currentIndex > 0) {
    const previousPhase = history[currentIndex - 1];
    
    // Update current phase
    session.currentPhase = previousPhase;
    
    // Update navigation state
    this.updateNavigationState();
    
    // Save
    this.stateManager.saveToLocalStorage();
    
    console.log(`⬅️ Navigated back to: ${previousPhase}`);
    return true;
  }
  
  return false;
}
  
  /**
   * Can navigate to phase?
   */
  canNavigateTo(phase: PhaseType): boolean {
  const session = this.stateManager.getSession();
  
  // Can always go to initial
  if (phase === 'initial') return true;

  // Can go to any phase in history
  if (session.navigation.history.includes(phase)) {
    return true;
  }

  // Map phase names to session.phases keys
  const phaseKeyMap: Record<string, string> = {
    'logo_prefs': 'logoPreferences',
    'tagline_prefs': 'taglinePreferences',
    'names': 'names',
    'domains': 'domains',
    'logos': 'logos',
    'taglines': 'taglines'
  };
  
  const phaseKey = (phaseKeyMap[phase] || phase) as keyof typeof session.phases;
  const phaseData = session.phases[phaseKey];
  
  // Can go to phases that are completed or in progress
  if (phaseData && (phaseData.status === 'completed' || phaseData.status === 'in_progress')) {
    return true;
  }
  
  // Define phase dependencies
  const dependencies: { [key in PhaseType]?: string[] } = {
    'names': [],
    'domains': ['names'],
    'logo_prefs': ['names'],
    'logos': ['logoPreferences'],
    'tagline_prefs': ['names'],
    'taglines': ['taglinePreferences'],
    'complete': []
  };
  
  const required = dependencies[phase] || [];
  
  // Check all dependencies are completed
  for (const dep of required) {
    const depData = session.phases[dep as keyof typeof session.phases];
    if (depData && depData.status !== 'completed' && depData.status !== 'skipped') {
      return false;
    }
  }
  
  // Check if service is selected
  const serviceMap: { [key in PhaseType]?: string } = {
    'names': 'name',
    'domains': 'domain',
    'logo_prefs': 'logo',
    'logos': 'logo',
    'tagline_prefs': 'tagline',
    'taglines': 'tagline'
  };
  
  const requiredService = serviceMap[phase];
  if (requiredService && !session.input.selectedServices.includes(requiredService)) {
    return false;
  }
  
  return true;
  }
  
  /**
   * Update navigation state
   */
  private updateNavigationState(): void {
    const session = this.stateManager.getSession();
    const history = session.navigation.history;
    
    session.navigation.canGoBack = history.length > 1;
    session.navigation.canGoForward = false; // Not implemented yet
    session.navigation.nextPhase = this.determineNextPhase();
  }
  
  /**
   * Determine next available phase
   */
  determineNextPhase(): PhaseType | null {
    const session = this.stateManager.getSession();
    
    const phaseOrder: PhaseType[] = [
      'initial',
      'names',
      'domains',
      'logo_prefs',
      'logos',
      'tagline_prefs',
      'taglines',
      'complete'
    ];
    
    const currentIndex = phaseOrder.indexOf(session.currentPhase);
    
    // Find next phase that's selected in services
    for (let i = currentIndex + 1; i < phaseOrder.length; i++) {
      const phase = phaseOrder[i];
      
      if (phase === 'complete') return 'complete';
      
      // Check if this phase's service is selected
      if (this.canNavigateTo(phase)) {
        return phase;
      }
    }
    
    return 'complete';
  }
  
  /**
   * Get next phase based on current selections
   */
  getNextPhaseFromInitial(): PhaseType {
    const session = this.stateManager.getSession();
    
    // If user said they don't have a name and selected name service
    if (!session.input.hasBusinessName && session.input.selectedServices.includes('name')) {
      return 'names';
    }
    
    // If domain selected (and we have or will skip name)
    if (session.input.selectedServices.includes('domain')) {
      return 'domains';
    }
    
    // If logo selected
    if (session.input.selectedServices.includes('logo')) {
      return 'logo_prefs';
    }
    
    // If tagline selected
    if (session.input.selectedServices.includes('tagline')) {
      return 'tagline_prefs';
    }
    
    // Shouldn't happen, but fallback
    return 'complete';
  }
  
  /**
   * Will changing this phase affect later choices?
   */
  willAffectLaterChoices(phase: PhaseType): boolean {
    const session = this.stateManager.getSession();
    
    const affectedBy: { [key in PhaseType]?: (keyof typeof session.phases)[] } = {
      'names': ['domains', 'logos', 'taglines'],
      'logo_prefs': ['logos'],
      'tagline_prefs': ['taglines']
    };
    
    const affected = affectedBy[phase] || [];
    
    // Check if any affected phases are completed
    for (const affectedPhase of affected) {
      if (session.phases[affectedPhase]?.status === 'completed') {
        return true;
      }
    }
    
    return false;
  }
  
  /**
   * Reset phases affected by going back
   */
  resetAffectedPhases(phase: PhaseType): void {
    const session = this.stateManager.getSession();
    
    const affectedBy: { [key in PhaseType]?: (keyof typeof session.phases)[] } = {
      'names': ['domains', 'logos', 'taglines'],
      'logo_prefs': ['logos'],
      'tagline_prefs': ['taglines']
    };
    
    const affected = affectedBy[phase] || [];
    
    // Reset affected phases
    for (const affectedPhase of affected) {
      this.stateManager.updatePhase(affectedPhase, {
        status: 'not_started',
        startedAt: null,
        completedAt: null
      } as any);
    }
    
    console.log(`🔄 Reset phases affected by changing ${phase}:`, affected);
  }
}