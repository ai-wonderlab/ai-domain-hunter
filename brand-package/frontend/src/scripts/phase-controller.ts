/**
 * Phase Controller - Handles API calls and workflow logic
 */

import { StateManager } from './state-manager';
import { NavigationManager } from './navigation-manager';
import { apiClient } from './api-client';
import type { PhaseType } from './state-manager';

export class PhaseController {
  private stateManager: StateManager;
  private navigationManager: NavigationManager;
  
  // DOMAIN FIX: Track what business name was used for domain generation
  private lastDomainGenerationName: string | null = null;
  
  // NAMES FIX: Track what description was used for names generation
  private lastNamesGenerationDesc: string | null = null;
  
  constructor(stateManager: StateManager, navigationManager: NavigationManager) {
    this.stateManager = stateManager;
    this.navigationManager = navigationManager;
  }
  
  /**
   * Get current session
   */
  getSession() {
    return this.stateManager.getSession();
  }
  
  /**
   * Start generation workflow
   */
  async startWorkflow(): Promise<void> {
    // Determine first phase
    const nextPhase = this.navigationManager.getNextPhaseFromInitial();
    
    // Navigate to first phase
    this.navigationManager.goToPhase(nextPhase);
    
    // Trigger the appropriate generation
    switch (nextPhase) {
      case 'names':
        await this.generateNames();
        break;
      case 'domains':
        await this.generateDomains();
        break;
      case 'logo_prefs':
        await this.prepareLogoPreferences();
        break;
      case 'tagline_prefs':
        await this.prepareTaglinePreferences();
        break;
    }
  }
  
  /**
   * Generate business names
   */
  async generateNames(): Promise<void> {
    const session = this.getSession();
    
    // NAMES FIX: Check if description has changed since last generation
    const descChanged = this.lastNamesGenerationDesc !== null && 
                       this.lastNamesGenerationDesc !== session.input.description;
    
    // Clear old names if description changed
    if (descChanged) {
      console.log(`Description changed from "${this.lastNamesGenerationDesc}" to "${session.input.description}". Clearing old names...`);
      this.stateManager.updatePhase('names', {
        generatedOptions: [],
        selectedName: null,
        status: 'not_started'
      });
    }
    
    // Skip if already generated for this description
    if (!descChanged && session.phases.names.generatedOptions.length > 0 && 
        this.lastNamesGenerationDesc === session.input.description) {
      console.log(`Names already generated for this description, skipping...`);
      return;
    }
    
    try {
      this.stateManager.updatePhase('names', { status: 'in_progress' });
      this.showLoader('Generating business names...', 'AI is creating unique options');
      
      // Call API
      const result = await apiClient.generateNames({
        description: session.input.description,
        industry: undefined,
        style: undefined
      });
      
      // STORE THE DESCRIPTION USED FOR THIS GENERATION
      this.lastNamesGenerationDesc = session.input.description;
      
      this.stateManager.updatePhase('names', {
        generatedOptions: result.names,
        generationId: result.generation_id,
        status: 'completed'
      });
      
      this.hideLoader();
      window.dispatchEvent(new CustomEvent('phase-data-ready'));
      
    } catch (error) {
      this.hideLoader();
      this.stateManager.updatePhase('names', { status: 'not_started' });
      this.showError('Failed to generate names: ' + (error as Error).message);
      console.error('Name generation error:', error);
    }
  }
  
  /**
   * Regenerate names with feedback
   */
  async regenerateNames(feedback: string): Promise<void> {
    const session = this.getSession();
    
    try {
      this.showLoader('Regenerating names...', 'Applying your feedback');
      
            // Call same endpoint, append feedback to description
      const result = await apiClient.generateNames({
        description: `${session.input.description}\n\nUser feedback: ${feedback}`,
        industry: undefined,
        style: undefined
      });
      
      // STORE THE DESCRIPTION USED FOR THIS GENERATION
      this.lastNamesGenerationDesc = session.input.description;
      
      // Update state with new results
      this.stateManager.updatePhase('names', {
        generatedOptions: result.names,
        generationId: result.generation_id
      });
      
      this.hideLoader();
      window.dispatchEvent(new CustomEvent('phase-data-ready'));
      
    } catch (error) {
      this.hideLoader();
      this.showError('Failed to regenerate: ' + (error as Error).message);
    }
  }
  
  /**
   * Generate/find available domains - WITH DOMAIN FIX
   */
  async generateDomains(): Promise<void> {
    const session = this.getSession();
    const businessName = session.input.businessName || 
                        session.phases.names.selectedName;

    if (!businessName) {
      throw new Error('No business name selected');
    }

    // DOMAIN FIX: Check if business name has changed since last generation
    const nameHasChanged = this.lastDomainGenerationName !== null && 
                          this.lastDomainGenerationName !== businessName;
    
    // Clear old domains if name changed
    if (nameHasChanged) {
      console.log(`Business name changed from "${this.lastDomainGenerationName}" to "${businessName}". Clearing old domains...`);
      this.stateManager.updatePhase('domains', {
        availableOptions: [],
        checkedVariations: [],
        checkRounds: 0,
        selectedDomain: null,
        status: 'not_started'
      });
    }

    // Skip if already have domains for this name
    if (!nameHasChanged && 
        session.phases.domains.availableOptions.length > 0 &&
        this.lastDomainGenerationName === businessName) {
      console.log(`Domains already generated for "${businessName}", skipping...`);
      return;
    }

    try {
      this.stateManager.updatePhase('domains', { status: 'in_progress' });
      
      const result = await apiClient.generateDomains({
        business_name: businessName,
        description: session.input.description
      });

      console.log('Domain generation result:', result);

      // DOMAIN FIX: Store the name used for this generation
      this.lastDomainGenerationName = businessName;

      this.stateManager.updatePhase('domains', {
        availableOptions: result.results,
        checkedVariations: [],
        checkRounds: result.rounds || 1,
        status: 'completed'
      });
      
      this.hideLoader();
      window.dispatchEvent(new CustomEvent('phase-data-ready'));
      
    } catch (error) {
      console.error('Failed to generate domains:', error);
      this.stateManager.updatePhase('domains', { status: 'not_started' });
      throw error;
    }
  }
  
  /**
   * Regenerate domains with feedback
   */
  async regenerateDomains(feedback: string): Promise<void> {
    const session = this.getSession();
    const businessName = session.input.businessName || 
                        session.phases.names.selectedName;
    
    if (!businessName) {
      throw new Error('No business name selected');
    }
    
    try {
      this.showLoader('Generating new domains...', 'Applying your feedback');
      
      // Get already checked domains to exclude
      const checkedDomains = session.phases.domains.availableOptions.map((d: any) => d.domain);
      
      // Call regenerate endpoint with feedback
      const result = await apiClient.regenerateDomains({
        business_name: businessName,
        description: session.input.description,
        feedback: feedback,
        exclude_domains: checkedDomains
      });
      
      // DOMAIN FIX: Update the last generation name
      this.lastDomainGenerationName = businessName;
      
      // Update state with new results
      this.stateManager.updatePhase('domains', {
        availableOptions: result.results,
        checkedVariations: session.phases.domains.checkedVariations,
        checkRounds: (session.phases.domains.checkRounds || 1) + 1
      });
      
      this.hideLoader();
      window.dispatchEvent(new CustomEvent('phase-data-ready'));
      
    } catch (error) {
      this.hideLoader();
      this.showError('Failed to regenerate: ' + (error as Error).message);
    }
  }
  
  /**
   * Prepare logo preferences (AI suggestions)
   */
  async prepareLogoPreferences(): Promise<void> {
    const session = this.getSession();
    const businessName = this.stateManager.getBusinessName();
    
    try {
      this.stateManager.updatePhase('logoPreferences', { status: 'in_progress' });
      
      this.showLoader('Analyzing your brand...', 'AI is suggesting logo styles and colors');
      
      // Call AI preference analyzer
      const suggestions = await apiClient.analyzePreferences({
        description: session.input.description,
        business_name: businessName || 'Unnamed Business',
        for_type: 'logo'
      });
      
      this.stateManager.updatePhase('logoPreferences', {
        aiSuggestions: suggestions
      });
      
      this.hideLoader();
      window.dispatchEvent(new CustomEvent('phase-data-ready'));
      
    } catch (error) {
      this.hideLoader();
      this.showError('Failed to analyze preferences: ' + (error as Error).message);
      console.error('Logo preferences error:', error);
    }
  }
  
  /**
   * Generate logos
   */
  async generateLogos(): Promise<void> {
    const session = this.stateManager.getSession();
    const businessName = this.stateManager.getBusinessName();
    
    // Check αν έχουμε ήδη logos
    if (session.phases.logos.generatedOptions.length > 0) {
      console.log('Logos already generated, skipping...');
      return;
    }
    
    if (!businessName) {
      this.showError('Business name is required for logo generation');
      return;
    }
    
    // Πάρε τα preferences από το logoPreferences phase
    const logoPrefs = session.phases.logoPreferences;
    const style = logoPrefs?.userChoice?.style || logoPrefs?.aiSuggestions?.style || 'modern';
    const userColors = logoPrefs?.userChoice?.colors || [];
    const aiColors = logoPrefs?.aiSuggestions?.colors || [];
    
    // Convert ColorSuggestion[] to string[] if needed
    const colors = userColors.length > 0 
      ? (typeof userColors[0] === 'string' ? userColors : userColors.map((c: any) => c.hex || c.color || c))
      : (aiColors.length > 0 
        ? (typeof aiColors[0] === 'string' ? aiColors : aiColors.map((c: any) => c.hex || c.color || c))
        : []);
    
    this.showLoader('Generating logo concepts...');
    
    try {
      const result = await apiClient.generateLogos({
        business_name: businessName,
        description: session.input.description,
        style: style,
        colors: colors
      });
      
      this.stateManager.updatePhase('logos', {
        status: 'completed',
        generatedOptions: result.logos
      });
      
      this.hideLoader();
      window.dispatchEvent(new CustomEvent('phase-changed', { 
        detail: { phase: 'logos' } 
      }));
    } catch (error) {
      console.error('Logo generation error:', error);
      this.showError('Failed to generate logos: ' + (error as Error).message);
      this.hideLoader();
    }
  }
  
  /**
   * Prepare tagline preferences
   */
  async prepareTaglinePreferences(): Promise<void> {
    const session = this.getSession();
    const businessName = this.stateManager.getBusinessName();
    
    try {
      this.stateManager.updatePhase('taglinePreferences', { status: 'in_progress' });
      
      this.showLoader('Analyzing your brand voice...');
      
      const suggestions = await apiClient.analyzePreferences({
        description: session.input.description,
        business_name: businessName || 'Unnamed Business',
        for_type: 'tagline'
      });
      
      this.stateManager.updatePhase('taglinePreferences', {
        aiSuggestions: suggestions
      });
      
      this.hideLoader();
      window.dispatchEvent(new CustomEvent('phase-data-ready'));
      
    } catch (error) {
      this.hideLoader();
      this.showError('Failed to analyze preferences: ' + (error as Error).message);
      console.error('Tagline preferences error:', error);
    }
  }
  
  /**
   * Generate taglines
   */
  async generateTaglines(): Promise<void> {
    const session = this.getSession();
    const businessName = this.stateManager.getBusinessName();
    const prefs = session.phases.taglinePreferences.userChoice;
    
    if (!businessName) {
      this.showError('Business name is required for tagline generation');
      return;
    }
    
    try {
      this.stateManager.updatePhase('taglines', { status: 'in_progress' });
      
      this.showLoader(`Creating taglines for "${businessName}"...`);
      
      const result = await apiClient.generateTaglines({
        description: session.input.description,
        business_name: businessName,
        tone: prefs?.tone
      });
      
      this.stateManager.updatePhase('taglines', {
        generatedOptions: result.taglines,
        generationId: result.generation_id
      });
      
      this.hideLoader();
      window.dispatchEvent(new CustomEvent('phase-data-ready'));
      
    } catch (error) {
      this.hideLoader();
      this.showError('Failed to generate taglines: ' + (error as Error).message);
      console.error('Tagline generation error:', error);
    }
  }
  
  /**
   * Navigate to a specific phase
   */
  async navigateToPhase(phase: PhaseType) {
    // DOMAIN FIX: Check if name changed when navigating to domains
    if (phase === 'domains') {
      await this.generateDomains();
    }
    
    this.navigationManager.goToPhase(phase);
  }
  
  /**
   * Load session - WITH DOMAIN FIX
   */
  loadSession(sessionId: string): void {
    this.stateManager.loadFromLocalStorage(sessionId);
    
    // DOMAIN FIX: Reset the domain generation tracker when loading a session
    const session = this.getSession();
    if (session.phases.domains.availableOptions.length > 0) {
      this.lastDomainGenerationName = session.input.businessName || 
                                      session.phases.names.selectedName;
    } else {
      this.lastDomainGenerationName = null;
    }
    
    // NAMES FIX: Reset the names generation tracker when loading a session
    if (session.phases.names.generatedOptions.length > 0) {
      this.lastNamesGenerationDesc = session.input.description;
    } else {
      this.lastNamesGenerationDesc = null;
    }
  }
  
  /**
   * Clear session - WITH DOMAIN FIX
   */
  clearSession(): void {
    this.stateManager.clearSession();
    // DOMAIN FIX: Clear the tracker
    this.lastDomainGenerationName = null;
    // NAMES FIX: Clear the tracker
    this.lastNamesGenerationDesc = null;
  }
  
  /**
   * Show global loader
   */
  private showLoader(message: string, tip?: string): void {
    const loader = document.getElementById('global-loader');
    const loaderMessage = document.getElementById('loader-message');
    const loaderTip = document.getElementById('loader-tip');
    
    if (loader) {
      loader.classList.remove('hidden');
    }
    
    if (loaderMessage) {
      loaderMessage.textContent = message;
    }
    
    if (loaderTip && tip) {
      loaderTip.textContent = `💡 ${tip}`;
    } else if (loaderTip) {
      loaderTip.textContent = '';
    }
  }
  
  /**
   * Hide global loader
   */
  private hideLoader(): void {
    const loader = document.getElementById('global-loader');
    if (loader) {
      loader.classList.add('hidden');
    }
  }
  
  /**
   * Select a logo
   */
  selectLogo(logoId: string): void {
    const session = this.stateManager.getSession();
    const logo = session.phases.logos.generatedOptions.find(l => l.id === logoId);
    
    if (!logo) {
      this.showError('Logo not found');
      return;
    }
    
    this.stateManager.updatePhase('logos', {
      selectedLogo: logo
    });
    
    // Update UI
    document.querySelectorAll('.logo-card').forEach(card => {
      card.classList.remove('selected');
    });
    document.querySelector(`[data-logo-id="${logoId}"]`)?.classList.add('selected');
    
    // Enable continue button
    const continueBtn = document.getElementById('continue-btn');
    if (continueBtn) {
      continueBtn.removeAttribute('disabled');
    }
    
    // Update preview
    window.dispatchEvent(new CustomEvent('phase-data-ready'));
  }

    /**
   * Download a logo
   */
  async downloadLogo(logoId: string): Promise<void> {
    const session = this.stateManager.getSession();
    const logo = session.phases.logos.generatedOptions.find((l: any) => l.id === logoId);
    
    if (!logo) {
      this.showError('Logo not found');
      return;
    }
    
    // Get the best quality URL
    const url = logo.urls?.png || logo.urls?.jpg || logo.urls?.svg;
    if (!url) {
      this.showError('No download URL available');
      return;
    }
    
    // Create download link
    const link = document.createElement('a');
    link.href = url;
    link.download = `${this.stateManager.getBusinessName()}-logo.png`;
    link.target = '_blank';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    this.showToast('Logo download started', 'success');
  }

  /**
   * Show toast helper
   */
  private showToast(message: string, type: 'success' | 'error' | 'info' = 'info'): void {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type} show`;
    toast.textContent = message;
    document.body.appendChild(toast);
    
    setTimeout(() => {
      toast.classList.remove('show');
      setTimeout(() => toast.remove(), 300);
    }, 3000);
  }
  
  /**
   * Show error toast
   */
  private showError(message: string): void {
    const toast = document.createElement('div');
    toast.className = 'toast toast-error show';
    toast.textContent = message;
    
    const container = document.getElementById('toast-container') || document.body;
    container.appendChild(toast);
    
    setTimeout(() => {
      toast.classList.remove('show');
      setTimeout(() => toast.remove(), 300);
    }, 5000);
  }
}