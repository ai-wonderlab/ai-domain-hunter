/**
 * Phase Controller - Handles API calls and workflow logic
 */

import { StateManager } from './state-manager';
import { NavigationManager } from './navigation-manager';
import { apiClient } from './api-client';

export class PhaseController {
  private stateManager: StateManager;
  private navigationManager: NavigationManager;
  
  constructor(stateManager: StateManager, navigationManager: NavigationManager) {
    this.stateManager = stateManager;
    this.navigationManager = navigationManager;
  }
  
  /**
   * Start generation workflow
   */
  async startWorkflow(): Promise<void> {
    const session = this.stateManager.getSession();
    
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
    const session = this.stateManager.getSession();
    
    try {
      // Mark phase as started
      this.stateManager.startPhase('names');
      
      // Show loading
      this.showLoader('Generating business names...', 'Analyzing thousands of successful brands');
      
      // Call API
      const result = await apiClient.generateNames({
        description: session.input.description,
        industry: undefined,
        style: undefined
      });
      
      // Update state with results
      this.stateManager.updatePhase('names', {
        generatedOptions: result.names,
        generationId: result.generation_id
      });
      
      // Hide loader
      this.hideLoader();
      
      // Trigger re-render
      window.dispatchEvent(new CustomEvent('phase-data-ready'));
      
    } catch (error) {
      this.hideLoader();
      this.showError('Failed to generate names: ' + (error as Error).message);
      console.error('Name generation error:', error);
    }
  }
  
  /**
   * Regenerate names with feedback
   */
  async regenerateNames(feedback: string): Promise<void> {
    const session = this.stateManager.getSession();
    
    try {
      this.showLoader('Regenerating names...', 'Applying your feedback');
      
      // Call same endpoint, append feedback to description
      const result = await apiClient.generateNames({
        description: `${session.input.description}\n\nUser feedback: ${feedback}`,
        industry: undefined,
        style: undefined
      });
      
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
   * Generate/find available domains
   */
  async generateDomains(): Promise<void> {
    const businessName = this.stateManager.getBusinessName();
    const session = this.stateManager.getSession();
    
    if (!businessName) {
      this.showError('Business name is required for domain search');
      return;
    }
    
    try {
      this.stateManager.startPhase('domains');
      
      this.showLoader(
        `Finding domains for "${businessName}"...`,
        'AI is generating variations and checking availability'
      );
      
      // Call NEW AI-powered domain generation
      const result = await apiClient.generateDomains({
        business_name: businessName,
        description: session.input.description
      });
      
      console.log('Domain generation result:', result);
      
      this.stateManager.updatePhase('domains', {
        availableOptions: result.results || [],
        checkedVariations: [],
        checkRounds: result.rounds || 1
      });
      
      this.hideLoader();
      window.dispatchEvent(new CustomEvent('phase-data-ready'));
      
    } catch (error) {
      this.hideLoader();
      this.showError('Failed to generate domains: ' + (error as Error).message);
      console.error('Domain generation error:', error);
    }
  }

  /**
   * Regenerate domains with feedback
   */
  async regenerateDomains(feedback: string): Promise<void> {
    const session = this.stateManager.getSession();
    const businessName = this.stateManager.getBusinessName();
    
    if (!businessName) {
      this.showError('Business name is required');
      return;
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
    const session = this.stateManager.getSession();
    const businessName = this.stateManager.getBusinessName();
    
    try {
      this.stateManager.startPhase('logoPreferences');
      
      this.showLoader('Analyzing your brand...', 'AI is suggesting logo styles and colors');
      
      // Call AI preference analyzer
      const suggestions = await apiClient.analyzePreferences({
        description: session.input.description,
        business_name: businessName || undefined,
        for: 'logo'
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
    const prefs = session.phases.logoPreferences.userChoice;
    
    if (!businessName) {
      this.showError('Business name is required for logo generation');
      return;
    }
    
    if (!prefs) {
      this.showError('Logo preferences are required');
      return;
    }
    
    try {
      this.stateManager.startPhase('logos');
      
      this.showLoader(
        `Creating logo concepts for "${businessName}"...`,
        'This may take 30-60 seconds'
      );
      
      const result = await apiClient.generateLogos({
        description: session.input.description,
        business_name: businessName,
        style: prefs.style,
        colors: prefs.colors
      });
      
      this.stateManager.updatePhase('logos', {
        generatedOptions: result.logos,
        generationId: result.generation_id
      });
      
      this.hideLoader();
      window.dispatchEvent(new CustomEvent('phase-data-ready'));
      
    } catch (error) {
      this.hideLoader();
      this.showError('Failed to generate logos: ' + (error as Error).message);
      console.error('Logo generation error:', error);
    }
  }
  
  /**
   * Prepare tagline preferences
   */
  async prepareTaglinePreferences(): Promise<void> {
    const session = this.stateManager.getSession();
    const businessName = this.stateManager.getBusinessName();
    
    try {
      this.stateManager.startPhase('taglinePreferences');
      
      this.showLoader('Analyzing your brand voice...');
      
      const suggestions = await apiClient.analyzePreferences({
        description: session.input.description,
        business_name: businessName || undefined,
        for: 'tagline'
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
    const session = this.stateManager.getSession();
    const businessName = this.stateManager.getBusinessName();
    const prefs = session.phases.taglinePreferences.userChoice;
    
    if (!businessName) {
      this.showError('Business name is required for tagline generation');
      return;
    }
    
    try {
      this.stateManager.startPhase('taglines');
      
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