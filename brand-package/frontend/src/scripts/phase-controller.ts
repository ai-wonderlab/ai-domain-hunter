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
  
  // Track what was used for last generation
  private lastDomainGenerationName: string | null = null;
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
   * ✅ SAVE CURRENT GENERATION TO HISTORY (Universal Method)
   * Καλείται ΠΑΝΤΑ πριν:
   * - Navigate away from a phase
   * - Regenerate
   * - Change selections
   */
  saveCurrentGenerationToHistory(phase: PhaseType): void {
    const session = this.getSession();
    
    switch (phase) {
      case 'names':
        const namesData = session.phases.names;
        if (namesData.generatedOptions.length > 0) {
          const history = namesData.generationHistory || [];
          
          // Check if already saved (avoid duplicates)
          const alreadySaved = history.some(
            h => h.timestamp > Date.now() - 5000 && 
                 h.names.length === namesData.generatedOptions.length
          );
          
          if (!alreadySaved) {
            history.push({
              timestamp: Date.now(),
              description: namesData.lastGeneratedDescription || session.input.description,
              names: [...namesData.generatedOptions] // Deep copy
            });
            
            console.log(`✅ Saved ${namesData.generatedOptions.length} names to history`);
            
            this.stateManager.updatePhase('names', {
              generationHistory: history
            });
          }
        }
        break;
        
      case 'domains':
        const domainsData = session.phases.domains;
        if (domainsData.availableOptions.length > 0) {
          const history = domainsData.generationHistory || [];
          
          const alreadySaved = history.some(
            h => h.timestamp > Date.now() - 5000 && 
                 h.domains.length === domainsData.availableOptions.length
          );
          
          if (!alreadySaved) {
            history.push({
              timestamp: Date.now(),
              businessName: domainsData.lastGeneratedBusinessName || this.stateManager.getBusinessName() || '',
              domains: [...domainsData.availableOptions]
            });
            
            console.log(`✅ Saved ${domainsData.availableOptions.length} domains to history`);
            
            this.stateManager.updatePhase('domains', {
              generationHistory: history
            });
          }
        }
        break;
        
      case 'logos':
        const logosData = session.phases.logos;
        if (logosData.generatedOptions.length > 0) {
          const history = logosData.generationHistory || [];
          
          const alreadySaved = history.some(
            h => h.timestamp > Date.now() - 5000 && 
                 h.logos.length === logosData.generatedOptions.length
          );
          
          if (!alreadySaved) {
            history.push({
              timestamp: Date.now(),
              preferences: logosData.lastGeneratedPreferences,
              logos: [...logosData.generatedOptions]
            });
            
            console.log(`✅ Saved ${logosData.generatedOptions.length} logos to history`);
            
            this.stateManager.updatePhase('logos', {
              generationHistory: history
            });
          }
        }
        break;
        
      case 'taglines':
        const taglinesData = session.phases.taglines;
        if (taglinesData.generatedOptions.length > 0) {
          const history = taglinesData.generationHistory || [];
          
          const alreadySaved = history.some(
            h => h.timestamp > Date.now() - 5000 && 
                 h.taglines.length === taglinesData.generatedOptions.length
          );
          
          if (!alreadySaved) {
            history.push({
              timestamp: Date.now(),
              preferences: taglinesData.lastGeneratedPreferences,
              taglines: [...taglinesData.generatedOptions]
            });
            
            console.log(`✅ Saved ${taglinesData.generatedOptions.length} taglines to history`);
            
            this.stateManager.updatePhase('taglines', {
              generationHistory: history
            });
          }
        }
        break;
    }
  }
  
  /**
   * Start generation workflow
   */
  async startWorkflow(): Promise<void> {
    const nextPhase = this.navigationManager.getNextPhaseFromInitial();
    this.navigationManager.goToPhase(nextPhase);
    
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
    
    const descChanged = this.lastNamesGenerationDesc !== null && 
                       this.lastNamesGenerationDesc !== session.input.description;
    
    // ✅ SAVE TO HISTORY πριν κάνεις οτιδήποτε
    if (session.phases.names.generatedOptions.length > 0) {
      this.saveCurrentGenerationToHistory('names');
      
      // Αν άλλαξε description, CLEAR current options
      if (descChanged) {
        console.log(`Description changed, clearing current names...`);
        this.stateManager.updatePhase('names', {
          generatedOptions: [],
          selectedName: null,
          status: 'not_started'
        });
      }
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
      
      const result = await apiClient.generateNames({
        description: session.input.description,
        industry: undefined,
        style: undefined
      });
      
      this.lastNamesGenerationDesc = session.input.description;
      
      this.stateManager.updatePhase('names', {
        generatedOptions: result.names,
        generationId: result.generation_id,
        status: 'completed',
        lastGeneratedDescription: session.input.description
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
    
    // ✅ SAVE TO HISTORY FIRST
    this.saveCurrentGenerationToHistory('names');
    
    try {
      this.showLoader('Regenerating names...', 'Applying your feedback');
      
      const result = await apiClient.generateNames({
        description: `${session.input.description}\n\nUser feedback: ${feedback}`,
        industry: undefined,
        style: undefined
      });
      
      this.lastNamesGenerationDesc = session.input.description;
      
      this.stateManager.updatePhase('names', {
        generatedOptions: result.names,
        generationId: result.generation_id,
        lastGeneratedDescription: session.input.description
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
    const session = this.getSession();
    const businessName = session.input.businessName || 
                        session.phases.names.selectedName;

    if (!businessName) {
      throw new Error('No business name selected');
    }

    const nameHasChanged = this.lastDomainGenerationName !== null && 
                          this.lastDomainGenerationName !== businessName;
    
    // ✅ SAVE TO HISTORY πριν κάνεις οτιδήποτε
    if (session.phases.domains.availableOptions.length > 0) {
      this.saveCurrentGenerationToHistory('domains');
      
      // Αν άλλαξε name, CLEAR current options
      if (nameHasChanged) {
        console.log(`Business name changed, clearing current domains...`);
        this.stateManager.updatePhase('domains', {
          availableOptions: [],
          checkedVariations: [],
          checkRounds: 0,
          selectedDomain: null,
          status: 'not_started'
        });
      }
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

      this.lastDomainGenerationName = businessName;

      this.stateManager.updatePhase('domains', {
        availableOptions: result.results,
        checkedVariations: [],
        checkRounds: result.rounds || 1,
        status: 'completed',
        lastGeneratedBusinessName: businessName
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
    
    // ✅ SAVE TO HISTORY FIRST
    this.saveCurrentGenerationToHistory('domains');
    
    try {
      this.showLoader('Generating new domains...', 'Applying your feedback');
      
      const checkedDomains = session.phases.domains.availableOptions.map((d: any) => d.domain);
      
      const result = await apiClient.regenerateDomains({
        business_name: businessName,
        description: session.input.description,
        feedback: feedback,
        exclude_domains: checkedDomains
      });
      
      this.lastDomainGenerationName = businessName;
      
      this.stateManager.updatePhase('domains', {
        availableOptions: result.results,
        checkedVariations: session.phases.domains.checkedVariations,
        checkRounds: (session.phases.domains.checkRounds || 1) + 1,
        lastGeneratedBusinessName: businessName
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
  /**
   * Generate logos
   */
  async generateLogos(): Promise<void> {
    const session = this.stateManager.getSession();
    const businessName = this.stateManager.getBusinessName();
    
    if (!businessName) {
      this.showError('Business name is required');
      return;
    }
    
    const logoPrefs = session.phases.logoPreferences;
    const userChoice = logoPrefs?.userChoice || {};
    
    const selectedStyles = userChoice.styles || [];
    const customStyle = userChoice.customStyle || '';
    const customColors = userChoice.customColors || '';
    const aiText = userChoice.aiText || '';
    
    const currentPreferences = {
      styles: selectedStyles,
      customStyle: customStyle,
      customColors: customColors,
      aiText: aiText
    };
    
    const lastPrefs = session.phases.logos.lastGeneratedPreferences;
    const preferencesChanged = JSON.stringify(currentPreferences) !== JSON.stringify(lastPrefs);
    
    // ✅ If no logos exist, generate
    if (session.phases.logos.generatedOptions.length === 0) {
      console.log('No logos exist, generating...');
    }
    // ✅ If preferences changed, save to history and regenerate
    else if (preferencesChanged) {
      console.log('Preferences changed, regenerating logos...');
      this.saveCurrentGenerationToHistory('logos');
    }
    // ✅ If preferences unchanged and we have logos, SKIP
    else {
      console.log('✓ Preferences unchanged, showing existing logos');
      return;
    }
    
    // Build prompt
    let finalStyle = selectedStyles.join(', ');
    if (customStyle) {
      finalStyle = finalStyle ? `${finalStyle}, ${customStyle}` : customStyle;
    }
    
    const enhancedDescription = [
      session.input.description,
      aiText ? `AI Analysis: ${aiText}` : '',
      customColors ? `Color preferences: ${customColors}` : ''
    ].filter(Boolean).join('\n\n');
    
    this.showLoader('Generating new logo concepts...');
    
    try {
      const result = await apiClient.generateLogos({
        business_name: businessName,
        description: enhancedDescription,
        style: finalStyle || 'modern',
        colors: []
      });
      
      this.stateManager.updatePhase('logos', {
        status: 'completed',
        generatedOptions: result.logos,
        lastGeneratedPreferences: currentPreferences
      });
      
      this.hideLoader();
      window.dispatchEvent(new CustomEvent('phase-changed', { 
        detail: { phase: 'logos' } 
      }));
    } catch (error) {
      console.error('Logo generation error:', error);
      this.showError('Failed to generate logos');
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
    
    const currentPreferences = {
      tone: prefs?.tone || 'professional'
    };
    
    // ✅ SAVE TO HISTORY πριν κάνεις οτιδήποτε
    if (session.phases.taglines.generatedOptions.length > 0) {
      this.saveCurrentGenerationToHistory('taglines');
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
        generationId: result.generation_id,
        lastGeneratedPreferences: currentPreferences
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
    // ✅ SAVE CURRENT PHASE TO HISTORY πριν φύγεις
    const currentPhase = this.getSession().currentPhase;
    if (['names', 'domains', 'logos', 'taglines'].includes(currentPhase)) {
      this.saveCurrentGenerationToHistory(currentPhase as any);
    }
    
    if (phase === 'domains') {
      await this.generateDomains();
    }
    
    this.navigationManager.goToPhase(phase);
  }
  
  /**
   * Load session
   */
  loadSession(sessionId: string): void {
    this.stateManager.loadFromLocalStorage(sessionId);
    
    const session = this.getSession();
    if (session.phases.domains.availableOptions.length > 0) {
      this.lastDomainGenerationName = session.input.businessName || 
                                      session.phases.names.selectedName;
    } else {
      this.lastDomainGenerationName = null;
    }
    
    if (session.phases.names.generatedOptions.length > 0) {
      this.lastNamesGenerationDesc = session.input.description;
    } else {
      this.lastNamesGenerationDesc = null;
    }
  }
  
  /**
   * Clear session
   */
  clearSession(): void {
    this.stateManager.clearSession();
    this.lastDomainGenerationName = null;
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
    
    document.querySelectorAll('.logo-option').forEach(card => {
      card.classList.remove('selected');
    });
    document.querySelector(`[data-logo-id="${logoId}"]`)?.classList.add('selected');
    
    const continueBtn = document.getElementById('continue-btn');
    if (continueBtn) {
      continueBtn.removeAttribute('disabled');
    }
    
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
    
    const url = logo.urls?.png || logo.urls?.jpg || logo.urls?.svg;
    if (!url) {
      this.showError('No download URL available');
      return;
    }
    
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