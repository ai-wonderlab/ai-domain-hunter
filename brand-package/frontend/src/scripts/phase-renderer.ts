/**
 * Phase Renderer - Renders UI for each phase
 */

/**
 * Phase Renderer - Renders UI for each phase
 */

import { StateManager, PhaseType, NameOption, DomainOption } from './state-manager';
import { NavigationManager } from './navigation-manager';
import { PhaseController } from './phase-controller';

export class PhaseRenderer {
  private stateManager: StateManager;
  private navigationManager: NavigationManager;
  private phaseController: PhaseController | null = null;
  private middlePanel: HTMLElement;
  private rightPanel: HTMLElement;
  
  constructor(
    stateManager: StateManager, 
    navigationManager: NavigationManager,
    phaseController?: PhaseController
  ) {
    this.stateManager = stateManager;
    this.navigationManager = navigationManager;
    this.phaseController = phaseController || null;
    
    this.middlePanel = document.querySelector('.middle-panel') as HTMLElement;
    this.rightPanel = document.querySelector('.right-panel') as HTMLElement;
    
    // Listen for data ready events
    window.addEventListener('phase-data-ready', () => {
      const session = this.stateManager.getSession();
      this.renderPhase(session.currentPhase);
    });
  }
  
  /**
   * Render phase based on type
   */
  renderPhase(phase: PhaseType): void {
    console.log(`🎨 Rendering phase: ${phase}`);
    
    switch (phase) {
      case 'initial':
        this.renderInitialForm();
        break;
      case 'names':
        this.renderNamesPhase();
        break;
      case 'domains':
        this.renderDomainsPhase();
        // Wait for DOM to be ready, then add listener
        setTimeout(() => {
            const regenerateDomainsBtn = document.getElementById('regenerate-domains-btn');
            const feedbackTextarea = document.getElementById('domain-feedback') as HTMLTextAreaElement;
            
            if (regenerateDomainsBtn && feedbackTextarea) {
            regenerateDomainsBtn.addEventListener('click', async () => {
                const feedback = feedbackTextarea.value;
                if (!feedback || feedback.trim().length < 5) {
                this.showToast('Please provide specific feedback', 'error');
                return;
                }
                
                // ✅ Add null check for phaseController
                if (this.phaseController) {
                await this.phaseController.regenerateDomains(feedback.trim());
                }
            });
            }
        }, 100);
        break;
      case 'logo_prefs':
        this.renderLogoPreferencesPhase();
        break;
      case 'logos':
        this.renderLogosPhase();
        break;
      case 'tagline_prefs':
        this.renderTaglinePreferencesPhase();
        break;
      case 'taglines':
        this.renderTaglinesPhase();
        break;
      case 'complete':
        this.renderCompletePhase();
        break;
    }
  }
  
  /**
   * Render Initial Form
   */
  private renderInitialForm(): void {
    const session = this.stateManager.getSession();
    
    this.middlePanel.innerHTML = `
      <div class="initial-form fade-in">
        <div class="form-header">
          <h1>✨ Create Your Brand Identity</h1>
          <p>Tell us about your business and we'll generate everything you need</p>
        </div>
        
        <div class="form-divider"></div>
        
        <!-- Section 1: Description -->
        <div class="form-section">
          <label class="section-label">
            <span class="section-number">1</span>
            <span class="section-title">Describe Your Business</span>
          </label>
          
          <textarea 
            id="business-description"
            class="glass-input description-input"
            placeholder="Tell us about your business idea, target audience, what makes you unique..."
            rows="6"
          >${session.input.description}</textarea>
          
          <div class="input-tip">
            💡 Tip: More detail = better results. Include your target audience, key features, and brand personality.
          </div>
        </div>
        
        <div class="form-divider"></div>
        
        <!-- Section 2: Business Name -->
        <div class="form-section">
          <label class="section-label">
            <span class="section-number">2</span>
            <span class="section-title">Business Name</span>
          </label>
          
          <div class="radio-group">
            <label class="radio-option ${!session.input.hasBusinessName ? 'selected' : ''}">
              <input 
                type="radio" 
                name="name-option" 
                value="generate" 
                ${!session.input.hasBusinessName ? 'checked' : ''}
              >
              <div class="radio-content">
                <span class="radio-icon">✨</span>
                <div>
                  <strong>Generate a name for me</strong>
                  <small>We'll create 10 unique name suggestions</small>
                </div>
              </div>
            </label>
            
            <label class="radio-option ${session.input.hasBusinessName ? 'selected' : ''}">
              <input 
                type="radio" 
                name="name-option" 
                value="have"
                ${session.input.hasBusinessName ? 'checked' : ''}
              >
              <div class="radio-content">
                <span class="radio-icon">📝</span>
                <div>
                  <strong>I already have a name</strong>
                  <small>Enter your business name below</small>
                </div>
              </div>
            </label>
          </div>
          
          <input 
            type="text" 
            id="existing-business-name" 
            class="glass-input conditional-input ${session.input.hasBusinessName ? '' : 'hidden'}"
            placeholder="Enter your business name"
            value="${session.input.businessName || ''}"
          >
        </div>
        
        <div class="form-divider"></div>
        
        <!-- Section 3: Services -->
        <div class="form-section">
          <label class="section-label">
            <span class="section-number">3</span>
            <span class="section-title">What Should We Create?</span>
          </label>
          
          <p class="section-description">Select all that apply (we recommend all for a complete brand package)</p>
          
          <div class="checkbox-grid">
            <label class="checkbox-option ${session.input.selectedServices.includes('name') ? 'checked' : ''}">
              <input 
                type="checkbox" 
                data-service="name"
                ${session.input.selectedServices.includes('name') ? 'checked' : ''}
              >
              <div class="option-content">
                <span class="option-icon">✨</span>
                <div class="option-text">
                  <strong>Business Names</strong>
                  <small>10 unique name suggestions</small>
                </div>
              </div>
            </label>
            
            <label class="checkbox-option ${session.input.selectedServices.includes('logo') ? 'checked' : ''}">
              <input 
                type="checkbox" 
                data-service="logo"
                ${session.input.selectedServices.includes('logo') ? 'checked' : ''}
              >
              <div class="option-content">
                <span class="option-icon">🎨</span>
                <div class="option-text">
                  <strong>Logo Designs</strong>
                  <small>3 professional logo concepts</small>
                </div>
              </div>
            </label>
            
            <label class="checkbox-option ${session.input.selectedServices.includes('color') ? 'checked' : ''}">
              <input 
                type="checkbox" 
                data-service="color"
                ${session.input.selectedServices.includes('color') ? 'checked' : ''}
              >
              <div class="option-content">
                <span class="option-icon">🌈</span>
                <div class="option-text">
                  <strong>Color Palettes</strong>
                  <small>3 harmonious color schemes</small>
                </div>
              </div>
            </label>
            
            <label class="checkbox-option ${session.input.selectedServices.includes('tagline') ? 'checked' : ''}">
              <input 
                type="checkbox" 
                data-service="tagline"
                ${session.input.selectedServices.includes('tagline') ? 'checked' : ''}
              >
              <div class="option-content">
                <span class="option-icon">💬</span>
                <div class="option-text">
                  <strong>Taglines</strong>
                  <small>5 memorable taglines</small>
                </div>
              </div>
            </label>
            
            <label class="checkbox-option ${session.input.selectedServices.includes('domain') ? 'checked' : ''}">
              <input 
                type="checkbox" 
                data-service="domain"
                ${session.input.selectedServices.includes('domain') ? 'checked' : ''}
              >
              <div class="option-content">
                <span class="option-icon">🌐</span>
                <div class="option-text">
                  <strong>Domain Check</strong>
                  <small>10 available .com/.ai domains</small>
                </div>
              </div>
            </label>
          </div>
        </div>
        
        <div class="form-divider"></div>
        
        <!-- Actions -->
        <div class="form-actions">
          <button id="start-generation-btn" class="primary-btn gradient-btn">
            ✨ Start Generation →
          </button>
        </div>
      </div>
    `;
    
    // Setup event listeners
    this.setupInitialFormListeners();
    
    // Clear right panel or show welcome
    this.rightPanel.innerHTML = `
      <div class="preview-placeholder">
        <div class="placeholder-icon">🎨</div>
        <h3>Your Brand Preview</h3>
        <p>Fill out the form to start creating your brand identity</p>
      </div>
    `;
  }
  
  /**
   * Setup Initial Form Event Listeners
   */
  private setupInitialFormListeners(): void {
    const session = this.stateManager.getSession();
    
    // Description input
    const descriptionInput = document.getElementById('business-description') as HTMLTextAreaElement;
    descriptionInput?.addEventListener('input', (e) => {
      this.stateManager.updateInput({
        description: (e.target as HTMLTextAreaElement).value
      });
    });
    
    // Radio buttons for name option
    const radioButtons = document.querySelectorAll('input[name="name-option"]');
    radioButtons.forEach(radio => {
      radio.addEventListener('change', (e) => {
        const value = (e.target as HTMLInputElement).value;
        const hasName = value === 'have';
        
        this.stateManager.updateInput({ hasBusinessName: hasName });
        
        // Show/hide name input
        const nameInput = document.getElementById('existing-business-name');
        if (nameInput) {
          if (hasName) {
            nameInput.classList.remove('hidden');
          } else {
            nameInput.classList.add('hidden');
          }
        }
        
        // Update radio option styling
        document.querySelectorAll('.radio-option').forEach(opt => {
          opt.classList.remove('selected');
        });
        (e.target as HTMLInputElement).closest('.radio-option')?.classList.add('selected');
      });
    });
    
    // Business name input
    const nameInput = document.getElementById('existing-business-name') as HTMLInputElement;
    nameInput?.addEventListener('input', (e) => {
      this.stateManager.updateInput({
        businessName: (e.target as HTMLInputElement).value
      });
    });
    
    // Service checkboxes
    const checkboxes = document.querySelectorAll('.checkbox-option input[type="checkbox"]');
    checkboxes.forEach(checkbox => {
      checkbox.addEventListener('change', (e) => {
        const input = e.target as HTMLInputElement;
        const service = input.getAttribute('data-service');
        
        if (!service) return;
        
        let services = [...session.input.selectedServices];
        
        if (input.checked) {
          if (!services.includes(service)) {
            services.push(service);
          }
          input.closest('.checkbox-option')?.classList.add('checked');
        } else {
          services = services.filter(s => s !== service);
          input.closest('.checkbox-option')?.classList.remove('checked');
        }
        
        this.stateManager.updateInput({ selectedServices: services });
      });
    });
    
    // Start generation button
    const startBtn = document.getElementById('start-generation-btn');
    startBtn?.addEventListener('click', () => {
      this.onStartGeneration();
    });
  }
  
  /**
   * Handle start generation
   */
  private onStartGeneration(): void {
    const errors = this.validateInitialForm();
    
    if (errors.length > 0) {
      this.showErrors(errors);
      return;
    }
    
    // Save state
    this.stateManager.saveToLocalStorage();
    
    // Determine first phase
    const nextPhase = this.navigationManager.getNextPhaseFromInitial();
    
    // Navigate
    this.navigationManager.goToPhase(nextPhase);
    
    // Trigger render (this will be handled by the main app)
    window.dispatchEvent(new CustomEvent('phase-changed', { 
      detail: { phase: nextPhase } 
    }));
  }
  
  /**
   * Validate initial form
   */
  private validateInitialForm(): Array<{ field: string; message: string }> {
    const errors: Array<{ field: string; message: string }> = [];
    const session = this.stateManager.getSession();
    
    // Description required, min 20 chars
    if (!session.input.description || session.input.description.length < 20) {
      errors.push({
        field: 'description',
        message: 'Please provide a detailed description (at least 20 characters)'
      });
    }
    
    // If user said they have a name, validate name input
    if (session.input.hasBusinessName && !session.input.businessName) {
      errors.push({
        field: 'businessName',
        message: 'Please enter your business name'
      });
    }
    
    // At least one service must be selected
    if (session.input.selectedServices.length === 0) {
      errors.push({
        field: 'services',
        message: 'Please select at least one service'
      });
    }
    
    return errors;
  }
  
  /**
   * Render Names Phase
   */
  private renderNamesPhase(): void {
    const session = this.stateManager.getSession();
    const names = session.phases.names.generatedOptions;
    
    this.middlePanel.innerHTML = `
      <div class="phase-content fade-in">
        <div class="phase-header">
          <h2>🎯 Choose Your Business Name</h2>
          <p>We generated 10 unique names based on your description</p>
        </div>
        
        <div class="phase-divider"></div>
        
        <div id="names-list" class="names-list">
          ${names.length === 0 ? this.renderLoadingState('Generating names...') : this.renderNameOptions(names)}
        </div>
        
        <div class="phase-divider"></div>
        
        <div class="feedback-section">
          <label>💡 Not happy? Tell us what to change:</label>
          <textarea 
            id="name-feedback"
            class="glass-input"
            placeholder="E.g., 'Make it shorter', 'More playful', 'Include the word food'..."
            rows="3"
          ></textarea>
          <button id="regenerate-names-btn" class="secondary-btn">
            🔄 Regenerate Names
          </button>
        </div>
        
        <div class="phase-divider"></div>
        
        <div class="phase-actions">
          <button id="back-btn" class="secondary-btn">
            ← Back
          </button>
          <button id="continue-btn" class="primary-btn gradient-btn" ${!session.phases.names.selectedName ? 'disabled' : ''}>
            Continue with Selected →
          </button>
        </div>
      </div>
    `;
    
    this.setupNamesPhaseListeners();
    this.updatePreviewPanel();
  }
  
  /**
   * Render name options
   */
  private renderNameOptions(names: NameOption[]): string {
    const session = this.stateManager.getSession();
    const selected = session.phases.names.selectedName;
    
    return names.map(name => `
      <label class="name-option ${selected === name.name ? 'selected' : ''}" data-name="${name.name}">
        <input 
          type="radio" 
          name="selected-name" 
          value="${name.name}"
          ${selected === name.name ? 'checked' : ''}
        >
        <div class="name-content">
          <div class="name-header">
            <h3>${this.escapeHtml(name.name)}</h3>
            <span class="name-score">
              <span class="score-value">${name.score.toFixed(1)}</span>/10
            </span>
          </div>
          <p class="name-reasoning">${this.escapeHtml(name.reasoning)}</p>
          <div class="name-meta">
            ${name.style ? `<span class="meta-tag">${this.escapeHtml(name.style)}</span>` : ''}
            ${name.memorability ? `<span class="meta-tag">Memorability: ${name.memorability}/10</span>` : ''}
          </div>
        </div>
      </label>
    `).join('');
  }
  
  /**
   * Setup Names Phase Listeners
   */
  private setupNamesPhaseListeners(): void {
    // Name selection
    const nameOptions = document.querySelectorAll('.name-option');
    nameOptions.forEach(option => {
      option.addEventListener('click', (e) => {
        const name = (e.currentTarget as HTMLElement).getAttribute('data-name');
        if (!name) return;
        
        // Update state
        this.stateManager.updatePhase('names', {
          selectedName: name
        });
        
        // Update UI
        nameOptions.forEach(opt => opt.classList.remove('selected'));
        (e.currentTarget as HTMLElement).classList.add('selected');
        
        // Enable continue button
        const continueBtn = document.getElementById('continue-btn');
        continueBtn?.removeAttribute('disabled');
        
        // Update preview
        this.updatePreviewPanel();
      });
    });
    
    // Regenerate names button
    const regenerateBtn = document.getElementById('regenerate-names-btn');
    regenerateBtn?.addEventListener('click', async () => {
      const feedbackTextarea = document.getElementById('name-feedback') as HTMLTextAreaElement;
      const feedback = feedbackTextarea?.value;
      
      if (!feedback || feedback.length < 10) {
        this.showToast('Please provide feedback on what to change (at least 10 characters)', 'error');
        return;
      }
      
      if (this.phaseController) {
        await this.phaseController.regenerateNames(feedback);
      }
    });
    
    // Back button
    const backBtn = document.getElementById('back-btn');
    backBtn?.addEventListener('click', () => {
      this.navigationManager.goBack();
      window.dispatchEvent(new CustomEvent('phase-changed'));
    });
    
    // Continue button
    const continueBtn = document.getElementById('continue-btn');
    continueBtn?.addEventListener('click', () => {
      this.stateManager.completePhase('names');
      const nextPhase = this.navigationManager.determineNextPhase();
      if (nextPhase) {
        this.navigationManager.goToPhase(nextPhase);
        window.dispatchEvent(new CustomEvent('phase-changed', { detail: { phase: nextPhase } }));
      }
    });
  }
  
  /**
   * Render Domains Phase
   */
    private renderDomainsPhase(): void {
    const session = this.stateManager.getSession();
    const domains = session.phases.domains.availableOptions;
    
    this.middlePanel.innerHTML = `
        <div class="phase-content fade-in">
        <div class="phase-header">
            <h2>🌐 Choose Your Domain</h2>
            <p>We found 10 available domains for "${this.escapeHtml(this.stateManager.getBusinessName() || '')}"</p>
        </div>
        
        <div class="phase-divider"></div>
        
        <div id="domains-list" class="domains-list">
            ${domains.length === 0 ? this.renderLoadingState('Finding available domains...') : this.renderDomainOptions(domains)}
        </div>
        
        ${domains.length > 0 ? `
            <div class="domains-info">
            ℹ️ Checked ${session.phases.domains.checkedVariations.length} variations across ${session.phases.domains.checkRounds} rounds to find these 10
            </div>
        ` : ''}
        
        <!-- ✅ ADD THIS REGENERATE SECTION HERE -->
        ${domains.length > 0 ? `
            <div class="regenerate-section">
            <h3>Not finding what you need?</h3>
            <p>Tell us what you're looking for and we'll generate new options</p>
            <textarea 
                id="domain-feedback" 
                placeholder="E.g., 'shorter names', 'include tech keywords', 'more professional', 'easier to spell'..."
                rows="3"
            ></textarea>
            <button id="regenerate-domains-btn" class="secondary-btn">
                🔄 Generate New Options
            </button>
            </div>
        ` : ''}
        
        <div class="phase-divider"></div>
        
        <div class="phase-actions">
            <button id="back-btn" class="secondary-btn">
            ← Back to Names
            </button>
            <button id="continue-btn" class="primary-btn gradient-btn" ${!session.phases.domains.selectedDomain ? 'disabled' : ''}>
            Continue →
            </button>
        </div>
        </div>
    `;
    
    this.setupDomainsPhaseListeners();
    this.updatePreviewPanel();
    }
  
  /**
   * Render domain options
   */
  private renderDomainOptions(domains: DomainOption[]): string {
    const session = this.stateManager.getSession();
    const selected = session.phases.domains.selectedDomain;
    
    // Separate premium (.com, .ai) from alternatives
    const premium = domains.filter(d => d.domain.endsWith('.com') || d.domain.endsWith('.ai'));
    const alternatives = domains.filter(d => !d.domain.endsWith('.com') && !d.domain.endsWith('.ai'));
    
    let html = '';
    
    // Premium section
    if (premium.length > 0) {
      html += `
        <div class="premium-section">
          <h3 class="section-title">⭐ PREMIUM (.com & .ai)</h3>
          ${premium.map(domain => this.renderDomainCard(domain, selected)).join('')}
        </div>
      `;
    }
    
    // Alternatives section
    if (alternatives.length > 0) {
      html += `
        <div class="alternatives-section">
          <h3 class="section-title">ALTERNATIVES</h3>
          ${alternatives.map(domain => this.renderDomainCard(domain, selected)).join('')}
        </div>
      `;
    }
    
    return html;
  }
  
  /**
   * Render single domain card
   */
  private renderDomainCard(domain: DomainOption, selected: string | null): string {
    const isSelected = selected === domain.domain;
    const isPremium = domain.domain.endsWith('.com') || domain.domain.endsWith('.ai');
    
    return `
      <label class="domain-option ${isSelected ? 'selected' : ''}" data-domain="${domain.domain}">
        <input 
          type="radio" 
          name="selected-domain" 
          value="${domain.domain}"
          ${isSelected ? 'checked' : ''}
        >
        <div class="domain-content">
          <div class="domain-header">
            <h4>${this.escapeHtml(domain.domain)}</h4>
            ${isPremium && domain.domain.endsWith('.com') ? '<span class="premium-badge">Most Popular</span>' : ''}
            ${isPremium && domain.domain.endsWith('.ai') ? '<span class="premium-badge ai-badge">Premium</span>' : ''}
          </div>
          <div class="domain-footer">
            <span class="domain-price">💰 ${this.escapeHtml(domain.price)}</span>
            ${domain.registrar_link ? `
              <a href="${this.escapeHtml(domain.registrar_link)}" target="_blank" class="registrar-link">
                View on Namecheap →
              </a>
            ` : ''}
          </div>
        </div>
      </label>
    `;
  }
  
  /**
   * Setup Domains Phase Listeners
   */
  private setupDomainsPhaseListeners(): void {
    // Domain selection
    const domainOptions = document.querySelectorAll('.domain-option');
    domainOptions.forEach(option => {
      option.addEventListener('click', (e) => {
        const domain = (e.currentTarget as HTMLElement).getAttribute('data-domain');
        if (!domain) return;
        
        // Update state
        this.stateManager.updatePhase('domains', {
          selectedDomain: domain
        });
        
        // Update UI
        domainOptions.forEach(opt => opt.classList.remove('selected'));
        (e.currentTarget as HTMLElement).classList.add('selected');
        
        // Enable continue button
        const continueBtn = document.getElementById('continue-btn');
        continueBtn?.removeAttribute('disabled');
        
        // Update preview
        this.updatePreviewPanel();
      });
    });
    
    // Back button
    const backBtn = document.getElementById('back-btn');
    backBtn?.addEventListener('click', () => {
      this.navigationManager.goBack();
      window.dispatchEvent(new CustomEvent('phase-changed'));
    });
    
    // Continue button
    const continueBtn = document.getElementById('continue-btn');
    continueBtn?.addEventListener('click', () => {
      this.stateManager.completePhase('domains');
      const nextPhase = this.navigationManager.determineNextPhase();
      if (nextPhase) {
        this.navigationManager.goToPhase(nextPhase);
        window.dispatchEvent(new CustomEvent('phase-changed', { detail: { phase: nextPhase } }));
      }
    });
  }
  
  /**
   * Render loading state
   */
  private renderLoadingState(message: string): string {
    return `
      <div class="loading-state">
        <div class="loading-spinner"></div>
        <p>${this.escapeHtml(message)}</p>
      </div>
    `;
  }
  
  /**
   * Update preview panel (right panel)
   */
  private updatePreviewPanel(): void {
    const session = this.stateManager.getSession();
    const businessName = this.stateManager.getBusinessName();
    const domain = session.phases.domains.selectedDomain;
    
    let html = '<div class="preview-content">';
    
    html += '<h3>Your Brand Preview</h3>';
    
    if (businessName) {
      html += `
        <div class="preview-item">
          <label>Business Name:</label>
          <div class="preview-value brand-name">${this.escapeHtml(businessName)}</div>
        </div>
      `;
    }
    
    if (domain) {
      html += `
        <div class="preview-item">
          <label>Domain:</label>
          <div class="preview-value domain-name">${this.escapeHtml(domain)}</div>
        </div>
      `;
    }
    
    // Add tips based on selections
    if (domain) {
      if (domain.endsWith('.com')) {
        html += `
          <div class="preview-tip">
            <strong>💡 Why .com?</strong>
            <ul>
              <li>Most trusted extension</li>
              <li>Best for SEO</li>
              <li>Universal recognition</li>
            </ul>
          </div>
        `;
      } else if (domain.endsWith('.ai')) {
        html += `
          <div class="preview-tip">
            <strong>💡 Why .ai?</strong>
            <ul>
              <li>Premium positioning</li>
              <li>Perfect for AI/tech</li>
              <li>Memorable & modern</li>
            </ul>
          </div>
        `;
      }
    }
    
    html += '</div>';
    
    this.rightPanel.innerHTML = html;
  }
  
private renderLogoPreferencesPhase(): void {
  const session = this.stateManager.getSession();
  const businessName = session.input.businessName || session.phases.names.selectedName;
  
  const html = `
    <div class="phase-content">
      <div class="phase-header">
        <h2>Logo Style & Colors</h2>
        <p>Define the visual identity for <strong>${businessName}</strong></p>
      </div>
      
      <div class="phase-divider"></div>
      
      <!-- AI Recommendation -->
      <div class="ai-section">
        <div class="section-header">
          <h3>AI Analysis</h3>
          <small class="text-secondary">You can modify or add your own details</small>
        </div>
        <textarea 
          id="ai-recommendation-text" 
          class="glass-input"
          rows="4"
          placeholder="Analyzing your brand..."
        ></textarea>
      </div>
      
      <div class="phase-divider"></div>
      
      <!-- Logo Styles Grid -->
      <div class="style-section">
        <h3>Select Style</h3>
        <div class="section-spacing"></div>
        <div class="style-grid">
          ${['Modern', 'Classic', 'Playful', 'Minimalist', 'Bold', 'Elegant'].map(style => `
            <div class="style-box" data-style="${style.toLowerCase()}">
              <input type="checkbox" name="logo-style" value="${style.toLowerCase()}" id="style-${style.toLowerCase()}">
              <span class="style-name">${style}</span>
              <span class="style-desc">${this.getCompactDescription(style)}</span>
            </div>
          `).join('')}
        </div>
        <div class="custom-input-section">
          <input 
            type="text" 
            id="custom-style-input" 
            class="glass-input compact-input"
            placeholder="Describe your preferred style (optional)"
          >
        </div>
      </div>
      
      <div class="phase-divider"></div>
      
      <!-- Color Palettes -->
      <div class="color-section">
        <h3>Select Colors</h3>
        <div class="section-spacing"></div>
        <div class="palette-grid" id="palette-grid">
          <div class="loader-text">Loading palettes...</div>
        </div>
        <div class="custom-input-section">
          <input 
            type="text" 
            id="custom-colors-input" 
            class="glass-input compact-input"
            placeholder="Describe your color preferences (optional)"
          >
        </div>
      </div>
      
      <div class="phase-divider"></div>
      
      <div class="phase-actions">
        <button id="back-btn" class="secondary-btn">Back</button>
        <button id="generate-logos-btn" class="primary-btn gradient-btn">
          Generate Logos
        </button>
      </div>
    </div>
  `;
  
  this.middlePanel.innerHTML = html;  // THIS CREATES THE style-box ELEMENTS
  this.initializeLogoPreferences();
  this.setupLogoPreferencesListeners();  // THIS FINDS THEM
}

  // Helper functions
  private getStyleOptions() {
    return [
      { value: 'modern', name: 'Modern', desc: 'Clean & Tech' },
      { value: 'classic', name: 'Classic', desc: 'Traditional' },
      { value: 'playful', name: 'Playful', desc: 'Fun & Friendly' },
      { value: 'minimalist', name: 'Minimalist', desc: 'Simple' },
      { value: 'bold', name: 'Bold', desc: 'Strong Impact' },
      { value: 'elegant', name: 'Elegant', desc: 'Sophisticated' }
    ];
  }

  private getGeometricShape(style: string): string {
    const shapes: {[key: string]: string} = {
      'modern': '<svg viewBox="0 0 40 40"><polygon points="20,5 35,15 35,25 20,35 5,25 5,15" fill="currentColor" opacity="0.7"/></svg>',
      'classic': '<svg viewBox="0 0 40 40"><rect x="10" y="10" width="20" height="20" fill="currentColor" opacity="0.7" rx="2"/></svg>',
      'playful': '<svg viewBox="0 0 40 40"><circle cx="20" cy="20" r="12" fill="currentColor" opacity="0.7"/></svg>',
      'minimalist': '<svg viewBox="0 0 40 40"><line x1="10" y1="20" x2="30" y2="20" stroke="currentColor" stroke-width="2"/></svg>',
      'bold': '<svg viewBox="0 0 40 40"><polygon points="20,8 32,32 8,32" fill="currentColor" opacity="0.7"/></svg>',
      'elegant': '<svg viewBox="0 0 40 40"><path d="M20,10 Q30,20 20,30 Q10,20 20,10" fill="currentColor" opacity="0.7"/></svg>'
    };
    return shapes[style] || '';
  }

private getCompactDescription(style: string): string {
  const descriptions: {[key: string]: string} = {
    'Modern': 'Clean & Tech',
    'Classic': 'Traditional',
    'Playful': 'Fun & Friendly',
    'Minimalist': 'Simple',
    'Bold': 'Strong Impact',
    'Elegant': 'Sophisticated'
  };
  return descriptions[style] || '';
}

  // Helper για style icons
  private getStyleIcon(style: string): string {
    const icons: {[key: string]: string} = {
      'modern': '🚀',
      'classic': '🏛️',
      'playful': '🎈',
      'minimalist': '○',
      'bold': '💪',
      'elegant': '✨'
    };
    return icons[style] || '🎨';
  }

  // Helper για style descriptions  
  private getStyleDescription(style: string): string {
    const descriptions: {[key: string]: string} = {
      'modern': 'Clean, tech-forward design',
      'classic': 'Timeless, traditional feel',
      'playful': 'Fun, approachable vibe',
      'minimalist': 'Simple, essential elements',
      'bold': 'Strong, impactful presence',
      'elegant': 'Sophisticated, refined look'
    };
    return descriptions[style] || '';
  }

  // Generate color palette variations
  private generateColorPalettes(baseColors: string[]): any[] {
    return [
      {
        name: 'AI Suggested',
        colors: baseColors
      },
      {
        name: 'Monochrome',
        colors: ['#000000', '#666666', '#CCCCCC', '#FFFFFF']
      },
      {
        name: 'Ocean',
        colors: ['#0077BE', '#00A8E8', '#00C9FF', '#B4E7F8']
      },
      {
        name: 'Forest',
        colors: ['#2D5016', '#73A942', '#AAD576', '#C5E99B']
      },
      {
        name: 'Sunset',
        colors: ['#FF6B35', '#F77825', '#FFB700', '#FFC857']
      },
      {
        name: 'Royal',
        colors: ['#3D1E6D', '#6B46C1', '#9D4EDD', '#C77DFF']
      }
    ];
  }

  // Helper to update state when selections change
  private updateLogoPreferencesState(): void {
    const style = (document.querySelector('input[name="logo-style"]:checked') as HTMLInputElement)?.value;
    const paletteIndex = (document.querySelector('input[name="color-palette"]:checked') as HTMLInputElement)?.value;
    
    if (style || paletteIndex) {
      document.getElementById('generate-logos-btn')?.removeAttribute('disabled');
    }
  }
  
  private renderLogosPhase(): void {
    const session = this.stateManager.getSession();
    const currentLogos = session.phases.logos.generatedOptions || [];
    const history = session.phases.logos.generationHistory || [];
    
    const html = `
      <div class="phase-content">
        <div class="phase-header">
          <h2>Logo Concepts</h2>
          <p>Choose your logo design</p>
        </div>
        
        <div class="phase-divider"></div>
        
        ${currentLogos.length === 0 ? `
          <div class="loading-state">
            <div class="loading-spinner"></div>
            <p>Creating logo concepts...</p>
          </div>
        ` : `
          <!-- Current Generation -->
          <div class="current-logos-section">
            <h3>Latest Designs</h3>
            <div class="logos-grid">
              ${currentLogos.map(logo => `
                <div class="logo-option ${session.phases.logos.selectedLogo?.id === logo.id ? 'selected' : ''}" data-logo-id="${logo.id}">
                  <div class="logo-image">
                    <img src="${logo.urls?.png || logo.urls?.jpg || '/placeholder-logo.png'}" 
                        alt="${logo.concept_name}"
                        onerror="this.src='/placeholder-logo.png'">
                  </div>
                  <h4>${logo.concept_name}</h4>
                  <p class="logo-description">${logo.description || ''}</p>
                  <div class="logo-actions">
                    <button class="select-logo-btn" data-logo-id="${logo.id}">
                      ${session.phases.logos.selectedLogo?.id === logo.id ? 'Selected ✓' : 'Select'}
                    </button>
                    <button class="download-logo-btn" data-logo-id="${logo.id}">
                      Download
                    </button>
                  </div>
                </div>
              `).join('')}
            </div>
          </div>
          
          ${history.length > 0 ? `
            <div class="phase-divider"></div>
            <details class="history-section">
              <summary>Previous Generations (${history.length})</summary>
              ${history.reverse().map((gen, idx) => `
                <div class="generation-group">
                  <h4>Generation from ${new Date(gen.timestamp).toLocaleString()}</h4>
                  <div class="logos-grid smaller">
                    ${gen.logos.map(logo => `
                      <div class="logo-option historical" data-logo-id="${logo.id}">
                        <div class="logo-image">
                          <img src="${logo.urls?.png || logo.urls?.jpg || '/placeholder-logo.png'}" 
                              alt="${logo.concept_name}"
                              onerror="this.src='/placeholder-logo.png'">
                        </div>
                        <h4>${logo.concept_name}</h4>
                        <div class="logo-actions">
                          <button class="select-logo-btn" data-logo-id="${logo.id}">Select</button>
                          <button class="download-logo-btn" data-logo-id="${logo.id}">Download</button>
                        </div>
                      </div>
                    `).join('')}
                  </div>
                </div>
              `).join('')}
            </details>
          ` : ''}
        `}
        
        <div class="phase-divider"></div>
        
        <div class="phase-actions">
          <button id="back-btn" class="secondary-btn">← Back to Style</button>
          <button id="regenerate-logos-btn" class="secondary-btn">🔄 Try Different Logos</button>
          <button id="continue-btn" class="primary-btn gradient-btn" ${!session.phases.logos.selectedLogo ? 'disabled' : ''}>
            Continue →
          </button>
        </div>
      </div>
    `;
    
    this.middlePanel.innerHTML = html;
    
    // Generate logos if none exist
    if (currentLogos.length === 0 && this.phaseController) {
      this.phaseController.generateLogos();
    }
    
    // Setup event listeners after DOM is ready
    setTimeout(() => {
      // Select buttons
      document.querySelectorAll('.select-logo-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
          const logoId = (e.currentTarget as HTMLElement).dataset.logoId;
          if (logoId && this.phaseController) {
            // Find logo in current or history
            let logo = currentLogos.find(l => l.id === logoId);
            if (!logo) {
              history.forEach(gen => {
                const found = gen.logos.find(l => l.id === logoId);
                if (found) logo = found;
              });
            }
            
            if (logo) {
              this.stateManager.updatePhase('logos', {
                selectedLogo: logo
              });
              
              // Update UI
              document.querySelectorAll('.logo-option').forEach(opt => {
                opt.classList.remove('selected');
              });
              document.querySelector(`[data-logo-id="${logoId}"]`)?.classList.add('selected');
              
              // Update button text
              document.querySelectorAll('.select-logo-btn').forEach(b => {
                const btnId = (b as HTMLElement).dataset.logoId;
                b.textContent = btnId === logoId ? 'Selected ✓' : 'Select';
              });
              
              // Enable continue
              document.getElementById('continue-btn')?.removeAttribute('disabled');
            }
          }
        });
      });
      
      // Download buttons
      document.querySelectorAll('.download-logo-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
          const logoId = (e.currentTarget as HTMLElement).dataset.logoId;
          if (logoId && this.phaseController) {
            this.phaseController.downloadLogo(logoId);
          }
        });
      });
      
      // Back button
      document.getElementById('back-btn')?.addEventListener('click', () => {
        this.navigationManager.goBack();
        window.dispatchEvent(new CustomEvent('phase-changed'));
      });
      
      // Regenerate button
      document.getElementById('regenerate-logos-btn')?.addEventListener('click', () => {
        // Clear preferences to force regeneration
        this.stateManager.updatePhase('logos', {
          lastGeneratedPreferences: null
        });
        if (this.phaseController) {
          this.phaseController.generateLogos();
        }
      });
      
      // Continue button
      document.getElementById('continue-btn')?.addEventListener('click', () => {
        this.stateManager.completePhase('logos');
        const nextPhase = this.navigationManager.determineNextPhase();
        if (nextPhase) {
          this.navigationManager.goToPhase(nextPhase);
          window.dispatchEvent(new CustomEvent('phase-changed', { detail: { phase: nextPhase } }));
        }
      });
    }, 100);
  }
  
  // Updated initialization function
  private async initializeLogoPreferences(): Promise<void> {
    const session = this.stateManager.getSession();
    const businessName = session.input.businessName || session.phases.names.selectedName;
    
    if (!businessName) {
      console.error('No business name available for preferences analysis');
      return;
    }
    
    // CHECK αν έχουμε ήδη
    if (session.phases.logoPreferences.aiSuggestions) {
      const existing = session.phases.logoPreferences.aiSuggestions;
      
      // Extract colors safely
      const colorStrings = existing.colors.map((c: any) => 
        typeof c === 'string' ? c : c.hex
      ) as string[];
      
      // Populate text
      const textArea = document.getElementById('ai-recommendation-text') as HTMLTextAreaElement;
      if (textArea) {
        const recommendationText = `For ${businessName}, ${existing.styleReasoning || existing.colorReasoning}\n\nRecommended style: ${existing.style}\nSuggested colors: ${colorStrings.join(', ')}`;
        textArea.value = recommendationText;
      }
      
      // Select saved style
      const savedStyle = session.phases.logoPreferences.userChoice?.style || existing.style;
      const styleInput = document.querySelector(`input[value="${savedStyle}"]`) as HTMLInputElement;
      if (styleInput) {
        styleInput.checked = true;
        styleInput.closest('.style-card')?.classList.add('selected');
      }
      
      // Show palettes
      const paletteGrid = document.getElementById('palette-grid');
      if (paletteGrid) {
        const palettes = this.generateColorPalettes(colorStrings);
        paletteGrid.innerHTML = palettes.map((palette, index) => `
          <div class="palette-box" data-palette="${palette.name}">
            <input type="checkbox" name="color-palette" value="${palette.name}" id="palette-${index}">
            <div class="color-swatches">
              ${palette.colors.map((color: string) => `
                <div class="color-swatch" style="background-color: ${color};"></div>
              `).join('')}
            </div>
            <div class="palette-name">${palette.name}</div>
          </div>
        `).join('');
      }
      
      document.getElementById('generate-logos-btn')?.removeAttribute('disabled');
      return;
    }
    
    // API call section stays the same
    try {
      const { apiClient } = await import('./api-client');
      
      const response = await apiClient.analyzePreferences({
        business_name: businessName,
        description: session.input.description,
        for_type: 'logo'
      });
      
      const recommendationText = `For ${businessName}, ${response.reasoning}\n\nRecommended style: ${response.style}\nSuggested colors: ${response.colors.join(', ')}`;
      
      const textArea = document.getElementById('ai-recommendation-text') as HTMLTextAreaElement;
      if (textArea) {
        textArea.value = recommendationText;
      }
      
      const suggestedInput = document.querySelector(`input[value="${response.style}"]`) as HTMLInputElement;
      if (suggestedInput) {
        suggestedInput.checked = true;
        suggestedInput.closest('.style-card')?.classList.add('selected');
      }
      
      const paletteGrid = document.getElementById('palette-grid');
      if (paletteGrid) {
        const palettes = this.generateColorPalettes(response.colors);
        paletteGrid.innerHTML = palettes.map((palette, index) => `
          <div class="palette-box ${index === 0 ? 'selected' : ''}" data-palette="${palette.name}">
            <input type="checkbox" name="color-palette" value="${palette.name}" id="palette-${index}" ${index === 0 ? 'checked' : ''}>
            <div class="color-swatches">
              ${palette.colors.map((color: string) => `
                <div class="color-swatch" style="background-color: ${color};"></div>
              `).join('')}
            </div>
            <div class="palette-name">${palette.name}</div>
          </div>
        `).join('');
      }
      
      document.getElementById('generate-logos-btn')?.removeAttribute('disabled');
      
      // SAVE to state
      this.stateManager.updatePhase('logoPreferences', {
        status: 'completed',
        aiSuggestions: response,
        userChoice: {
          style: response.style,
          colors: response.colors,
          customized: false
        }
      });
      
    } catch (error) {
      console.error('Failed to get preferences:', error);
      document.getElementById('generate-logos-btn')?.removeAttribute('disabled');
    }
  }
  
  /**
   * Setup Logo Preferences Phase Listeners
   */
  private setupLogoPreferencesListeners(): void {
    // Style box clicks - NOW THE .style-box ELEMENTS EXIST IN DOM
    const styleBoxes = document.querySelectorAll('.style-box');
    styleBoxes.forEach(box => {
      box.addEventListener('click', () => {
        const checkbox = box.querySelector('input[type="checkbox"]') as HTMLInputElement;
        checkbox.checked = !checkbox.checked;
        
        if (checkbox.checked) {
          box.classList.add('selected');
        } else {
          box.classList.remove('selected');
        }
        
        this.updateLogoPreferencesState();
      });
    });

    // Palette box clicks
    const paletteBoxes = document.querySelectorAll('.palette-box');
    paletteBoxes.forEach(box => {
      box.addEventListener('click', () => {
        const checkbox = box.querySelector('input[type="checkbox"]') as HTMLInputElement;
        checkbox.checked = !checkbox.checked;
        
        if (checkbox.checked) {
          box.classList.add('selected');
        } else {
          box.classList.remove('selected');
        }
        
        this.updateLogoPreferencesState();
      });
    });

    // Generate button - always enabled
    document.getElementById('generate-logos-btn')?.addEventListener('click', () => {
      this.collectAndSavePreferences();
    });
    
    // Back button
    document.getElementById('back-btn')?.addEventListener('click', () => {
      this.navigationManager.goBack();
      window.dispatchEvent(new CustomEvent('phase-changed'));
    });
  }

  private collectAndSavePreferences(): void {
    const selectedStyles = Array.from(document.querySelectorAll('input[name="logo-style"]:checked'))
      .map(cb => (cb as HTMLInputElement).value)
      .filter(v => v !== 'none');
      
    const customStyle = (document.getElementById('custom-style-input') as HTMLInputElement)?.value;
    const customColors = (document.getElementById('custom-colors-input') as HTMLInputElement)?.value;
    const aiText = (document.getElementById('ai-recommendation-text') as HTMLTextAreaElement)?.value;
    
    this.stateManager.updatePhase('logoPreferences', {
      userChoice: {
        styles: selectedStyles,
        customStyle: customStyle,
        customColors: customColors,
        aiText: aiText,
        customized: true
      }
    });
    
    // Continue to next phase
    this.stateManager.completePhase('logoPreferences');
    const nextPhase = this.navigationManager.determineNextPhase();
    if (nextPhase) {
      this.navigationManager.goToPhase(nextPhase);
      window.dispatchEvent(new CustomEvent('phase-changed', { detail: { phase: nextPhase } }));
    }
  }
  
  private renderTaglinePreferencesPhase(): void {
    this.middlePanel.innerHTML = '<div class="phase-content">Tagline Preferences Phase - Coming Soon</div>';
  }
  
  private renderTaglinesPhase(): void {
    this.middlePanel.innerHTML = '<div class="phase-content">Taglines Phase - Coming Soon</div>';
  }
  
  private renderCompletePhase(): void {
    this.middlePanel.innerHTML = '<div class="phase-content">Complete Phase - Coming Soon</div>';
  }
  
  /**
   * Show toast notification
   */
  private showToast(message: string, type: 'success' | 'error' | 'info' = 'info'): void {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    
    document.body.appendChild(toast);
    
    setTimeout(() => {
      toast.classList.add('show');
    }, 100);
    
    setTimeout(() => {
      toast.classList.remove('show');
      setTimeout(() => toast.remove(), 300);
    }, 3000);
  }
  
  /**
   * Show errors
   */
  private showErrors(errors: Array<{ field: string; message: string }>): void {
    errors.forEach(error => {
      this.showToast(error.message, 'error');
    });
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