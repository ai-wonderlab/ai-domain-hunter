import { api } from './api';
import { authManager } from './auth';
import { 
  showLoader, 
  hideLoader, 
  animateList 
} from './animations';
import type { 
  GeneratePackageRequest, 
  GeneratePackageResponse
} from '@/types/api.types';

export class StudioApp {
  private currentStep = 1;
  private generationData: Partial<GeneratePackageRequest> = {
    include_services: ['name', 'logo', 'color', 'tagline', 'domain']
  };
  private generatedPackage: GeneratePackageResponse | null = null;

  constructor() {
    this.init();
  }

  private init() {
    this.setupStepNavigation();
    this.setupFormInputs();
    this.setupGenerateButton();
    this.setupThemeToggle();
    this.updateAuthStatus();
    
    // Load any saved data
    this.loadSavedData();
    
    // Load history
    this.loadHistory();
  }

  /* ========================================
     STEP NAVIGATION
     ======================================== */

  private setupStepNavigation() {
    const steps = document.querySelectorAll('.step');
    const nextBtns = document.querySelectorAll('.next-step');
    const prevBtns = document.querySelectorAll('.prev-step');

    // Step indicator clicks
    steps.forEach((step, index) => {
      step.addEventListener('click', () => {
        const stepNum = index + 1;
        if (stepNum <= this.currentStep) {
          this.goToStep(stepNum);
        }
      });
    });

    // Next buttons
    nextBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        if (this.validateStep(this.currentStep)) {
          this.goToStep(this.currentStep + 1);
        }
      });
    });

    // Previous buttons
    prevBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        this.goToStep(this.currentStep - 1);
      });
    });
  }

  private goToStep(step: number) {
    if (step < 1 || step > 3) return;

    // Update step indicators
    document.querySelectorAll('.step').forEach((el, i) => {
      el.classList.toggle('active', i + 1 === step);
      el.classList.toggle('completed', i + 1 < step);
    });

    // Hide all step content
    document.querySelectorAll('.step-content').forEach((el) => {
      el.classList.add('hidden');
    });

    // Show current step
    const currentContent = document.querySelector(`.step-content[data-step="${step}"]`);
    currentContent?.classList.remove('hidden');

    this.currentStep = step;
  }

  private validateStep(step: number): boolean {
    switch (step) {
      case 1: {
        const description = (document.getElementById('description') as HTMLTextAreaElement)?.value;
        if (!description || description.length < 20) {
          this.showError('Please provide a detailed description (at least 20 characters)');
          return false;
        }
        this.generationData.description = description;
        
        const industry = (document.getElementById('industry') as HTMLSelectElement)?.value;
        if (industry) {
          this.generationData.industry = industry;
        }
        return true;
      }
      
      case 2: {
        const selectedStyle = document.querySelector('.style-card.selected');
        if (!selectedStyle) {
          this.showError('Please select a style');
          return false;
        }
        return true;
      }
      
      default:
        return true;
    }
  }

  /* ========================================
     FORM INPUTS
     ======================================== */

  private setupFormInputs() {
    // Business description
    const descriptionInput = document.getElementById('description') as HTMLTextAreaElement;
    descriptionInput?.addEventListener('input', (e) => {
      this.generationData.description = (e.target as HTMLTextAreaElement).value;
      this.saveToLocalStorage();
    });

    // Industry selection
    const industrySelect = document.getElementById('industry') as HTMLSelectElement;
    industrySelect?.addEventListener('change', (e) => {
      this.generationData.industry = (e.target as HTMLSelectElement).value;
      this.saveToLocalStorage();
    });

    // Style cards
    document.querySelectorAll('.style-card').forEach(card => {
      card.addEventListener('click', () => {
        document.querySelectorAll('.style-card').forEach(c => c.classList.remove('selected'));
        card.classList.add('selected');
        
        const style = card.getAttribute('data-style');
        if (!this.generationData.style_preferences) {
          this.generationData.style_preferences = {};
        }
        this.generationData.style_preferences.logo_style = style || undefined;
        this.saveToLocalStorage();
      });
    });

    // Color themes
    document.querySelectorAll('.color-theme').forEach(theme => {
      theme.addEventListener('click', () => {
        document.querySelectorAll('.color-theme').forEach(t => t.classList.remove('selected'));
        theme.classList.add('selected');
        
        const colorTheme = theme.getAttribute('data-theme');
        if (!this.generationData.style_preferences) {
          this.generationData.style_preferences = {};
        }
        this.generationData.style_preferences.color_theme = colorTheme || undefined;
        this.saveToLocalStorage();
      });
    });

    // Generation options checkboxes
    document.querySelectorAll('.option-item input[type="checkbox"]').forEach(checkbox => {
      checkbox.addEventListener('change', (e) => {
        const service = (e.target as HTMLInputElement).getAttribute('data-service');
        if (!service) return;

        if (!this.generationData.include_services) {
          this.generationData.include_services = [];
        }

        if ((e.target as HTMLInputElement).checked) {
          if (!this.generationData.include_services.includes(service)) {
            this.generationData.include_services.push(service);
          }
        } else {
          this.generationData.include_services = this.generationData.include_services.filter(
            s => s !== service
          );
        }
        this.saveToLocalStorage();
      });
    });
  }

  /* ========================================
     GENERATION
     ======================================== */

  private setupGenerateButton() {
    const generateBtn = document.getElementById('generate-btn');
    generateBtn?.addEventListener('click', () => {
      if (!this.validateStep(3)) return;

      if (api.isAuthenticated()) {
        this.startGeneration();
      } else {
        authManager.showAuthModal(() => {
          this.updateAuthStatus();
          this.startGeneration();
        });
      }
    });
  }

  private async startGeneration() {
    if (!this.validateStep(this.currentStep)) return;

    // Check if services are selected
    if (!this.generationData.include_services || this.generationData.include_services.length === 0) {
      this.showError('Please select at least one service to generate');
      return;
    }

    showLoader();
    this.setStatus('generating', 'Generating your brand package...');
    
    // CRITICAL: Get the correct container
    const previewContent = document.querySelector('.preview-content');
    if (!previewContent) {
      console.error('Preview content container not found!');
      hideLoader();
      this.showError('UI error: Preview container not found');
      return;
    }
    
    // Clear previous results
    previewContent.innerHTML = '<div class="generation-progress"><p>Generating your brand identity...</p></div>';

    try {
      const response = await api.generatePackage(this.generationData as GeneratePackageRequest);
      
      this.generatedPackage = response;
      this.displayResults(response);
      this.setStatus('ready', 'Generation complete!');
      hideLoader();
      
      // Save to history
      this.saveToHistory(response);
    } catch (error) {
      hideLoader();
      this.setStatus('ready', 'Ready');
      this.showError('Generation failed. Please try again.');
      console.error('Generation error:', error);
      
      // Show error in preview
      if (previewContent) {
        previewContent.innerHTML = `
          <div class="preview-placeholder">
            <div class="placeholder-icon">❌</div>
            <p>Generation failed. Please try again.</p>
          </div>
        `;
      }
    }
  }

  private setStatus(state: 'ready' | 'generating' | 'error', text: string) {
    const statusDot = document.querySelector('.status-dot');
    const statusText = document.getElementById('status-text');
    
    if (statusDot) {
      statusDot.classList.remove('generating');
      if (state === 'generating') {
        statusDot.classList.add('generating');
      }
    }
    
    if (statusText) {
      statusText.textContent = text;
    }
  }

  /* ========================================
     DISPLAY RESULTS (CRITICAL FIX)
     ======================================== */

  private displayResults(data: GeneratePackageResponse) {
    // CRITICAL: Get the correct container
    const container = document.querySelector('.preview-content');
    if (!container) {
      console.error('Preview content container not found!');
      this.showError('UI error: Cannot display results');
      return;
    }

    let html = '<div class="results-container">';

    // Business name header
    if (data.business_name) {
      html += `
        <div class="result-section">
          <h2>${this.escapeHtml(data.business_name)}</h2>
        </div>
      `;
    }

    // Display names
    if (data.names && data.names.length > 0) {
      html += '<div class="result-section"><h3>Business Names</h3><div class="names-grid">';
      data.names.forEach(name => {
        html += `
          <div class="name-option" data-id="${name.id}">
            <h4>${this.escapeHtml(name.name)}</h4>
            <div class="score">${name.score.toFixed(1)}/10</div>
            <p class="reasoning">${this.escapeHtml(name.reasoning)}</p>
          </div>
        `;
      });
      html += '</div></div>';
    }

    // Display logos
    if (data.logos && data.logos.length > 0) {
      html += '<div class="result-section"><h3>Logo Concepts</h3><div class="logos-grid">';
      data.logos.forEach(logo => {
        html += `
          <div class="logo-option" data-id="${logo.id}">
            <img src="${this.escapeHtml(logo.urls.png)}" alt="${this.escapeHtml(logo.concept_name)}" loading="lazy">
            <h4>${this.escapeHtml(logo.concept_name)}</h4>
            <p>${this.escapeHtml(logo.description)}</p>
          </div>
        `;
      });
      html += '</div></div>';
    }

    // Display color palettes
    if (data.color_palettes && data.color_palettes.length > 0) {
      html += '<div class="result-section"><h3>Color Palettes</h3><div class="palettes-grid">';
      data.color_palettes.forEach(palette => {
        html += `
          <div class="palette-option" data-id="${palette.id}">
            <h4>${this.escapeHtml(palette.name)}</h4>
            <div class="colors-row">
              ${palette.colors.map(color => `
                <div class="color-swatch" style="background: ${this.escapeHtml(color.hex)}" title="${this.escapeHtml(color.name)} - ${this.escapeHtml(color.hex)}"></div>
              `).join('')}
            </div>
            <p>${this.escapeHtml(palette.description)}</p>
          </div>
        `;
      });
      html += '</div></div>';
    }

    // Display taglines
    if (data.taglines && data.taglines.length > 0) {
      html += '<div class="result-section"><h3>Taglines</h3><div class="taglines-list">';
      data.taglines.forEach(tagline => {
        html += `
          <div class="tagline-option" data-id="${tagline.id}">
            <p class="tagline-text">"${this.escapeHtml(tagline.text)}"</p>
            <span class="tagline-tone">${this.escapeHtml(tagline.tone)}</span>
          </div>
        `;
      });
      html += '</div></div>';
    }

    // Download button
    html += `
      <div class="result-section">
        <button class="action-btn download-btn" id="download-package-btn">
          Download Complete Package
        </button>
      </div>
    `;

    html += '</div>';
    
    // CRITICAL: Replace innerHTML of the correct container
    container.innerHTML = html;

    // Animate items in
    const items = container.querySelectorAll('.name-option, .logo-option, .palette-option, .tagline-option');
    if (items.length > 0) {
      animateList(items);
    }

    // Setup download button
    const downloadBtn = document.getElementById('download-package-btn');
    downloadBtn?.addEventListener('click', () => this.downloadPackage());
  }

  /* ========================================
     UTILITIES
     ======================================== */

  private escapeHtml(text: string): string {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  private showError(message: string) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-toast';
    errorDiv.textContent = message;
    document.body.appendChild(errorDiv);
    
    requestAnimationFrame(() => {
      errorDiv.classList.add('show');
    });
    
    setTimeout(() => {
      errorDiv.classList.remove('show');
      setTimeout(() => errorDiv.remove(), 300);
    }, 3000);
  }

  private showSuccess(message: string) {
    const successDiv = document.createElement('div');
    successDiv.className = 'error-toast success-toast';
    successDiv.textContent = message;
    document.body.appendChild(successDiv);
    
    requestAnimationFrame(() => {
      successDiv.classList.add('show');
    });
    
    setTimeout(() => {
      successDiv.classList.remove('show');
      setTimeout(() => successDiv.remove(), 300);
    }, 3000);
  }

  private updateAuthStatus() {
    const user = api.getUser();
    const statusEl = document.getElementById('auth-status');
    if (statusEl) {
      if (user) {
        statusEl.innerHTML = `
          <span>${this.escapeHtml(user.email)}</span>
          <span>${user.generation_count}/${user.generation_limit} left</span>
        `;
      } else {
        statusEl.innerHTML = '<span>Not signed in</span>';
      }
    }
  }

  private setupThemeToggle() {
    const toggle = document.getElementById('theme-toggle');
    toggle?.addEventListener('click', () => {
      const html = document.documentElement;
      const currentTheme = html.getAttribute('data-theme');
      const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
      html.setAttribute('data-theme', newTheme);
      localStorage.setItem('theme', newTheme);
      
      // Update logos - use direct SVG files
      const logoSrc = newTheme === 'dark' ? '/logo-dark.svg' : '/logo-light.svg';
      const logos = document.querySelectorAll<HTMLImageElement>('.logo-img, .logo-img-small, .loader-logo');
      logos.forEach(logo => {
        logo.src = logoSrc;
      });
    });
  }

  private downloadPackage() {
    if (!this.generatedPackage) {
      this.showError('No package to download');
      return;
    }

    // Create a JSON file with the package data
    const dataStr = JSON.stringify(this.generatedPackage, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = `brand-package-${this.generatedPackage.project_id}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    
    this.showSuccess('Package downloaded!');
  }

  private saveToLocalStorage() {
    localStorage.setItem('generation_draft', JSON.stringify(this.generationData));
  }

  private loadSavedData() {
    const saved = localStorage.getItem('generation_draft');
    if (saved) {
      try {
        this.generationData = JSON.parse(saved);
        
        // Restore form values
        const descriptionInput = document.getElementById('description') as HTMLTextAreaElement;
        if (descriptionInput && this.generationData.description) {
          descriptionInput.value = this.generationData.description;
        }
        
        const industrySelect = document.getElementById('industry') as HTMLSelectElement;
        if (industrySelect && this.generationData.industry) {
          industrySelect.value = this.generationData.industry;
        }
      } catch (e) {
        console.error('Failed to load saved data:', e);
      }
    }
  }

  private saveToHistory(packageData: GeneratePackageResponse) {
    const history = JSON.parse(localStorage.getItem('generation_history') || '[]');
    history.unshift({
      id: packageData.project_id,
      name: packageData.business_name,
      created_at: packageData.created_at,
      thumbnail: packageData.logos?.[0]?.urls?.png || null
    });
    
    // Keep only last 10
    if (history.length > 10) {
      history.length = 10;
    }
    
    localStorage.setItem('generation_history', JSON.stringify(history));
    this.loadHistory();
  }

  private loadHistory() {
    const history = JSON.parse(localStorage.getItem('generation_history') || '[]');
    const container = document.querySelector('.saved-items');
    
    if (!container) return;
    
    if (history.length === 0) {
      container.innerHTML = '<p class="empty-state">No saved brands yet</p>';
      return;
    }
    
    container.innerHTML = history.map((item: any) => `
      <div class="history-item" data-id="${item.id}">
        ${item.thumbnail ? `<img src="${this.escapeHtml(item.thumbnail)}" alt="${this.escapeHtml(item.name)}">` : ''}
        <div>
          <strong>${this.escapeHtml(item.name)}</strong>
          <small>${new Date(item.created_at).toLocaleDateString()}</small>
        </div>
      </div>
    `).join('');
  }
}

// Make it globally available
(window as any).StudioApp = StudioApp;