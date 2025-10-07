import { api } from './api';

export class AuthManager {
  private modal: HTMLElement | null = null;

  showAuthModal(onSuccess: () => void) {
    this.modal = this.createAuthModal(onSuccess);
    document.body.appendChild(this.modal);
    
    // Trigger animation
    requestAnimationFrame(() => {
      this.modal?.classList.add('show');
    });
  }

  private createAuthModal(onSuccess: () => void): HTMLElement {
    const modal = document.createElement('div');
    modal.className = 'auth-modal';
    modal.innerHTML = `
      <div class="modal-content">
        <button class="modal-close">&times;</button>
        <h2>Continue to Generate</h2>
        <p class="modal-subtitle">Enter your email to save your brand package</p>
        
        <div class="auth-tabs">
          <button class="auth-tab active" data-tab="login">Sign In</button>
          <button class="auth-tab" data-tab="register">Create Account</button>
        </div>
        
        <form id="auth-form">
          <div class="input-group">
            <input 
              type="email" 
              id="auth-email" 
              class="glass-input"
              placeholder="your@email.com" 
              required
              autocomplete="email"
            />
          </div>
          
          <button type="submit" class="action-btn">
            <span id="auth-btn-text">Continue</span>
          </button>
        </form>
        
        <p class="auth-note">
          Free tier includes 2 brand package generations
        </p>
      </div>
    `;

    // Close button handler
    const closeBtn = modal.querySelector('.modal-close');
    closeBtn?.addEventListener('click', () => this.closeModal());

    // Click outside to close
    modal.addEventListener('click', (e) => {
      if (e.target === modal) {
        this.closeModal();
      }
    });

    // Tab switching
    const tabs = modal.querySelectorAll('.auth-tab');
    tabs.forEach(tab => {
      tab.addEventListener('click', (e) => {
        tabs.forEach(t => t.classList.remove('active'));
        (e.target as HTMLElement).classList.add('active');
      });
    });

    // Form submission
    const form = modal.querySelector('#auth-form') as HTMLFormElement;
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      await this.handleAuth(modal, onSuccess);
    });

    return modal;
  }

  private async handleAuth(modal: HTMLElement, onSuccess: () => void) {
    const emailInput = modal.querySelector('#auth-email') as HTMLInputElement;
    const email = emailInput.value.trim();
    
    if (!email) {
      this.showError('Please enter your email');
      return;
    }

    const activeTab = modal.querySelector('.auth-tab.active');
    const isLogin = activeTab?.getAttribute('data-tab') === 'login';
    
    const btnText = modal.querySelector('#auth-btn-text');
    if (btnText) btnText.textContent = 'Loading...';
    
    try {
      if (isLogin) {
        await api.login(email);
      } else {
        await api.register(email);
      }
      
      this.closeModal();
      this.showSuccess(isLogin ? 'Welcome back!' : 'Account created!');
      onSuccess();
    } catch (error) {
      if (btnText) btnText.textContent = 'Try Again';
      this.showError(error instanceof Error ? error.message : 'Authentication failed');
      console.error('Auth error:', error);
    }
  }

  private closeModal() {
    if (this.modal) {
      this.modal.classList.remove('show');
      setTimeout(() => {
        this.modal?.remove();
        this.modal = null;
      }, 300);
    }
  }

  private showError(message: string) {
    const toast = document.createElement('div');
    toast.className = 'error-toast';
    toast.textContent = message;
    document.body.appendChild(toast);
    
    requestAnimationFrame(() => {
      toast.classList.add('show');
    });
    
    setTimeout(() => {
      toast.classList.remove('show');
      setTimeout(() => toast.remove(), 300);
    }, 3000);
  }

  private showSuccess(message: string) {
    const toast = document.createElement('div');
    toast.className = 'error-toast success-toast';
    toast.textContent = message;
    document.body.appendChild(toast);
    
    requestAnimationFrame(() => {
      toast.classList.add('show');
    });
    
    setTimeout(() => {
      toast.classList.remove('show');
      setTimeout(() => toast.remove(), 300);
    }, 3000);
  }

  checkAuth(): boolean {
    return api.isAuthenticated();
  }

  logout() {
    api.logout();
    window.location.href = '/';
  }
}

export const authManager = new AuthManager();