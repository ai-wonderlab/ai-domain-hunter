import './styles/style.css';
import './styles/animations.css';
import './scripts/liquid-bg';

// Initialize theme
const initTheme = () => {
  const savedTheme = localStorage.getItem('theme') || 'light';
  document.documentElement.setAttribute('data-theme', savedTheme);
  
  // Update logo based on theme - use direct SVG files
  const logoSrc = savedTheme === 'dark' ? '/logo-dark.svg' : '/logo-light.svg';
  
  const logos = document.querySelectorAll<HTMLImageElement>('#logo, #studio-logo, #loader-logo, .logo-img, .logo-img-small, .loader-logo');
  logos.forEach(logo => {
    logo.src = logoSrc;
  });
};

// Theme toggle handler
const setupThemeToggle = () => {
  const themeToggle = document.getElementById('theme-toggle');
  if (!themeToggle) return;
  
  themeToggle.addEventListener('click', () => {
    const html = document.documentElement;
    const currentTheme = html.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    
    html.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    
    // Update logos - use direct SVG files
    const logoSrc = newTheme === 'dark' ? '/logo-dark.svg' : '/logo-light.svg';
    
    const logos = document.querySelectorAll<HTMLImageElement>('#logo, #studio-logo, #loader-logo, .logo-img, .logo-img-small, .loader-logo');
    logos.forEach(logo => {
      logo.src = logoSrc;
    });
  });
};

// Landing page handlers
const setupLandingPage = () => {
  const tryFreeBtn = document.getElementById('try-free');
  const goStudioBtn = document.getElementById('go-studio');
  
  if (tryFreeBtn) {
    tryFreeBtn.addEventListener('click', () => {
      // For free trial, go directly to studio
      window.location.href = '/studio.html';
    });
  }
  
  if (goStudioBtn) {
    goStudioBtn.addEventListener('click', () => {
      window.location.href = '/studio.html';
    });
  }
};

// Studio page handlers
const setupStudioPage = () => {
  // Import and initialize studio app
  import('./scripts/studio-app').then(() => {
    console.log('✅ Studio app loaded');
  }).catch(error => {
    console.error('Failed to load studio app:', error);
  });
};

// Initialize based on current page
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  setupThemeToggle();
  
  // Add loaded class for animations
  setTimeout(() => {
    document.body.classList.add('loaded');
  }, 100);
  
  const currentPath = window.location.pathname;
  
  if (currentPath.includes('studio')) {
    setupStudioPage();
  } else {
    setupLandingPage();
  }
});