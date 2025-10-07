import { animate, stagger, spring } from 'motion';

export function initLandingAnimations() {
  // Container entrance
  animate(
    '.container',
    { opacity: [0, 1], y: [50, 0] },
    { duration: 0.8, easing: spring() }
  );

  // Feature pills with stagger
  animate(
    '.pill',
    { opacity: [0, 1], scale: [0.8, 1] },
    { delay: stagger(0.1, { start: 0.5 }), easing: spring() }
  );
}

export function animateGenerationItem(element: HTMLElement) {
  animate(
    element,
    { 
      opacity: [0, 1], 
      y: [20, 0],
      scale: [0.95, 1]
    },
    { 
      duration: 0.6,
      easing: spring({ stiffness: 300, damping: 30 })
    }
  );
}

export function animateStepTransition(
  exitElement: HTMLElement, 
  enterElement: HTMLElement
) {
  // Exit animation
  animate(
    exitElement,
    { opacity: [1, 0], x: [0, -30] },
    { duration: 0.3 }
  ).finished.then(() => {
    exitElement.classList.add('hidden');
    
    // Enter animation
    enterElement.classList.remove('hidden');
    animate(
      enterElement,
      { opacity: [0, 1], x: [30, 0] },
      { duration: 0.3 }
    );
  });
}

export function showLoader() {
  const loader = document.getElementById('loading');
  if (loader) {
    loader.classList.remove('hidden');
    
    // Spin animation for loader
    const loaderEl = loader.querySelector('.loader');
    if (loaderEl) {
      animate(
        loaderEl,
        { rotate: [0, 360] },
        { duration: 1.2, repeat: Infinity, easing: 'linear' }
      );
    }
  }
}

export function hideLoader() {
  const loader = document.getElementById('loading');
  if (loader) {
    animate(
      loader,
      { opacity: [1, 0] },
      { duration: 0.3 }
    ).finished.then(() => {
      loader.classList.add('hidden');
    });
  }
}

export function pulseElement(element: HTMLElement) {
  animate(
    element,
    { scale: [1, 1.05, 1] },
    { duration: 0.3 }
  );
}

export function shakeElement(element: HTMLElement) {
  animate(
    element,
    { x: [0, -10, 10, -10, 10, 0] },
    { duration: 0.5 }
  );
}

export function fadeIn(element: HTMLElement) {
  animate(
    element,
    { opacity: [0, 1] },
    { duration: 0.5 }
  );
}

export function fadeOut(element: HTMLElement): Promise<void> {
  return animate(
    element,
    { opacity: [1, 0] },
    { duration: 0.3 }
  ).finished;
}

export function slideInUp(element: HTMLElement) {
  animate(
    element,
    { opacity: [0, 1], y: [30, 0] },
    { duration: 0.5, easing: spring() }
  );
}

export function slideInDown(element: HTMLElement) {
  animate(
    element,
    { opacity: [0, 1], y: [-30, 0] },
    { duration: 0.5, easing: spring() }
  );
}

export function animateList(elements: NodeListOf<Element> | Element[]) {
  animate(
    elements,
    { opacity: [0, 1], y: [20, 0] },
    { delay: stagger(0.1), easing: spring() }
  );
}

export function progressBar(element: HTMLElement, duration: number = 2000) {
  animate(
    element,
    { width: ['0%', '100%'] },
    { duration: duration / 1000, easing: 'ease-out' }
  );
}