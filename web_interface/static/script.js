// Theme Management
const themeToggle = document.getElementById('theme-toggle');
const html = document.documentElement;
const logo = document.getElementById('logo');

themeToggle.addEventListener('click', () => {
    const currentTheme = html.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    
    html.setAttribute('data-theme', newTheme);
    logo.src = `/logo/${newTheme}`;
    localStorage.setItem('theme', newTheme);
});

// Initialize theme on page load
document.addEventListener('DOMContentLoaded', () => {
    const savedTheme = localStorage.getItem('theme') || 'light';
    html.setAttribute('data-theme', savedTheme);
    logo.src = `/logo/${savedTheme}`;
});

// Tab Switching
document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        // Remove active class from all tabs
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        // Add active to clicked tab
        tab.classList.add('active');
        
        const mode = tab.getAttribute('data-mode');
        
        // Show/hide content
        if (mode === 'classic') {
            document.getElementById('classic-mode').classList.remove('hidden');
            document.getElementById('business-mode').classList.add('hidden');
        } else {
            document.getElementById('classic-mode').classList.add('hidden');
            document.getElementById('business-mode').classList.remove('hidden');
        }
        
        // Hide results when switching
        document.getElementById('results').classList.add('hidden');
        const backdrop = document.querySelector('.overlay-backdrop');
        if (backdrop) backdrop.classList.remove('active');
    });
});

// Create backdrop if not exists
function createBackdrop() {
    const existing = document.querySelector('.overlay-backdrop');
    if (existing) return existing;
    
    const backdrop = document.createElement('div');
    backdrop.className = 'overlay-backdrop';
    document.body.appendChild(backdrop);
    return backdrop;
}

// Classic Hunt
document.getElementById('start-hunt').addEventListener('click', async () => {
    const count = document.getElementById('domain-count').value;
    const strategy = document.getElementById('strategy').value;
    const method = document.getElementById('check-method').value;
    const container = document.querySelector('.container');
    
    // Enable loading state
    container.classList.add('loading-active');
    document.getElementById('loading').classList.remove('hidden');
    
    let loadingMessage = `Generating ${count} domains...`;
    switch(method) {
        case 'hybrid':
            loadingMessage += ' Using API + smart anti-bot protection';
            break;
        case 'smart':
            loadingMessage += ' Using adaptive checking strategy';
            break;
        case 'dns':
            loadingMessage += ' Fast DNS-only checking';
            break;
        case 'robust':
            loadingMessage += ' Comprehensive marketplace analysis';
            break;
    }
    
    document.getElementById('loading-text').textContent = loadingMessage;
    document.getElementById('results').classList.add('hidden');
    
    try {
        const response = await fetch('/api/hunt', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                count: parseInt(count),
                mode: strategy,
                check_strategy: method
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            displayClassicResults(data.domains);
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        alert('Hunt failed: ' + error.message);
    } finally {
        container.classList.remove('loading-active');
        document.getElementById('loading').classList.add('hidden');
    }
});

// Business Idea Domains
document.getElementById('find-domains').addEventListener('click', async () => {
    const idea = document.getElementById('idea').value.trim();
    const container = document.querySelector('.container');
    
    if (!idea) {
        alert('Please describe your business idea');
        return;
    }
    
    // Enable loading state
    container.classList.add('loading-active');
    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('loading-text').textContent = 'Analyzing your idea and finding perfect domains...';
    document.getElementById('results').classList.add('hidden');
    
    try {
        const response = await fetch('/api/suggest', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ idea })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            displayBusinessResults(data);
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        alert('Failed to generate suggestions: ' + error.message);
    } finally {
        container.classList.remove('loading-active');
        document.getElementById('loading').classList.add('hidden');
    }
});

// script.js - Updated display functions

// Display function update - simpler animation
function displayClassicResults(domains) {
    const container = document.querySelector('.container');
    const resultsDiv = document.getElementById('results');
    
    // Prepare results
    const available = domains.filter(d => d.available).length;
    document.getElementById('results-title').textContent = 
        `Found ${available} available domains`;
    
    // Add close button if not exists
    if (!resultsDiv.querySelector('.results-close')) {
        const closeBtn = document.createElement('button');
        closeBtn.className = 'results-close';
        closeBtn.innerHTML = '×';
        closeBtn.onclick = () => {
            container.classList.remove('results-active');
            resultsDiv.classList.remove('show');
            
            setTimeout(() => {
                resultsDiv.classList.add('hidden');
            }, 400);
        };
        resultsDiv.insertBefore(closeBtn, resultsDiv.firstChild);
    }
    
    // Populate domains
    const listDiv = document.getElementById('domains-list');
    listDiv.innerHTML = '';
    
    domains.sort((a, b) => {
        if (a.available && !b.available) return -1;
        if (!a.available && b.available) return 1;
        return (b.score || 0) - (a.score || 0);
    });
    
    domains.slice(0, 100).forEach(item => {
        const domainDiv = document.createElement('div');
        domainDiv.className = 'domain-item';
        
        domainDiv.innerHTML = `
            <div class="domain-info">
                <h3>${item.domain}</h3>
                ${item.score ? `<span class="domain-score">Score: ${item.score.toFixed(1)}/10</span>` : ''}
            </div>
            <div>
                <span class="domain-status ${item.available ? 'available' : 'taken'}">
                    ${item.available ? 'Available' : 'Taken'}
                </span>
            </div>
        `;
        
        listDiv.appendChild(domainDiv);
    });
    
    // Show results
    resultsDiv.classList.remove('hidden');
    
    // Animate both containers
    setTimeout(() => {
        container.classList.add('results-active');
        resultsDiv.classList.add('show');
    }, 10);
}

// Display Business Idea Results - same logic
function displayBusinessResults(data) {
    const container = document.querySelector('.container');
    const resultsDiv = document.getElementById('results');
    const suggestions = data.suggestions || [];
    const availableDomains = suggestions.filter(d => d.available);
    
    if (availableDomains.length === 0) {
        alert('No available domains found. Try a different idea.');
        return;
    }
    
    resultsDiv.classList.remove('hidden');
    
    setTimeout(() => {
        container.classList.add('results-active');
        resultsDiv.classList.add('show');
    }, 10);
    
    document.getElementById('results-title').textContent = 
        `Found ${availableDomains.length} available domains`;
    
    if (!resultsDiv.querySelector('.results-close')) {
        const closeBtn = document.createElement('button');
        closeBtn.className = 'results-close';
        closeBtn.innerHTML = '×';
        closeBtn.onclick = () => {
            container.classList.remove('results-active');
            resultsDiv.classList.remove('show');
            
            setTimeout(() => {
                resultsDiv.classList.add('hidden');
            }, 400);
        };
        resultsDiv.insertBefore(closeBtn, resultsDiv.firstChild);
    }
    
    const listDiv = document.getElementById('domains-list');
    listDiv.innerHTML = '';
    
    availableDomains.forEach(item => {
        const domainDiv = document.createElement('div');
        domainDiv.className = 'domain-item';
        
        domainDiv.innerHTML = `
            <div class="domain-info">
                <h3>${item.domain}</h3>
                <span class="domain-score">Score: ${item.score ? item.score.toFixed(1) : 'N/A'}</span>
            </div>
            <div>
                <span class="domain-status available">
                    Available
                </span>
            </div>
        `;
        
        listDiv.appendChild(domainDiv);
    });
}