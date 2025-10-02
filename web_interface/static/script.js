// Theme Management
const themeToggle = document.getElementById('theme-toggle');
const html = document.documentElement;
const logo = document.getElementById('logo');
const loaderLogo = document.getElementById('loader-logo');

themeToggle.addEventListener('click', () => {
    const currentTheme = html.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    
    html.setAttribute('data-theme', newTheme);
    logo.src = `/logo/${newTheme}`;
    if (loaderLogo) {
        loaderLogo.src = `/logo/${newTheme}`;
    }
    localStorage.setItem('theme', newTheme);
});

// Initialize theme on page load
document.addEventListener('DOMContentLoaded', () => {
    setTimeout(() => {
        document.body.classList.add('loaded');
    }, 2000);
    
    const savedTheme = localStorage.getItem('theme') || 'light';
    html.setAttribute('data-theme', savedTheme);
    logo.src = `/logo/${savedTheme}`;
    if (loaderLogo) {
        loaderLogo.src = `/logo/${savedTheme}`;
    }
    
    // Initialize custom dropdown arrows
    initializeDropdownArrows();
    
    // Add close button handler for search details
    const detailsClose = document.querySelector('.details-close');
    if (detailsClose) {
        detailsClose.addEventListener('click', () => {
            const container = document.querySelector('.container');
            const detailsPanel = document.getElementById('search-details');
            
            container.classList.remove('results-active');
            detailsPanel.classList.remove('show');
            
            setTimeout(() => {
                detailsPanel.classList.add('hidden');
            }, 400);
        });
    }
});

// Custom dropdown arrow functionality
function initializeDropdownArrows() {
    const selectWrappers = document.querySelectorAll('.select-wrapper');
    
    selectWrappers.forEach(wrapper => {
        const select = wrapper.querySelector('select');
        if (select) {
            // Add focus class when select is focused
            select.addEventListener('focus', () => {
                wrapper.classList.add('focused');
            });
            
            // Remove focus class when select loses focus
            select.addEventListener('blur', () => {
                wrapper.classList.remove('focused');
            });
        }
    });
    
    // Handle select focus for arrow animation - alternative approach
    document.querySelectorAll('select').forEach(select => {
        const wrapper = select.closest('.select-wrapper');
        if (wrapper) {
            select.addEventListener('focus', () => {
                wrapper.classList.add('focused');
            });
            
            select.addEventListener('blur', () => {
                wrapper.classList.remove('focused');
            });
        }
    });
}

// Supabase setup
const SUPABASE_URL = 'https://pmzxlnpmapqlwsphiath.supabase.co';
const SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBtenhsbnBtYXBxbHdzcGhpYXRoIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTkzOTYwMDUsImV4cCI6MjA3NDk3MjAwNX0.lmLmBgEz0O5dfKeu_4HwOxNMJq0IrXVFiSAKKCs2OCM';
const supabase = window.supabase.createClient(SUPABASE_URL, SUPABASE_KEY);

// Helper to save search
async function saveSearch(type, params, results) {
    // 1. Create search record
    const { data: search } = await supabase
        .from('searches')
        .insert({
            type: type,
            input_data: params,
            results_summary: {
                total: results.length,
                available: results.filter(d => d.available).length
            }
        })
        .select()
        .single();
    
    // 2. Save domains
    if (search && results.length > 0) {
        const domains = results.map(d => ({
            search_id: search.id,
            domain: d.domain,
            available: d.available,
            score: d.score || 0
        }));
        
        await supabase.from('domains').insert(domains);
    }
    
    return search?.id;
}

// Fix the clear history button handler
document.getElementById('clear-history').addEventListener('click', async () => {
    const confirmDelete = confirm(
        'Are you sure you want to delete all history?\n\n' +
        'This will permanently remove:\n' +
        '• All search history\n' + 
        '• All domain records\n' +
        '• All analytics data\n\n' +
        'This action cannot be undone.'
    );
    
    if (!confirmDelete) return;
    
    try {
        const btn = document.getElementById('clear-history');
        btn.textContent = 'Clearing...';
        btn.disabled = true;
        
        // Delete all domains first (due to foreign key)
        const { error: domainError } = await supabase
            .from('domains')
            .delete()
            .not('id', 'is', null); // This matches all records where id is not null
        
        if (domainError) throw domainError;
        
        // Delete all searches
        const { error: searchError } = await supabase
            .from('searches')
            .delete()
            .not('id', 'is', null); // This matches all records
        
        if (searchError) throw searchError;
        
        // Reset the UI
        document.getElementById('total-hunts').textContent = '0';
        document.getElementById('domains-checked').textContent = '0';
        document.getElementById('success-rate').textContent = '0%';
        document.getElementById('avg-score').textContent = '0.0';
        document.getElementById('top-domains-list').innerHTML = '<p>No available domains yet</p>';
        document.getElementById('recent-activity').innerHTML = '<p>No activity yet</p>';
        
        // Success message
        btn.textContent = 'History Cleared!';
        setTimeout(() => {
            btn.textContent = 'Clear All History';
            btn.disabled = false;
        }, 2000);
        
    } catch (error) {
        console.error('Error clearing history:', error);
        alert('Failed to clear history: ' + error.message);
        
        // Reset button
        document.getElementById('clear-history').textContent = 'Clear All History';
        document.getElementById('clear-history').disabled = false;
    }
});

// Tab Switching
document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        // Remove active class from all tabs
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        // Add active to clicked tab
        tab.classList.add('active');
        
        const mode = tab.getAttribute('data-mode');
        
        // Hide all modes
        document.getElementById('classic-mode').classList.add('hidden');
        document.getElementById('business-mode').classList.add('hidden');
        document.getElementById('analytics-mode').classList.add('hidden');
        
        // Show selected mode
        if (mode === 'classic') {
            document.getElementById('classic-mode').classList.remove('hidden');
        } else if (mode === 'business') {
            document.getElementById('business-mode').classList.remove('hidden');
        } else if (mode === 'analytics') {
            document.getElementById('analytics-mode').classList.remove('hidden');
            loadAnalytics(); // Load fresh data
        }
        
        // Hide results when switching
        // document.getElementById('results').classList.add('hidden');
        // const backdrop = document.querySelector('.overlay-backdrop');
        // if (backdrop) backdrop.classList.remove('active');
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
    
    // Return container to center immediately when starting new hunt
    container.classList.remove('results-active');
    
    // Hide results panel if it's open
    const resultsDiv = document.getElementById('results');
    resultsDiv.classList.remove('show');
    setTimeout(() => {
        resultsDiv.classList.add('hidden');
    }, 400);
    
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

async function getStats() {
    const { data } = await supabase
        .from('searches')
        .select('*')
        .order('created_at', { ascending: false })
        .limit(5);
    
    console.log('Recent searches:', data);
}

// Business Idea Domains
document.getElementById('find-domains').addEventListener('click', async () => {
    const idea = document.getElementById('idea').value.trim();
    const container = document.querySelector('.container');
    
    if (!idea) {
        alert('Please describe your business idea');
        return;
    }
    
    // Return container to center immediately
    container.classList.remove('results-active');
    
    // Hide results panel if it's open
    const resultsDiv = document.getElementById('results');
    resultsDiv.classList.remove('show');
    setTimeout(() => {
        resultsDiv.classList.add('hidden');
    }, 400);
    
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
async function displayClassicResults(domains) {
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

    // Save to Supabase
    const searchId = await saveSearch('classic', 
        { count: document.getElementById('domain-count').value },
        domains
    );
    if (searchId) {
        console.log('Search saved:', searchId);
    }
}

// Display Business Idea Results - same logic
async function displayBusinessResults(data) {
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

    // Save to Supabase
    const searchId = await saveSearch('business',
        { idea: document.getElementById('idea').value },
        data.suggestions || []
    );
    if (searchId) {
        console.log('Search saved:', searchId);
    }
}

async function loadAnalytics() {
    try {
        // Get all searches
        const { data: searches, error: searchError } = await supabase
            .from('searches')
            .select('*')
            .order('created_at', { ascending: false });
        
        if (searchError) throw searchError;
        
        // Get all domains
        const { data: domains, error: domainError } = await supabase
            .from('domains')
            .select('*');
        
        if (domainError) throw domainError;
        
        // Calculate stats
        const totalHunts = searches.length;
        const totalDomains = domains.length;
        const availableDomains = domains.filter(d => d.available).length;
        const successRate = totalDomains > 0 ? Math.round((availableDomains / totalDomains) * 100) : 0;
        const avgScore = domains.reduce((acc, d) => acc + (d.score || 0), 0) / (domains.length || 1);
        
        // Update stats
        document.getElementById('total-hunts').textContent = totalHunts;
        document.getElementById('domains-checked').textContent = totalDomains;
        document.getElementById('success-rate').textContent = successRate + '%';
        document.getElementById('avg-score').textContent = avgScore.toFixed(1);
        
        // Get top available domains
        const topDomains = domains
            .filter(d => d.available)
            .sort((a, b) => (b.score || 0) - (a.score || 0))
            .slice(0, 10);
        
        // Display top domains
        const topDomainsHtml = topDomains.map(d => `
            <div class="compact-domain-item">
                <div class="compact-domain-name">${d.domain}</div>
                <div class="compact-domain-score">
                    <span class="score-badge">${(d.score || 0).toFixed(1)}</span>
                </div>
            </div>
        `).join('');
        
        document.getElementById('top-domains-list').innerHTML = topDomainsHtml || '<p>No available domains yet</p>';
        
        // Recent activity - make items clickable
        const recentActivity = searches.slice(0, 5).map(s => {
            const time = new Date(s.created_at).toLocaleString();
            const type = s.type === 'classic' ? 'Classic Hunt' : 'Business Search';
            const summary = s.results_summary;
            return `
                <div class="activity-item" onclick="showSearchDetails('${s.id}')" style="cursor: pointer;">
                    <div class="activity-time">${time}</div>
                    <div class="activity-description">
                        ${type} - Found ${summary?.available || 0} available out of ${summary?.total || 0}
                    </div>
                </div>
            `;
        }).join('');
        
        document.getElementById('recent-activity').innerHTML = recentActivity || '<p>No activity yet</p>';
        
    } catch (error) {
        console.error('Error loading analytics:', error);
    }
}

// New function to show search details
async function showSearchDetails(searchId) {
    try {
        // Fetch search details
        const { data: search } = await supabase
            .from('searches')
            .select('*')
            .eq('id', searchId)
            .single();
        
        // Fetch associated domains
        const { data: domains } = await supabase
            .from('domains')
            .select('*')
            .eq('search_id', searchId)
            .order('score', { ascending: false });
        
        // Populate details panel
        document.getElementById('details-title').textContent = 
            search.type === 'classic' ? 'Classic Hunt Details' : 'Business Search Details';
        
        document.getElementById('detail-type').textContent = 
            search.type === 'classic' ? 'Classic Hunt' : 'Business Search';
        
        document.getElementById('detail-date').textContent = 
            new Date(search.created_at).toLocaleString();
        
        document.getElementById('detail-total').textContent = 
            search.results_summary?.total || 0;
        
        document.getElementById('detail-available').textContent = 
            search.results_summary?.available || 0;
        
        // Populate domains list
        const domainsList = document.getElementById('detail-domains-list');
        domainsList.innerHTML = domains.map(d => `
            <div class="domain-item">
                <div class="domain-info">
                    <h3>${d.domain}</h3>
                    ${d.score ? `<span class="domain-score">Score: ${d.score.toFixed(1)}/10</span>` : ''}
                </div>
                <div>
                    <span class="domain-status ${d.available ? 'available' : 'taken'}">
                        ${d.available ? 'Available' : 'Taken'}
                    </span>
                </div>
            </div>
        `).join('');
        
        // Show details panel
        const container = document.querySelector('.container');
        const detailsPanel = document.getElementById('search-details');
        
        container.classList.add('results-active');
        detailsPanel.classList.remove('hidden');
        setTimeout(() => {
            detailsPanel.classList.add('show');
        }, 10);
        
    } catch (error) {
        console.error('Error loading search details:', error);
    }
}