# 🌟 AI Domain Finder

An AI-powered domain discovery system that helps you find the perfect .ai domain for your next project. Features intelligent domain generation, real-time availability checking, and business idea-based suggestions.

![Domain Finder Preview](<Screenshot 2025-09-30 at 7.18.50 PM.png>)

## ✨ Features

### 🎯 Two Discovery Modes

**Classic Hunt**
- Generate 10-500 domains at once
- Multiple AI generation strategies (Hybrid, Emergent, Basic)
- Smart availability checking with multiple methods
- Domain scoring based on quality metrics

**Business Idea Mode**
- Describe your business concept
- Get AI-curated domain suggestions
- Context-aware naming based on your industry
- Relevance scoring for each suggestion

### 🔍 Advanced Checking Methods
- **Hybrid**: API + Smart anti-bot protection
- **Smart**: Adaptive checking strategy
- **DNS**: Fast DNS-only checking
- **Robust**: Comprehensive marketplace analysis

### 🎨 Modern UI/UX
- Glass-morphism design
- Dark/Light theme toggle
- Responsive layout (desktop & mobile)
- Smooth animations
- Real-time loading states

## 🛠 Tech Stack

### Backend
- **Python 3.8+** with Flask
- **Refined Domain Hunter** - Custom domain generation engine
- **OpenRouter AI** - Multi-model AI integration
- **Async/Await** for concurrent operations
- **WhoAPI & WhoisXML** for domain checking

### Frontend
- **Vanilla JavaScript** (ES6+)
- **CSS3** with glass-morphism effects
- **Three.js** for animated backgrounds
- **Responsive design** with mobile-first approach

### AI Models
- **Claude 3** (Anthropic) via OpenRouter
- **GPT-4** (OpenAI) via OpenRouter
- **Gemini** for research analysis

## 📦 Installation

### Prerequisites
```bash
python >= 3.8
pip >= 20.0
```

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/ai-domain-finder.git
cd ai-domain-finder
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
Create a `.env` file in the root directory:
```env
# OpenRouter API (Required)
OPENROUTER_API_KEY=your_openrouter_key_here

# Domain Checking APIs (Optional but recommended)
WHOAPI_KEY=your_whoapi_key_here
WHOISXML_API_KEY=your_whoisxml_key_here

# Gemini API (Optional)
GEMINI_API_KEY=your_gemini_key_here
```

### 4. Run the application
```bash
python app.py
```

Visit `http://localhost:8001` in your browser.

## 🔧 Configuration

### Domain Generation Settings
Edit `DomainConfig` in `refined_domain_hunter.py`:
```python
config = DomainConfig()
config.min_length = 3
config.max_length = 15
config.use_stealth_mode = True
config.check_strategy = 'hybrid'
```

### AI Model Selection
Configure in `UnifiedAIManager`:
```python
ai_manager = UnifiedAIManager(config)
ai_manager.default_model = 'claude'  # or 'gpt'
```

## 📚 API Endpoints

### `/api/hunt` - Classic Domain Hunt
```javascript
POST /api/hunt
{
    "count": 50,           // Number of domains (10-500)
    "mode": "hybrid",      // Generation strategy
    "check_strategy": "smart"  // Checking method
}
```

### `/api/suggest` - Business Idea Suggestions
```javascript
POST /api/suggest
{
    "idea": "AI-powered app for designers"
}
```

## 📁 Project Structure
```
ai-domain-finder/
├── app.py                      # Flask application
├── api_handler.py              # API request handling
├── refined_domain_hunter.py    # Core domain generation engine
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables
├── static/
│   ├── style.css              # Main styles
│   ├── script.js              # Frontend logic
│   └── liquid-bg.js           # Three.js background
├── templates/
│   └── index.html             # Main HTML template
├── logos/
│   ├── light.png              # Light theme logo
│   └── dark.png               # Dark theme logo
└── merged_output.txt          # Research context
```

## 🚀 Usage

### Classic Hunt Mode
1. Select "Classic Hunt" tab
2. Choose number of domains (10-500)
3. Select generation strategy
4. Pick checking method
5. Click "Start Hunting"
6. View results with availability status

### Business Idea Mode
1. Select "Business Idea" tab
2. Describe your business concept
3. Click "Find Perfect Domains"
4. Review AI-curated suggestions

## 🎨 Customization

### Theme Colors
Edit CSS variables in `style.css`:
```css
:root[data-theme="light"] {
    --bg-primary: #ffffff;
    --text-primary: #000000;
    --glass-bg: rgba(255, 255, 255, 0.25);
}
```

### Logo
Replace files in `/logos/` directory:
- `light.png` - Logo for light theme
- `dark.png` - Logo for dark theme

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- OpenRouter for AI model access
- WhoAPI & WhoisXML for domain checking services
- Three.js for beautiful animations
- The open-source community

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Contact: ioannis@viralpassion.gr
