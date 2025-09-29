# 🚀 AI Domain Hunter - Refined Edition

A sophisticated AI-powered domain discovery and evaluation system that leverages Claude, GPT-4, and Gemini to find high-value .ai domains through pattern recognition, market analysis, and intelligent scoring.

## 🎯 What This System Does

This system represents months of research and optimization to create the ultimate domain hunting tool. It:

1. **Analyzes 32 Research Documents**: Uses Gemini's 1M token capacity to process comprehensive domain research
2. **Discovers Patterns**: AI models identify valuable patterns without templates - letting insights emerge from data
3. **Generates Premium Domains**: Claude and GPT create domains based on discovered patterns
4. **Verifies Availability**: Three-tier checking system (DNS → WHOIS → Marketplace)
5. **Scores Investment Potential**: Dual-model evaluation for accurate domain valuation

## 📊 Key Features

### Three Generation Modes
- **Basic**: Simple keyword combinations for quick results
- **Hybrid**: 40% proven patterns + 40% current trends + 20% creative combinations
- **Emergent**: AI discovers its own patterns without human templates
- **Enhanced**: Uses Gemini to analyze merged research document for data-driven insights

### Three Checking Strategies
- **DNS**: Fast checking via DNS records (quick but less accurate)
- **Smart**: DNS + marketplace scan for promising domains (balanced)
- **Robust**: DNS + WHOIS + full marketplace verification (most accurate)

### Advanced Features
- **Previous Domain Memory**: Never checks the same domain twice
- **Marketplace Integration**: Checks Afternic, Sedo, Dan.com, GoDaddy
- **Performance Tracking**: Detailed timing and metrics
- **Organized Output**: Separate folders for logs, results, debug data
- **Rate Limiting**: Respects API and WHOIS server limits

## 🛠 Installation

### Prerequisites
- Python 3.8+
- OpenRouter API key (get from https://openrouter.ai)

### Step 1: Install Dependencies
```bashpip install openai aiohttp pandas dnspython python-whois python-dotenv

### Step 2: Set Up API Key
Create a `.env` file in the project directory:
```bashecho "OPENROUTER_API_KEY=your-key-here" > .env

### Step 3: Prepare Research Document (Optional but Recommended)
Place your `merged_output.txt` (combined research documents) in the project directory for enhanced analysis.

## 🎮 Usage

### Quick Start - Test Run
```bashpython hunt.py quick  # 50 domains, fast DNS checking

### Standard Hunt
```bashpython hunt.py standard  # 200 domains, balanced approach

### Deep Hunt with Verification
```bashpython hunt.py deep  # 500 domains with WHOIS verification

### Enhanced Hunt (BEST - Uses Research Document)
```bashpython hunt.py enhanced  # Uses merged_output.txt for pattern discovery

### Custom Parameters
```bashpython hunt.py --count 300 --mode hybrid --strategy robust --score-top 100Parameters:
--count      Number of domains to generate (50-1000)
--mode       Generation mode: basic|hybrid|emergent|enhanced
--strategy   Checking strategy: dns|smart|robust
--score-top  Number of domains to score with AI (10-200)
--document   Path to research document (default: merged_output.txt)

## 📁 File StructureProject/
├── refined_domain_hunter.py  # Main system engine
├── hunt.py                   # Command-line launcher
├── test_refined.py          # Testing without API calls
├── merged_output.txt        # Research document (32 docs combined)
├── .env                     # API key configuration
│
├── logs/                    # Execution logs (auto-created)
├── results/                 # Domain results CSV files (auto-created)
├── debug/                   # Debug info and analysis (auto-created)
└── data/                    # Cache and temporary files (auto-created)

## 🔄 How It Works

### Phase 1: Document Analysis (Enhanced Mode Only)
1. Loads `merged_output.txt` containing 32 research documents
2. Sends entire document to Gemini for comprehensive analysis
3. Gemini identifies patterns, trends, and opportunities
4. Analysis saved to debug folder for review

### Phase 2: Domain Generation
1. **Enhanced Mode**: Uses Gemini's analysis to inform generation
2. **Other Modes**: Uses predefined strategies or emergent discovery
3. Claude and GPT generate domains based on mode
4. Excludes previously checked domains automatically

### Phase 3: Availability Checking
1. **DNS Check**: Quick verification of DNS records
2. **WHOIS Check**: Legal registration verification (robust mode)
3. **Marketplace Check**: Scans major domain marketplaces
4. Results cached to avoid duplicate checks

### Phase 4: Domain Scoring
1. Both Claude and GPT evaluate each domain
2. Scoring based on:
   - Cognitive fluency and memorability
   - Strategic inevitability
   - Market potential
   - Pattern matches from research
3. Final score is average of both models

### Phase 5: Results Output
1. Main CSV with all domains and scores
2. Separate CSV with available domains only
3. Performance metrics in debug folder
4. Detailed logs of entire execution

## 📊 Output Files

### Results Folder
- `hunt_[mode]_[strategy]_[timestamp].csv` - All domains with scores
- `available_domains_[timestamp].csv` - Only available domains
- `enhanced_available_[timestamp].csv` - Top available with detailed scoring

### Debug Folder  
- `gemini_analysis_[timestamp].txt` - Full document analysis
- `performance_[timestamp].json` - Execution metrics
- `enhanced_analysis_meta_[timestamp].json` - Analysis metadata

### Logs Folder
- `domain_hunt_[timestamp].log` - Complete execution log

## ⚙️ Configuration

The system uses sensible defaults but can be customized:

### Models Used
- **Claude**: anthropic/claude-3-5-sonnet-20241022
- **GPT**: openai/gpt-4o  
- **Gemini**: google/gemini-2.0-flash-exp

### Rate Limits
- DNS: 50 domains per batch, 1 second delay
- WHOIS: 10 domains max, 2 second delay
- Marketplace: 5 domains max, 2 second delay
- API calls: Automatic rate limiting

## 🏆 Best Practices

1. **Always use Enhanced Mode** if you have research documents
2. **Start with Smart strategy** for balanced speed/accuracy
3. **Use Robust strategy** for domains you're serious about
4. **Check logs** for detailed execution information
5. **Review debug folder** to understand AI reasoning
6. **Don't exceed 500 domains** per run to avoid rate limits

## 🔍 Understanding Results

### Domain Status Types
- `dns_available` - No DNS record found (potentially available)
- `dns_taken` - DNS record exists (likely taken)
- `whois_available` - WHOIS confirms availability
- `whois_registered` - WHOIS shows registration
- `marketplace_listed` - Found on domain marketplace

### Score Interpretation
- **8-10**: Premium domain, high investment potential
- **6-8**: Good domain, worth considering
- **4-6**: Average domain, niche use cases
- **0-4**: Low value, probably skip

## 🐛 Troubleshooting

### "API key not found"
- Check `.env` file exists and contains valid key
- Ensure key starts with `sk-or-v1-`

### "Import error"
- Run: `pip install -r requirements_refined.txt`
- Ensure Python 3.8+

### "merged_output.txt not found"  
- Enhanced mode requires research document
- Use standard mode without it

### Rate limit errors
- Reduce `--count` parameter
- Use `dns` strategy instead of `robust`
- Wait before running again

## 📈 Performance Tips

1. **For Speed**: Use `dns` strategy with lower count
2. **For Accuracy**: Use `robust` strategy with WHOIS
3. **For Quality**: Use `enhanced` mode with research
4. **For Discovery**: Use `emergent` mode to find new patterns

## 🎯 Research Methodology

This system is based on extensive research including:
- Analysis of 10,000+ domain sales
- Study of cognitive fluency in naming
- Market trend analysis
- Cultural and linguistic factors
- Investment pattern recognition

The AI models don't just combine keywords - they understand what makes domains valuable based on real market data.

## 📝 License

This project is for educational and research purposes. Please respect:
- API rate limits and terms of service
- WHOIS server usage policies  
- Domain marketplace terms
- Intellectual property rights

## 🤝 Acknowledgments

Built with:
- OpenRouter API for multi-model access
- Claude 3.5 Sonnet for generation and evaluation
- GPT-4 for generation and evaluation
- Gemini 2.0 Flash for document analysis
- DNS and WHOIS libraries for verification

---

Created with extensive research and optimization for finding premium .ai domains.


domain_hunt_[timestamp].log - Complete execution log

⚙️ Configuration
The system uses sensible defaults but can be customized:
Models Used

Claude: anthropic/claude-3-5-sonnet-20241022
GPT: openai/gpt-4o
Gemini: google/gemini-2.0-flash-exp

Rate Limits

DNS: 50 domains per batch, 1 second delay
WHOIS: 10 domains max, 2 second delay
Marketplace: 5 domains max, 2 second delay
API calls: Automatic rate limiting

🏆 Best Practices

Always use Enhanced Mode if you have research documents
Start with Smart strategy for balanced speed/accuracy
Use Robust strategy for domains you're serious about
Check logs for detailed execution information
Review debug folder to understand AI reasoning
Don't exceed 500 domains per run to avoid rate limits

🔍 Understanding Results
Domain Status Types

dns_available - No DNS record found (potentially available)
dns_taken - DNS record exists (likely taken)
whois_available - WHOIS confirms availability
whois_registered - WHOIS shows registration
marketplace_listed - Found on domain marketplace

Score Interpretation

8-10: Premium domain, high investment potential
6-8: Good domain, worth considering
4-6: Average domain, niche use cases
0-4: Low value, probably skip

🐛 Troubleshooting
"API key not found"

Check .env file exists and contains valid key
Ensure key starts with sk-or-v1-

"Import error"

Run: pip install -r requirements_refined.txt
Ensure Python 3.8+

"merged_output.txt not found"

Enhanced mode requires research document
Use standard mode without it

Rate limit errors

Reduce --count parameter
Use dns strategy instead of robust
Wait before running again

📈 Performance Tips

For Speed: Use dns strategy with lower count
For Accuracy: Use robust strategy with WHOIS
For Quality: Use enhanced mode with research
For Discovery: Use emergent mode to find new patterns

🎯 Research Methodology
This system is based on extensive research including:

Analysis of 10,000+ domain sales
Study of cognitive fluency in naming
Market trend analysis
Cultural and linguistic factors
Investment pattern recognition

The AI models don't just combine keywords - they understand what makes domains valuable based on real market data.
📝 License
This project is for educational and research purposes. Please respect:

API rate limits and terms of service
WHOIS server usage policies
Domain marketplace terms
Intellectual property rights

🤝 Acknowledgments
Built with:

OpenRouter API for multi-model access
Claude 3.5 Sonnet for generation and evaluation
GPT-4 for generation and evaluation
Gemini 2.0 Flash for document analysis
DNS and WHOIS libraries for verification


Created with extensive research and optimization for finding premium .ai domains.

## 🚀 **Η ΚΑΛΥΤΕΡΗ ΕΝΤΟΛΗ - The Ultimate Domain Hunt**

Για το πιο δυνατό domain hunting με ΟΛΑ τα features:
```bash
python hunt.py enhanced --count 500 --strategy robust --score-top 100
Ή αν θέλεις να το κάνεις πιο explicit:
bashpython hunt.py --count 500 --mode enhanced --strategy robust --score-top 100 --document merged_output.txt