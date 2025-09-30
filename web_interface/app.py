from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import asyncio
import sys
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from web_interface.api_handler import DomainSuggestionAPI
from refined_domain_hunter import RefinedDomainHunter, DomainConfig

app = Flask(__name__)
CORS(app)

api = DomainSuggestionAPI()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/logo/<theme>')
def serve_logo(theme):
    logo_map = {
        'light': 'logo_icon-02.svg',
        'dark': 'logo_icon-01.svg'
    }
    
    logo_file = logo_map.get(theme, 'logo_icon-02.svg')
    logo_path = Path(__file__).parent / 'public' / 'assets' / 'logo' / 'svg' / logo_file
    
    if logo_path.exists():
        return send_file(str(logo_path), mimetype='image/svg+xml')
    return '', 404

@app.route('/favicon.ico')
def favicon():
    # Try PNG first
    png_path = Path(__file__).parent / 'public' / 'assets' / 'logo' / 'png' / 'logo-02.png'
    if png_path.exists():
        return send_file(str(png_path), mimetype='image/png')
    
    return '', 404

@app.route('/api/suggest', methods=['POST'])
def suggest_domains():
    data = request.json
    idea = data.get('idea', '')
    
    if not idea:
        return jsonify({'error': 'Please describe your idea'}), 400
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(api.generate_suggestions(idea))
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/hunt', methods=['POST'])
def classic_hunt():
    data = request.json
    count = data.get('count', 50)
    mode = data.get('mode', 'hybrid')
    check_strategy = data.get('check_strategy', 'smart')
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Create hunter with API + DNS ONLY - NO MARKETPLACE CHECKING
        config = DomainConfig()
        config.whoapi_key = os.getenv('WHOAPI_KEY', '')
        config.whoisxml_api_key = os.getenv('WHOISXML_API_KEY', '')
        config.use_stealth_mode = False  # FORCE DISABLE stealth mode
        config.check_strategy = 'api_only'  # API + DNS only, no marketplace
        
        hunter = RefinedDomainHunter(config)
        
        # Force disable ALL marketplace checking
        if hasattr(hunter, 'checker'):
            hunter.checker.enable_stealth = False
            hunter.checker.stealth_budget = 0
            # Also disable easy marketplace checking
            if hasattr(hunter.checker, 'enable_marketplace'):
                hunter.checker.enable_marketplace = False
        
        df = loop.run_until_complete(
            hunter.hunt(
                count=count,
                mode=mode,
                check_strategy=check_strategy,
                score_top_n=min(count//2, 50)  # Score half or max 50
            )
        )
        
        # Cleanup
        loop.run_until_complete(hunter.cleanup())
        
        domains = df.to_dict('records')
        
        return jsonify({
            'domains': domains,
            'total': len(domains),
            'available': len([d for d in domains if d.get('available', False)])
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8001)