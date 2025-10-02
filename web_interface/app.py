from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import asyncio
import sys
from pathlib import Path
import os
from dotenv import load_dotenv
from supabase import create_client
import json
from datetime import datetime
import uuid

load_dotenv()

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_SERVICE_KEY')  # Service key για backend
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

sys.path.insert(0, str(Path(__file__).parent.parent))

from web_interface.api_handler import DomainSuggestionAPI
from refined_domain_hunter import RefinedDomainHunter, DomainConfig

app = Flask(__name__)
CORS(app)

api = DomainSuggestionAPI()

def get_existing_classic_domains():
    """Get all domains from previous classic hunts"""
    if not supabase:
        return set()
    
    try:
        # Get all classic searches
        searches = supabase.table('searches').select('id').eq('type', 'classic').execute()
        
        all_domains = set()
        for search in searches.data:
            # Get domains for each search
            domains = supabase.table('domains').select('domain').eq('search_id', search['id']).execute()
            for d in domains.data:
                all_domains.add(d['domain'])
        
        print(f"📊 Loaded {len(all_domains)} existing classic domains from Supabase")
        return all_domains
        
    except Exception as e:
        print(f"Error loading domains: {e}")
        return set()

def upload_hunt_files(search_id, df):
    """Upload CSV and logs to Supabase Storage"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save CSV
        csv_filename = f"{search_id}/results_{timestamp}.csv"
        csv_data = df.to_csv(index=False)
        
        supabase.storage.from_('domain-hunt-files').upload(
            csv_filename,
            csv_data.encode(),
            {'content-type': 'text/csv'}
        )
        
        # Save logs if they exist
        log_files = Path('logs').glob('*.log')
        for log_file in list(log_files)[-1:]:  # Only latest log
            with open(log_file, 'r') as f:
                log_content = f.read()
            
            log_filename = f"{search_id}/log_{timestamp}.txt"
            supabase.storage.from_('domain-hunt-files').upload(
                log_filename,
                log_content.encode(),
                {'content-type': 'text/plain'}
            )
        
        return True
    except Exception as e:
        print(f"Upload failed: {e}")
        return False

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
        
        # Save to Supabase Storage
        if supabase and results.get('suggestions'):
            try:
                import uuid
                from datetime import datetime
                import pandas as pd
                
                # Create DataFrame from results
                df = pd.DataFrame(results['suggestions'])
                
                # Upload to storage
                search_id = str(uuid.uuid4())
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                csv_content = df.to_csv(index=False)
                file_path = f"business/{search_id}/results_{timestamp}.csv"
                
                supabase.storage.from_('domain-hunt-files').upload(
                    file_path,
                    csv_content.encode('utf-8'),
                    {'content-type': 'text/csv'}
                )
                
                print(f"📤 Uploaded business results to: {file_path}")
                
            except Exception as e:
                print(f"Storage upload failed: {e}")
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/hunt', methods=['POST'])
@app.route('/api/hunt', methods=['POST'])
def classic_hunt():
    data = request.json
    count = data.get('count', 50)
    mode = data.get('mode', 'hybrid')
    check_strategy = data.get('check_strategy', 'smart')
    
    # Get existing domains from Supabase
    existing_domains = get_existing_classic_domains()
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        config = DomainConfig()
        config.whoapi_key = os.getenv('WHOAPI_KEY', '')
        config.whoisxml_api_key = os.getenv('WHOISXML_API_KEY', '')
        config.use_stealth_mode = False
        config.check_strategy = 'api_only'
        
        hunter = RefinedDomainHunter(config)
        
        # Add existing domains to exclude them
        if existing_domains:
            hunter.checker.searched_domains.update(existing_domains)
            print(f"🚫 Excluding {len(existing_domains)} already-hunted domains")
        
        # Disable marketplace checking
        if hasattr(hunter, 'checker'):
            hunter.checker.enable_stealth = False
            hunter.checker.stealth_budget = 0
            if hasattr(hunter.checker, 'enable_marketplace'):
                hunter.checker.enable_marketplace = False
        
        df = loop.run_until_complete(
            hunter.hunt(
                count=count,
                mode=mode,
                check_strategy=check_strategy,
                score_top_n=min(count//2, 50)
            )
        )
        
        loop.run_until_complete(hunter.cleanup())
        domains = df.to_dict('records')

        # Save to Supabase
        if supabase:
            try:
                search_data = supabase.table('searches').insert({
                    'type': 'classic',
                    'input_data': data,
                    'results_summary': {
                        'total': len(domains),
                        'available': len([d for d in domains if d.get('available', False)])
                    }
                }).execute()
                
                search_id = search_data.data[0]['id']
                
                # Upload ALL files
                upload_all_hunt_files(search_id, df)
                
            except Exception as e:
                print(f"Supabase error: {e}")

        return jsonify({
            'domains': domains,
            'total': len(domains),
            'available': len([d for d in domains if d.get('available', False)])
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def upload_all_hunt_files(search_id, df):
    """Upload ALL hunt files to Supabase Storage"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        uploaded_files = []
        
        # 1. Upload main CSV results
        csv_data = df.to_csv(index=False)
        csv_path = f"hunts/{search_id}/results_{timestamp}.csv"
        supabase.storage.from_('domain-hunt-files').upload(
            csv_path, csv_data.encode(), {'content-type': 'text/csv'}
        )
        uploaded_files.append(csv_path)
        
        # 2. Upload latest log file
        log_dir = Path('logs')
        if log_dir.exists():
            log_files = sorted(log_dir.glob('*.log'), key=lambda x: x.stat().st_mtime)
            if log_files:
                latest_log = log_files[-1]
                with open(latest_log, 'r') as f:
                    log_content = f.read()
                log_path = f"hunts/{search_id}/logs/{latest_log.name}"
                supabase.storage.from_('domain-hunt-files').upload(
                    log_path, log_content.encode(), {'content-type': 'text/plain'}
                )
                uploaded_files.append(log_path)
        
        # 3. Upload results folder CSVs
        results_dir = Path('results')
        if results_dir.exists():
            for csv_file in results_dir.glob('*.csv'):
                with open(csv_file, 'r') as f:
                    content = f.read()
                file_path = f"hunts/{search_id}/results/{csv_file.name}"
                supabase.storage.from_('domain-hunt-files').upload(
                    file_path, content.encode(), {'content-type': 'text/csv'}
                )
                uploaded_files.append(file_path)
        
        # 4. Upload debug files
        debug_dir = Path('debug')
        if debug_dir.exists():
            for json_file in debug_dir.glob('*.json'):
                with open(json_file, 'r') as f:
                    content = f.read()
                file_path = f"hunts/{search_id}/debug/{json_file.name}"
                supabase.storage.from_('domain-hunt-files').upload(
                    file_path, content.encode(), {'content-type': 'application/json'}
                )
                uploaded_files.append(file_path)
        
        print(f"✅ Uploaded {len(uploaded_files)} files to Storage:")
        for f in uploaded_files:
            print(f"   📄 {f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Storage upload failed: {e}")
        return False

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8001)