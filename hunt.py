#!/usr/bin/env python3
"""
Quick Launcher for Refined Domain Hunter
Provides easy commands for different hunting scenarios
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from refined_domain_hunter import RefinedDomainHunter, DomainConfig
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all dependencies are installed:")
    print("pip install -r requirements_refined.txt")
    sys.exit(1)

def create_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Refined AI Domain Hunter - Streamlined domain discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hunt.py quick                    # Quick 50 domain hunt
  python hunt.py standard                 # Standard 200 domain hunt  
  python hunt.py deep --count 500         # Deep 500 domain hunt
  python hunt.py emergent --mode emergent # Emergent AI discovery
  python hunt.py enhanced                 # Enhanced with document analysis
  python hunt.py --count 100 --mode enhanced --document merged_output.txt
        """
    )
    
    # Preset commands
    parser.add_argument('preset', nargs='?', 
                       choices=['quick', 'standard', 'deep', 'emergent', 'enhanced'],
                       help='Preset hunting configurations')
    
    # Custom parameters
    parser.add_argument('--count', type=int, default=200,
                       help='Number of domains to generate (default: 200)')
    
    parser.add_argument('--mode', choices=['basic', 'hybrid', 'emergent', 'enhanced'],
                       default='hybrid',
                       help='Generation mode (default: hybrid)')
    
    parser.add_argument('--strategy', choices=['dns', 'smart', 'robust'],
                       default='smart',
                       help='Checking strategy (default: smart)')
    
    parser.add_argument('--score-top', type=int, default=50,
                       help='Number of top domains to score (default: 50)')
    
    parser.add_argument('--document', type=str, default="merged_output.txt",
                       help='Research document for enhanced analysis (default: merged_output.txt)')
    
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without executing')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    return parser

async def run_hunt(args):
    """Execute the domain hunt with given parameters"""
    
    # Apply preset configurations
    if args.preset == 'quick':
        count, mode, strategy, score_top = 50, 'basic', 'dns', 20
    elif args.preset == 'standard':
        count, mode, strategy, score_top = 200, 'hybrid', 'smart', 50
    elif args.preset == 'deep':
        count, mode, strategy, score_top = 500, 'hybrid', 'robust', 100
    elif args.preset == 'emergent':
        count, mode, strategy, score_top = 300, 'emergent', 'smart', 75
    elif args.preset == 'enhanced':
        count, mode, strategy, score_top = 200, 'enhanced', 'smart', 50
    else:
        # Use custom parameters
        count = args.count
        mode = args.mode
        strategy = args.strategy
        score_top = args.score_top
    
    # Show configuration
    print("🎯 HUNT CONFIGURATION")
    print("=" * 40)
    print(f"Domains: {count}")
    print(f"Mode: {mode}")
    print(f"Strategy: {strategy}")
    print(f"Score top: {score_top}")
    if mode == 'enhanced' or args.preset == 'enhanced':
        print(f"Document: {args.document}")
    print("=" * 40)
    
    if args.dry_run:
        print("🔍 DRY RUN - Would execute with above settings")
        return
    
    # Execute hunt
    try:
        config = DomainConfig()
        hunter = RefinedDomainHunter(config)
        
        # Choose hunt method based on mode
        if mode == 'enhanced' or args.preset == 'enhanced':
            # Check if document exists
            from pathlib import Path
            if not Path(args.document).exists():
                print(f"❌ Document {args.document} not found!")
                print("Enhanced mode requires merged_output.txt or specified document")
                return None
            
            df = await hunter.hunt_with_document_analysis(
                count=count,
                check_strategy=strategy,
                score_top_n=score_top,
                document_path=args.document
            )
        else:
            df = await hunter.hunt(
                count=count,
                mode=mode,
                check_strategy=strategy,
                score_top_n=score_top
            )
        
        print(f"\n✅ Hunt completed successfully!")
        print(f"📊 Results: {len(df)} domains processed")
        
        available = df[df['available'] == True]
        if not available.empty:
            print(f"💎 {len(available)} available domains found!")
            
        return df
        
    except Exception as e:
        print(f"❌ Hunt failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return None

def main():
    """Main launcher function"""
    parser = create_parser()
    args = parser.parse_args()
    
    print("🚀 Refined Domain Hunter Launcher")
    
    # Check for API key
    config = DomainConfig()
    if not config.openrouter_key:
        print("❌ OpenRouter API key not found!")
        print("Set environment variable: export OPENROUTER_API_KEY='your-key'")
        print("Or create a .env file with: OPENROUTER_API_KEY=your-key")
        sys.exit(1)
    
    # Run the hunt
    result = asyncio.run(run_hunt(args))
    
    if result is not None:
        print("\n📁 Check the 'results/' folder for detailed output")
        print("📊 Check the 'logs/' folder for execution logs")
        print("🔧 Check the 'debug/' folder for debug information")
    
    return 0 if result is not None else 1

if __name__ == "__main__":
    sys.exit(main())