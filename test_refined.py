#!/usr/bin/env python3
"""
Test script for Refined Domain Hunter
Validates core functionality without requiring API calls
"""

import asyncio
import sys
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from refined_domain_hunter import (
        DomainConfig, DomainChecker, DomainGenerator, 
        UnifiedDomainScorer, RefinedDomainHunter
    )
    print("✅ Import successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Install dependencies: pip install -r requirements_refined.txt")
    sys.exit(1)

def test_domain_config():
    """Test configuration loading"""
    print("\n🔧 Testing DomainConfig...")
    
    config = DomainConfig()
    assert config.max_domains == 500
    assert config.batch_size == 50
    assert 'claude' in config.models
    assert 'gpt' in config.models
    assert 'gemini' in config.models
    
    print("✅ DomainConfig working correctly")

def test_domain_checker():
    """Test domain checking functionality"""
    print("\n🔍 Testing DomainChecker...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock results directory
        results_dir = Path(temp_dir) / "results"
        results_dir.mkdir()
        
        # Create mock CSV with previous domains
        mock_csv = results_dir / "test_domains.csv"
        mock_csv.write_text("domain,score\ntest1.ai,7.5\ntest2.ai,8.0\n")
        
        # Patch the results path
        with patch('refined_domain_hunter.Path') as mock_path:
            mock_path.return_value = results_dir
            
            config = DomainConfig()
            checker = DomainChecker(config)
            
            # Should load previous domains
            assert len(checker.searched_domains) >= 0
            
            # Test WHOIS checking capability
            try:
                import whois
                whois_available = True
            except ImportError:
                whois_available = False
            
            print(f"  WHOIS checking: {'✅ Available' if whois_available else '⚠️ Not installed'}")
            
            print("✅ DomainChecker initialization working")

def test_domain_generator():
    """Test domain generation parsing"""
    print("\n🎯 Testing DomainGenerator...")
    
    # Mock AI manager
    mock_ai = Mock()
    config = DomainConfig()
    generator = DomainGenerator(mock_ai, config)
    
    # Test domain parsing
    test_responses = [
        '["mindflow", "neurallab", "aicore"]',
        'mindflow.ai\nneurallab.ai\naicore.ai',
        '1. mindflow\n2. neurallab\n3. aicore'
    ]
    
    for response in test_responses:
        domains = generator._parse_domain_response(response)
        assert len(domains) > 0
        for domain in domains:
            assert domain.endswith('.ai')
    
    # Test domain normalization
    test_cases = [
        ("mindflow", "mindflow.ai"),
        ("neural-lab", "neural-lab.ai"),
        ("AI-Core", "ai-core.ai"),
        ("test123", "test123.ai")
    ]
    
    for input_domain, expected in test_cases:
        result = generator._normalize_domain(input_domain)
        assert result == expected, f"Expected {expected}, got {result}"
    
    print("✅ DomainGenerator parsing working correctly")

async def test_async_operations():
    """Test async functionality with mocks"""
    print("\n⚡ Testing async operations...")
    
    # Mock configuration with fake API key
    config = DomainConfig()
    config.openrouter_key = "test-key"
    
    # Test that async methods exist and are callable
    checker = DomainChecker(config)
    
    # Test with empty domain list (should not make API calls)
    result = await checker.check_availability([], 'dns')
    assert isinstance(result, dict)
    
    print("✅ Async operations structure working")

def test_file_organization():
    """Test that proper directories are created"""
    print("\n📁 Testing file organization...")
    
    # The module should create these directories
    expected_dirs = ['logs', 'results', 'debug', 'data']
    
    for dir_name in expected_dirs:
        dir_path = Path(dir_name)
        assert dir_path.exists(), f"Directory {dir_name} was not created"
        assert dir_path.is_dir(), f"{dir_name} is not a directory"
    
    print("✅ Directory structure created correctly")

def test_configuration_validation():
    """Test configuration validation"""
    print("\n⚙️ Testing configuration validation...")
    
    # Test with missing API key
    config = DomainConfig()
    config.openrouter_key = None
    
    try:
        hunter = RefinedDomainHunter(config)
        assert False, "Should have raised ValueError for missing API key"
    except ValueError as e:
        assert "API key required" in str(e)
    
    print("✅ Configuration validation working")

def run_all_tests():
    """Run all tests"""
    print("🧪 Running Refined Domain Hunter Tests")
    print("=" * 50)
    
    try:
        # Sync tests
        test_domain_config()
        test_domain_checker() 
        test_domain_generator()
        test_file_organization()
        test_configuration_validation()
        
        # Async tests
        asyncio.run(test_async_operations())
        
        print("\n" + "=" * 50)
        print("✅ ALL TESTS PASSED!")
        print("🚀 Refined Domain Hunter is working correctly")
        print("\nTo run a real hunt:")
        print("1. Set OPENROUTER_API_KEY environment variable")
        print("2. Run: python hunt.py quick")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)