"""
End-to-End Workflow Tests
"""
import pytest
import asyncio
from typing import Dict, Any
from fastapi.testclient import TestClient

from main import app
from services.package_service import PackageService
from database.client import get_supabase


class TestE2EWorkflows:
    """End-to-end workflow tests"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self, client):
        """Get authenticated headers"""
        # Register test user
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "e2e_test@example.com",
                "full_name": "E2E Test User"
            }
        )
        assert response.status_code == 200
        token = response.json()["access_token"]
        
        return {"Authorization": f"Bearer {token}"}
    
    def test_complete_brand_generation_workflow(self, client, auth_headers):
        """Test complete brand generation workflow"""
        
        # Step 1: Generate business names
        response = client.post(
            "/api/v1/generation/names",
            headers=auth_headers,
            json={
                "description": "An AI-powered fitness app that creates personalized workout plans",
                "industry": "health_fitness",
                "style": "modern",
                "keywords": ["fit", "ai", "personal"],
                "count": 5
            }
        )
        assert response.status_code == 200
        names_result = response.json()
        assert len(names_result["names"]) > 0
        
        # Pick first name
        selected_name = names_result["names"][0]["name"]
        
        # Step 2: Check domain availability
        response = client.post(
            "/api/v1/generation/domains",
            headers=auth_headers,
            json={
                "domains": [
                    f"{selected_name.lower()}.com",
                    f"{selected_name.lower()}.ai",
                    f"get{selected_name.lower()}.com"
                ]
            }
        )
        assert response.status_code == 200
        domains_result = response.json()
        assert "results" in domains_result
        
        # Step 3: Generate logo
        response = client.post(
            "/api/v1/generation/logos",
            headers=auth_headers,
            json={
                "business_name": selected_name,
                "description": "AI fitness app",
                "style": "modern",
                "colors": ["#4A90E2", "#50E3C2"],
                "count": 2
            }
        )
        assert response.status_code == 200
        logos_result = response.json()
        assert len(logos_result["logos"]) > 0
        
        # Step 4: Generate color palette
        response = client.post(
            "/api/v1/generation/colors",
            headers=auth_headers,
            json={
                "business_name": selected_name,
                "description": "AI fitness app",
                "theme": "vibrant",
                "count": 3
            }
        )
        assert response.status_code == 200
        colors_result = response.json()
        assert len(colors_result["palettes"]) > 0
        
        # Step 5: Generate taglines
        response = client.post(
            "/api/v1/generation/taglines",
            headers=auth_headers,
            json={
                "business_name": selected_name,
                "description": "AI fitness app",
                "tone": "inspirational",
                "target_audience": "Fitness enthusiasts who want personalized training",
                "count": 5
            }
        )
        assert response.status_code == 200
        taglines_result = response.json()
        assert len(taglines_result["taglines"]) > 0
    
    def test_package_generation_workflow(self, client, auth_headers):
        """Test package generation workflow"""
        
        # Generate complete package
        response = client.post(
            "/api/v1/generation/package",
            headers=auth_headers,
            json={
                "description": "A sustainable fashion marketplace connecting eco-conscious consumers with ethical brands",
                "industry": "fashion",
                "style_preferences": {
                    "name_style": "modern",
                    "logo_style": "minimalist",
                    "color_theme": "earth",
                    "tagline_tone": "inspirational"
                },
                "include_services": ["name", "logo", "color", "tagline"]
            }
        )
        
        assert response.status_code == 200
        package_result = response.json()
        
        # Verify package contents
        assert "project_id" in package_result
        assert "business_name" in package_result
        assert package_result.get("names") is not None
        assert package_result.get("logos") is not None
        assert package_result.get("color_palettes") is not None
        assert package_result.get("taglines") is not None
    
    def test_regeneration_workflow(self, client, auth_headers):
        """Test regeneration with feedback workflow"""
        
        # Step 1: Generate initial names
        response = client.post(
            "/api/v1/generation/names",
            headers=auth_headers,
            json={
                "description": "A meditation app for busy professionals",
                "count": 3
            }
        )
        assert response.status_code == 200
        initial_result = response.json()
        generation_id = initial_result["generation_id"]
        
        # Step 2: Regenerate with feedback
        response = client.post(
            "/api/v1/generation/regenerate/name",
            headers=auth_headers,
            json={
                "generation_id": generation_id,
                "feedback": "Make the names more calming and zen-like, avoid tech-sounding names"
            }
        )
        assert response.status_code == 200
        regenerated = response.json()
        assert "result" in regenerated
    
    def test_project_workflow(self, client, auth_headers):
        """Test project management workflow"""
        
        # Step 1: Create project
        response = client.post(
            "/api/v1/projects",
            headers=auth_headers,
            json={
                "name": "My Startup Brand",
                "description": "Brand assets for my new startup"
            }
        )
        assert response.status_code == 200
        project = response.json()
        project_id = project["id"]
        
        # Step 2: Generate assets for project
        response = client.post(
            "/api/v1/generation/names",
            headers=auth_headers,
            json={
                "description": "Tech startup",
                "project_id": project_id,
                "count": 3
            }
        )
        assert response.status_code == 200
        
        # Step 3: Get project details
        response = client.get(
            f"/api/v1/projects/{project_id}",
            headers=auth_headers
        )
        assert response.status_code == 200
        project_details = response.json()
        assert len(project_details["generations"]) > 0
        
        # Step 4: Update project
        response = client.put(
            f"/api/v1/projects/{project_id}",
            headers=auth_headers,
            json={
                "status": "completed"
            }
        )
        assert response.status_code == 200
    
    def test_rate_limiting(self, client, auth_headers):
        """Test rate limiting"""
        
        # Make multiple rapid requests
        responses = []
        for i in range(15):
            response = client.post(
                "/api/v1/generation/names",
                headers=auth_headers,
                json={
                    "description": f"Test business {i}",
                    "count": 1
                }
            )
            responses.append(response)
            
            # Should hit rate limit eventually
            if response.status_code == 429:
                break
        
        # Verify rate limit was hit
        rate_limited = any(r.status_code == 429 for r in responses)
        assert rate_limited, "Rate limit should be triggered"
    
    @pytest.mark.asyncio
    async def test_async_generation_workflow(self):
        """Test async generation workflow"""
        
        service = PackageService()
        
        # Test async package generation
        result = await service.generate(
            description="An AI-powered education platform",
            user_id="test_user_async",
            include_services=["name", "tagline"]
        )
        
        assert result is not None
        assert "business_name" in result
        assert "taglines" in result
    
    def test_error_handling_workflow(self, client, auth_headers):
        """Test error handling in workflows"""
        
        # Test with invalid data
        response = client.post(
            "/api/v1/generation/names",
            headers=auth_headers,
            json={
                "description": "x",  # Too short
                "count": 100  # Too many
            }
        )
        assert response.status_code == 400
        
        # Test with missing required fields
        response = client.post(
            "/api/v1/generation/taglines",
            headers=auth_headers,
            json={
                "description": "Test business"
                # Missing business_name
            }
        )
        assert response.status_code == 400
        
        # Test unauthorized access
        response = client.get(
            "/api/v1/user/profile"
            # No auth headers
        )
        assert response.status_code == 401
    
    def test_user_journey_workflow(self, client):
        """Test complete user journey from signup to generation"""
        
        # Step 1: Register
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "journey_test@example.com",
                "full_name": "Journey Test"
            }
        )
        assert response.status_code == 200
        auth_data = response.json()
        token = auth_data["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Step 2: Check usage
        response = client.get(
            "/api/v1/user/usage",
            headers=headers
        )
        assert response.status_code == 200
        usage = response.json()
        assert usage["remaining"] > 0
        
        # Step 3: Generate content
        response = client.post(
            "/api/v1/generation/names",
            headers=headers,
            json={
                "description": "Online learning platform for coding",
                "count": 3
            }
        )
        assert response.status_code == 200
        
        # Step 4: Check history
        response = client.get(
            "/api/v1/user/history",
            headers=headers
        )
        assert response.status_code == 200
        history = response.json()
        assert len(history["recent_generations"]) > 0
        
        # Step 5: Check updated usage
        response = client.get(
            "/api/v1/user/usage",
            headers=headers
        )
        assert response.status_code == 200
        updated_usage = response.json()
        assert updated_usage["generation_count"] > usage["generation_count"]