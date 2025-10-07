# Brand Package Generator - Backend Architecture Documentation

**Version:** 1.0.0  
**Framework:** FastAPI (Python 3.11+)  
**Database:** Supabase (PostgreSQL)  
**Deployment:** Railway / Render

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture Philosophy](#architecture-philosophy)
3. [Complete Directory Structure](#complete-directory-structure)
4. [Layer-by-Layer Breakdown](#layer-by-layer-breakdown)
5. [Tech Stack](#tech-stack)
6. [Setup & Installation](#setup--installation)
7. [Configuration](#configuration)
8. [Database Schema](#database-schema)
9. [API Documentation](#api-documentation)
10. [Core Components](#core-components)
11. [Service Layer](#service-layer)
12. [Data Flow](#data-flow)
13. [Development Workflow](#development-workflow)
14. [Testing Strategy](#testing-strategy)
15. [Deployment](#deployment)
16. [Security](#security)
17. [Performance & Optimization](#performance--optimization)
18. [Troubleshooting](#troubleshooting)

---

## 🎯 Project Overview

### What is Brand Package Generator?

An AI-powered platform that generates complete brand identities for startups in minutes. The backend provides RESTful APIs for:

- **Business Name Generation** - AI creates 10 brandable names based on business description
- **Domain Availability Checking** - Real-time verification across multiple registrars
- **Logo Generation** - Creates 3 unique logo concepts using multiple AI providers
- **Color Palette Creation** - Generates accessible, psychology-based color schemes
- **Tagline Generation** - Creates 5 compelling tagline variations
- **Complete Brand Packages** - Orchestrates all services for end-to-end brand creation

### Key Features

✅ **Modular Architecture** - Each service is independent and reusable  
✅ **AI-Agnostic** - Works with any text/image generation API  
✅ **Failover System** - Automatic backup if primary APIs fail  
✅ **Rate Limiting** - Built-in usage limits (2 free generations per user)  
✅ **Authentication** - JWT-based auth via Supabase  
✅ **Caching** - Intelligent research pattern caching  
✅ **Production-Ready** - Comprehensive error handling, logging, and monitoring  

---

## 🏗️ Architecture Philosophy

### Design Principles

1. **Separation of Concerns** - Clear boundaries between layers
2. **Single Responsibility** - Each module has one job
3. **Dependency Injection** - Services receive dependencies, don't create them
4. **API-First Design** - Backend is headless, frontend-agnostic
5. **Fail-Safe** - Graceful degradation when external APIs fail
6. **Scalability** - Horizontal scaling through stateless services

### Architecture Pattern: Layered Monolith

```
┌─────────────────────────────────────────────┐
│            API Layer (FastAPI)              │  ← HTTP Interface
├─────────────────────────────────────────────┤
│         Middleware (Auth, Logging)          │  ← Cross-cutting
├─────────────────────────────────────────────┤
│       Service Layer (Business Logic)        │  ← Core Logic
├─────────────────────────────────────────────┤
│    Core Components (AI, Research, Cache)    │  ← Infrastructure
├─────────────────────────────────────────────┤
│     Repository Layer (Data Access)          │  ← Data Operations
├─────────────────────────────────────────────┤
│       Database Layer (Supabase)             │  ← Persistence
└─────────────────────────────────────────────┘
```

### Why This Architecture?

- **MVP Speed** - Monolith is faster to develop and deploy
- **Simple Deployment** - One service to manage
- **Easy Debugging** - All code in one place
- **Cost-Effective** - Single server/container
- **Future-Proof** - Can extract to microservices later

---

## 📁 Complete Directory Structure

```
backend/
├── main.py                              # Application entry point
├── requirements.txt                     # Python dependencies
├── .env                                 # Environment variables (not in git)
├── .env.example                         # Environment template
├── Dockerfile                           # Container configuration
├── railway.toml                         # Railway deployment config
├── pytest.ini                           # Pytest configuration
├── alembic.ini                          # Database migration config
├── README.md                            # This file
├── .gitignore                           # Git ignore rules
│
├── core/                                # 🧠 Core infrastructure
│   ├── __init__.py
│   ├── ai_manager.py                    # Centralized AI operations
│   ├── research_loader.py               # Intelligence pattern loader
│   ├── failover_manager.py              # API failover orchestration
│   └── exceptions.py                    # Custom exception classes
│
├── services/                            # 💼 Business logic layer
│   ├── __init__.py
│   ├── base_service.py                  # Abstract base for all services
│   ├── name_service.py                  # Name generation logic
│   ├── domain_service.py                # Domain checking logic
│   ├── logo_service.py                  # Logo generation logic
│   ├── color_service.py                 # Color palette logic
│   ├── tagline_service.py               # Tagline generation logic
│   └── package_service.py               # Orchestrates complete packages
│
├── database/                            # 🗄️ Data persistence layer
│   ├── __init__.py
│   ├── client.py                        # Supabase client setup
│   ├── models.py                        # SQLAlchemy & Pydantic models
│   ├── repositories/                    # Repository pattern
│   │   ├── __init__.py
│   │   ├── base_repository.py           # Abstract repository
│   │   ├── user_repository.py           # User CRUD operations
│   │   ├── project_repository.py        # Project CRUD operations
│   │   ├── generation_repository.py     # Generation tracking
│   │   └── asset_repository.py          # Asset management
│   └── migrations/                      # Alembic migrations
│       └── versions/                    # Migration version files
│
├── api/                                 # 🌐 HTTP interface layer
│   ├── __init__.py
│   ├── dependencies.py                  # FastAPI dependencies
│   ├── routes/                          # API endpoints
│   │   ├── __init__.py
│   │   ├── auth_routes.py               # Authentication endpoints
│   │   ├── generation_routes.py         # Generation endpoints
│   │   ├── user_routes.py               # User management
│   │   ├── project_routes.py            # Project management
│   │   └── health_routes.py             # Health check
│   └── schemas/                         # Request/response models
│       ├── __init__.py
│       ├── base_schema.py               # Base Pydantic models
│       ├── request_schemas.py           # API request schemas
│       ├── response_schemas.py          # API response schemas
│       └── error_schemas.py             # Error response schemas
│
├── middleware/                          # 🛡️ Cross-cutting concerns
│   ├── __init__.py
│   ├── auth.py                          # JWT authentication
│   ├── rate_limiter.py                  # Rate limiting logic
│   ├── error_handler.py                 # Global error handling
│   ├── cors.py                          # CORS configuration
│   └── logging_middleware.py            # Request/response logging
│
├── config/                              # ⚙️ Configuration management
│   ├── __init__.py
│   ├── settings.py                      # Environment-based settings
│   ├── constants.py                     # Application constants
│   └── logging_config.py                # Logging configuration
│
├── utils/                               # 🔧 Utility functions
│   ├── __init__.py
│   ├── storage.py                       # File upload/download
│   ├── validators.py                    # Input validation
│   ├── formatters.py                    # Data formatting
│   ├── image_utils.py                   # Image processing
│   ├── text_parsers.py                  # Text parsing
│   ├── security.py                      # Security utilities
│   └── logger.py                        # Custom logger
│
├── research/                            # 📚 Intelligence patterns
│   ├── __init__.py
│   ├── loader.py                        # Pattern loading logic
│   ├── documents/                       # Raw research files
│   │   ├── name_intelligence.txt        # Name generation patterns
│   │   ├── logo_intelligence.txt        # Logo design principles
│   │   ├── color_intelligence.txt       # Color psychology
│   │   └── tagline_intelligence.txt     # Tagline frameworks
│   └── cache/                           # Processed pattern cache
│       └── .gitkeep
│
├── workers/                             # ⚡ Background tasks
│   ├── __init__.py
│   ├── celery_app.py                    # Celery configuration
│   ├── schedulers.py                    # Scheduled tasks
│   └── tasks/                           # Task definitions
│       ├── __init__.py
│       ├── generation_tasks.py          # Async generation tasks
│       └── cleanup_tasks.py             # Cleanup tasks
│
└── tests/                               # 🧪 Testing suite
    ├── __init__.py
    ├── conftest.py                      # Pytest configuration
    ├── unit/                            # Unit tests
    │   ├── test_services/               # Service tests
    │   ├── test_core/                   # Core component tests
    │   └── test_utils/                  # Utility tests
    ├── integration/                     # Integration tests
    │   ├── test_api/                    # API endpoint tests
    │   └── test_database/               # Database tests
    └── e2e/                             # End-to-end tests
        └── test_workflows.py            # Complete workflow tests
```

---

## 🔍 Layer-by-Layer Breakdown

### 1. API Layer (`/api`)

**Purpose:** HTTP interface for all client interactions

**Components:**

- **Routes** - Define API endpoints
  - `auth_routes.py` - `/api/auth/*` - Login, signup, logout
  - `generation_routes.py` - `/api/generate/*` - All generation endpoints
  - `user_routes.py` - `/api/users/*` - User profile management
  - `project_routes.py` - `/api/projects/*` - Project CRUD
  - `health_routes.py` - `/health` - Health check endpoint

- **Schemas** - Request/response validation using Pydantic
  - Input validation happens here
  - Output formatting happens here
  - Type safety enforced

- **Dependencies** - Reusable FastAPI dependencies
  - `get_current_user()` - Validates JWT token
  - `get_db_session()` - Provides database session
  - `check_rate_limit()` - Enforces usage limits

**Example:**
```python
# api/routes/generation_routes.py
@router.post("/generate/names", response_model=NameGenerationResponse)
async def generate_names(
    request: NameGenerationRequest,
    user: User = Depends(get_current_user),
    name_service: NameService = Depends(get_name_service)
):
    # Route delegates to service
    result = await name_service.generate(
        description=request.description,
        industry=request.industry,
        user_id=user.id
    )
    return result
```

---

### 2. Service Layer (`/services`)

**Purpose:** Contains all business logic, orchestrates operations

**Design Pattern:** Service Layer Pattern

**Key Services:**

1. **NameService** - Business name generation
   - Loads name generation patterns
   - Calls AI Manager for text generation
   - Validates and scores results
   - Returns 10 unique names

2. **DomainService** - Domain availability checking
   - Parallel DNS checks
   - API verification (WhoisXML, WhoAPI)
   - Marketplace scraping (optional)
   - Returns availability + pricing

3. **LogoService** - Logo concept generation
   - Calls Image API via Failover Manager
   - Generates 3 variations
   - Handles PNG/JPG/SVG formats
   - Stores in Supabase Storage

4. **ColorService** - Color palette creation
   - Generates 5-6 colors with hex codes
   - Ensures WCAG accessibility
   - Applies color psychology
   - Returns palette with reasoning

5. **TaglineService** - Tagline generation
   - Uses copywriting frameworks
   - Generates 5 variations
   - Scores for memorability
   - Returns ranked taglines

6. **PackageService** - Orchestrates complete packages
   - Calls all services in sequence
   - Handles errors gracefully
   - Saves complete package to database
   - Returns unified response

**Base Service Pattern:**
```python
# services/base_service.py
class BaseService:
    """All services inherit from this"""
    
    def __init__(self, ai_manager, research_loader, db_client):
        self.ai = ai_manager
        self.patterns = research_loader
        self.db = db_client
    
    async def generate(self, **kwargs):
        raise NotImplementedError()
    
    def _validate_input(self, data):
        """Common validation logic"""
        pass
    
    async def _save_result(self, result):
        """Common persistence logic"""
        pass
```

**Why Service Layer?**
- ✅ Business logic separated from HTTP layer
- ✅ Services are framework-agnostic (can be used in CLI, workers, etc.)
- ✅ Easy to test in isolation
- ✅ Reusable across different interfaces

---

### 3. Core Components (`/core`)

**Purpose:** Infrastructure components shared across services

#### **AIManager** (`core/ai_manager.py`)

Centralized AI operations manager - Singleton pattern

**Responsibilities:**
- Manages all text generation API calls (OpenRouter)
- Handles multiple models (Claude, GPT, Gemini)
- Implements parallel generation for speed
- Tracks API usage and costs
- Handles retries and errors

**Key Methods:**
```python
class AIManager:
    @staticmethod
    async def generate_text(prompt: str, model: str, temperature: float = 0.7) -> str
        """Single model text generation"""
    
    @staticmethod
    async def generate_parallel(prompts: List[Tuple[str, str]]) -> Dict[str, str]
        """Parallel generation from multiple models"""
    
    @staticmethod
    async def generate_with_context(prompt: str, context: str, model: str) -> str
        """Generation with research pattern context"""
```

**Why Singleton?**
- One instance shared across all services
- Prevents multiple API client initializations
- Centralized rate limiting and error handling

#### **ResearchLoader** (`core/research_loader.py`)

Loads and caches intelligence patterns from research documents

**Responsibilities:**
- Reads research documents from `/research/documents/`
- Processes with Gemini to extract patterns
- Caches processed patterns for 7 days
- Provides patterns to services on demand

**Key Methods:**
```python
class ResearchLoader:
    @staticmethod
    def get_patterns(component: str) -> str
        """Get cached patterns for a component"""
    
    @staticmethod
    async def analyze_and_cache(file_path: str, component: str) -> str
        """Process and cache a research document"""
    
    @staticmethod
    def load_all_patterns() -> None
        """Load all patterns on startup"""
```

**Caching Strategy:**
- File-based cache with 7-day TTL
- Lazy loading (load only when needed)
- Shared across all service instances

#### **FailoverManager** (`core/failover_manager.py`)

Manages multiple image generation APIs with automatic failover

**Responsibilities:**
- Maintains list of image APIs (Flux, Ideogram, Stability AI, DALL-E)
- Attempts primary API first
- Automatically fails over to backup if error
- Tracks success/failure rates
- Optimizes cost (prefers cheaper APIs)

**API Priority:**
```
1. Flux (primary) - Fastest, cheapest
2. Ideogram (backup) - Good quality
3. Stability AI (backup) - Reliable
4. DALL-E (last resort) - Most expensive
```

**Key Methods:**
```python
class FailoverManager:
    async def generate_with_failover(prompt: str, style: str = "realistic") -> bytes
        """Try APIs in order until success"""
    
    def record_success(api_name: str) -> None
        """Track successful API call"""
    
    def record_failure(api_name: str) -> None
        """Track failed API call"""
    
    def get_stats() -> Dict
        """Get API success/failure statistics"""
```

---

### 4. Database Layer (`/database`)

**Purpose:** All data persistence logic

#### **Models** (`database/models.py`)

Contains both SQLAlchemy ORM models and Pydantic schemas

**Key Models:**

1. **User** - User account information
   ```python
   class User(Base):
       id: UUID (primary key)
       email: str (unique)
       username: str (unique, optional)
       generations_remaining: int (default: 2)
       is_premium: bool (default: False)
       created_at: datetime
       updated_at: datetime
   ```

2. **Project** - User's branding projects
   ```python
   class Project(Base):
       id: UUID (primary key)
       user_id: UUID (foreign key)
       name: str
       description: str
       industry: str (optional)
       status: enum (draft, in_progress, completed)
       created_at: datetime
       updated_at: datetime
   ```

3. **Generation** - Individual generation records
   ```python
   class Generation(Base):
       id: UUID (primary key)
       project_id: UUID (foreign key)
       user_id: UUID (foreign key)
       generation_type: enum (name, logo, color, tagline, package)
       input_data: JSONB
       output_data: JSONB
       status: enum (pending, success, failed)
       error_message: str (optional)
       created_at: datetime
   ```

4. **Asset** - Generated files (logos, etc.)
   ```python
   class Asset(Base):
       id: UUID (primary key)
       generation_id: UUID (foreign key)
       project_id: UUID (foreign key)
       asset_type: enum (logo, image, document)
       file_url: str
       file_format: str (png, jpg, svg)
       file_size: int
       created_at: datetime
   ```

#### **Repositories** (`database/repositories/`)

**Design Pattern:** Repository Pattern

Abstracts database operations from services. Each repository handles one entity.

**Base Repository:**
```python
# database/repositories/base_repository.py
class BaseRepository:
    def __init__(self, db_client):
        self.db = db_client
    
    async def create(self, data: dict) -> Model
    async def get_by_id(self, id: UUID) -> Optional[Model]
    async def update(self, id: UUID, data: dict) -> Model
    async def delete(self, id: UUID) -> bool
    async def list(self, filters: dict) -> List[Model]
```

**Specific Repositories:**
- `UserRepository` - User CRUD operations
- `ProjectRepository` - Project CRUD operations
- `GenerationRepository` - Generation tracking
- `AssetRepository` - Asset management

**Why Repository Pattern?**
- ✅ Database logic separated from business logic
- ✅ Services don't know about SQL
- ✅ Easy to swap databases later
- ✅ Cleaner testing (can mock repositories)

---

### 5. Middleware Layer (`/middleware`)

**Purpose:** Cross-cutting concerns applied to all requests

#### **Authentication** (`middleware/auth.py`)

Validates JWT tokens from Supabase

```python
async def verify_token(request: Request) -> User:
    """Extract and verify JWT from Authorization header"""
    token = request.headers.get("Authorization")
    if not token:
        raise AuthenticationError()
    
    # Verify with Supabase
    user_data = supabase.auth.get_user(token)
    return User(**user_data)
```

#### **Rate Limiter** (`middleware/rate_limiter.py`)

Enforces usage limits (2 free generations per user)

```python
async def check_rate_limit(user: User, generation_type: str):
    """Check if user has remaining generations"""
    if user.is_premium:
        return True  # Unlimited for premium users
    
    if user.generations_remaining <= 0:
        raise RateLimitExceeded("No generations remaining")
    
    # Decrement counter
    await user_repo.update(user.id, {
        "generations_remaining": user.generations_remaining - 1
    })
```

#### **Error Handler** (`middleware/error_handler.py`)

Catches all exceptions and formats responses consistently

```python
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch all exceptions and return JSON response"""
    
    if isinstance(exc, CustomException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.message, "code": exc.error_code}
        )
    
    # Log unexpected errors
    logger.error(f"Unexpected error: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )
```

#### **Logging Middleware** (`middleware/logging_middleware.py`)

Logs all requests and responses

```python
async def log_requests(request: Request, call_next):
    """Log request and response details"""
    request_id = str(uuid4())
    
    logger.info(f"[{request_id}] {request.method} {request.url}")
    
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    logger.info(f"[{request_id}] Status: {response.status_code} - Duration: {duration:.2f}s")
    
    return response
```

---

### 6. Configuration Layer (`/config`)

**Purpose:** Centralized configuration management

#### **Settings** (`config/settings.py`)

Uses Pydantic for environment variable validation

```python
class Settings(BaseSettings):
    # Application
    app_env: str = "development"
    app_name: str = "Brand Package Generator"
    app_version: str = "1.0.0"
    
    # Database
    supabase_url: str
    supabase_key: str
    
    # AI APIs
    text_model_1: Optional[str] = None
    text_model_2: Optional[str] = None
    image_api_1_provider: Optional[str] = None
    image_api_1_key: Optional[str] = None
    
    # Rate Limiting
    rate_limit_generations: int = 2
    
    class Config:
        env_file = ".env"

# Global settings instance
settings = Settings()
```

**Benefits:**
- ✅ Type-safe environment variables
- ✅ Automatic validation on startup
- ✅ Default values for optional settings
- ✅ Easy to test (can override settings)

#### **Constants** (`config/constants.py`)

Hard-coded application constants

```python
# Generation limits
MAX_NAMES_PER_REQUEST = 10
MAX_LOGOS_PER_REQUEST = 3
MAX_TAGLINES_PER_REQUEST = 5

# File sizes
MAX_FILE_SIZE_MB = 5
ALLOWED_IMAGE_FORMATS = ["png", "jpg", "jpeg", "svg"]

# Caching
CACHE_TTL_DAYS = 7
```

---

### 7. Utilities Layer (`/utils`)

**Purpose:** Shared helper functions

**Key Utilities:**

- `storage.py` - File upload/download to Supabase Storage
- `validators.py` - Input validation (email, domain, etc.)
- `formatters.py` - Data formatting (JSON, CSV, etc.)
- `image_utils.py` - Image processing (resize, convert, etc.)
- `text_parsers.py` - Parse AI responses (extract JSON, etc.)
- `security.py` - Password hashing, token generation

---

### 8. Research Layer (`/research`)

**Purpose:** Intelligence pattern management

**Structure:**
- `documents/` - Raw research files (TXT format)
- `cache/` - Processed patterns (JSON format)
- `loader.py` - Loading and processing logic

**How it Works:**

1. **Startup:**
   - `ResearchLoader.load_all_patterns()` is called
   - Checks if patterns are cached
   - If not, processes documents with Gemini
   - Caches results for 7 days

2. **Service Usage:**
   - Services request patterns: `patterns = ResearchLoader.get_patterns("name")`
   - Patterns are injected into AI prompts
   - Improves generation quality

**Example Pattern:**
```
# research/documents/name_intelligence.txt

BRANDABLE NAME PATTERNS:
- Short (4-8 characters)
- Memorable and pronounceable
- Avoid hyphens and numbers
- Use word combinations (e.g., "FaceBook", "YouTube")
- Consider domain availability

NAMING FRAMEWORKS:
1. Descriptive (e.g., "PayPal")
2. Invented (e.g., "Kodak")
3. Metaphorical (e.g., "Amazon")
4. Acronyms (e.g., "IBM")
5. Founder names (e.g., "Tesla")

AVOID:
- Generic terms
- Trademark conflicts
- Difficult spellings
```

---

### 9. Workers Layer (`/workers`)

**Purpose:** Background task processing (optional for MVP)

**When to Use:**
- Long-running operations (>30 seconds)
- Scheduled tasks (cleanup, reports)
- Asynchronous processing (email sending)

**Technologies:**
- Celery (task queue)
- Redis (message broker)

**Example Tasks:**
```python
# workers/tasks/generation_tasks.py

@celery_app.task
async def generate_complete_package_async(project_id: str, user_id: str):
    """Generate complete brand package in background"""
    
    package_service = PackageService()
    result = await package_service.generate_complete_package(
        project_id=project_id,
        user_id=user_id
    )
    
    # Send email notification when done
    await send_completion_email(user_id, result)
```

---

### 10. Testing Layer (`/tests`)

**Purpose:** Comprehensive testing

**Testing Pyramid:**

```
        /\
       /  \      E2E Tests (Few)
      /____\     - Complete workflows
     /      \    - User journeys
    /________\   Integration Tests (Some)
   /          \  - API endpoints
  /____________\ - Database operations
 /              \
/________________\ Unit Tests (Many)
                   - Service functions
                   - Utilities
                   - Validators
```

**Test Types:**

1. **Unit Tests** (`tests/unit/`)
   - Test individual functions
   - Mock external dependencies
   - Fast execution

2. **Integration Tests** (`tests/integration/`)
   - Test layer interactions
   - Use test database
   - Slower but more realistic

3. **E2E Tests** (`tests/e2e/`)
   - Test complete user workflows
   - Simulate real usage
   - Slowest but most comprehensive

---

## 🛠️ Tech Stack

### Core Technologies

| Category | Technology | Version | Purpose |
|----------|-----------|---------|---------|
| **Framework** | FastAPI | 0.104+ | Web framework |
| **Language** | Python | 3.11+ | Programming language |
| **Database** | PostgreSQL | 15+ | Primary database |
| **ORM** | SQLAlchemy | 2.0+ | Database ORM |
| **Validation** | Pydantic | 2.5+ | Data validation |
| **Migration** | Alembic | 1.12+ | Database migrations |

### External Services

| Service | Purpose | Provider |
|---------|---------|----------|
| **Supabase** | Database + Auth + Storage | Supabase |
| **OpenRouter** | Text generation APIs | OpenRouter |
| **Flux/Ideogram** | Image generation | Multiple |
| **WhoisXML** | Domain checking | WhoisXML API |

### Supporting Libraries

```
# AI & APIs
openai==1.3.7                # OpenAI SDK (used with OpenRouter)
httpx==0.25.1                # Async HTTP client
aiohttp==3.9.0               # Alternative async HTTP

# Authentication & Security
python-jose==3.3.0           # JWT handling
passlib==1.7.4               # Password hashing

# Image Processing
pillow==10.1.0               # Image manipulation
cairosvg==2.7.1              # SVG processing

# Utilities
python-dotenv==1.0.0         # Environment variables
loguru==0.7.2                # Logging

# Development
pytest==7.4.3                # Testing framework
black==23.11.0               # Code formatting
flake8==6.1.0                # Linting
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.11 or higher
- PostgreSQL (or Supabase account)
- Git
- Virtual environment tool (venv, virtualenv, conda)

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/brand-generator-backend.git
cd brand-generator-backend
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Mac/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your credentials
nano .env
```

**Required Environment Variables:**
```bash
# Database
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
SUPABASE_SERVICE_KEY=your-service-key

# AI APIs
TEXT_API_KEY=your-openrouter-key
TEXT_MODEL_1=anthropic/claude-3-5-sonnet-20241022

IMAGE_API_1_PROVIDER=flux
IMAGE_API_1_KEY=your-flux-key

# Security
JWT_SECRET_KEY=your-super-secret-key-change-this

# App
APP_ENV=development
DEBUG=true
```

### Step 5: Initialize Database

```bash
# Create initial migration
alembic revision --autogenerate -m "initial schema"

# Apply migrations
alembic upgrade head
```

### Step 6: Run the Server

```bash
# Development mode (with auto-reload)
python main.py

# Production mode
uvicorn main:app --host 0.0.0.0 --port 8000
```

Server should start at: `http://localhost:8000`

### Step 7: Verify Installation

```bash
# Health check
curl http://localhost:8000/health

# API documentation
open http://localhost:8000/docs
```

---

## 🔐 Configuration

### Environment Variables Explained

#### Application Settings
```bash
APP_ENV=development              # development | staging | production
APP_NAME="Brand Package Generator"
APP_VERSION="1.0.0"
DEBUG=true                       # Enable debug mode
LOG_LEVEL=INFO                   # DEBUG | INFO | WARNING | ERROR
```

#### Server Settings
```bash
HOST=0.0.0.0                     # Bind address
PORT=8000                        # Server port
RELOAD=true                      # Auto-reload on code changes
```

#### Database Settings
```bash
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_ANON_KEY=eyJhbGc...    # Public key for client
SUPABASE_SERVICE_KEY=eyJhbGc... # Private key for server
```

#### Text Generation APIs
```bash
TEXT_API_PROVIDER=openrouter     # API provider name
TEXT_API_KEY=sk-or-v1-xxx        # API key
TEXT_API_URL=https://openrouter.ai/api/v1

# Models (up to 5)
TEXT_MODEL_1=anthropic/claude-3-5-sonnet-20241022
TEXT_MODEL_2=openai/gpt-4o
TEXT_MODEL_3=google/gemini-exp-1121
```

#### Image Generation APIs
```bash
# Primary API
IMAGE_API_1_PROVIDER=flux
IMAGE_API_1_KEY=your-key
IMAGE_API_1_URL=https://api.flux.ai/v1
IMAGE_API_1_MODEL=flux-pro

# Backup API
IMAGE_API_2_PROVIDER=ideogram
IMAGE_API_2_KEY=your-key
```

#### Domain Checking APIs
```bash
DOMAIN_API_1_PROVIDER=whoisxml
DOMAIN_API_1_KEY=your-key
```

#### Rate Limiting
```bash
RATE_LIMIT_GENERATIONS=2         # Free generations per user
RATE_LIMIT_WINDOW_MINUTES=1440   # Reset window (24 hours)
```

#### Security
```bash
JWT_SECRET_KEY=your-secret-key   # For signing tokens
JWT_ALGORITHM=HS256              # Signing algorithm
ACCESS_TOKEN_EXPIRE_MINUTES=1440 # Token validity (24 hours)
```

#### Frontend
```bash
FRONTEND_URL=http://localhost:3000
CORS_ORIGINS=["http://localhost:3000","http://localhost:3001"]
```

---

## 🗄️ Database Schema

### Entity Relationship Diagram

```
┌─────────────┐       ┌──────────────┐       ┌────────────────┐
│    User     │──1:N──│   Project    │──1:N──│  Generation    │
│             │       │              │       │                │
│ id          │       │ id           │       │ id             │
│ email       │       │ user_id (FK) │       │ project_id(FK) │
│ username    │       │ name         │       │ user_id (FK)   │
│ gens_remain │       │ description  │       │ type           │
│ is_premium  │       │ industry     │       │ input_data     │
│ created_at  │       │ status       │       │ output_data    │
└─────────────┘       └──────────────┘       │ status         │
                                             └────────────────┘
                                                     │
                                                     │1:N
                                                     │
                                             ┌────────────────┐
                                             │     Asset      │
                                             │                │
                                             │ id             │
                                             │ generation_id  │
                                             │ project_id     │
                                             │ asset_type     │
                                             │ file_url       │
                                             │ file_format    │
                                             └────────────────┘
```

### Table: users

```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE,
    generations_remaining INTEGER DEFAULT 2,
    is_premium BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_premium ON users(is_premium);
```

### Table: projects

```sql
CREATE TABLE projects (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    industry VARCHAR(100),
    status VARCHAR(50) DEFAULT 'draft',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_projects_user_id ON projects(user_id);
CREATE INDEX idx_projects_status ON projects(status);
```

### Table: generations

```sql
CREATE TABLE generations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    generation_type VARCHAR(50) NOT NULL,
    input_data JSONB,
    output_data JSONB,
    status VARCHAR(50) DEFAULT 'pending',
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_generations_project_id ON generations(project_id);
CREATE INDEX idx_generations_user_id ON generations(user_id);
CREATE INDEX idx_generations_type ON generations(generation_type);
CREATE INDEX idx_generations_status ON generations(status);
```

### Table: assets

```sql
CREATE TABLE assets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    generation_id UUID NOT NULL REFERENCES generations(id) ON DELETE CASCADE,
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    asset_type VARCHAR(50) NOT NULL,
    file_url TEXT NOT NULL,
    file_format VARCHAR(10),
    file_size INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_assets_generation_id ON assets(generation_id);
CREATE INDEX idx_assets_project_id ON assets(project_id);
CREATE INDEX idx_assets_type ON assets(asset_type);
```

---

## 🌐 API Documentation

### Authentication

All API endpoints (except `/health`) require authentication.

**Header:**
```
Authorization: Bearer <your-jwt-token>
```

### Endpoints Overview

| Endpoint | Method | Purpose | Rate Limited |
|----------|--------|---------|--------------|
| `/health` | GET | Health check | No |
| `/api/auth/signup` | POST | Create account | No |
| `/api/auth/login` | POST | Login | No |
| `/api/auth/logout` | POST | Logout | No |
| `/api/generate/names` | POST | Generate names | Yes |
| `/api/generate/logos` | POST | Generate logos | Yes |
| `/api/generate/colors` | POST | Generate colors | Yes |
| `/api/generate/taglines` | POST | Generate taglines | Yes |
| `/api/generate/package` | POST | Generate complete package | Yes |
| `/api/domains/check` | POST | Check domains | No |
| `/api/users/me` | GET | Get current user | No |
| `/api/users/history` | GET | Get generation history | No |
| `/api/projects` | GET/POST | List/create projects | No |
| `/api/projects/{id}` | GET/PUT/DELETE | Manage project | No |

### Example: Generate Names

**Request:**
```bash
curl -X POST "http://localhost:8000/api/generate/names" \
  -H "Authorization: Bearer your-token" \
  -H "Content-Type: application/json" \
  -d '{
    "description": "AI-powered recipe discovery app for home cooks",
    "industry": "food tech",
    "style": "modern"
  }'
```

**Response:**
```json
{
  "generation_id": "550e8400-e29b-41d4-a716-446655440000",
  "names": [
    {
      "name": "FlavorAI",
      "reasoning": "Combines 'Flavor' with AI, memorable and brandable",
      "score": 9.2,
      "domain_available": true
    },
    {
      "name": "RecipeFlow",
      "reasoning": "Suggests smooth recipe discovery experience",
      "score": 8.7,
      "domain_available": false
    }
    // ... 8 more names
  ],
  "processing_time": 2.3,
  "models_used": ["claude-3-5-sonnet", "gpt-4o"]
}
```

### Example: Generate Complete Package

**Request:**
```bash
curl -X POST "http://localhost:8000/api/generate/package" \
  -H "Authorization: Bearer your-token" \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Sustainable fashion marketplace for vintage clothing",
    "industry": "e-commerce",
    "preferences": {
      "style": "modern minimalist",
      "colors": "earthy tones"
    }
  }'
```

**Response:**
```json
{
  "project_id": "123e4567-e89b-12d3-a456-426614174000",
  "generation_id": "550e8400-e29b-41d4-a716-446655440000",
  "names": [...],
  "logos": [
    {
      "variation": 1,
      "url": "https://storage.supabase.co/...",
      "format": "png"
    }
  ],
  "colors": {
    "primary": "#8B7355",
    "secondary": "#E8D5C4",
    "accent": "#2F4F4F",
    // ... more colors
  },
  "taglines": [...],
  "processing_time": 15.7,
  "status": "completed"
}
```

### Error Responses

All errors follow this format:

```json
{
  "error": "Human-readable error message",
  "code": "ERROR_CODE",
  "details": {
    "field": "Additional context"
  }
}
```

**Common Error Codes:**

| Code | Status | Meaning |
|------|--------|---------|
| `AUTH_REQUIRED` | 401 | Missing or invalid token |
| `RATE_LIMIT_EXCEEDED` | 429 | No generations remaining |
| `VALIDATION_ERROR` | 422 | Invalid input data |
| `RESOURCE_NOT_FOUND` | 404 | Project/generation not found |
| `INTERNAL_ERROR` | 500 | Server error |

---

## 🔄 Data Flow

### Complete Generation Flow

```
1. CLIENT REQUEST
   ↓
   POST /api/generate/names
   Headers: Authorization: Bearer <token>
   Body: { description, industry }

2. MIDDLEWARE LAYER
   ↓
   auth.py → Verify JWT token → Extract user
   rate_limiter.py → Check generations_remaining
   logging_middleware.py → Log request

3. API LAYER
   ↓
   generation_routes.py → Validate request schema
   dependencies.py → Inject NameService

4. SERVICE LAYER
   ↓
   NameService.generate()
   ├─ Load name patterns from ResearchLoader
   ├─ Build prompt with patterns
   ├─ Call AIManager.generate_text()
   │  └─ OpenRouter API call
   ├─ Parse and validate results
   ├─ Score names
   └─ Save to database via Repository

5. CORE COMPONENTS
   ↓
   AIManager → Make API call
   ResearchLoader → Provide cached patterns
   
6. DATABASE LAYER
   ↓
   GenerationRepository.create()
   └─ INSERT INTO generations

7. RESPONSE
   ↓
   Format response → JSON
   └─ Return to client

8. MIDDLEWARE (RESPONSE)
   ↓
   logging_middleware.py → Log response
   error_handler.py → Format any errors
```

### Package Generation Flow (Complex)

```
PackageService.generate_complete_package()
│
├── 1. NameService.generate()
│   └── Returns: 10 names
│
├── 2. DomainService.check_batch()
│   └── Returns: Availability for all names
│
├── 3. LogoService.generate()
│   ├── FailoverManager tries Flux API
│   ├── If fails → tries Ideogram
│   ├── If fails → tries Stability AI
│   └── Returns: 3 logo variations
│
├── 4. ColorService.generate()
│   └── Returns: 5-6 color palette
│
├── 5. TaglineService.generate()
│   └── Returns: 5 taglines
│
├── 6. Save complete package
│   ├── Create Project
│   ├── Create Generation records
│   ├── Upload Assets to Supabase Storage
│   └── Link everything together
│
└── 7. Return unified response
```

---

## 🧪 Development Workflow

### Daily Development

```bash
# 1. Pull latest changes
git pull origin main

# 2. Activate virtual environment
source venv/bin/activate

# 3. Install new dependencies (if any)
pip install -r requirements.txt

# 4. Run migrations (if any)
alembic upgrade head

# 5. Start development server
python main.py

# Server runs with auto-reload enabled
```

### Making Changes

```bash
# 1. Create feature branch
git checkout -b feature/new-service

# 2. Make your changes
# Edit files...

# 3. Format code
black .
isort .

# 4. Check linting
flake8

# 5. Run tests
pytest

# 6. Commit changes
git add .
git commit -m "feat: add new service"

# 7. Push to remote
git push origin feature/new-service
```

### Database Changes

```bash
# 1. Modify models in database/models.py
# Add/remove/modify table columns

# 2. Generate migration
alembic revision --autogenerate -m "add new column"

# 3. Review generated migration
# Check database/migrations/versions/<timestamp>_add_new_column.py

# 4. Apply migration
alembic upgrade head

# 5. Rollback if needed
alembic downgrade -1
```

### Adding New Service

```bash
# 1. Create service file
touch backend/services/new_service.py

# 2. Implement service
# Inherit from BaseService
class NewService(BaseService):
    async def generate(self, **kwargs):
        # Implementation
        pass

# 3. Create route
touch backend/api/routes/new_routes.py

# 4. Add route to main.py
# app.include_router(new_routes.router)

# 5. Create tests
touch backend/tests/unit/test_services/test_new_service.py

# 6. Run tests
pytest tests/unit/test_services/test_new_service.py
```

---

## 🧪 Testing Strategy

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_services/test_name_service.py

# Run with coverage
pytest --cov=. --cov-report=html

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run with verbose output
pytest -v

# Run specific test
pytest tests/unit/test_services/test_name_service.py::test_generate_names
```

### Writing Tests

**Unit Test Example:**
```python
# tests/unit/test_services/test_name_service.py
import pytest
from unittest.mock import Mock, AsyncMock
from services.name_service import NameService

@pytest.fixture
def mock_ai_manager():
    """Mock AI Manager"""
    mock = AsyncMock()
    mock.generate_text.return_value = '["FlavorAI", "RecipeFlow"]'
    return mock

@pytest.fixture
def name_service(mock_ai_manager):
    """Create NameService with mocked dependencies"""
    return NameService(
        ai_manager=mock_ai_manager,
        research_loader=Mock(),
        db_client=Mock()
    )

@pytest.mark.asyncio
async def test_generate_names_success(name_service):
    """Test successful name generation"""
    result = await name_service.generate(
        description="AI recipe app",
        industry="food tech"
    )
    
    assert len(result['names']) == 10
    assert all(isinstance(name['score'], float) for name in result['names'])
    assert result['status'] == 'completed'

@pytest.mark.asyncio
async def test_generate_names_empty_description(name_service):
    """Test validation error for empty description"""
    with pytest.raises(ValueError, match="Description cannot be empty"):
        await name_service.generate(description="")
```

**Integration Test Example:**
```python
# tests/integration/test_api/test_generation_endpoints.py
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

@pytest.fixture
def auth_headers():
    """Get authentication headers"""
    # Login and get token
    response = client.post("/api/auth/login", json={
        "email": "test@example.com",
        "password": "testpass123"
    })
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

def test_generate_names_endpoint(auth_headers):
    """Test name generation endpoint"""
    response = client.post(
        "/api/generate/names",
        json={
            "description": "AI recipe app",
            "industry": "food tech"
        },
        headers=auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "names" in data
    assert len(data["names"]) == 10
    assert "generation_id" in data

def test_generate_names_requires_auth():
    """Test authentication requirement"""
    response = client.post(
        "/api/generate/names",
        json={"description": "Test"}
    )
    
    assert response.status_code == 401
```

---

## 🚀 Deployment

### Railway Deployment

**1. Install Railway CLI:**
```bash
npm install -g @railway/cli
railway login
```

**2. Initialize Project:**
```bash
railway init
```

**3. Configure Environment:**
```bash
# Set all environment variables
railway variables set SUPABASE_URL=https://...
railway variables set TEXT_API_KEY=sk-...
# ... set all variables from .env
```

**4. Deploy:**
```bash
railway up
```

**5. View Logs:**
```bash
railway logs
```

### Docker Deployment

**Build Image:**
```bash
docker build -t brand-generator-backend .
```

**Run Container:**
```bash
docker run -d \
  -p 8000:8000 \
  --env-file .env \
  --name brand-backend \
  brand-generator-backend
```

**Docker Compose:**
```yaml
# docker-compose.yml
version: '3.8'

services:
  backend:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - ./research:/app/research
    restart: unless-stopped
```

Run with:
```bash
docker-compose up -d
```

---

## 🔒 Security

### Security Measures Implemented

1. **Authentication**
   - JWT tokens from Supabase
   - Token expiration (24 hours)
   - Refresh token mechanism

2. **Authorization**
   - User-scoped data access
   - Rate limiting per user
   - Premium tier access control

3. **Input Validation**
   - Pydantic schema validation
   - SQL injection prevention (ORM)
   - XSS prevention (output encoding)

4. **API Security**
   - CORS configuration
   - Rate limiting
   - Request size limits

5. **Data Security**
   - Password hashing (bcrypt)
   - Environment variable secrets
   - Secure file uploads

### Security Best Practices

**Don't:**
- ❌ Commit `.env` file to git
- ❌ Store API keys in code
- ❌ Use default secret keys
- ❌ Disable CORS in production
- ❌ Log sensitive data

**Do:**
- ✅ Use environment variables
- ✅ Rotate API keys regularly
- ✅ Enable HTTPS in production
- ✅ Monitor rate limit abuse
- ✅ Keep dependencies updated

---

## ⚡ Performance & Optimization

### Caching Strategy

1. **Research Patterns** - Cached for 7 days
2. **AI Responses** - Optional response caching
3. **Database Queries** - Connection pooling

### Performance Tips

**1. Parallel Processing:**
```python
# Generate names from multiple models in parallel
results = await asyncio.gather(
    ai_manager.generate_text(prompt, "claude"),
    ai_manager.generate_text(prompt, "gpt-4")
)
```

**2. Batch Operations:**
```python
# Check multiple domains at once
domains = ["example1.ai", "example2.ai", "example3.ai"]
results = await domain_service.check_batch(domains)
```

**3. Database Optimization:**
```python
# Use proper indexes
CREATE INDEX idx_generations_user_id ON generations(user_id);

# Use connection pooling
engine = create_engine(url, pool_size=20, max_overflow=40)
```

**4. Response Compression:**
```python
# Enable gzip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)
```

### Monitoring

**Key Metrics to Track:**
- API response times
- AI API latency
- Database query performance
- Error rates
- Rate limit hits
- User generation counts

---

## 🐛 Troubleshooting

### Common Issues

**1. Server Won't Start**

```bash
# Check if port is in use
lsof -i :8000

# Kill process using port
kill -9 <PID>

# Check environment variables
python -c "from config.settings import settings; print(settings)"
```

**2. Database Connection Failed**

```bash
# Test Supabase connection
python -c "from database.client import init_supabase; init_supabase()"

# Check credentials
echo $SUPABASE_URL
echo $SUPABASE_KEY
```

**3. AI API Errors**

```bash
# Test API key
curl https://openrouter.ai/api/v1/models \
  -H "Authorization: Bearer $TEXT_API_KEY"

# Check API quota/billing
```

**4. Rate Limit Issues**

```python
# Reset user's generation count
from database.repositories.user_repository import UserRepository

user_repo = UserRepository()
await user_repo.update(user_id, {"generations_remaining": 2})
```

**5. Migration Errors**

```bash
# Check current migration
alembic current

# Show migration history
alembic history

# Rollback to previous
alembic downgrade -1

# Reset and recreate
alembic downgrade base
alembic upgrade head
```

### Debug Mode

Enable detailed logging:

```python
# config/settings.py
DEBUG = True
LOG_LEVEL = "DEBUG"

# Run with verbose output
python main.py --log-level DEBUG
```

### Getting Help

1. Check logs: `tail -f logs/app.log`
2. Enable debug mode
3. Test individual components
4. Review API documentation
5. Check GitHub issues

---

## 📚 Additional Resources

### Documentation

- **FastAPI Docs:** https://fastapi.tiangolo.com/
- **Supabase Docs:** https://supabase.com/docs
- **Pydantic Docs:** https://docs.pydantic.dev/
- **SQLAlchemy Docs:** https://docs.sqlalchemy.org/

### Related Repositories

- Frontend Repository: `brand-generator-frontend`
- Documentation: `brand-generator-docs`

### Contributing

See `CONTRIBUTING.md` for guidelines.

### License

MIT License - See `LICENSE` file for details.

---

## 📞 Support

- **Email:** support@yourdomain.com
- **Documentation:** https://docs.yourdomain.com
- **GitHub Issues:** https://github.com/yourusername/brand-generator-backend/issues

---

**Last Updated:** October 6, 2025  
**Version:** 1.0.0  
**Maintained by:** Your Team Name