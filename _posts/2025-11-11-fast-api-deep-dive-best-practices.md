---
layout: post
title: "🌊 FastAPI: Deep Dive & Best Practices"
description: "Concise, clear, and validated revision notes on FastAPI — Pydantic schemas, dependency injection, async patterns, and practical best practices on building high-performance APIs."
author: technical_notes
date: 2025-11-11 22:00:00 +0530
categories: [Notes, FastAPI]
tags: [FastAPI, Python, Web Framework, ASYNC, REST API, API Development, Web Development, OpenAPI, Swagger, pydantic, dependency-injection, microservices, async-await]
image: /assets/img/posts/fastapi-logo.png
toc: true
math: true
mermaid: true
---

# FastAPI: Deep Dive & Best Practices

## Introduction

FastAPI is a modern, high-performance web framework for building APIs with Python 3.7+ based on standard Python type hints. Created by Sebastián Ramírez, FastAPI combines the speed of Starlette for web routing with the robustness of Pydantic for data validation, resulting in one of the fastest Python frameworks available—rivaling Node.js and Go in performance benchmarks. FastAPI's automatic interactive documentation (Swagger UI and ReDoc), native async support, and intuitive dependency injection system make it an excellent choice for building production-ready APIs, microservices, and machine learning model deployments. This comprehensive guide explores FastAPI from foundational concepts to advanced patterns, empowering you to build fast, secure, and maintainable APIs.

---

## Table 1: FastAPI Ecosystem Terminology Mapping

Different contexts and documentation use varying terminology for similar FastAPI concepts:

| Standard Term | Alternative Names | Context/Usage |
|--------------|-------------------|---------------|
| **Path Operation** | Endpoint, Route, API Endpoint, Route Handler | HTTP endpoint function |
| **Path Operation Function** | Endpoint Function, Route Function, Handler | Decorated function handling requests |
| **Path Parameter** | URL Parameter, Route Parameter, Dynamic Segment | Variable in URL path |
| **Query Parameter** | URL Parameter, Query String Parameter | Key-value in URL query string |
| **Request Body** | Payload, Body, POST Data, JSON Body | Data sent in request body |
| **Response Model** | Return Model, Output Schema, Response Schema | Pydantic model for responses |
| **Dependency** | Injectable, Dependency Injection, DI | Reusable function or class |
| **Dependency Injection** | DI, Inversion of Control, IoC | Design pattern for dependencies |
| **Pydantic Model** | Schema, Data Model, BaseModel, DTO | Data validation class |
| **Field Validation** | Data Validation, Type Validation, Constraint | Input data checking |
| **Interactive Docs** | Swagger UI, API Docs, OpenAPI Docs | Auto-generated documentation |
| **Path Operation Decorator** | Route Decorator, HTTP Method Decorator | @app.get(), @app.post(), etc. |
| **ASGI** | Asynchronous Server Gateway Interface | Async server standard |
| **Middleware** | Interceptor, Filter, HTTP Middleware | Request/response processor |
| **Background Task** | Async Task, Deferred Task, Background Job | Non-blocking background work |
| **Security Scheme** | Auth Scheme, Authentication, Authorization | Security mechanism |
| **OAuth2** | OAuth, Token Authentication | Authorization framework |
| **JWT** | JSON Web Token, Bearer Token | Authentication token |
| **CORS** | Cross-Origin Resource Sharing | Cross-domain requests |
| **OpenAPI** | Swagger, API Specification, OAS | API documentation standard |

---

## Table 2: Hierarchical FastAPI Concept Structure

This table organizes FastAPI concepts from high-level abstractions to specific implementations:

| Level | Category | Term | Parent Concept | Description |
|-------|----------|------|----------------|-------------|
| **L1** | Framework | FastAPI | - | Modern Python web framework |
| **L2** | Foundation | Starlette | FastAPI | ASGI framework for routing |
| **L2** | Foundation | Pydantic | FastAPI | Data validation library |
| **L2** | Foundation | Uvicorn | FastAPI | ASGI server |
| **L3** | Application | FastAPI Instance | Application | Main application object |
| **L3** | Application | APIRouter | Application | Modular route organizer |
| **L3** | Application | Middleware | Application | Request/response processors |
| **L4** | Routing | Path Operations | FastAPI Instance | HTTP endpoint definitions |
| **L4** | Routing | HTTP Methods | Path Operations | GET, POST, PUT, DELETE, etc. |
| **L4** | Routing | Path Parameters | Path Operations | Dynamic URL segments |
| **L4** | Routing | Query Parameters | Path Operations | URL query string values |
| **L5** | HTTP Methods | @app.get() | Decorators | GET request handler |
| **L5** | HTTP Methods | @app.post() | Decorators | POST request handler |
| **L5** | HTTP Methods | @app.put() | Decorators | PUT request handler |
| **L5** | HTTP Methods | @app.delete() | Decorators | DELETE request handler |
| **L5** | HTTP Methods | @app.patch() | Decorators | PATCH request handler |
| **L5** | HTTP Methods | @app.options() | Decorators | OPTIONS request handler |
| **L5** | HTTP Methods | @app.head() | Decorators | HEAD request handler |
| **L6** | Data Validation | BaseModel | Pydantic | Base class for schemas |
| **L6** | Data Validation | Field | Pydantic | Field-level validation |
| **L6** | Data Validation | validator | Pydantic | Custom validation decorator |
| **L6** | Data Validation | root_validator | Pydantic | Model-level validation |
| **L7** | Dependency Injection | Depends | DI System | Dependency declaration |
| **L7** | Dependency Injection | Dependency Functions | DI System | Injectable callables |
| **L7** | Dependency Injection | Dependency Classes | DI System | Injectable classes |
| **L7** | Dependency Injection | Sub-dependencies | DI System | Nested dependencies |
| **L7** | Dependency Injection | Scoped Dependencies | DI System | Request-scoped instances |
| **L8** | Request Components | Path() | Parameter Types | Path parameter with validation |
| **L8** | Request Components | Query() | Parameter Types | Query parameter with validation |
| **L8** | Request Components | Body() | Parameter Types | Request body with validation |
| **L8** | Request Components | Header() | Parameter Types | HTTP header with validation |
| **L8** | Request Components | Cookie() | Parameter Types | Cookie with validation |
| **L8** | Request Components | Form() | Parameter Types | Form data with validation |
| **L8** | Request Components | File() | Parameter Types | File upload with validation |
| **L9** | Security | OAuth2PasswordBearer | Security Schemes | OAuth2 with password flow |
| **L9** | Security | OAuth2PasswordRequestForm | Security Schemes | Login form data |
| **L9** | Security | HTTPBasic | Security Schemes | Basic authentication |
| **L9** | Security | HTTPBearer | Security Schemes | Bearer token authentication |
| **L9** | Security | APIKeyHeader | Security Schemes | API key in header |
| **L9** | Security | APIKeyCookie | Security Schemes | API key in cookie |
| **L9** | Security | APIKeyQuery | Security Schemes | API key in query |
| **L10** | Response | JSONResponse | Response Types | JSON response |
| **L10** | Response | HTMLResponse | Response Types | HTML response |
| **L10** | Response | PlainTextResponse | Response Types | Plain text response |
| **L10** | Response | RedirectResponse | Response Types | HTTP redirect |
| **L10** | Response | FileResponse | Response Types | File download |
| **L10** | Response | StreamingResponse | Response Types | Streaming response |
| **L11** | Documentation | OpenAPI | Auto-documentation | API specification |
| **L11** | Documentation | Swagger UI | Auto-documentation | Interactive API docs |
| **L11** | Documentation | ReDoc | Auto-documentation | Alternative API docs |
| **L11** | Documentation | JSON Schema | Auto-documentation | Data model schemas |

---

## FastAPI Request-Response Lifecycle

Understanding the request-response lifecycle is crucial for debugging and optimization.

### Complete Request-Response Flow

```mermaid
graph TD
    A[Client Request] --> B{ASGI Server - Uvicorn}
    B --> C[FastAPI Application]
    C --> D[Middleware Chain]
    D --> E[Router Matching]
    
    E --> F{Path Found?}
    F -->|No| G[404 Not Found]
    F -->|Yes| H[Extract Parameters]
    
    H --> I[Path Parameters]
    H --> J[Query Parameters]
    H --> K[Request Body]
    H --> L[Headers/Cookies]
    
    I --> M[Pydantic Validation]
    J --> M
    K --> M
    L --> M
    
    M --> N{Validation Success?}
    N -->|No| O[422 Validation Error]
    N -->|Yes| P[Resolve Dependencies]
    
    P --> Q[Execute Dependency Functions]
    Q --> R{Dependency Success?}
    R -->|No| S[Dependency Error]
    R -->|Yes| T[Inject Dependencies]
    
    T --> U[Execute Path Operation Function]
    U --> V[Generate Response]
    
    V --> W[Response Model Validation]
    W --> X[Serialize to JSON]
    X --> Y[Apply Response Middleware]
    Y --> Z[Send Response to Client]
    
    style O fill:#ffe1e1
    style S fill:#ffe1e1
    style G fill:#ffe1e1
    style M fill:#fff4e1
    style W fill:#fff4e1
    style U fill:#e1ffe1
    style Z fill:#e1f5ff
```

---

## Phase 1: FastAPI Fundamentals

### 1.1 Installation and Setup

```bash
# Install FastAPI and Uvicorn
pip install fastapi uvicorn[standard]

# Or with optional dependencies
pip install "fastapi[all]"

# Install additional packages
pip install python-multipart  # For form data
pip install python-jose[cryptography]  # For JWT
pip install passlib[bcrypt]  # For password hashing
pip install sqlalchemy  # For database ORM
pip install alembic  # For database migrations
```

### 1.2 Hello World Application

```python
# main.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}
```

**Running the Application:**

```bash
# Development server with auto-reload
uvicorn main:app --reload

# Production server
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# With custom settings
uvicorn main:app --reload --log-level debug --access-log
```

**Access Points:**
- API: http://127.0.0.1:8000
- Interactive docs (Swagger): http://127.0.0.1:8000/docs
- Alternative docs (ReDoc): http://127.0.0.1:8000/redoc
- OpenAPI schema: http://127.0.0.1:8000/openapi.json

### 1.3 Understanding Async vs Sync

**When to Use Async:**

```python
# Use async for I/O-bound operations
@app.get("/users/{user_id}")
async def get_user(user_id: int):
    # Database query (async)
    user = await db.fetch_user(user_id)
    
    # External API call (async)
    data = await httpx_client.get(f"https://api.example.com/users/{user_id}")
    
    # File I/O (async)
    async with aiofiles.open('file.txt', 'r') as f:
        content = await f.read()
    
    return user
```

**When to Use Sync:**

```python
# Use sync (def) for CPU-bound operations or blocking code
@app.get("/compute")
def heavy_computation():
    # CPU-intensive task
    result = complex_calculation()
    
    # Blocking library (not async-compatible)
    data = blocking_lib.process()
    
    return {"result": result}
```

**Performance Considerations:**

- **async def**: Runs in event loop, non-blocking I/O
- **def**: Runs in thread pool, blocks event loop
- FastAPI handles both automatically
- Mixing is fine, but understand the performance implications

---

## Phase 2: Path Operations and Parameters

### 2.1 Path Parameters

```python
from fastapi import FastAPI, Path
from typing import Annotated

app = FastAPI()

# Basic path parameter
@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}

# Path parameter with validation
@app.get("/users/{user_id}")
async def read_user(
    user_id: Annotated[int, Path(
        title="User ID",
        description="The ID of the user to retrieve",
        ge=1,  # Greater than or equal to 1
        le=1000  # Less than or equal to 1000
    )]
):
    return {"user_id": user_id}

# Multiple path parameters
@app.get("/users/{user_id}/items/{item_id}")
async def read_user_item(user_id: int, item_id: int):
    return {"user_id": user_id, "item_id": item_id}

# String path parameter
@app.get("/files/{file_path:path}")
async def read_file(file_path: str):
    return {"file_path": file_path}
# Example: /files/home/user/documents/file.txt
```

### 2.2 Query Parameters

```python
from fastapi import Query

# Basic query parameters
@app.get("/items/")
async def read_items(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}
# URL: /items/?skip=0&limit=10

# Optional query parameter
@app.get("/items/")
async def read_items(q: str | None = None):
    if q:
        return {"q": q}
    return {"q": "Not provided"}

# Query parameter with validation
@app.get("/items/")
async def read_items(
    q: Annotated[str | None, Query(
        title="Query string",
        description="Query string for searching",
        min_length=3,
        max_length=50,
        pattern="^[a-zA-Z0-9_]+$"
    )] = None,
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 10
):
    return {"q": q, "skip": skip, "limit": limit}

# Multiple query parameters of same name (list)
@app.get("/items/")
async def read_items(
    tag: Annotated[list[str], Query()] = []
):
    return {"tags": tag}
# URL: /items/?tag=python&tag=fastapi&tag=api
```

### 2.3 Request Body with Pydantic

```python
from pydantic import BaseModel, Field, EmailStr
from typing import Annotated
from datetime import datetime

# Define Pydantic model
class User(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    full_name: str | None = None
    age: int = Field(..., ge=0, le=150)
    created_at: datetime = Field(default_factory=datetime.now)

# POST endpoint with request body
@app.post("/users/")
async def create_user(user: User):
    return user

# Multiple body parameters
class Item(BaseModel):
    name: str
    price: float

@app.post("/items/")
async def create_item(
    item: Item,
    user: User,
    importance: Annotated[int, Body(ge=1, le=10)]
):
    return {"item": item, "user": user, "importance": importance}

# Nested models
class Address(BaseModel):
    street: str
    city: str
    country: str
    postal_code: str

class UserWithAddress(BaseModel):
    username: str
    email: EmailStr
    address: Address  # Nested model

@app.post("/users/complete/")
async def create_user_complete(user: UserWithAddress):
    return user
```

### 2.4 Form Data and File Uploads

```python
from fastapi import Form, File, UploadFile

# Form data (application/x-www-form-urlencoded)
@app.post("/login/")
async def login(
    username: Annotated[str, Form()],
    password: Annotated[str, Form()]
):
    return {"username": username}

# File upload
@app.post("/files/")
async def create_file(file: Annotated[bytes, File()]):
    return {"file_size": len(file)}

# Upload with UploadFile (better for large files)
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    content = await file.read()
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": len(content)
    }

# Multiple file uploads
@app.post("/uploadfiles/")
async def create_upload_files(files: list[UploadFile]):
    return {"filenames": [file.filename for file in files]}

# Form + File together
@app.post("/files/annotated/")
async def create_file_annotated(
    file: UploadFile,
    token: Annotated[str, Form()],
    description: Annotated[str, Form()] = ""
):
    return {
        "token": token,
        "filename": file.filename,
        "description": description
    }
```

---

## Phase 3: Response Models and Status Codes

### 3.1 Response Models

```python
from pydantic import BaseModel

class UserIn(BaseModel):
    username: str
    password: str
    email: EmailStr

class UserOut(BaseModel):
    username: str
    email: EmailStr
    # Note: password is excluded

# Response model (automatic validation and documentation)
@app.post("/users/", response_model=UserOut)
async def create_user(user: UserIn):
    # Save user (password would be hashed)
    return user  # Password automatically excluded

# Response model with list
class Item(BaseModel):
    name: str
    price: float

@app.get("/items/", response_model=list[Item])
async def read_items():
    return [
        {"name": "Item 1", "price": 10.5},
        {"name": "Item 2", "price": 20.0}
    ]

# Response model with status code
@app.post("/items/", response_model=Item, status_code=201)
async def create_item(item: Item):
    return item

# Union response models
from typing import Union

class BaseUser(BaseModel):
    username: str

class AdminUser(BaseUser):
    is_admin: bool = True

class RegularUser(BaseUser):
    is_admin: bool = False

@app.get("/users/{user_id}", response_model=Union[AdminUser, RegularUser])
async def read_user(user_id: int):
    if user_id == 1:
        return AdminUser(username="admin")
    return RegularUser(username="regular_user")
```

### 3.2 Status Codes

```python
from fastapi import status

# Using status code constants
@app.post("/items/", status_code=status.HTTP_201_CREATED)
async def create_item(item: Item):
    return item

# Common status codes
@app.get("/items/{item_id}")
async def read_item(item_id: int):
    if item_id not in items:
        return Response(status_code=status.HTTP_404_NOT_FOUND)
    return items[item_id]

# Status code with HTTPException
from fastapi import HTTPException

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    if item_id not in items:
        raise HTTPException(
            status_code=404,
            detail="Item not found",
            headers={"X-Error": "Item not found header"}
        )
    return items[item_id]
```

### 3.3 Custom Responses

```python
from fastapi.responses import (
    JSONResponse,
    HTMLResponse,
    PlainTextResponse,
    RedirectResponse,
    FileResponse,
    StreamingResponse
)

# HTML response
@app.get("/html", response_class=HTMLResponse)
async def get_html():
    return """
    <html>
        <head><title>FastAPI</title></head>
        <body><h1>Hello from FastAPI</h1></body>
    </html>
    """

# Plain text response
@app.get("/text", response_class=PlainTextResponse)
async def get_text():
    return "Hello World"

# Redirect response
@app.get("/redirect")
async def redirect():
    return RedirectResponse(url="/docs")

# File download
@app.get("/download")
async def download_file():
    return FileResponse(
        path="file.pdf",
        filename="download.pdf",
        media_type="application/pdf"
    )

# Streaming response
import io

@app.get("/stream")
async def stream_data():
    async def generate():
        for i in range(10):
            yield f"data: {i}\n\n"
            await asyncio.sleep(1)
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

---

## Phase 4: Dependency Injection

### 4.1 Basic Dependencies

```python
from fastapi import Depends
from typing import Annotated

# Simple dependency function
async def common_parameters(q: str | None = None, skip: int = 0, limit: int = 10):
    return {"q": q, "skip": skip, "limit": limit}

# Using dependency
@app.get("/items/")
async def read_items(commons: Annotated[dict, Depends(common_parameters)]):
    return commons

@app.get("/users/")
async def read_users(commons: Annotated[dict, Depends(common_parameters)]):
    return commons

# Class as dependency
class CommonQueryParams:
    def __init__(self, q: str | None = None, skip: int = 0, limit: int = 10):
        self.q = q
        self.skip = skip
        self.limit = limit

@app.get("/items/")
async def read_items(commons: Annotated[CommonQueryParams, Depends()]):
    return commons
```

### 4.2 Sub-Dependencies

```python
# Sub-dependencies (dependencies of dependencies)
def query_extractor(q: str | None = None):
    return q

def query_or_cookie_extractor(
    q: Annotated[str, Depends(query_extractor)],
    last_query: Annotated[str | None, Cookie()] = None
):
    if not q:
        return last_query
    return q

@app.get("/items/")
async def read_query(
    query_or_default: Annotated[str, Depends(query_or_cookie_extractor)]
):
    return {"q_or_cookie": query_or_default}
```

### 4.3 Dependencies for Authentication

```python
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Dependency that checks authentication
async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
    user = await decode_token(token)  # Your token decoding logic
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user

# Protected endpoint
@app.get("/users/me")
async def read_users_me(current_user: Annotated[User, Depends(get_current_user)]):
    return current_user

# Dependency with additional checks
async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)]
):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

@app.get("/users/me/items/")
async def read_own_items(
    current_user: Annotated[User, Depends(get_current_active_user)]
):
    return [{"item_id": "Foo", "owner": current_user.username}]
```

### 4.4 Database Dependencies

```python
from sqlalchemy.orm import Session

# Database session dependency
def get_db():
    db = SessionLocal()
    try:
        yield db  # yield allows cleanup after request
    finally:
        db.close()

# Using database dependency
@app.get("/users/")
async def read_users(
    db: Annotated[Session, Depends(get_db)],
    skip: int = 0,
    limit: int = 100
):
    users = db.query(User).offset(skip).limit(limit).all()
    return users

@app.post("/users/")
async def create_user(
    user: UserCreate,
    db: Annotated[Session, Depends(get_db)]
):
    db_user = User(**user.dict())
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user
```

### 4.5 Dependency Injection for Validation

```python
# Dependency for validation
async def valid_post_id(post_id: int, db: Annotated[Session, Depends(get_db)]):
    post = db.query(Post).filter(Post.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    return post

# Use validated dependency
@app.get("/posts/{post_id}")
async def read_post(post: Annotated[Post, Depends(valid_post_id)]):
    return post

@app.put("/posts/{post_id}")
async def update_post(
    post: Annotated[Post, Depends(valid_post_id)],
    update_data: PostUpdate,
    db: Annotated[Session, Depends(get_db)]
):
    for key, value in update_data.dict(exclude_unset=True).items():
        setattr(post, key, value)
    db.commit()
    db.refresh(post)
    return post

# Chain dependencies for complex validation
async def parse_jwt_data(token: Annotated[str, Depends(oauth2_scheme)]):
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    return {"user_id": payload["sub"]}

async def valid_owned_post(
    post: Annotated[Post, Depends(valid_post_id)],
    token_data: Annotated[dict, Depends(parse_jwt_data)]
):
    if post.creator_id != token_data["user_id"]:
        raise HTTPException(status_code=403, detail="Not authorized")
    return post

@app.delete("/posts/{post_id}")
async def delete_post(
    post: Annotated[Post, Depends(valid_owned_post)],
    db: Annotated[Session, Depends(get_db)]
):
    db.delete(post)
    db.commit()
    return {"message": "Post deleted"}
```

---

## Phase 5: Pydantic Models and Validation

### 5.1 Advanced Pydantic Models

```python
from pydantic import BaseModel, Field, validator, root_validator
from datetime import datetime
from typing import List

class Item(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = Field(None, max_length=500)
    price: float = Field(..., gt=0, description="Price must be greater than 0")
    tax: float | None = Field(None, ge=0, le=100)
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    
    # Custom validator
    @validator('name')
    def name_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Name cannot be empty')
        return v.strip()
    
    # Validator with other field values
    @validator('tax')
    def tax_must_be_reasonable(cls, v, values):
        if 'price' in values and v and v > values['price']:
            raise ValueError('Tax cannot exceed price')
        return v
    
    # Root validator (validates entire model)
    @root_validator
    def check_price_and_tax(cls, values):
        price = values.get('price')
        tax = values.get('tax')
        if price and tax and (price + tax) > 10000:
            raise ValueError('Total (price + tax) cannot exceed 10000')
        return values
    
    # Config class
    class Config:
        schema_extra = {
            "example": {
                "name": "Laptop",
                "description": "A powerful laptop",
                "price": 999.99,
                "tax": 99.99,
                "tags": ["electronics", "computers"]
            }
        }
```

### 5.2 Field Validation

```python
from pydantic import EmailStr, HttpUrl, constr, conint, confloat

class User(BaseModel):
    username: constr(min_length=3, max_length=50, regex="^[a-zA-Z0-9_]+$")
    email: EmailStr
    website: HttpUrl | None = None
    age: conint(ge=18, le=120)
    score: confloat(ge=0.0, le=100.0)
    bio: constr(max_length=500) | None = None

# Using Field for more control
class Product(BaseModel):
    name: str = Field(..., example="Laptop")
    price: float = Field(..., gt=0, example=999.99)
    quantity: int = Field(..., ge=0, example=10)
    discount: float = Field(0, ge=0, le=100, example=10)
    
    @property
    def final_price(self) -> float:
        return self.price * (1 - self.discount / 100)
```

### 5.3 Nested and Complex Models

```python
# Nested models
class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str = Field(..., regex="^\d{5}(-\d{4})?$")

class Company(BaseModel):
    name: str
    address: Address
    employees: int = Field(..., gt=0)

class Employee(BaseModel):
    name: str
    email: EmailStr
    company: Company
    skills: List[str] = Field(default_factory=list)

# List of nested models
class Team(BaseModel):
    name: str
    members: List[Employee]
    
# Dict of models
class Organization(BaseModel):
    name: str
    departments: dict[str, Team]

# Recursive models
class Category(BaseModel):
    name: str
    subcategories: List['Category'] = []

Category.update_forward_refs()  # Required for recursive models
```

### 5.4 Model Inheritance and Composition

```python
# Model inheritance
class BaseUser(BaseModel):
    username: str
    email: EmailStr
    full_name: str | None = None
    
class UserCreate(BaseUser):
    password: str = Field(..., min_length=8)
    
class UserInDB(BaseUser):
    id: int
    hashed_password: str
    created_at: datetime
    is_active: bool = True
    
class UserOut(BaseUser):
    id: int
    created_at: datetime
    is_active: bool

# Model composition with Optional fields
class PatchUser(BaseModel):
    username: str | None = None
    email: EmailStr | None = None
    full_name: str | None = None
    
    class Config:
        # Only include fields that were explicitly set
        schema_extra = {
            "example": {
                "username": "newusername"
            }
        }
```

---

## Phase 6: Error Handling and Validation

### 6.1 HTTPException

```python
from fastapi import HTTPException, status

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    if item_id not in items_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Item {item_id} not found",
            headers={"X-Error": "Custom error header"}
        )
    return items_db[item_id]

# Custom exception with details
@app.get("/users/{user_id}")
async def read_user(user_id: int):
    if user_id not in users_db:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Not Found",
                "message": f"User with id {user_id} does not exist",
                "user_id": user_id,
                "suggestions": ["Check the user ID", "Try another ID"]
            }
        )
    return users_db[user_id]
```

### 6.2 Custom Exception Handlers

```python
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

# Custom exception class
class ItemNotFoundException(Exception):
    def __init__(self, item_id: int):
        self.item_id = item_id

# Custom exception handler
@app.exception_handler(ItemNotFoundException)
async def item_not_found_handler(request: Request, exc: ItemNotFoundException):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Item Not Found",
            "item_id": exc.item_id,
            "message": f"Item with ID {exc.item_id} was not found"
        }
    )

# Override default validation error handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "detail": exc.errors(),
            "body": exc.body
        }
    )

# Override default HTTP exception handler
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Exception",
            "message": exc.detail,
            "path": str(request.url)
        }
    )

# Generic exception handler
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "type": type(exc).__name__
        }
    )
```

### 6.3 Validation Error Responses

```python
from fastapi.encoders import jsonable_encoder

# Customize validation error response format
@app.exception_handler(RequestValidationError)
async def custom_validation_handler(request: Request, exc: RequestValidationError):
    errors = []
    for error in exc.errors():
        errors.append({
            "field": " -> ".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })
    
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "errors": errors,
            "request_body": jsonable_encoder(exc.body)
        }
    )
```

---

## Phase 7: Authentication and Security

### 7.1 OAuth2 with Password Flow

```python
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta

# Security configuration
SECRET_KEY = "your-secret-key-here"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Password hashing
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

# Token creation
def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# User authentication
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str | None = None

async def authenticate_user(username: str, password: str):
    user = await get_user_from_db(username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    user = await get_user_from_db(username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

# Login endpoint
@app.post("/token", response_model=Token)
async def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    user = await authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# Protected endpoints
@app.get("/users/me")
async def read_users_me(current_user: Annotated[User, Depends(get_current_user)]):
    return current_user
```

### 7.2 API Key Authentication

```python
from fastapi.security import APIKeyHeader

API_KEY = "your-api-key-here"
api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: Annotated[str, Depends(api_key_header)]):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )
    return api_key

# Protected endpoint
@app.get("/protected")
async def protected_route(api_key: Annotated[str, Depends(verify_api_key)]):
    return {"message": "Access granted"}
```

### 7.3 Role-Based Access Control (RBAC)

```python
from enum import Enum

class Role(str, Enum):
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"

class User(BaseModel):
    username: str
    role: Role

# Role checker dependency
class RoleChecker:
    def __init__(self, allowed_roles: list[Role]):
        self.allowed_roles = allowed_roles
    
    def __call__(self, user: Annotated[User, Depends(get_current_user)]):
        if user.role not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Operation not permitted"
            )
        return user

# Admin-only endpoint
@app.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    current_user: Annotated[User, Depends(RoleChecker([Role.ADMIN]))]
):
    return {"message": f"User {user_id} deleted"}

# Admin or User endpoint
@app.get("/dashboard")
async def dashboard(
    current_user: Annotated[User, Depends(RoleChecker([Role.ADMIN, Role.USER]))]
):
    return {"message": "Welcome to dashboard"}
```

### 7.4 CORS Configuration

```python
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://example.com"],  # Specific origins
    # allow_origins=["*"],  # Allow all origins (not recommended for production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Or more restrictive
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://example.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)
```

---

## Phase 8: Background Tasks and Async Operations

### 8.1 Background Tasks

```python
from fastapi import BackgroundTasks
import time

# Background task function
def write_notification(email: str, message: str):
    time.sleep(5)  # Simulate slow operation
    with open("log.txt", mode="a") as log:
        log.write(f"Notification to {email}: {message}\n")

# Endpoint with background task
@app.post("/send-notification/{email}")
async def send_notification(
    email: str,
    background_tasks: BackgroundTasks
):
    background_tasks.add_task(write_notification, email, "Registration successful")
    return {"message": "Notification will be sent"}

# Multiple background tasks
@app.post("/signup/")
async def signup(
    user: UserCreate,
    background_tasks: BackgroundTasks
):
    # Create user
    db_user = await create_user_in_db(user)
    
    # Add multiple background tasks
    background_tasks.add_task(send_welcome_email, user.email)
    background_tasks.add_task(send_admin_notification, user.username)
    background_tasks.add_task(initialize_user_resources, db_user.id)
    
    return {"message": "User created", "user_id": db_user.id}

# Background task with dependency
async def process_data(item_id: int, db: Session):
    # Long-running data processing
    item = db.query(Item).filter(Item.id == item_id).first()
    # Process item...
    item.processed = True
    db.commit()

@app.post("/items/{item_id}/process")
async def process_item(
    item_id: int,
    background_tasks: BackgroundTasks,
    db: Annotated[Session, Depends(get_db)]
):
    background_tasks.add_task(process_data, item_id, db)
    return {"message": "Processing started"}
```

### 8.2 Async Database Operations

```python
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Async database setup
DATABASE_URL = "postgresql+asyncpg://user:password@localhost/dbname"

engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

# Async database dependency
async def get_async_db():
    async with AsyncSessionLocal() as session:
        yield session

# Async CRUD operations
@app.get("/users/{user_id}")
async def read_user(
    user_id: int,
    db: Annotated[AsyncSession, Depends(get_async_db)]
):
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.post("/users/")
async def create_user(
    user: UserCreate,
    db: Annotated[AsyncSession, Depends(get_async_db)]
):
    db_user = User(**user.dict())
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    return db_user
```

### 8.3 Async External API Calls

```python
import httpx

# Async HTTP client
async def fetch_external_data(url: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()

@app.get("/external/{resource_id}")
async def get_external_resource(resource_id: int):
    url = f"https://api.example.com/resources/{resource_id}"
    data = await fetch_external_data(url)
    return data

# Multiple concurrent requests
@app.get("/aggregate")
async def aggregate_data():
    async with httpx.AsyncClient() as client:
        # Concurrent requests
        response1 = client.get("https://api1.example.com/data")
        response2 = client.get("https://api2.example.com/data")
        response3 = client.get("https://api3.example.com/data")
        
        # Wait for all
        results = await asyncio.gather(response1, response2, response3)
        
        return {
            "data1": results[0].json(),
            "data2": results[1].json(),
            "data3": results[2].json()
        }
```

---

## Phase 9: Middleware and Event Handlers

### 9.1 Custom Middleware

```python
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
import time

# Custom timing middleware
class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response

app.add_middleware(TimingMiddleware)

# Request logging middleware
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Log request
        print(f"Request: {request.method} {request.url}")
        
        # Process request
        response = await call_next(request)
        
        # Log response
        print(f"Response: {response.status_code}")
        
        return response

app.add_middleware(LoggingMiddleware)

# Authentication middleware
class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Skip auth for public routes
        if request.url.path in ["/docs", "/openapi.json", "/login"]:
            return await call_next(request)
        
        # Check authentication
        token = request.headers.get("Authorization")
        if not token or not token.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"detail": "Unauthorized"}
            )
        
        return await call_next(request)

app.add_middleware(AuthMiddleware)
```

### 9.2 Event Handlers

```python
@app.on_event("startup")
async def startup_event():
    print("Application starting up...")
    # Initialize database connection pool
    # Load ML models
    # Setup caching
    # Connect to external services

@app.on_event("shutdown")
async def shutdown_event():
    print("Application shutting down...")
    # Close database connections
    # Save state
    # Clean up resources

# Lifespan context manager (modern approach)
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting up...")
    # Initialize resources
    db_pool = await create_db_pool()
    ml_models = await load_ml_models()
    
    # Make available to app
    app.state.db_pool = db_pool
    app.state.ml_models = ml_models
    
    yield
    
    # Shutdown
    print("Shutting down...")
    await db_pool.close()

app = FastAPI(lifespan=lifespan)
```

---

## Phase 10: Testing FastAPI Applications

### 10.1 Testing with TestClient

```python
from fastapi.testclient import TestClient

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}

def test_create_item():
    response = client.post(
        "/items/",
        json={"name": "Test Item", "price": 10.5}
    )
    assert response.status_code == 201
    assert response.json()["name"] == "Test Item"

def test_read_item_not_found():
    response = client.get("/items/999")
    assert response.status_code == 404

def test_authentication():
    # Login
    response = client.post(
        "/token",
        data={"username": "testuser", "password": "testpass"}
    )
    assert response.status_code == 200
    token = response.json()["access_token"]
    
    # Authenticated request
    response = client.get(
        "/users/me",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
```

### 10.2 Async Testing

```python
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_async_endpoint():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/")
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_create_user_async():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post(
            "/users/",
            json={"username": "testuser", "email": "test@example.com"}
        )
    assert response.status_code == 201
```

### 10.3 Dependency Override for Testing

```python
from fastapi import Depends

# Original dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Test dependency
def override_get_db():
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()

# Override in tests
app.dependency_overrides[get_db] = override_get_db

def test_with_overridden_db():
    response = client.get("/users/")
    assert response.status_code == 200

# Reset overrides after tests
app.dependency_overrides = {}
```

---

## Phase 11: Project Structure and Best Practices

### 11.1 Recommended Project Structure

```
myproject/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app instance
│   ├── config.py               # Configuration
│   ├── dependencies.py         # Shared dependencies
│   ├── api/
│   │   ├── __init__.py
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   ├── endpoints/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── users.py
│   │   │   │   ├── items.py
│   │   │   │   └── auth.py
│   │   │   └── router.py       # API v1 router
│   │   └── v2/                 # API versioning
│   ├── core/
│   │   ├── __init__.py
│   │   ├── security.py         # Security utilities
│   │   └── config.py           # Core configuration
│   ├── models/
│   │   ├── __init__.py
│   │   ├── user.py             # SQLAlchemy models
│   │   └── item.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── user.py             # Pydantic schemas
│   │   └── item.py
│   ├── crud/
│   │   ├── __init__.py
│   │   ├── user.py             # CRUD operations
│   │   └── item.py
│   ├── db/
│   │   ├── __init__.py
│   │   ├── base.py             # Database base
│   │   └── session.py          # Session management
│   └── tests/
│       ├── __init__.py
│       ├── conftest.py
│       ├── test_users.py
│       └── test_items.py
├── alembic/                    # Database migrations
│   ├── versions/
│   └── env.py
├── .env                        # Environment variables
├── .env.example
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
└── README.md
```

### 11.2 Main Application Setup

```python
# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.router import api_router
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(
    api_router,
    prefix=settings.API_V1_STR
)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

### 11.3 Configuration Management

```python
# app/core/config.py
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    PROJECT_NAME: str = "My FastAPI Project"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "API Description"
    API_V1_STR: str = "/api/v1"
    
    # Database
    DATABASE_URL: str
    
    # Security
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000"]
    
    # External APIs
    EXTERNAL_API_KEY: str
    EXTERNAL_API_URL: str
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
```

### 11.4 Router Organization

```python
# app/api/v1/router.py
from fastapi import APIRouter
from app.api.v1.endpoints import users, items, auth

api_router = APIRouter()

api_router.include_router(
    auth.router,
    prefix="/auth",
    tags=["authentication"]
)

api_router.include_router(
    users.router,
    prefix="/users",
    tags=["users"]
)

api_router.include_router(
    items.router,
    prefix="/items",
    tags=["items"]
)
```

```python
# app/api/v1/endpoints/users.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app import crud, schemas
from app.api import deps

router = APIRouter()

@router.get("/", response_model=List[schemas.User])
def read_users(
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: schemas.User = Depends(deps.get_current_active_user)
):
    users = crud.user.get_multi(db, skip=skip, limit=limit)
    return users

@router.post("/", response_model=schemas.User, status_code=201)
def create_user(
    user_in: schemas.UserCreate,
    db: Session = Depends(deps.get_db)
):
    user = crud.user.get_by_email(db, email=user_in.email)
    if user:
        raise HTTPException(
            status_code=400,
            detail="User with this email already exists"
        )
    user = crud.user.create(db, obj_in=user_in)
    return user

@router.get("/{user_id}", response_model=schemas.User)
def read_user(
    user_id: int,
    db: Session = Depends(deps.get_db),
    current_user: schemas.User = Depends(deps.get_current_active_user)
):
    user = crud.user.get(db, id=user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
```

---

## Phase 12: Performance Optimization

### 12.1 Database Query Optimization

```python
from sqlalchemy.orm import selectinload, joinedload

# Bad: N+1 query problem
@app.get("/users")
async def get_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return [
        {
            "user": user,
            "posts": user.posts  # N additional queries!
        }
        for user in users
    ]

# Good: Eager loading
@app.get("/users")
async def get_users(db: Session = Depends(get_db)):
    users = db.query(User).options(
        selectinload(User.posts)  # Single additional query
    ).all()
    return users

# Using joinedload for one-to-one
users = db.query(User).options(
    joinedload(User.profile)  # Single query with JOIN
).all()
```

### 12.2 Response Caching

```python
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
from redis import asyncio as aioredis

# Setup cache
@app.on_event("startup")
async def startup():
    redis = aioredis.from_url("redis://localhost")
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")

# Cache endpoint response
@app.get("/items")
@cache(expire=60)  # Cache for 60 seconds
async def get_items(db: Session = Depends(get_db)):
    items = db.query(Item).all()
    return items

# Cache with custom key
@app.get("/user/{user_id}")
@cache(expire=300)
async def get_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    return user
```

### 12.3 Connection Pooling

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Configure connection pool
engine = create_engine(
    DATABASE_URL,
    pool_size=20,  # Number of permanent connections
    max_overflow=10,  # Max temporary connections
    pool_pre_ping=True,  # Test connections before use
    pool_recycle=3600,  # Recycle connections after 1 hour
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
```

### 12.4 Response Compression

```python
from fastapi.middleware.gzip import GZipMiddleware

app.add_middleware(GZipMiddleware, minimum_size=1000)
```

---

## Best Practices Summary

### ✅ DO

1. **Use Type Hints Everywhere**
   ```python
   async def get_user(user_id: int) -> User:
   ```

2. **Use Pydantic for Validation**
   ```python
   class UserCreate(BaseModel):
       username: str = Field(..., min_length=3)
   ```

3. **Use Dependency Injection**
   ```python
   async def get_db() -> Generator:
       db = SessionLocal()
       try:
           yield db
       finally:
           db.close()
   ```

4. **Use Response Models**
   ```python
   @app.post("/users/", response_model=UserOut)
   ```

5. **Handle Errors Properly**
   ```python
   raise HTTPException(status_code=404, detail="Not found")
   ```

6. **Use Async Where Appropriate**
   ```python
   async def fetch_data():
       async with httpx.AsyncClient() as client:
           return await client.get(url)
   ```

7. **Organize Code with Routers**
   ```python
   router = APIRouter(prefix="/users", tags=["users"])
   ```

8. **Document Your API**
   ```python
   @app.get("/items/", summary="Get all items")
   async def read_items():
       """
       Retrieve all items from the database.
       """
   ```

9. **Use Environment Variables**
   ```python
   from pydantic_settings import BaseSettings
   ```

10. **Write Tests**
    ```python
    def test_read_main():
        response = client.get("/")
        assert response.status_code == 200
    ```

### ❌ DON'T

1. **Don't Use Mutable Default Arguments**
   ```python
   # Bad
   async def get_items(tags: list = []):  # Mutable default!
   
   # Good
   async def get_items(tags: list | None = None):
       if tags is None:
           tags = []
   ```

2. **Don't Block the Event Loop**
   ```python
   # Bad
   @app.get("/")
   async def root():
       time.sleep(5)  # Blocks event loop!
   
   # Good
   @app.get("/")
   def root():  # Use sync function for blocking code
       time.sleep(5)
   ```

3. **Don't Ignore Validation Errors**
   ```python
   # Bad
   @app.post("/items/")
   async def create_item(item: dict):  # No validation
   
   # Good
   @app.post("/items/")
   async def create_item(item: ItemCreate):  # Validated
   ```

4. **Don't Hardcode Configuration**
   ```python
   # Bad
   SECRET_KEY = "my-secret-key"  # Hardcoded
   
   # Good
   SECRET_KEY = os.getenv("SECRET_KEY")  # From environment
   ```

5. **Don't Forget Error Handling**
   ```python
   # Bad
   @app.get("/users/{user_id}")
   async def get_user(user_id: int, db: Session = Depends(get_db)):
       return db.query(User).filter(User.id == user_id).first()
   
   # Good
   @app.get("/users/{user_id}")
   async def get_user(user_id: int, db: Session = Depends(get_db)):
       user = db.query(User).filter(User.id == user_id).first()
       if not user:
           raise HTTPException(status_code=404, detail="User not found")
       return user
   ```

6. **Don't Return Sensitive Data**
   ```python
   # Bad
   @app.get("/users/me")
   async def get_me(user: User = Depends(get_current_user)):
       return user  # Includes password hash!
   
   # Good
   @app.get("/users/me", response_model=UserOut)
   async def get_me(user: User = Depends(get_current_user)):
       return user  # Password excluded by response model
   ```

7. **Don't Create Database Sessions Incorrectly**
   ```python
   # Bad
   db = SessionLocal()  # Global session
   
   # Good
   def get_db():
       db = SessionLocal()
       try:
           yield db
       finally:
           db.close()
   ```

8. **Don't Mix Sync and Async Incorrectly**
   ```python
   # Bad
   async def get_user():
       return sync_db_call()  # Blocks event loop
   
   # Good
   def get_user():  # Use sync function
       return sync_db_call()
   ```

9. **Don't Ignore Security**
   ```python
   # Bad
   @app.get("/admin")
   async def admin_panel():  # No authentication
   
   # Good
   @app.get("/admin")
   async def admin_panel(user: User = Depends(get_admin_user)):
   ```

10. **Don't Skip Testing**
    ```python
    # Write comprehensive tests for all endpoints
    ```

---

## Performance Benchmarks

### FastAPI vs Other Frameworks

Based on TechEmpower benchmarks:

| Framework | Requests/sec | Latency (avg) |
|-----------|-------------|---------------|
| FastAPI (Uvicorn) | ~65,000 | 1.5ms |
| Flask | ~10,000 | 10ms |
| Django | ~8,000 | 12ms |
| Node.js (Express) | ~50,000 | 2ms |
| Go (Gin) | ~70,000 | 1.4ms |

**FastAPI Performance Formula:**

$
\text{Throughput} = \frac{\text{Workers} \times \text{Concurrent Connections}}{\text{Average Response Time}}
$

**Optimal Worker Count:**

$
\text{Workers} = (2 \times \text{CPU Cores}) + 1
$

Example for 4-core system:
$
\text{Workers} = (2 \times 4) + 1 = 9
$

```bash
# Run with optimal workers
uvicorn main:app --workers 9 --host 0.0.0.0 --port 8000
```

---

## Production Deployment

### Using Gunicorn with Uvicorn Workers

```bash
# Install
pip install gunicorn uvicorn[standard]

# Run with Gunicorn + Uvicorn workers
gunicorn app.main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --access-logfile - \
    --error-logfile - \
    --log-level info
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY ./app ./app

# Expose port
EXPOSE 8000

# Run with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/dbname
      - SECRET_KEY=${SECRET_KEY}
    depends_on:
      - db
    restart: unless-stopped

  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
      POSTGRES_DB: dbname
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    restart: unless-stopped

volumes:
  postgres_data:
```

### Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fastapi-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fastapi
  template:
    metadata:
      labels:
        app: fastapi
    spec:
      containers:
      - name: fastapi
        image: myapp:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: database-url
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: fastapi-service
spec:
  selector:
    app: fastapi
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

---

## Advanced Patterns

### WebSocket Support

```python
from fastapi import WebSocket, WebSocketDisconnect

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.broadcast(f"Client {client_id}: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"Client {client_id} disconnected")
```

### GraphQL Integration

```python
from strawberry.fastapi import GraphQLRouter
import strawberry

@strawberry.type
class User:
    id: int
    username: str
    email: str

@strawberry.type
class Query:
    @strawberry.field
    def user(self, id: int) -> User:
        return get_user_from_db(id)
    
    @strawberry.field
    def users(self) -> list[User]:
        return get_all_users_from_db()

@strawberry.type
class Mutation:
    @strawberry.mutation
    def create_user(self, username: str, email: str) -> User:
        return create_user_in_db(username, email)

schema = strawberry.Schema(query=Query, mutation=Mutation)
graphql_app = GraphQLRouter(schema)

app.include_router(graphql_app, prefix="/graphql")
```

### Server-Sent Events (SSE)

```python
from sse_starlette.sse import EventSourceResponse
import asyncio

@app.get("/stream")
async def stream_events(request: Request):
    async def event_generator():
        while True:
            if await request.is_disconnected():
                break
            
            # Generate event data
            data = {"message": "Current time", "time": datetime.now().isoformat()}
            
            yield {
                "event": "message",
                "data": json.dumps(data)
            }
            
            await asyncio.sleep(1)
    
    return EventSourceResponse(event_generator())
```

---

## Monitoring and Observability

### Logging

```python
import logging
from fastapi import Request

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response: {response.status_code}")
    return response

# Log in endpoints
@app.post("/users/")
async def create_user(user: UserCreate):
    logger.info(f"Creating user: {user.username}")
    try:
        result = await create_user_in_db(user)
        logger.info(f"User created successfully: {result.id}")
        return result
    except Exception as e:
        logger.error(f"Error creating user: {str(e)}", exc_info=True)
        raise
```

### Prometheus Metrics

```python
from prometheus_fastapi_instrumentator import Instrumentator

# Add metrics endpoint
Instrumentator().instrument(app).expose(app)

# Custom metrics
from prometheus_client import Counter, Histogram

request_counter = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

@app.middleware("http")
async def track_metrics(request: Request, call_next):
    with request_duration.labels(
        method=request.method,
        endpoint=request.url.path
    ).time():
        response = await call_next(request)
    
    request_counter.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    return response
```

### Health Checks

```python
@app.get("/health")
async def health_check(db: Session = Depends(get_db)):
    # Check database
    try:
        db.execute("SELECT 1")
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
    
    # Check Redis
    try:
        redis_client.ping()
        redis_status = "healthy"
    except Exception as e:
        redis_status = f"unhealthy: {str(e)}"
    
    return {
        "status": "healthy" if db_status == "healthy" and redis_status == "healthy" else "unhealthy",
        "database": db_status,
        "redis": redis_status,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/readiness")
async def readiness_check():
    # Check if app is ready to serve traffic
    return {"ready": True}

@app.get("/liveness")
async def liveness_check():
    # Check if app is alive
    return {"alive": True}
```

---

## Command Reference

### Development

```bash
# Run development server
uvicorn main:app --reload

# Run with custom host/port
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Run with log level
uvicorn main:app --reload --log-level debug

# Generate requirements
pip freeze > requirements.txt

# Run tests
pytest

# Run tests with coverage
pytest --cov=app --cov-report=html
```

### Production

```bash
# Run with Gunicorn + Uvicorn workers
gunicorn app.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Docker build and run
docker build -t myapp .
docker run -p 8000:8000 myapp

# Docker Compose
docker-compose up -d
docker-compose logs -f
docker-compose down
```

---

## Conclusion

FastAPI has revolutionized Python web development by combining modern Python features (type hints, async/await) with exceptional performance and developer experience. Its automatic API documentation, intuitive dependency injection, and Pydantic integration make it an excellent choice for building production-ready APIs.

**Key Takeaways:**

1. **Type hints are essential** - Enable validation, documentation, and IDE support
2. **Pydantic models** - Automatic validation and serialization
3. **Dependency injection** - Clean, testable, reusable code
4. **Async/await** - High performance for I/O-bound operations
5. **Automatic documentation** - Interactive API docs out of the box
6. **Response models** - Control output, exclude sensitive data
7. **Security built-in** - OAuth2, JWT, API keys made easy
8. **Testing friendly** - TestClient and dependency overrides
9. **Production ready** - Async, performant, scalable
10. **Modern Python** - Leverages latest language features

By following the practices outlined in this guide and understanding FastAPI's architecture and patterns, you'll be equipped to build fast, secure, and maintainable APIs that scale from prototypes to production systems handling millions of requests.

---

## References

<a href="https://fastapi.tiangolo.com/" target="_blank">FastAPI Official Documentation</a>

<a href="https://fastapi.tiangolo.com/tutorial/" target="_blank">FastAPI Tutorial - User Guide</a>

<a href="https://fastapi.tiangolo.com/advanced/" target="_blank">FastAPI Advanced User Guide</a>

<a href="https://docs.pydantic.dev/" target="_blank">Pydantic Documentation</a>

<a href="https://www.starlette.io/" target="_blank">Starlette Documentation</a>

<a href="https://www.uvicorn.org/" target="_blank">Uvicorn ASGI Server</a>

<a href="https://fastapi.tiangolo.com/deployment/" target="_blank">FastAPI Deployment Guide</a>

<a href="https://fastapi.tiangolo.com/tutorial/dependencies/" target="_blank">FastAPI Dependency Injection</a>

<a href="https://fastapi.tiangolo.com/tutorial/security/" target="_blank">FastAPI Security and Authentication</a>

<a href="https://github.com/tiangolo/fastapi" target="_blank">FastAPI GitHub Repository</a>

<a href="https://fastapi.tiangolo.com/async/" target="_blank">Async/Await in FastAPI</a>

<a href="https://fastapi.tiangolo.com/tutorial/sql-databases/" target="_blank">FastAPI with SQL Databases</a>

---
