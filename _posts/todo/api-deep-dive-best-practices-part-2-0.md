type: string
                  format: email
                  example: john@example.com
                password:
                  type: string
                  format: password
                  example: SecurePass123!
      responses:
        '200':
          description: Successfully authenticated
          content:
            application/json:
              schema:
                type: object
                properties:
                  access_token:
                    type: string
                    example: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
                  refresh_token:
                    type: string
                    example: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
                  token_type:
                    type: string
                    example: Bearer
                  expires_in:
                    type: integer
                    example: 3600
        '401':
          $ref: '#/components/responses/UnauthorizedError'
        '429':
          $ref: '#/components/responses/RateLimitError'

  /users:
    get:
      tags:
        - Users
      summary: List all users
      description: Retrieve a paginated list of users
      operationId: getUsers
      security:
        - BearerAuth: []
      parameters:
        - name: page
          in: query
          description: Page number
          schema:
            type: integer
            minimum: 1
            default: 1
        - name: limit
          in: query
          description: Number of items per page
          schema:
            type: integer
            minimum: 1
            maximum: 100
            default: 20
        - name: sort
          in: query
          description: Sort field and direction
          schema:
            type: string
            example: -created_at
        - name: filter[role]
          in: query
          description: Filter by role
          schema:
            type: string
            enum: [admin, editor, user]
      responses:
        '200':
          description: Successful response
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    type: array
                    items:
                      $ref: '#/components/schemas/User'
                  meta:
                    $ref: '#/components/schemas/PaginationMeta'
                  links:
                    $ref: '#/components/schemas/PaginationLinks'
        '401':
          $ref: '#/components/responses/UnauthorizedError'
        '403':
          $ref: '#/components/responses/ForbiddenError'
          
    post:
      tags:
        - Users
      summary: Create a new user
      description: Register a new user account
      operationId: createUser
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/UserCreate'
      responses:
        '201':
          description: User created successfully
          headers:
            Location:
              description: URL of the created user
              schema:
                type: string
                example: /api/v1/users/123
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    $ref: '#/components/schemas/User'
        '400':
          $ref: '#/components/responses/ValidationError'
        '409':
          description: User already exists
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'

  /users/{userId}:
    parameters:
      - name: userId
        in: path
        required: true
        description: User ID
        schema:
          type: integer
          example: 123
          
    get:
      tags:
        - Users
      summary: Get user by ID
      description: Retrieve detailed information about a specific user
      operationId: getUserById
      security:
        - BearerAuth: []
      responses:
        '200':
          description: Successful response
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    $ref: '#/components/schemas/User'
        '404':
          $ref: '#/components/responses/NotFoundError'
          
    put:
      tags:
        - Users
      summary: Update user
      description: Replace entire user resource
      operationId: updateUser
      security:
        - BearerAuth: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/UserUpdate'
      responses:
        '200':
          description: User updated successfully
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    $ref: '#/components/schemas/User'
        '400':
          $ref: '#/components/responses/ValidationError'
        '404':
          $ref: '#/components/responses/NotFoundError'
          
    delete:
      tags:
        - Users
      summary: Delete user
      description: Permanently delete a user account
      operationId: deleteUser
      security:
        - BearerAuth: []
      responses:
        '204':
          description: User deleted successfully
        '404':
          $ref: '#/components/responses/NotFoundError'

components:
  securitySchemes:
    BearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
    ApiKeyAuth:
      type: apiKey
      in: header
      name: X-API-Key

  schemas:
    User:
      type: object
      properties:
        id:
          type: integer
          readOnly: true
          example: 123
        name:
          type: string
          example: John Doe
        email:
          type: string
          format: email
          example: john@example.com
        role:
          type: string
          enum: [admin, editor, user]
          example: user
        created_at:
          type: string
          format: date-time
          readOnly: true
          example: 2025-11-15T10:30:00Z
        updated_at:
          type: string
          format: date-time
          readOnly: true
          example: 2025-11-15T10:30:00Z
          
    UserCreate:
      type: object
      required:
        - name
        - email
        - password
      properties:
        name:
          type: string
          minLength: 2
          maxLength: 100
          example: John Doe
        email:
          type: string
          format: email
          example: john@example.com
        password:
          type: string
          format: password
          minLength: 8
          example: SecurePass123!
        role:
          type: string
          enum: [user, editor]
          default: user
          
    UserUpdate:
      type: object
      required:
        - name
        - email
      properties:
        name:
          type: string
          minLength: 2
          maxLength: 100
        email:
          type: string
          format: email
          
    Error:
      type: object
      properties:
        error:
          type: object
          properties:
            code:
              type: string
              example: VALIDATION_ERROR
            message:
              type: string
              example: Request validation failed
            details:
              type: array
              items:
                type: object
                properties:
                  field:
                    type: string
                  message:
                    type: string
                  value:
                    oneOf:
                      - type: string
                      - type: number
            request_id:
              type: string
              example: req_abc123
            timestamp:
              type: string
              format: date-time
              
    PaginationMeta:
      type: object
      properties:
        total:
          type: integer
          example: 100
        page:
          type: integer
          example: 1
        per_page:
          type: integer
          example: 20
        total_pages:
          type: integer
          example: 5
          
    PaginationLinks:
      type: object
      properties:
        self:
          type: string
          example: /api/v1/users?page=1
        first:
          type: string
          example: /api/v1/users?page=1
        prev:
          type: string
          nullable: true
          example: null
        next:
          type: string
          nullable: true
          example: /api/v1/users?page=2
        last:
          type: string
          example: /api/v1/users?page=5

  responses:
    UnauthorizedError:
      description: Authentication required or failed
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            error:
              code: UNAUTHORIZED
              message: Access token is required
              
    ForbiddenError:
      description: Insufficient permissions
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            error:
              code: FORBIDDEN
              message: Insufficient permissions
              
    NotFoundError:
      description: Resource not found
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            error:
              code: NOT_FOUND
              message: User not found
              
    ValidationError:
      description: Validation failed
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            error:
              code: VALIDATION_ERROR
              message: Request validation failed
              details:
                - field: email
                  message: Email address is invalid
                  value: invalid-email
                  
    RateLimitError:
      description: Rate limit exceeded
      headers:
        X-RateLimit-Limit:
          description: Request limit per time window
          schema:
            type: integer
        X-RateLimit-Remaining:
          description: Remaining requests in current window
          schema:
            type: integer
        X-RateLimit-Reset:
          description: Time when the rate limit resets (Unix timestamp)
          schema:
            type: integer
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
```

### Setting Up Swagger UI

```javascript
// Swagger UI setup
const swaggerUi = require('swagger-ui-express');
const YAML = require('yamljs');
const swaggerDocument = YAML.load('./openapi.yaml');

app.use('/api-docs', swaggerUi.serve, swaggerUi.setup(swaggerDocument, {
  customCss: '.swagger-ui .topbar { display: none }',
  customSiteTitle: "API Documentation",
  customfavIcon: "/favicon.ico"
}));
```

### Code Examples in Multiple Languages

Include examples for popular languages and frameworks:

**cURL:**
```bash
curl -X POST https://api.example.com/v1/users \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "name": "John Doe",
    "email": "john@example.com",
    "password": "SecurePass123!"
  }'
```

**Python (requests):**
```python
import requests

url = "https://api.example.com/v1/users"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer YOUR_TOKEN"
}
data = {
    "name": "John Doe",
    "email": "john@example.com",
    "password": "SecurePass123!"
}

response = requests.post(url, json=data, headers=headers)
print(response.json())
```

**JavaScript (fetch):**
```javascript
const response = await fetch('https://api.example.com/v1/users', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer YOUR_TOKEN'
  },
  body: JSON.stringify({
    name: 'John Doe',
    email: 'john@example.com',
    password: 'SecurePass123!'
  })
});

const data = await response.json();
console.log(data);
```

**Java (OkHttp):**
```java
OkHttpClient client = new OkHttpClient();

MediaType JSON = MediaType.get("application/json; charset=utf-8");
String json = "{\"name\":\"John Doe\",\"email\":\"john@example.com\",\"password\":\"SecurePass123!\"}";

RequestBody body = RequestBody.create(json, JSON);
Request request = new Request.Builder()
    .url("https://api.example.com/v1/users")
    .addHeader("Authorization", "Bearer YOUR_TOKEN")
    .post(body)
    .build();

try (Response response = client.newCall(request).execute()) {
    System.out.println(response.body().string());
}
```

## API Testing

### Unit Testing

```javascript
const request = require('supertest');
const app = require('../app');
const db = require('../models');

describe('User API', () => {
  let authToken;
  
  beforeAll(async () => {
    await db.sync({ force: true });
    
    // Create test user and get token
    const response = await request(app)
      .post('/api/auth/login')
      .send({
        email: 'test@example.com',
        password: 'TestPass123!'
      });
    
    authToken = response.body.access_token;
  });
  
  afterAll(async () => {
    await db.close();
  });
  
  describe('GET /api/v1/users', () => {
    it('should return list of users', async () => {
      const response = await request(app)
        .get('/api/v1/users')
        .set('Authorization', `Bearer ${authToken}`)
        .expect('Content-Type', /json/)
        .expect(200);
      
      expect(response.body).toHaveProperty('data');
      expect(Array.isArray(response.body.data)).toBe(true);
      expect(response.body).toHaveProperty('meta');
    });
    
    it('should support pagination', async () => {
      const response = await request(app)
        .get('/api/v1/users?page=1&limit=10')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);
      
      expect(response.body.meta).toHaveProperty('page', 1);
      expect(response.body.meta).toHaveProperty('per_page', 10);
    });
    
    it('should return 401 without authentication', async () => {
      await request(app)
        .get('/api/v1/users')
        .expect(401);
    });
  });
  
  describe('POST /api/v1/users', () => {
    it('should create a new user', async () => {
      const userData = {
        name: 'Jane Doe',
        email: 'jane@example.com',
        password: 'SecurePass123!'
      };
      
      const response = await request(app)
        .post('/api/v1/users')
        .send(userData)
        .expect('Content-Type', /json/)
        .expect(201);
      
      expect(response.body.data).toHaveProperty('id');
      expect(response.body.data.name).toBe(userData.name);
      expect(response.body.data.email).toBe(userData.email);
      expect(response.body.data).not.toHaveProperty('password');
    });
    
    it('should validate email format', async () => {
      const response = await request(app)
        .post('/api/v1/users')
        .send({
          name: 'Test User',
          email: 'invalid-email',
          password: 'SecurePass123!'
        })
        .expect(400);
      
      expect(response.body.error).toHaveProperty('code', 'VALIDATION_ERROR');
      expect(response.body.error.details).toEqual(
        expect.arrayContaining([
          expect.objectContaining({ field: 'email' })
        ])
      );
    });
    
    it('should prevent duplicate emails', async () => {
      const userData = {
        name: 'Duplicate User',
        email: 'jane@example.com',
        password: 'SecurePass123!'
      };
      
      await request(app)
        .post('/api/v1/users')
        .send(userData)
        .expect(409);
    });
  });
});
```

### Integration Testing

```javascript
describe('User Workflow Integration Tests', () => {
  it('should complete full user lifecycle', async () => {
    // 1. Register new user
    const registerResponse = await request(app)
      .post('/api/v1/users')
      .send({
        name: 'Integration Test User',
        email: 'integration@example.com',
        password: 'SecurePass123!'
      })
      .expect(201);
    
    const userId = registerResponse.body.data.id;
    
    // 2. Login
    const loginResponse = await request(app)
      .post('/api/auth/login')
      .send({
        email: 'integration@example.com',
        password: 'SecurePass123!'
      })
      .expect(200);
    
    const token = loginResponse.body.access_token;
    
    // 3. Get profile
    const profileResponse = await request(app)
      .get('/api/v1/profile')
      .set('Authorization', `Bearer ${token}`)
      .expect(200);
    
    expect(profileResponse.body.data.id).toBe(userId);
    
    // 4. Update profile
    await request(app)
      .patch('/api/v1/profile')
      .set('Authorization', `Bearer ${token}`)
      .send({ name: 'Updated Integration User' })
      .expect(200);
    
    // 5. Create post
    const postResponse = await request(app)
      .post('/api/v1/posts')
      .set('Authorization', `Bearer ${token}`)
      .send({
        title: 'Test Post',
        content: 'This is a test post'
      })
      .expect(201);
    
    const postId = postResponse.body.data.id;
    
    // 6. Get user's posts
    const postsResponse = await request(app)
      .get(`/api/v1/users/${userId}/posts`)
      .set('Authorization', `Bearer ${token}`)
      .expect(200);
    
    expect(postsResponse.body.data).toHaveLength(1);
    expect(postsResponse.body.data[0].id).toBe(postId);
    
    // 7. Delete post
    await request(app)
      .delete(`/api/v1/posts/${postId}`)
      .set('Authorization', `Bearer ${token}`)
      .expect(204);
  });
});
```

### Load Testing with Artillery

```yaml
# artillery.yml
config:
  target: "https://api.example.com"
  phases:
    - duration: 60
      arrivalRate: 10
      name: "Warm up"
    - duration: 120
      arrivalRate: 50
      name: "Sustained load"
    - duration: 60
      arrivalRate: 100
      name: "Peak load"
  processor: "./test-helpers.js"
  
scenarios:
  - name: "User workflow"
    flow:
      - post:
          url: "/api/auth/login"
          json:
            email: "{{ $randomEmail() }}"
            password: "TestPass123!"
          capture:
            - json: "$.access_token"
              as: "token"
      
      - get:
          url: "/api/v1/users"
          headers:
            Authorization: "Bearer {{ token }}"
          
      - think: 2
      
      - post:
          url: "/api/v1/posts"
          headers:
            Authorization: "Bearer {{ token }}"
          json:
            title: "{{ $randomString() }}"
            content: "{{ $randomString() }}"
```

```javascript
// test-helpers.js
module.exports = {
  $randomEmail: () => {
    return `user${Math.random().toString(36).substr(2, 9)}@example.com`;
  },
  $randomString: () => {
    return Math.random().toString(36).substr(2, 15);
  }
};
```

## API Monitoring and Observability

### Logging

```javascript
const winston = require('winston');

// Winston logger setup
const logger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.json()
  ),
  transports: [
    new winston.transports.File({ filename: 'error.log', level: 'error' }),
    new winston.transports.File({ filename: 'combined.log' }),
  ],
});

if (process.env.NODE_ENV !== 'production') {
  logger.add(new winston.transports.Console({
    format: winston.format.simple()
  }));
}

// Comprehensive logging middleware
app.use((req, res, next) => {
  const startTime = Date.now();
  req.id = req.headers['x-request-id'] || crypto.randomUUID();
  res.setHeader('X-Request-ID', req.id);
  
  // Log request
  logger.info('API Request', {
    request_id: req.id,
    method: req.method,
    url: req.url,
    ip: req.ip,
    user_agent: req.get('user-agent'),
    user_id: req.user?.id
  });
  
  // Capture response
  const originalSend = res.send;
  res.send = function(data) {
    const duration = Date.now() - startTime;
    
    logger.info('API Response', {
      request_id: req.id,
      status_code: res.statusCode,
      duration_ms: duration,
      response_size: Buffer.byteLength(data)
    });
    
    originalSend.call(this, data);
  };
  
  next();
});

// Error logging
app.use((err, req, res, next) => {
  logger.error('API Error', {
    request_id: req.id,
    error: {
      message: err.message,
      stack: err.stack,
      code: err.code
    },
    request: {
      method: req.method,
      url: req.url,
      user_id: req.user?.id
    }
  });
  
  res.status(err.statusCode || 500).json({
    error: {
      message: err.message,
      request_id: req.id
    }
  });
});
```

### Performance Metrics

```javascript
const prometheus = require('prom-client');

// Create metrics
const register = new prometheus.Registry();

const httpRequestDuration = new prometheus.Histogram({
  name: 'http_request_duration_ms',
  help: 'Duration of HTTP requests in ms',
  labelNames: ['method', 'route', 'status_code'],
  buckets: [10, 50, 100, 200, 500, 1000, 2000, 5000]
});

const httpRequestTotal = new prometheus.Counter({
  name: 'http_requests_total',
  help: 'Total number of HTTP requests',
  labelNames: ['method', 'route', 'status_code']
});

const activeConnections = new prometheus.Gauge({
  name: 'active_connections',
  help: 'Number of active connections'
});

register.registerMetric(httpRequestDuration);
register.registerMetric(httpRequestTotal);
register.registerMetric(activeConnections);

// Metrics middleware
app.use((req, res, next) => {
  const start = Date.now();
  activeConnections.inc();
  
  res.on('finish', () => {
    const duration = Date.now() - start;
    const route = req.route ? req.route.path : req.url;
    
    httpRequestDuration
      .labels(req.method, route, res.statusCode)
      .observe(duration);
      
    httpRequestTotal
      .labels(req.method, route, res.statusCode)
      .inc();
      
    activeConnections.dec();
  });
  
  next();
});

// Metrics endpoint
app.get('/metrics', async (req, res) => {
  res.set('Content-Type', register.contentType);
  res.end(await register.metrics());
});
```

### Health Check Endpoints

```javascript
// Basic health check
app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Detailed health check
app.get('/health/detailed', async (req, res) => {
  const checks = {
    uptime: process.uptime(),
    timestamp: new Date().toISOString(),
    status: 'ok',
    checks: {}
  };
  
  // Database check
  try {
    await db.authenticate();
    checks.checks.database = { status: 'ok' };
  } catch (err) {
    checks.checks.database = { status: 'error', message: err.message };
    checks.status = 'degraded';
  }
  
  // Redis check
  try {
    await redisClient.ping();
    checks.checks.redis = { status: 'ok' };
  } catch (err) {
    checks.checks.redis = { status: 'error', message: err.message };
    checks.status = 'degraded';
  }
  
  // Memory check
  const memUsage = process.memoryUsage();
  checks.checks.memory = {
    status: memUsage.heapUsed / memUsage.heapTotal < 0.9 ? 'ok' : 'warning',
    heap_used_mb: Math.round(memUsage.heapUsed / 1024 / 1024),
    heap_total_mb: Math.round(memUsage.heapTotal / 1024 / 1024)
  };
  
  const statusCode = checks.status === 'ok' ? 200 : 503;
  res.status(statusCode).json(checks);
});
```

## Performance Optimization

### Caching Strategies

```javascript
const Redis = require('ioredis');
const redis = new Redis();

// Cache middleware
const cacheMiddleware = (duration = 300) => {
  return async (req, res, next) => {
    if (req.method !== 'GET') {
      return next();
    }
    
    const key = `cache:${req.url}`;
    
    try {
      const cachedResponse = await redis.get(key);
      
      if (cachedResponse) {
        res.setHeader('X-Cache', 'HIT');
        return res.json(JSON.parse(cachedResponse));
      }
      
      res.setHeader('X-Cache', 'MISS');
      
      // Intercept response
      const originalSend = res.json;
      res.json = function(data) {
        redis.setex(key, duration, JSON.stringify(data));
        originalSend.call(this, data);
      };
      
      next();
    } catch (err) {
      next();
    }
  };
};

// Usage
app.get('/api/v1/products', cacheMiddleware(600), async (req, res) => {
  const products = await db.products.findAll();
  res.json({ data: products });
});

// Cache invalidation
const invalidateCache = async (pattern) => {
  const keys = await redis.keys(pattern);
  if (keys.length > 0) {
    await redis.del(...keys);
  }
};

// Usage
app.post('/api/v1/products', async (req, res) => {
  const product = await db.products.create(req.body);
  await invalidateCache('cache:/api/v1/products*');
  res.status(201).json({ data: product });
});
```

### Database Query Optimization

```javascript
// Eager loading to prevent N+1 queries
app.get('/api/v1/users/:id/posts', async (req, res) => {
  // GOOD: Single query with joins
  const user = await db.users.findByPk(req.params.id, {
    include: [
      {
        model: db.posts,
        include: [db.comments] // Nested eager loading
      }
    ]
  });
  
  res.json({ data: user.posts });
});

// Field selection to reduce payload size
app.get('/api/v1/users', async (req, res) => {
  const fields = req.query.fields ? req.query.fields.split(',') : null;
  
  const users = await db.users.findAll({
    attributes: fields || ['id', 'name', 'email']
  });
  
  res.json({ data: users });
});
```

### Compression

```javascript
const compression = require('compression');

// Enable gzip compression
app.use(compression({
  filter: (req, res) => {
    if (req.headers['x-no-compression']) {
      return false;
    }
    return compression.filter(req, res);
  },
  level: 6 // Compression level (0-9)
}));
```

## Advanced Patterns

### API Gateway

```javascript
const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');

const gateway = express();

// Service registry
const services = {
  users: 'http://localhost:3001',
  posts: 'http://localhost:3002',
  comments: 'http://localhost:3003'
};

// Authentication middleware
gateway.use(authenticateJWT);

// Rate limiting
gateway.use(globalLimiter);

// Route requests to appropriate microservices
Object.entries(services).forEach(([name, target]) => {
  gateway.use(`/api/${name}`, createProxyMiddleware({
    target,
    changeOrigin: true,
    pathRewrite: {
      [`^/api/${name}`]: ''
    },
    onProxyReq: (proxyReq, req) => {
      // Add user info to forwarded request
      if (req.user) {
        proxyReq.setHeader('X-User-ID', req.user.id);
        proxyReq.setHeader('X-User-Role', req.user.role);
      }
    },
    onError: (err, req, res) => {
      logger.error(`Proxy error for ${name}:`, err);
      res.status(503).json({
        error: {
          message: 'Service temporarily unavailable',
          service: name
        }
      });
    }
  }));
});

gateway.listen(3000, () => {
  console.log('API Gateway running on port 3000');
});
```

### Webhooks

```javascript
// Webhook management
app.post('/api/v1/webhooks', authenticateJWT, async (req, res) => {
  const { url, events, secret } = req.body;
  
  const webhook = await db.webhooks.create({
    user_id: req.user.id,
    url,
    events,
    secret: secret || crypto.randomBytes(32).toString('hex'),
    active: true
  });
  
  res.status(201).json({ data: webhook });
});

// Webhook delivery
const deliverWebhook = async (webhook, event, data) => {
  const payload = {
    id: crypto.randomUUID(),
    event,
    data,
    timestamp: new Date().toISOString()
  };
  
  // Sign---
title: "API: Deep Dive & Best Practices - Part 2"
date: 2025-11-15
categories: [Software Engineering, API Development, Web Services]
tags: [api-security, authentication, authorization, api-documentation, api-testing, api-monitoring, performance-optimization]
math: true
---

## Introduction to Part 2

This is the second part of the comprehensive API guide. Part 1 covered API fundamentals, architectural styles, REST design principles, and versioning strategies. Part 2 focuses on security, documentation, testing, monitoring, and advanced implementation patterns.

## API Security

### Authentication Mechanisms

#### API Keys

Simplest form of authentication - unique identifier for each client.

```http
# Header-based
GET /api/v1/users HTTP/1.1
X-API-Key: sk_live_abc123xyz789

# Query parameter (less secure)
GET /api/v1/users?api_key=sk_live_abc123xyz789

# Implementation (Express)
const API_KEYS = {
  'sk_live_abc123xyz789': { name: 'Production App', tier: 'premium' },
  'sk_test_def456uvw012': { name: 'Test App', tier: 'free' }
};

const authenticateApiKey = (req, res, next) => {
  const apiKey = req.headers['x-api-key'] || req.query.api_key;
  
  if (!apiKey) {
    return res.status(401).json({
      error: { message: 'API key is required' }
    });
  }
  
  const keyInfo = API_KEYS[apiKey];
  
  if (!keyInfo) {
    return res.status(401).json({
      error: { message: 'Invalid API key' }
    });
  }
  
  req.apiKey = apiKey;
  req.client = keyInfo;
  next();
};

app.use('/api', authenticateApiKey);
```

**Best Practices:**
- Use HTTPS only
- Implement key rotation
- Different keys for different environments
- Monitor and log key usage
- Set expiration dates
- Implement rate limiting per key

#### OAuth 2.0

Industry-standard protocol for authorization.

**OAuth 2.0 Flows:**

1. **Authorization Code Flow** (most secure, for web apps)

```http
# Step 1: Redirect user to authorization server
GET /oauth/authorize?
    response_type=code&
    client_id=abc123&
    redirect_uri=https://myapp.com/callback&
    scope=read write&
    state=xyz789

# Step 2: User approves, redirected back with code
https://myapp.com/callback?code=auth_code_123&state=xyz789

# Step 3: Exchange code for access token
POST /oauth/token HTTP/1.1
Content-Type: application/x-www-form-urlencoded

grant_type=authorization_code&
code=auth_code_123&
client_id=abc123&
client_secret=secret456&
redirect_uri=https://myapp.com/callback

# Response
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "refresh_token": "refresh_token_xyz",
  "scope": "read write"
}

# Step 4: Use access token
GET /api/v1/users HTTP/1.1
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

2. **Client Credentials Flow** (for server-to-server)

```http
POST /oauth/token HTTP/1.1
Content-Type: application/x-www-form-urlencoded

grant_type=client_credentials&
client_id=abc123&
client_secret=secret456&
scope=read

# Response
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600
}
```

3. **Refresh Token Flow**

```http
POST /oauth/token HTTP/1.1
Content-Type: application/x-www-form-urlencoded

grant_type=refresh_token&
refresh_token=refresh_token_xyz&
client_id=abc123&
client_secret=secret456

# Response
{
  "access_token": "new_access_token...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "refresh_token": "new_refresh_token..."
}
```

**OAuth 2.0 Implementation (Node.js with oauth2-server):**

```javascript
const OAuth2Server = require('oauth2-server');
const Request = OAuth2Server.Request;
const Response = OAuth2Server.Response;

// OAuth model implementation
const model = {
  getClient: async (clientId, clientSecret) => {
    // Fetch from database
    const client = await db.clients.findOne({ 
      where: { id: clientId, secret: clientSecret } 
    });
    
    return client ? {
      id: client.id,
      grants: ['authorization_code', 'refresh_token'],
      redirectUris: client.redirect_uris
    } : null;
  },
  
  saveToken: async (token, client, user) => {
    // Save to database
    return await db.tokens.create({
      access_token: token.accessToken,
      access_token_expires_at: token.accessTokenExpiresAt,
      refresh_token: token.refreshToken,
      refresh_token_expires_at: token.refreshTokenExpiresAt,
      client_id: client.id,
      user_id: user.id
    });
  },
  
  getAccessToken: async (accessToken) => {
    const token = await db.tokens.findOne({
      where: { access_token: accessToken },
      include: ['user', 'client']
    });
    
    return token ? {
      accessToken: token.access_token,
      accessTokenExpiresAt: token.access_token_expires_at,
      user: token.user,
      client: token.client
    } : null;
  },
  
  verifyScope: async (token, scope) => {
    // Verify token has required scope
    return token.scope.includes(scope);
  }
};

const oauth = new OAuth2Server({ model });

// Authorization endpoint
app.get('/oauth/authorize', async (req, res) => {
  const request = new Request(req);
  const response = new Response(res);
  
  try {
    const code = await oauth.authorize(request, response);
    res.json({ code });
  } catch (err) {
    res.status(err.code || 500).json(err);
  }
});

// Token endpoint
app.post('/oauth/token', async (req, res) => {
  const request = new Request(req);
  const response = new Response(res);
  
  try {
    const token = await oauth.token(request, response);
    res.json(token);
  } catch (err) {
    res.status(err.code || 500).json(err);
  }
});

// Protected endpoint
app.get('/api/v1/users', async (req, res) => {
  const request = new Request(req);
  const response = new Response(res);
  
  try {
    const token = await oauth.authenticate(request, response);
    // Token is valid, proceed
    res.json({ user: token.user });
  } catch (err) {
    res.status(err.code || 401).json({ error: 'Unauthorized' });
  }
});
```

#### JWT (JSON Web Tokens)

Self-contained tokens that carry information about the user.

**JWT Structure:**

```
Header.Payload.Signature

eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.
eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.
SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c
```

**Decoded JWT:**

```json
// Header
{
  "alg": "HS256",
  "typ": "JWT"
}

// Payload
{
  "sub": "1234567890",
  "name": "John Doe",
  "email": "john@example.com",
  "role": "admin",
  "iat": 1516239022,
  "exp": 1516242622
}

// Signature (HMAC SHA256)
HMACSHA256(
  base64UrlEncode(header) + "." + base64UrlEncode(payload),
  secret
)
```

**JWT Implementation:**

```javascript
const jwt = require('jsonwebtoken');
const bcrypt = require('bcrypt');

const JWT_SECRET = process.env.JWT_SECRET;
const JWT_EXPIRY = '1h';
const REFRESH_TOKEN_EXPIRY = '7d';

// Login endpoint
app.post('/api/auth/login', async (req, res) => {
  const { email, password } = req.body;
  
  // Find user
  const user = await db.users.findOne({ where: { email } });
  
  if (!user) {
    return res.status(401).json({
      error: { message: 'Invalid credentials' }
    });
  }
  
  // Verify password
  const validPassword = await bcrypt.compare(password, user.password_hash);
  
  if (!validPassword) {
    return res.status(401).json({
      error: { message: 'Invalid credentials' }
    });
  }
  
  // Generate access token
  const accessToken = jwt.sign(
    {
      sub: user.id,
      email: user.email,
      role: user.role,
      type: 'access'
    },
    JWT_SECRET,
    { expiresIn: JWT_EXPIRY }
  );
  
  // Generate refresh token
  const refreshToken = jwt.sign(
    {
      sub: user.id,
      type: 'refresh'
    },
    JWT_SECRET,
    { expiresIn: REFRESH_TOKEN_EXPIRY }
  );
  
  // Store refresh token (optional, for revocation)
  await db.refresh_tokens.create({
    token: refreshToken,
    user_id: user.id,
    expires_at: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000)
  });
  
  res.json({
    access_token: accessToken,
    refresh_token: refreshToken,
    token_type: 'Bearer',
    expires_in: 3600
  });
});

// JWT authentication middleware
const authenticateJWT = async (req, res, next) => {
  const authHeader = req.headers.authorization;
  
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(401).json({
      error: { message: 'Access token is required' }
    });
  }
  
  const token = authHeader.substring(7);
  
  try {
    const decoded = jwt.verify(token, JWT_SECRET);
    
    // Check token type
    if (decoded.type !== 'access') {
      return res.status(401).json({
        error: { message: 'Invalid token type' }
      });
    }
    
    // Attach user info to request
    req.user = {
      id: decoded.sub,
      email: decoded.email,
      role: decoded.role
    };
    
    next();
  } catch (err) {
    if (err.name === 'TokenExpiredError') {
      return res.status(401).json({
        error: { 
          message: 'Access token expired',
          code: 'TOKEN_EXPIRED'
        }
      });
    }
    
    return res.status(401).json({
      error: { message: 'Invalid access token' }
    });
  }
};

// Refresh token endpoint
app.post('/api/auth/refresh', async (req, res) => {
  const { refresh_token } = req.body;
  
  if (!refresh_token) {
    return res.status(400).json({
      error: { message: 'Refresh token is required' }
    });
  }
  
  try {
    const decoded = jwt.verify(refresh_token, JWT_SECRET);
    
    if (decoded.type !== 'refresh') {
      return res.status(401).json({
        error: { message: 'Invalid token type' }
      });
    }
    
    // Check if refresh token exists in database (optional)
    const storedToken = await db.refresh_tokens.findOne({
      where: { token: refresh_token, user_id: decoded.sub }
    });
    
    if (!storedToken || storedToken.revoked) {
      return res.status(401).json({
        error: { message: 'Refresh token revoked or invalid' }
      });
    }
    
    // Generate new access token
    const user = await db.users.findByPk(decoded.sub);
    
    const newAccessToken = jwt.sign(
      {
        sub: user.id,
        email: user.email,
        role: user.role,
        type: 'access'
      },
      JWT_SECRET,
      { expiresIn: JWT_EXPIRY }
    );
    
    res.json({
      access_token: newAccessToken,
      token_type: 'Bearer',
      expires_in: 3600
    });
  } catch (err) {
    return res.status(401).json({
      error: { message: 'Invalid refresh token' }
    });
  }
});

// Protected routes
app.use('/api', authenticateJWT);

app.get('/api/v1/profile', (req, res) => {
  res.json({
    user: req.user
  });
});
```

**JWT Best Practices:**
- Keep tokens short-lived (15 min - 1 hour)
- Use refresh tokens for long-term access
- Store JWT secret securely
- Never store sensitive data in payload
- Implement token revocation mechanism
- Use HTTPS only
- Validate all claims (exp, iat, iss, aud)
- Consider using asymmetric algorithms (RS256) for multi-service architectures

### Authorization

#### Role-Based Access Control (RBAC)

```javascript
// Define roles and permissions
const ROLES = {
  admin: ['users:read', 'users:write', 'users:delete', 'posts:*'],
  editor: ['posts:read', 'posts:write', 'users:read'],
  user: ['posts:read', 'users:read:own']
};

// Authorization middleware
const authorize = (requiredPermission) => {
  return (req, res, next) => {
    const userRole = req.user.role;
    const permissions = ROLES[userRole] || [];
    
    const hasPermission = permissions.some(permission => {
      // Exact match
      if (permission === requiredPermission) return true;
      
      // Wildcard match
      const [resource, action] = permission.split(':');
      const [reqResource, reqAction] = requiredPermission.split(':');
      
      if (resource === reqResource && action === '*') return true;
      if (resource === '*' && action === reqAction) return true;
      if (permission === '*') return true;
      
      return false;
    });
    
    if (!hasPermission) {
      return res.status(403).json({
        error: {
          message: 'Insufficient permissions',
          required: requiredPermission
        }
      });
    }
    
    next();
  };
};

// Usage
app.get('/api/v1/users', 
  authenticateJWT,
  authorize('users:read'),
  (req, res) => {
    // Handler
  }
);

app.delete('/api/v1/users/:id',
  authenticateJWT,
  authorize('users:delete'),
  (req, res) => {
    // Handler
  }
);
```

#### Attribute-Based Access Control (ABAC)

```javascript
// More granular control based on attributes
const checkPolicy = async (user, resource, action, context) => {
  // Example policies
  const policies = [
    {
      effect: 'allow',
      conditions: {
        user: { role: 'admin' },
        action: '*'
      }
    },
    {
      effect: 'allow',
      conditions: {
        user: { role: 'editor' },
        resource: { type: 'post' },
        action: ['read', 'write']
      }
    },
    {
      effect: 'allow',
      conditions: {
        user: { role: 'user' },
        resource: { type: 'post', owner_id: user.id },
        action: ['read', 'write']
      }
    },
    {
      effect: 'deny',
      conditions: {
        resource: { status: 'archived' },
        action: 'write'
      }
    }
  ];
  
  // Evaluate policies
  for (const policy of policies) {
    const matches = evaluateConditions(policy.conditions, { user, resource, action, context });
    
    if (matches) {
      return policy.effect === 'allow';
    }
  }
  
  return false; // Default deny
};

const evaluateConditions = (conditions, data) => {
  // Check user conditions
  if (conditions.user) {
    for (const [key, value] of Object.entries(conditions.user)) {
      if (data.user[key] !== value) return false;
    }
  }
  
  // Check resource conditions
  if (conditions.resource) {
    for (const [key, value] of Object.entries(conditions.resource)) {
      if (data.resource[key] !== value) return false;
    }
  }
  
  // Check action
  if (conditions.action) {
    if (Array.isArray(conditions.action)) {
      if (!conditions.action.includes(data.action)) return false;
    } else if (conditions.action !== '*' && conditions.action !== data.action) {
      return false;
    }
  }
  
  return true;
};

// Middleware
const authorizeABAC = (resourceType, action) => {
  return async (req, res, next) => {
    let resource = { type: resourceType };
    
    // Fetch resource if ID is provided
    if (req.params.id) {
      const fetchedResource = await db[resourceType].findByPk(req.params.id);
      if (!fetchedResource) {
        return res.status(404).json({
          error: { message: `${resourceType} not found` }
        });
      }
      resource = { ...resource, ...fetchedResource.toJSON() };
    }
    
    const allowed = await checkPolicy(
      req.user,
      resource,
      action,
      { ip: req.ip, timestamp: new Date() }
    );
    
    if (!allowed) {
      return res.status(403).json({
        error: { message: 'Access denied' }
      });
    }
    
    req.resource = resource;
    next();
  };
};

// Usage
app.put('/api/v1/posts/:id',
  authenticateJWT,
  authorizeABAC('post', 'write'),
  async (req, res) => {
    // User can only edit if they own the post or are admin
    const post = await db.posts.update(req.body, {
      where: { id: req.params.id }
    });
    res.json({ data: post });
  }
);
```

### API Security Best Practices

#### Input Validation

```javascript
const { body, param, query, validationResult } = require('express-validator');

// Validation middleware
const validateRequest = (req, res, next) => {
  const errors = validationResult(req);
  
  if (!errors.isEmpty()) {
    return res.status(400).json({
      error: {
        message: 'Validation failed',
        details: errors.array().map(err => ({
          field: err.param,
          message: err.msg,
          value: err.value
        }))
      }
    });
  }
  
  next();
};

// Example endpoint with validation
app.post('/api/v1/users',
  [
    body('email')
      .isEmail()
      .normalizeEmail()
      .withMessage('Valid email is required'),
    body('password')
      .isLength({ min: 8 })
      .matches(/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])/)
      .withMessage('Password must be at least 8 characters with uppercase, lowercase, number, and special character'),
    body('name')
      .trim()
      .isLength({ min: 2, max: 100 })
      .withMessage('Name must be between 2 and 100 characters'),
    body('age')
      .optional()
      .isInt({ min: 0, max: 150 })
      .withMessage('Age must be between 0 and 150')
  ],
  validateRequest,
  async (req, res) => {
    // Create user
    const user = await db.users.create(req.body);
    res.status(201).json({ data: user });
  }
);

// Sanitization
const sanitizeInput = (input) => {
  if (typeof input === 'string') {
    // Remove HTML tags
    input = input.replace(/<[^>]*>/g, '');
    // Remove SQL injection patterns
    input = input.replace(/(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC)\b)/gi, '');
  }
  return input;
};
```

#### Rate Limiting

```javascript
const rateLimit = require('express-rate-limit');
const RedisStore = require('rate-limit-redis');
const redis = require('redis');

const redisClient = redis.createClient({
  host: 'localhost',
  port: 6379
});

// Global rate limiter
const globalLimiter = rateLimit({
  store: new RedisStore({
    client: redisClient,
    prefix: 'rl:global:'
  }),
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // 100 requests per window
  message: {
    error: {
      message: 'Too many requests, please try again later',
      retry_after: 900
    }
  },
  standardHeaders: true,
  legacyHeaders: false,
  handler: (req, res) => {
    res.status(429).json({
      error: {
        message: 'Rate limit exceeded',
        retry_after: Math.ceil(req.rateLimit.resetTime / 1000)
      }
    });
  }
});

// Endpoint-specific rate limiter
const authLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 5, // 5 login attempts per 15 minutes
  skipSuccessfulRequests: true
});

// API key-based rate limiter
const apiKeyLimiter = (req, res, next) => {
  const tier = req.client?.tier || 'free';
  
  const limits = {
    free: { windowMs: 60 * 1000, max: 10 }, // 10 per minute
    premium: { windowMs: 60 * 1000, max: 100 }, // 100 per minute
    enterprise: { windowMs: 60 * 1000, max: 1000 } // 1000 per minute
  };
  
  const limiter = rateLimit({
    ...limits[tier],
    keyGenerator: (req) => req.apiKey,
    store: new RedisStore({
      client: redisClient,
      prefix: `rl:apikey:${tier}:`
    })
  });
  
  return limiter(req, res, next);
};

// Apply rate limiters
app.use('/api', globalLimiter);
app.post('/api/auth/login', authLimiter);
app.use('/api', authenticateApiKey, apiKeyLimiter);
```

#### CORS Configuration

```javascript
const cors = require('cors');

// Simple CORS
app.use(cors({
  origin: 'https://example.com',
  methods: ['GET', 'POST', 'PUT', 'DELETE'],
  allowedHeaders: ['Content-Type', 'Authorization'],
  credentials: true,
  maxAge: 86400 // 24 hours
}));

// Dynamic CORS
const whitelist = [
  'https://example.com',
  'https://app.example.com',
  'https://admin.example.com'
];

const corsOptions = {
  origin: (origin, callback) => {
    if (!origin || whitelist.includes(origin)) {
      callback(null, true);
    } else {
      callback(new Error('Not allowed by CORS'));
    }
  },
  credentials: true,
  optionsSuccessStatus: 200
};

app.use(cors(corsOptions));

// Preflight handling
app.options('*', cors());
```

#### Security Headers

```javascript
const helmet = require('helmet');

// Basic security headers
app.use(helmet());

// Detailed configuration
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      styleSrc: ["'self'", "'unsafe-inline'"],
      scriptSrc: ["'self'"],
      imgSrc: ["'self'", 'data:', 'https:'],
    }
  },
  hsts: {
    maxAge: 31536000,
    includeSubDomains: true,
    preload: true
  },
  frameguard: {
    action: 'deny'
  },
  noSniff: true,
  xssFilter: true
}));

// Custom security headers
app.use((req, res, next) => {
  res.setHeader('X-API-Version', '1.0');
  res.setHeader('X-Request-ID', req.id);
  res.removeHeader('X-Powered-By');
  next();
});
```

#### HTTPS Enforcement

```javascript
// Redirect HTTP to HTTPS
app.use((req, res, next) => {
  if (req.header('x-forwarded-proto') !== 'https' && process.env.NODE_ENV === 'production') {
    res.redirect(301, `https://${req.header('host')}${req.url}`);
  } else {
    next();
  }
});

// Strict Transport Security
app.use((req, res, next) => {
  res.setHeader('Strict-Transport-Security', 'max-age=31536000; includeSubDomains; preload');
  next();
});
```

#### SQL Injection Prevention

```javascript
// Use parameterized queries (Sequelize example)
const users = await db.users.findAll({
  where: {
    email: req.query.email // Safe - parameterized
  }
});

// NEVER do this:
// const users = await db.query(`SELECT * FROM users WHERE email = '${req.query.email}'`);

// For raw queries, use replacements
const users = await db.query(
  'SELECT * FROM users WHERE email = :email AND status = :status',
  {
    replacements: { email: req.query.email, status: 'active' },
    type: QueryTypes.SELECT
  }
);
```

#### Encryption and Hashing

```javascript
const bcrypt = require('bcrypt');
const crypto = require('crypto');

// Password hashing
const hashPassword = async (password) => {
  const salt = await bcrypt.genSalt(12);
  return bcrypt.hash(password, salt);
};

const verifyPassword = async (password, hash) => {
  return bcrypt.compare(password, hash);
};

// Data encryption (AES-256)
const ENCRYPTION_KEY = process.env.ENCRYPTION_KEY; // 32 bytes
const IV_LENGTH = 16;

const encrypt = (text) => {
  const iv = crypto.randomBytes(IV_LENGTH);
  const cipher = crypto.createCipheriv('aes-256-cbc', Buffer.from(ENCRYPTION_KEY), iv);
  let encrypted = cipher.update(text);
  encrypted = Buffer.concat([encrypted, cipher.final()]);
  return iv.toString('hex') + ':' + encrypted.toString('hex');
};

const decrypt = (text) => {
  const parts = text.split(':');
  const iv = Buffer.from(parts.shift(), 'hex');
  const encrypted = Buffer.from(parts.join(':'), 'hex');
  const decipher = crypto.createDecipheriv('aes-256-cbc', Buffer.from(ENCRYPTION_KEY), iv);
  let decrypted = decipher.update(encrypted);
  decrypted = Buffer.concat([decrypted, decipher.final()]);
  return decrypted.toString();
};

// Usage for sensitive data
app.post('/api/v1/payment-methods', async (req, res) => {
  const encryptedCardNumber = encrypt(req.body.card_number);
  
  const paymentMethod = await db.payment_methods.create({
    user_id: req.user.id,
    card_number: encryptedCardNumber,
    card_type: req.body.card_type
  });
  
  res.status(201).json({ data: paymentMethod });
});
```

## API Documentation

### OpenAPI (Swagger) Specification

```yaml
openapi: 3.0.3
info:
  title: User Management API
  description: |
    Comprehensive API for managing users, authentication, and profiles.
    
    ## Authentication
    This API uses Bearer token authentication. Include the token in the Authorization header:
    ```
    Authorization: Bearer YOUR_TOKEN_HERE
    ```
    
    ## Rate Limiting
    - Free tier: 10 requests per minute
    - Premium tier: 100 requests per minute
    - Enterprise tier: 1000 requests per minute
  version: 1.0.0
  contact:
    name: API Support
    email: api-support@example.com
    url: https://example.com/support
  license:
    name: Apache 2.0
    url: https://www.apache.org/licenses/LICENSE-2.0.html

servers:
  - url: https://api.example.com/v1
    description: Production server
  - url: https://staging-api.example.com/v1
    description: Staging server
  - url: http://localhost:3000/api/v1
    description: Development server

tags:
  - name: Authentication
    description: Authentication and authorization operations
  - name: Users
    description: User management operations
  - name: Posts
    description: Blog post operations

paths:
  /auth/login:
    post:
      tags:
        - Authentication
      summary: User login
      description: Authenticate user and receive access token
      operationId: login
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required:
                - email
                - password
              properties:
                email:
                  type: string
