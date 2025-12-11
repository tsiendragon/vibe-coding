# API Specification Document

## API Overview
- **Service Name**: [Service Name]
- **Version**: v1.0.0
- **Base URL**: `https://api.example.com/v1`
- **Protocol**: HTTPS
- **Authentication**: JWT Bearer Token

## Authentication

### Login
```http
POST /auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "password123"
}
```

**Response**:
```json
{
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### Refresh Token
```http
POST /auth/refresh
Authorization: Bearer {refresh_token}
```

## API Endpoints

### User Management

#### Get User Profile
```http
GET /users/me
Authorization: Bearer {access_token}
```

**Response**:
```json
{
  "id": "uuid",
  "email": "user@example.com",
  "name": "John Doe",
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:00:00Z"
}
```

#### Update User Profile
```http
PUT /users/me
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "name": "Jane Doe",
  "phone": "+1234567890"
}
```

### Resource Operations

#### List Resources
```http
GET /resources?page=1&limit=20&sort=created_at&order=desc
Authorization: Bearer {access_token}
```

**Query Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| page | integer | No | Page number (default: 1) |
| limit | integer | No | Items per page (default: 20, max: 100) |
| sort | string | No | Sort field |
| order | string | No | Sort order (asc/desc) |
| search | string | No | Search query |

**Response**:
```json
{
  "items": [...],
  "total": 100,
  "page": 1,
  "pages": 5,
  "limit": 20
}
```

#### Create Resource
```http
POST /resources
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "name": "Resource Name",
  "description": "Resource Description",
  "metadata": {}
}
```

#### Get Resource by ID
```http
GET /resources/{id}
Authorization: Bearer {access_token}
```

#### Update Resource
```http
PUT /resources/{id}
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "name": "Updated Name",
  "description": "Updated Description"
}
```

#### Delete Resource
```http
DELETE /resources/{id}
Authorization: Bearer {access_token}
```

## Error Responses

### Error Format
```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable error message",
    "details": {},
    "request_id": "uuid"
  }
}
```

### Common Error Codes
| Code | HTTP Status | Description |
|------|-------------|-------------|
| UNAUTHORIZED | 401 | Missing or invalid authentication |
| FORBIDDEN | 403 | Insufficient permissions |
| NOT_FOUND | 404 | Resource not found |
| VALIDATION_ERROR | 422 | Request validation failed |
| RATE_LIMIT_EXCEEDED | 429 | Too many requests |
| INTERNAL_ERROR | 500 | Internal server error |

## Rate Limiting
- **Default Rate Limit**: 1000 requests per hour
- **Authenticated Rate Limit**: 5000 requests per hour
- **Headers**:
  - `X-RateLimit-Limit`: Maximum requests
  - `X-RateLimit-Remaining`: Remaining requests
  - `X-RateLimit-Reset`: Reset timestamp

## Pagination
All list endpoints support pagination with the following structure:
```json
{
  "items": [...],
  "pagination": {
    "total": 1000,
    "page": 1,
    "pages": 50,
    "limit": 20,
    "has_next": true,
    "has_prev": false
  }
}
```

## Versioning
API versioning is handled through the URL path:
- Current version: `/v1`
- Legacy support: Minimum 6 months deprecation notice

## WebSocket Events

### Connection
```javascript
const ws = new WebSocket('wss://api.example.com/ws');
ws.send(JSON.stringify({
  type: 'auth',
  token: 'bearer_token'
}));
```

### Event Types
| Event | Description | Payload |
|-------|-------------|---------|
| resource.created | New resource created | Resource object |
| resource.updated | Resource updated | Resource object |
| resource.deleted | Resource deleted | { id: "uuid" } |
| notification | System notification | Notification object |

## Security Considerations
- All endpoints require HTTPS
- JWT tokens expire after 1 hour
- Refresh tokens expire after 30 days
- API keys must be kept secure
- CORS is configured for allowed origins only

## Performance SLA
- **Uptime**: 99.9%
- **Response Time**: < 200ms (P95)
- **Throughput**: 10,000 requests/second

## Change Log
| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024-01-01 | Initial release |
| 1.1.0 | 2024-02-01 | Added WebSocket support |
| 1.2.0 | 2024-03-01 | Enhanced filtering options |