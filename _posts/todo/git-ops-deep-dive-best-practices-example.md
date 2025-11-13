## Complete Example: E-Commerce Platform

This comprehensive example demonstrates a real-world e-commerce platform using GitOps best practices with multiple repositories, environments, and services.

### Platform Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        E-Commerce Platform                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Frontend │  │  User    │  │ Product  │  │   Cart   │       │
│  │   App    │  │ Service  │  │ Service  │  │ Service  │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
│                                                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │  Order   │  │ Payment  │  │  Notify  │  │  Search  │       │
│  │ Service  │  │ Service  │  │ Service  │  │ Service  │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │            Platform Services (Ingress, Monitoring)       │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

### Repository Organization

```
github.com/ecommerce-platform/

├── app-frontend                    # React frontend application
├── app-user-service               # User management microservice
├── app-product-catalog            # Product catalog microservice
├── app-cart-service               # Shopping cart microservice
├── app-order-service              # Order processing microservice
├── app-payment-service            # Payment processing microservice
├── app-notification-service       # Notification system
├── app-search-service             # Search engine service
├── infrastructure                  # Terraform infrastructure code
├── gitops-config                  # Kubernetes manifests (GitOps)
└── shared-libraries               # Common code libraries
```

---

### 1. Application Repository: User Service

```
app-user-service/
├── .github/
│   └── workflows/
│       ├── ci.yml                 # Continuous Integration
│       ├── build.yml              # Docker build and push
│       ├── security-scan.yml      # Trivy + Snyk scanning
│       └── release.yml            # Tag-based releases
├── src/
│   ├── main/
│   │   └── java/
│   │       └── com/
│   │           └── ecommerce/
│   │               └── user/
│   │                   ├── controller/
│   │                   │   ├── UserController.java
│   │                   │   ├── AuthController.java
│   │                   │   └── ProfileController.java
│   │                   ├── service/
│   │                   │   ├── UserService.java
│   │                   │   ├── AuthService.java
│   │                   │   └── EmailService.java
│   │                   ├── repository/
│   │                   │   ├── UserRepository.java
│   │                   │   └── SessionRepository.java
│   │                   ├── model/
│   │                   │   ├── User.java
│   │                   │   ├── Session.java
│   │                   │   └── Profile.java
│   │                   ├── dto/
│   │                   │   ├── UserDTO.java
│   │                   │   ├── LoginRequest.java
│   │                   │   └── RegisterRequest.java
│   │                   ├── security/
│   │                   │   ├── JwtTokenProvider.java
│   │                   │   └── SecurityConfig.java
│   │                   ├── exception/
│   │                   │   ├── UserNotFoundException.java
│   │                   │   └── GlobalExceptionHandler.java
│   │                   └── UserServiceApplication.java
│   └── test/
│       └── java/
│           └── com/
│               └── ecommerce/
│                   └── user/
│                       ├── controller/
│                       ├── service/
│                       └── integration/
├── src/main/resources/
│   ├── application.yml
│   ├── application-dev.yml
│   ├── application-prod.yml
│   └── db/migration/              # Flyway migrations
│       ├── V1__Create_users_table.sql
│       ├── V2__Create_sessions_table.sql
│       └── V3__Add_email_verification.sql
├── Dockerfile
├── .dockerignore
├── pom.xml                        # Maven configuration
├── README.md
├── CHANGELOG.md
└── docs/
    ├── api.md                     # API documentation
    └── architecture.md            # Service architecture
```

**Dockerfile**:

```dockerfile
# Build stage
FROM maven:3.9-eclipse-temurin-17 AS build
WORKDIR /app
COPY pom.xml .
RUN mvn dependency:go-offline
COPY src ./src
RUN mvn clean package -DskipTests

# Runtime stage
FROM eclipse-temurin:17-jre-alpine
WORKDIR /app

# Create non-root user
RUN addgroup -g 1001 appuser && \
    adduser -u 1001 -G appuser -s /bin/sh -D appuser

# Copy JAR
COPY --from=build /app/target/*.jar app.jar

# Change ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:8080/actuator/health || exit 1

EXPOSE 8080

ENTRYPOINT ["java", "-XX:+UseContainerSupport", "-XX:MaxRAMPercentage=75.0", "-jar", "app.jar"]
```

**CI/CD Workflow**:

```yaml
# .github/workflows/ci.yml
name: CI Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ecommerce-platform/user-service

jobs:
  test:
    name: Test & Quality
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
          POSTGRES_DB: user_service_test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up JDK 17
        uses: actions/setup-java@v4
        with:
          java-version: '17'
          distribution: 'temurin'
          cache: maven
      
      - name: Run tests
        run: mvn test
        env:
          SPRING_PROFILES_ACTIVE: test
          DB_URL: jdbc:postgresql://localhost:5432/user_service_test
      
      - name: Generate coverage report
        run: mvn jacoco:report
      
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          files: ./target/site/jacoco/jacoco.xml
      
      - name: SonarQube Scan
        run: |
          mvn sonar:sonar \
            -Dsonar.projectKey=user-service \
            -Dsonar.host.url=${{ secrets.SONAR_HOST_URL }} \
            -Dsonar.login=${{ secrets.SONAR_TOKEN }}
  
  build:
    name: Build & Push Image
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push'
    permissions:
      contents: read
      packages: write
    
    outputs:
      image-tag: ${{ steps.meta.outputs.tags }}
      image-digest: ${{ steps.build.outputs.digest }}
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Log in to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha,prefix={{branch}}-
            type=raw,value=latest,enable={{is_default_branch}}
      
      - name: Build and push
        id: build
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          provenance: true
          sbom: true
      
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          format: 'sarif'
          output: 'trivy-results.sarif'
      
      - name: Upload Trivy results to GitHub Security
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-results.sarif'
      
      - name: Sign image with Cosign
        run: |
          cosign sign --yes \
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}@${{ steps.build.outputs.digest }}
  
  update-gitops:
    name: Update GitOps Repo
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
      - name: Checkout GitOps repo
        uses: actions/checkout@v4
        with:
          repository: ecommerce-platform/gitops-config
          token: ${{ secrets.GITOPS_PAT }}
          path: gitops
      
      - name: Update image tag
        run: |
          cd gitops/apps/user-service/overlays/dev
          kustomize edit set image user-service=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
      
      - name: Commit and push
        run: |
          cd gitops
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add .
          git commit -m "chore(user-service): update to ${{ github.sha }}"
          git push
```

---

### 2. Infrastructure Repository

```
infrastructure/
├── .github/
│   └── workflows/
│       ├── terraform-plan.yml
│       ├── terraform-apply.yml
│       └── terraform-destroy.yml
├── terraform/
│   ├── modules/
│   │   ├── networking/
│   │   │   ├── vpc/
│   │   │   │   ├── main.tf
│   │   │   │   ├── variables.tf
│   │   │   │   ├── outputs.tf
│   │   │   │   └── README.md
│   │   │   ├── security-groups/
│   │   │   │   ├── main.tf
│   │   │   │   ├── variables.tf
│   │   │   │   └── outputs.tf
│   │   │   └── nat-gateway/
│   │   │       ├── main.tf
│   │   │       ├── variables.tf
│   │   │       └── outputs.tf
│   │   ├── compute/
│   │   │   ├── eks/
│   │   │   │   ├── main.tf
│   │   │   │   ├── variables.tf
│   │   │   │   ├── outputs.tf
│   │   │   │   ├── node-groups.tf
│   │   │   │   ├── irsa.tf
│   │   │   │   └── README.md
│   │   │   └── ec2/
│   │   │       ├── main.tf
│   │   │       ├── variables.tf
│   │   │       └── outputs.tf
│   │   ├── database/
│   │   │   ├── rds/
│   │   │   │   ├── main.tf
│   │   │   │   ├── variables.tf
│   │   │   │   ├── outputs.tf
│   │   │   │   ├── parameter-group.tf
│   │   │   │   └── README.md
│   │   │   ├── elasticache/
│   │   │   │   ├── main.tf
│   │   │   │   ├── variables.tf
│   │   │   │   └── outputs.tf
│   │   │   └── documentdb/
│   │   │       ├── main.tf
│   │   │       ├── variables.tf
│   │   │       └── outputs.tf
│   │   ├── storage/
│   │   │   ├── s3/
│   │   │   │   ├── main.tf
│   │   │   │   ├── variables.tf
│   │   │   │   ├── outputs.tf
│   │   │   │   └── lifecycle-rules.tf
│   │   │   └── efs/
│   │   │       ├── main.tf
│   │   │       ├── variables.tf
│   │   │       └── outputs.tf
│   │   ├── messaging/
│   │   │   ├── sqs/
│   │   │   │   ├── main.tf
│   │   │   │   ├── variables.tf
│   │   │   │   └── outputs.tf
│   │   │   ├── sns/
│   │   │   │   ├── main.tf
│   │   │   │   ├── variables.tf
│   │   │   │   └── outputs.tf
│   │   │   └── eventbridge/
│   │   │       ├── main.tf
│   │   │       ├── variables.tf
│   │   │       └── outputs.tf
│   │   └── monitoring/
│   │       ├── cloudwatch/
│   │       │   ├── main.tf
│   │       │   ├── variables.tf
│   │       │   ├── outputs.tf
│   │       │   └── alarms.tf
│   │       └── xray/
│   │           ├── main.tf
│   │           ├── variables.tf
│   │           └── outputs.tf
│   ├── environments/
│   │   ├── dev/
│   │   │   ├── main.tf
│   │   │   ├── variables.tf
│   │   │   ├── terraform.tfvars
│   │   │   ├── backend.tf
│   │   │   ├── outputs.tf
│   │   │   └── locals.tf
│   │   ├── staging/
│   │   │   ├── main.tf
│   │   │   ├── variables.tf
│   │   │   ├── terraform.tfvars
│   │   │   ├── backend.tf
│   │   │   ├── outputs.tf
│   │   │   └── locals.tf
│   │   └── prod/
│   │       ├── main.tf
│   │       ├── variables.tf
│   │       ├── terraform.tfvars
│   │       ├── backend.tf
│   │       ├── outputs.tf
│   │       └── locals.tf
│   └── global/
│       ├── route53/
│       │   ├── main.tf
│       │   ├── variables.tf
│       │   └── outputs.tf
│       ├── iam/
│       │   ├── main.tf
│       │   ├── roles.tf
│       │   ├── policies.tf
│       │   └── outputs.tf
│       └── cloudfront/
│           ├── main.tf
│           ├── variables.tf
│           └── outputs.tf
├── scripts/
│   ├── init-backend.sh
│   ├── validate-all.sh
│   └── cost-estimate.sh
└── docs/
    ├── architecture/
    │   ├── network-diagram.png
    │   ├── security-model.md
    │   └── disaster-recovery.md
    ├── runbooks/
    │   ├── database-recovery.md
    │   ├── cluster-upgrade.md
    │   └── incident-response.md
    └── README.md
```

**Production Environment Configuration**:

```hcl
# terraform/environments/prod/main.tf
terraform {
  required_version = ">= 1.6.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
  }
  
  backend "s3" {
    bucket         = "ecommerce-terraform-state"
    key            = "prod/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-lock"
  }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Environment = "production"
      ManagedBy   = "Terraform"
      Project     = "ecommerce-platform"
      CostCenter  = "engineering"
    }
  }
}

locals {
  cluster_name = "ecommerce-prod-cluster"
  
  common_tags = {
    Environment = "production"
    Project     = "ecommerce-platform"
  }
}

# VPC and Networking
module "vpc" {
  source = "../../modules/networking/vpc"
  
  name               = "${local.cluster_name}-vpc"
  cidr_block         = "10.0.0.0/16"
  availability_zones = ["us-east-1a", "us-east-1b", "us-east-1c"]
  
  public_subnets  = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  private_subnets = ["10.0.11.0/24", "10.0.12.0/24", "10.0.13.0/24"]
  database_subnets = ["10.0.21.0/24", "10.0.22.0/24", "10.0.23.0/24"]
  
  enable_nat_gateway   = true
  single_nat_gateway   = false
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = local.common_tags
}

# EKS Cluster
module "eks" {
  source = "../../modules/compute/eks"
  
  cluster_name    = local.cluster_name
  cluster_version = "1.28"
  
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
  
  cluster_endpoint_public_access  = true
  cluster_endpoint_private_access = true
  
  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent = true
    }
  }
  
  node_groups = {
    general = {
      name           = "general"
      instance_types = ["t3.large"]
      capacity_type  = "ON_DEMAND"
      
      min_size     = 3
      max_size     = 10
      desired_size = 5
      
      labels = {
        role = "general"
        workload = "applications"
      }
      
      tags = {
        "k8s.io/cluster-autoscaler/enabled" = "true"
        "k8s.io/cluster-autoscaler/${local.cluster_name}" = "owned"
      }
    }
    
    compute_optimized = {
      name           = "compute"
      instance_types = ["c5.xlarge"]
      capacity_type  = "SPOT"
      
      min_size     = 0
      max_size     = 20
      desired_size = 2
      
      labels = {
        role = "compute"
        workload = "batch-jobs"
      }
      
      taints = [{
        key    = "workload"
        value  = "compute"
        effect = "NoSchedule"
      }]
    }
  }
  
  tags = local.common_tags
}

# RDS PostgreSQL for User Service
module "user_service_db" {
  source = "../../modules/database/rds"
  
  identifier = "user-service-prod"
  
  engine         = "postgres"
  engine_version = "15.4"
  instance_class = "db.r6g.xlarge"
  
  allocated_storage     = 100
  max_allocated_storage = 500
  storage_encrypted     = true
  
  db_name  = "userservice"
  username = "admin"
  port     = 5432
  
  vpc_id                    = module.vpc.vpc_id
  db_subnet_group_name      = module.vpc.database_subnet_group
  vpc_security_group_ids    = [module.vpc.database_security_group_id]
  
  multi_az               = true
  backup_retention_period = 30
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  enabled_cloudwatch_logs_exports = ["postgresql", "upgrade"]
  
  performance_insights_enabled = true
  monitoring_interval         = 60
  
  deletion_protection = true
  skip_final_snapshot = false
  
  tags = merge(local.common_tags, {
    Service = "user-service"
  })
}

# ElastiCache Redis for Session Management
module "redis_cache" {
  source = "../../modules/database/elasticache"
  
  cluster_id           = "ecommerce-sessions-prod"
  engine               = "redis"
  engine_version       = "7.0"
  node_type            = "cache.r6g.large"
  num_cache_nodes      = 3
  parameter_group_name = "default.redis7"
  port                 = 6379
  
  subnet_group_name   = module.vpc.elasticache_subnet_group
  security_group_ids  = [module.vpc.cache_security_group_id]
  
  automatic_failover_enabled = true
  multi_az_enabled          = true
  
  snapshot_retention_limit = 7
  snapshot_window          = "03:00-04:00"
  maintenance_window       = "sun:04:00-sun:05:00"
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  
  tags = local.common_tags
}

# S3 Bucket for Product Images
module "product_images_bucket" {
  source = "../../modules/storage/s3"
  
  bucket = "ecommerce-product-images-prod"
  
  versioning_enabled = true
  
  lifecycle_rules = [{
    id      = "transition-to-ia"
    enabled = true
    
    transition = [{
      days          = 90
      storage_class = "STANDARD_IA"
    }]
  }]
  
  server_side_encryption_configuration = {
    rule = {
      apply_server_side_encryption_by_default = {
        sse_algorithm = "AES256"
      }
    }
  }
  
  cors_rule = [{
    allowed_headers = ["*"]
    allowed_methods = ["GET", "HEAD"]
    allowed_origins = ["https://www.ecommerce.com"]
    expose_headers  = ["ETag"]
    max_age_seconds = 3000
  }]
  
  tags = local.common_tags
}

# SQS Queue for Order Processing
module "order_queue" {
  source = "../../modules/messaging/sqs"
  
  name                      = "order-processing-prod"
  visibility_timeout_seconds = 300
  message_retention_seconds = 1209600  # 14 days
  max_message_size          = 262144   # 256 KB
  delay_seconds             = 0
  receive_wait_time_seconds = 20
  
  redrive_policy = {
    deadLetterTargetArn = module.order_dlq.queue_arn
    maxReceiveCount     = 3
  }
  
  tags = local.common_tags
}

# Dead Letter Queue
module "order_dlq" {
  source = "../../modules/messaging/sqs"
  
  name                      = "order-processing-dlq-prod"
  message_retention_seconds = 1209600
  
  tags = merge(local.common_tags, {
    Queue = "dead-letter"
  })
}

# CloudWatch Alarms
module "alarms" {
  source = "../../modules/monitoring/cloudwatch"
  
  cluster_name = local.cluster_name
  
  alarms = {
    high_cpu = {
      comparison_operator = "GreaterThanThreshold"
      evaluation_periods  = 2
      metric_name        = "CPUUtilization"
      namespace          = "AWS/EKS"
      period             = 300
      statistic          = "Average"
      threshold          = 80
      alarm_description  = "EKS cluster CPU utilization is too high"
    }
    
    database_connections = {
      comparison_operator = "GreaterThanThreshold"
      evaluation_periods  = 1
      metric_name        = "DatabaseConnections"
      namespace          = "AWS/RDS"
      period             = 300
      statistic          = "Average"
      threshold          = 80
      alarm_description  = "RDS connection count is high"
    }
  }
  
  sns_topic_arn = module.sns_alerts.topic_arn
  
  tags = local.common_tags
}

# Outputs
output "cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
}

output "cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "database_endpoint" {
  description = "RDS database endpoint"
  value       = module.user_service_db.db_instance_endpoint
  sensitive   = true
}

output "redis_endpoint" {
  description = "ElastiCache Redis endpoint"
  value       = module.redis_cache.cache_nodes
  sensitive   = true
}

output "s3_bucket_name" {
  description = "S3 bucket for product images"
  value       = module.product_images_bucket.bucket_id
}
```

---

### 3. GitOps Configuration Repository

```
gitops-config/
├── .github/
│   └── workflows/
│       ├── validate-manifests.yml
│       ├── policy-check.yml
│       └── sync-status.yml
├── clusters/
│   ├── dev-cluster/
│   │   ├── flux-system/
│   │   │   ├── gotk-components.yaml
│   │   │   ├── gotk-sync.yaml
│   │   │   └── kustomization.yaml
│   │   ├── infrastructure.yaml
│   │   └── applications.yaml
│   ├── staging-cluster/
│   │   ├── flux-system/
│   │   ├── infrastructure.yaml
│   │   └── applications.yaml
│   └── prod-cluster/
│       ├── flux-system/
│       ├── infrastructure.yaml
│       └── applications.yaml
├── infrastructure/
│   ├── base/
│   │   ├── namespaces/
│   │   │   ├── kustomization.yaml
│   │   │   ├── monitoring.yaml
│   │   │   ├── ingress.yaml
│   │   │   ├── cert-manager.yaml
│   │   │   └── external-secrets.yaml
│   │   ├── ingress-nginx/
│   │   │   ├── kustomization.yaml
│   │   │   ├── helmrepository.yaml
│   │   │   └── helmrelease.yaml
│   │   ├── cert-manager/
│   │   │   ├── kustomization.yaml
│   │   │   ├── helmrepository.yaml
│   │   │   ├── helmrelease.yaml
│   │   │   └── clusterissuer.yaml
│   │   ├── external-secrets/
│   │   │   ├── kustomization.yaml
│   │   │   ├── helmrepository.yaml
│   │   │   ├── helmrelease.yaml
│   │   │   └── secretstore.yaml
│   │   ├── prometheus/
│   │   │   ├── kustomization.yaml
│   │   │   ├── helmrepository.yaml
│   │   │   └── helmrelease.yaml
│   │   └── grafana/
│   │       ├── kustomization.yaml
│   │       ├── helmrepository.yaml
│   │       └── helmrelease.yaml
│   └── overlays/
│       ├── dev/
│       │   └── kustomization.yaml
│       ├── staging/
│       │   └── kustomization.yaml
│       └── prod/
│           ├── kustomization.yaml
│           └── ingress-patch.yaml
├── apps/
│   ├── frontend/
│   │   ├── base/
│   │   │   ├── kustomization.yaml
│   │   │   ├── namespace.yaml
│   │   │   ├── deployment.yaml
│   │   │   ├── service.yaml
│   │   │   ├── ingress.yaml
│   │   │   ├── configmap.yaml
│   │   │   └── hpa.yaml
│   │   └── overlays/
│   │       ├── dev/
│   │       │   ├── kustomization.yaml
│   │       │   ├── patch-replicas.yaml
│   │       │   └── configmap-patch.yaml
│   │       ├── staging/
│   │       │   ├── kustomization.yaml
│   │       │   ├── patch-replicas.yaml
│   │       │   └── configmap-patch.yaml
│   │       └── prod/
│   │           ├── kustomization.yaml
│   │           ├── patch-replicas.yaml
│   │           ├── patch-resources.yaml
│   │           ├── configmap-patch.yaml
│   │           └── pdb.yaml
│   ├── user-service/
│   │   ├── base/
│   │   │   ├── kustomization.yaml
│   │   │   ├── namespace.yaml
│   │   │   ├── deployment.yaml
│   │   │   ├── service.yaml
│   │   │   ├── servicemonitor.yaml
│   │   │   ├── configmap.yaml
│   │   │   └── externalsecret.yaml
│   │   └── overlays/
│   │       ├── dev/
│   │       ├── staging/
│   │       └── prod/
│   ├── product-catalog/
│   │   ├── base/
│   │   └── overlays/
│   ├── cart-service/
│   │   ├── base/
│   │   └── overlays/
│   ├── order-service/
│   │   ├── base/
│   │   └── overlays/
│   ├── payment-service/
│   │   ├── base/
│   │   └── overlays/
│   ├── notification-service/
│   │   ├── base/
│   │   └── overlays/
│   └── search-service/
│       ├── base/
│       └── overlays/
├── policies/
│   ├── kustomization.yaml
│   ├── require-pod-security.yaml
│   ├── require-resource-limits.yaml
│   ├── disallow-latest-tag.yaml
│   └── require-probes.yaml
└── docs/
    ├── deployment-guide.md
    ├── rollback-procedures.md
    └── troubleshooting.md
```

---

### Complete User Service Kubernetes Manifests

**Base Deployment**:

```yaml
# apps/user-service/base/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: user-service
  labels:
    app.kubernetes.io/name: user-service
    environment: production
```

```yaml
# apps/user-service/base/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: user-service
  namespace: user-service
  labels:
    app: user-service
    version: v1
spec:
  replicas: 3
  revisionHistoryLimit: 10
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: user-service
  template:
    metadata:
      labels:
        app: user-service
        version: v1
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8080"
        prometheus.io/path: "/actuator/prometheus"
    spec:
      serviceAccountName: user-service
      securityContext:
        runAsNonRoot: true
        runAsUser: 1001
        fsGroup: 2000
        seccompProfile:
          type: RuntimeDefault
      
      initContainers:
      - name: wait-for-db
        image: busybox:1.36
        command:
        - sh
        - -c
        - |
          until nc -z -v -w30 $DB_HOST $DB_PORT; do
            echo "Waiting for database connection..."
            sleep 2
          done
        env:
        - name: DB_HOST
          valueFrom:
            configMapKeyRef:
              name: user-service-config
              key: DB_HOST
        - name: DB_PORT
          valueFrom:
            configMapKeyRef:
              name: user-service-config
              key: DB_PORT
      
      containers:
      - name: user-service
        image: ghcr.io/ecommerce-platform/user-service:latest
        imagePullPolicy: IfNotPresent
        ports:
        - name: http
          containerPort: 8080
          protocol: TCP
        - name: management
          containerPort: 8081
          protocol: TCP
        
        env:
        - name: SPRING_PROFILES_ACTIVE
          value: "production"
        - name: JAVA_OPTS
          value: "-XX:+UseContainerSupport -XX:MaxRAMPercentage=75.0 -XX:+UseG1GC"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: user-service-db-credentials
              key: url
        - name: DATABASE_USERNAME
          valueFrom:
            secretKeyRef:
              name: user-service-db-credentials
              key: username
        - name: DATABASE_PASSWORD
          valueFrom:
            secretKeyRef:
              name: user-service-db-credentials
              key: password
        - name: REDIS_HOST
          valueFrom:
            configMapKeyRef:
              name: user-service-config
              key: REDIS_HOST
        - name: REDIS_PORT
          valueFrom:
            configMapKeyRef:
              name: user-service-config
              key: REDIS_PORT
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: user-service-jwt
              key: secret
        
        envFrom:
        - configMapRef:
            name: user-service-config
        
        resources:
          requests:
            cpu: 200m
            memory: 512Mi
          limits:
            cpu: 1000m
            memory: 1Gi
        
        livenessProbe:
          httpGet:
            path: /actuator/health/liveness
            port: management
            scheme: HTTP
          initialDelaySeconds: 60
          periodSeconds: 10
          timeoutSeconds: 5
          successThreshold: 1
          failureThreshold: 3
        
        readinessProbe:
          httpGet:
            path: /actuator/health/readiness
            port: management
            scheme: HTTP
          initialDelaySeconds: 30
          periodSeconds: 5
          timeoutSeconds: 3
          successThreshold: 1
          failureThreshold: 3
        
        startupProbe:
          httpGet:
            path: /actuator/health/liveness
            port: management
          initialDelaySeconds: 0
          periodSeconds: 10
          timeoutSeconds: 3
          successThreshold: 1
          failureThreshold: 30
        
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          runAsNonRoot: true
          runAsUser: 1001
          capabilities:
            drop:
            - ALL
        
        volumeMounts:
        - name: tmp
          mountPath: /tmp
        - name: logs
          mountPath: /app/logs
      
      volumes:
      - name: tmp
        emptyDir: {}
      - name: logs
        emptyDir: {}
      
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - user-service
              topologyKey: kubernetes.io/hostname
      
      topologySpreadConstraints:
      - maxSkew: 1
        topologyKey: topology.kubernetes.io/zone
        whenUnsatisfiable: ScheduleAnyway
        labelSelector:
          matchLabels:
            app: user-service
```

```yaml
# apps/user-service/base/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: user-service
  namespace: user-service
  labels:
    app: user-service
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
spec:
  type: ClusterIP
  ports:
  - name: http
    port: 80
    targetPort: http
    protocol: TCP
  - name: management
    port: 8081
    targetPort: management
    protocol: TCP
  selector:
    app: user-service
  sessionAffinity: None
```

```yaml
# apps/user-service/base/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: user-service-config
  namespace: user-service
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  DB_HOST: "user-service-db.rds.amazonaws.com"
  DB_PORT: "5432"
  DB_NAME: "userservice"
  REDIS_HOST: "ecommerce-sessions.cache.amazonaws.com"
  REDIS_PORT: "6379"
  CACHE_TTL: "3600"
  SESSION_TIMEOUT: "1800"
  MAX_LOGIN_ATTEMPTS: "5"
  TOKEN_EXPIRATION: "86400"
```

```yaml
# apps/user-service/base/externalsecret.yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: user-service-db-credentials
  namespace: user-service
spec:
  secretStoreRef:
    name: aws-secrets-manager
    kind: SecretStore
  target:
    name: user-service-db-credentials
    creationPolicy: Owner
  refreshInterval: 1h
  data:
  - secretKey: url
    remoteRef:
      key: prod/user-service/database
      property: url
  - secretKey: username
    remoteRef:
      key: prod/user-service/database
      property: username
  - secretKey: password
    remoteRef:
      key: prod/user-service/database
      property: password

---
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: user-service-jwt
  namespace: user-service
spec:
  secretStoreRef:
    name: aws-secrets-manager
    kind: SecretStore
  target:
    name: user-service-jwt
    creationPolicy: Owner
  data:
  - secretKey: secret
    remoteRef:
      key: prod/user-service/jwt
      property: secret
```

```yaml
# apps/user-service/base/servicemonitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: user-service
  namespace: user-service
  labels:
    app: user-service
spec:
  selector:
    matchLabels:
      app: user-service
  endpoints:
  - port: management
    path: /actuator/prometheus
    interval: 30s
    scrapeTimeout: 10s
```

```yaml
# apps/user-service/base/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

namespace: user-service

resources:
  - namespace.yaml
  - deployment.yaml
  - service.yaml
  - configmap.yaml
  - externalsecret.yaml
  - servicemonitor.yaml

commonLabels:
  app.kubernetes.io/name: user-service
  app.kubernetes.io/component: backend
  app.kubernetes.io/managed-by: flux

images:
  - name: ghcr.io/ecommerce-platform/user-service
    newTag: latest
```

---

**Production Overlay**:

```yaml
# apps/user-service/overlays/prod/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

namespace: user-service

bases:
  - ../../base

patches:
  - path: patch-replicas.yaml
  - path: patch-resources.yaml

resources:
  - hpa.yaml
  - pdb.yaml
  - networkpolicy.yaml

images:
  - name: ghcr.io/ecommerce-platform/user-service
    newTag: v1.2.3

configMapGenerator:
  - name: user-service-config
    behavior: merge
    literals:
      - ENVIRONMENT=production
      - LOG_LEVEL=WARN
      - CACHE_TTL=7200
      - MAX_CONNECTIONS=100

labels:
  - pairs:
      environment: production
      tier: backend
```

```yaml
# apps/user-service/overlays/prod/patch-replicas.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: user-service
spec:
  replicas: 5
```

```yaml
# apps/user-service/overlays/prod/patch-resources.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: user-service
spec:
  template:
    spec:
      containers:
      - name: user-service
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 2000m
            memory: 2Gi
```

```yaml
# apps/user-service/overlays/prod/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: user-service
  namespace: user-service
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: user-service
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "1000"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
      - type: Pods
        value: 2
        periodSeconds: 60
      selectPolicy: Min
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
      - type: Pods
        value: 4
        periodSeconds: 30
      selectPolicy: Max
```

```yaml
# apps/user-service/overlays/prod/pdb.yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: user-service
  namespace: user-service
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: user-service
```

```yaml
# apps/user-service/overlays/prod/networkpolicy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: user-service
  namespace: user-service
spec:
  podSelector:
    matchLabels:
      app: user-service
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8080
  - from:
    - namespaceSelector:
        matchLabels:
          name: monitoring
    ports:
    - protocol: TCP
      port: 8081
  egress:
  - to:
    - namespaceSelector: {}
    ports:
    - protocol: TCP
      port: 53
    - protocol: UDP
      port: 53
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
  - to:
    - namespaceSelector: {}
    ports:
    - protocol: TCP
      port: 443
```

---

### Cluster Bootstrap Configuration

```yaml
# clusters/prod-cluster/infrastructure.yaml
apiVersion: kustomize.toolkit.fluxcd.io/v1
kind: Kustomization
metadata:
  name: infrastructure
  namespace: flux-system
spec:
  interval: 10m
  retryInterval: 1m
  timeout: 5m
  sourceRef:
    kind: GitRepository
    name: flux-system
  path: ./infrastructure/overlays/prod
  prune: true
  wait: true
  healthChecks:
  - apiVersion: apps/v1
    kind: Deployment
    name: ingress-nginx-controller
    namespace: ingress-nginx
  - apiVersion: apps/v1
    kind: Deployment
    name: cert-manager
    namespace: cert-manager
```

```yaml
# clusters/prod-cluster/applications.yaml
apiVersion: kustomize.toolkit.fluxcd.io/v1
kind: Kustomization
metadata:
  name: applications
  namespace: flux-system
spec:
  interval: 5m
  retryInterval: 1m
  timeout: 10m
  sourceRef:
    kind: GitRepository
    name: flux-system
  path: ./apps
  prune: true
  wait: false
  dependsOn:
  - name: infrastructure
  postBuild:
    substitute:
      CLUSTER_NAME: prod-cluster
      ENVIRONMENT: production
  patches:
  - patch: |
      - op: add
        path: /spec/template/metadata/labels/cluster
        value: prod-cluster
    target:
      kind: Deployment
```

---

### Deployment Workflow

**Step 1: Developer Commits Code**

```bash
# Developer working on user-service
cd app-user-service
git checkout -b feature/add-email-verification

# Make changes
vim src/main/java/com/ecommerce/user/service/EmailService.java

# Commit changes
git add .
git commit -m "feat(user-service): add email verification"
git push origin feature/add-email-verification
```

**Step 2: Create Pull Request**
- CI pipeline runs tests
- Security scans execute
- Code review by team
- Approval required

**Step 3: Merge to Main**
```bash
# After approval
git checkout main
git merge feature/add-email-verification
git push origin main
```

**Step 4: CI Builds Docker Image**
- Build multi-stage Docker image
- Run Trivy security scan
- Push to GitHub Container Registry
- Tag with git SHA and semantic version

**Step 5: Update GitOps Repository**
```bash
# CI updates GitOps repo automatically
cd gitops-config/apps/user-service/overlays/dev
kustomize edit set image user-service=ghcr.io/ecommerce-platform/user-service:abc123
git commit -m "chore(user-service): update to abc123"
git push
```

**Step 6: Flux Syncs Changes**
```bash
# Flux detects change in Git
flux reconcile kustomization applications --with-source

# Monitor deployment
kubectl get deployment user-service -n user-service -w

# Check pod status
kubectl get pods -n user-service -l app=user-service
```

**Step 7: Verify Deployment**
```bash
# Check rollout status
kubectl rollout status deployment/user-service -n user-service

# Test health endpoint
kubectl port-forward -n user-service svc/user-service 8080:80
curl http://localhost:8080/actuator/health

# Check logs
kubectl logs -n user-service -l app=user-service --tail=100 -f
```

**Step 8: Promote to Staging**
```bash
# Copy dev config to staging
cd gitops-config/apps/user-service/overlays
cp dev/kustomization.yaml staging/
# Update image tag and environment-specific values
vim staging/kustomization.yaml

git add staging/
git commit -m "chore(user-service): promote abc123 to staging"
git push
```
# Organization structure
github.com/ecommerce/

**Step 9: Production Deployment**
```bash
# After testing in staging, promote to prod
cd gitops-config/apps/user-service/overlays
cp staging/kustomization.yaml prod/
# Update for production settings
vim prod/kustomization.yaml

# Create PR for production
git checkout -b deploy/user-service-v1.2.3
git add prod/
git commit -m "deploy(user-service): release v1.2.3 to production"
git push origin deploy/user-service-v1.2.3

# After approval and merge, Flux deploys to production
```

---

### Monitoring and Observability

**Grafana Dashboard Configuration**:

```yaml
# infrastructure/base/grafana/dashboards/user-service.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: user-service-dashboard
  namespace: monitoring
  labels:
    grafana_dashboard: "1"
data:
  user-service.json: |
    {
      "dashboard": {
        "title": "User Service Metrics",
        "panels": [
          {
            "title": "Request Rate",
            "targets": [{
              "expr": "rate(http_requests_total{service=\"user-service\"}[5m])"
            }]
          },
          {
            "title": "Error Rate",
            "targets": [{
              "expr": "rate(http_requests_total{service=\"user-service\",status=~\"5..\"}[5m])"
            }]
          },
          {
            "title": "Response Time (p95)",
            "targets": [{
              "expr": "histogram_quantile(0.95, http_request_duration_seconds_bucket{service=\"user-service\"})"
            }]
          },
          {
            "title": "Database Connections",
            "targets": [{
              "expr": "hikaricp_connections_active{service=\"user-service\"}"
            }]
          }
        ]
      }
    }
```

**Prometheus Alerts**:

```yaml
# infrastructure/base/prometheus/rules/user-service-alerts.yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: user-service-alerts
  namespace: monitoring
spec:
  groups:
  - name: user-service
    interval: 30s
    rules:
    - alert: HighErrorRate
      expr: |
        rate(http_requests_total{service="user-service",status=~"5.."}[5m]) > 0.05
      for: 5m
      labels:
        severity: critical
        service: user-service
      annotations:
        summary: "High error rate in user-service"
        description: "Error rate is {{ $value | humanizePercentage }} (threshold: 5%)"
    
    - alert: HighResponseTime
      expr: |
        histogram_quantile(0.95, 
          rate(http_request_duration_seconds_bucket{service="user-service"}[5m])
        ) > 1
      for: 10m
      labels:
        severity: warning
        service: user-service
      annotations:
        summary: "High response time in user-service"
        description: "P95 latency is {{ $value | humanizeDuration }}"
    
    - alert: PodCrashLooping
      expr: |
        rate(kube_pod_container_status_restarts_total{
          namespace="user-service",
          pod=~"user-service-.*"
        }[15m]) > 0
      for: 5m
      labels:
        severity: critical
        service: user-service
      annotations:
        summary: "Pod {{ $labels.pod }} is crash looping"
        description: "Pod has restarted {{ $value }} times in the last 15 minutes"
    
    - alert: DatabaseConnectionPoolExhausted
      expr: |
        hikaricp_connections_active{service="user-service"} / 
        hikaricp_connections_max{service="user-service"} > 0.9
      for: 5m
      labels:
        severity: warning
        service: user-service
      annotations:
        summary: "Database connection pool near capacity"
        description: "Connection pool is {{ $value | humanizePercentage }} full"
```

---

### Disaster Recovery and Rollback

**Rollback Procedure**:

```bash
# Option 1: Git revert
cd gitops-config
git log --oneline apps/user-service/overlays/prod
git revert abc123
git push

# Option 2: Flux rollback
flux suspend kustomization applications
kubectl set image deployment/user-service \
  user-service=ghcr.io/ecommerce-platform/user-service:v1.2.2 \
  -n user-service
flux resume kustomization applications

# Option 3: Kubectl rollback
kubectl rollout undo deployment/user-service -n user-service
kubectl rollout undo deployment/user-service -n user-service --to-revision=3
```

**Backup Strategy**:

```bash
# Backup all Kubernetes resources
kubectl get all --all-namespaces -o yaml > cluster-backup.yaml

# Backup Flux applications
flux export kustomization --all > flux-backup.yaml

# Backup to S3
aws s3 cp cluster-backup.yaml s3://ecommerce-backups/$(date +%Y%m%d)/
```

---

### Complete Deployment Checklist

**Pre-Deployment**:
- [ ] All tests passing in CI
- [ ] Security scans clean
- [ ] Code review approved
- [ ] Documentation updated
- [ ] Database migrations tested
- [ ] Rollback plan documented

**Deployment**:
- [ ] Deploy to dev environment
- [ ] Run smoke tests
- [ ] Deploy to staging
- [ ] Run full test suite
- [ ] Performance testing
- [ ] Create production PR
- [ ] Get stakeholder approval
- [ ] Deploy to production
- [ ] Monitor metrics

**Post-Deployment**:
- [ ] Verify health checks
- [ ] Check error rates
- [ ] Monitor response times
- [ ] Review logs
- [ ] Update CHANGELOG
- [ ] Notify team
- [ ] Document any issues

---

### Cost Optimization

**Resource Right-Sizing**:

```yaml
# Production resources based on actual usage
resources:
  requests:
    cpu: 500m      # Based on average: 350m + 40% buffer
    memory: 1Gi    # Based on average: 750Mi + 33% buffer
  limits:
    cpu: 2000m     # 4x requests for burst capacity
    memory: 2Gi    # 2x requests for headroom
```

**Autoscaling Configuration**:

```yaml
# HPA targets based on load testing
minReplicas: 3     # Minimum for high availability
maxReplicas: 20    # Maximum based on budget
targetCPUUtilizationPercentage: 70    # Sweet spot for cost/performance
```

**Cost Monitoring**:

```bash
# Kubernetes cost allocation
kubectl cost --namespace user-service --window 7d

# AWS cost by tag
aws ce get-cost-and-usage \
  --time-period Start=2025-01-01,End=2025-01-31 \
  --granularity MONTHLY \
  --filter file://cost-filter.json \
  --metrics BlendedCost
```

---

This complete e-commerce platform example demonstrates:

✅ **Multi-repository architecture** with clear separation of concerns
✅ **Complete CI/CD pipeline** from code commit to production
✅ **Production-ready Kubernetes manifests** with security, monitoring, and autoscaling
✅ **GitOps workflow** using Flux for automated deployments
✅ **Infrastructure as Code** with Terraform for AWS resources
✅ **Comprehensive monitoring** with Prometheus and Grafana
✅ **Security best practices** including secrets management and network policies
✅ **Disaster recovery** procedures and rollback strategies
✅ **Cost optimization** with right-sized resources and autoscaling

This architecture supports a scalable, secure, and maintainable e-commerce platform using GitOps principles.
