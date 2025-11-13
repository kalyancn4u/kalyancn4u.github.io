---
title: "🌊 GitOps: Deep Dive & Best Practices"
layout: post
author: technical_post
date: 2025-11-12 7:30:00 +0530
categories: [Notes, GitOps]
tags: [Git, GitHub, GitLab, GitHub Actions, GitOps, Best Practices, CI CD, DevOps, Automation, ]
description: "Concise, clear, and validated revision notes on GitOps (Git, GitHub, GitLab) — structured for beginners and practitioners."
toc: true
math: true
mermaid: true
---

## Table of Contents
1. [Introduction](#introduction)
2. [Core Concepts](#core-concepts)
3. [GitOps Principles](#gitops-principles)
4. [Git Fundamentals](#git-fundamentals)
5. [GitHub and GitHub Actions](#github-and-github-actions)
6. [GitLab CI/CD](#gitlab-cicd)
7. [Repository Structure](#repository-structure)
8. [Branching Strategies](#branching-strategies)
9. [CI/CD Pipeline Design](#cicd-pipeline-design)
10. [Infrastructure as Code](#infrastructure-as-code)
11. [Deployment Strategies](#deployment-strategies)
12. [Security Best Practices](#security-best-practices)
13. [Monitoring and Observability](#monitoring-and-observability)
14. [GitOps Tools](#gitops-tools)
15. [Best Practices](#best-practices)
16. [Common Pitfalls](#common-pitfalls)
17. [Jargon Tables](#jargon-tables)

---

## Introduction

GitOps is a modern operational framework that leverages Git as the single source of truth for declarative infrastructure and application code. It extends DevOps practices by using Git repositories to manage infrastructure configuration and application deployment, enabling teams to deliver software faster, more reliably, and with greater auditability.

### Directory Structure Best Practices

**Use Folders, Not Branches**:
- Avoid environment branches (dev, staging, prod)
- Use directories to organize environments
- Easier to see all variants simultaneously
- Simpler promotion between environments

```
k8s/
├── base/                     # Common configuration
│   ├── deployment.yaml
│   ├── service.yaml
│   └── kustomization.yaml
└── overlays/
    ├── dev/
    │   ├── kustomization.yaml
    │   └── patch-replicas.yaml
    ├── staging/
    │   ├── kustomization.yaml
    │   └── patch-replicas.yaml
    └── prod/
        ├── kustomization.yaml
        └── patch-replicas.yaml
```

**WET vs DRY Configuration**:

**DRY (Don't Repeat Yourself)**: Use templates and generators
- Pros: Less repetition, easier updates
- Cons: Harder to review, requires processing

**WET (Write Everything Twice)**: Explicit configuration files
- Pros: Easy to review, no processing needed
- Cons: More files, potential inconsistencies

**Recommendation**: Use WET for GitOps
- Changes are visible in pull requests
- No hidden logic or transformations
- Config Sync applies exactly what's in Git

---

## Branching Strategies

### Trunk-Based Development

**Recommended for GitOps**: Single main branch with short-lived feature branches.

**Principles**:
- Main branch is always deployable
- Feature branches live < 2 days
- Small, incremental changes
- Continuous integration
- Feature flags for incomplete features

```
main ─────●─────●─────●─────●─────●─────
           \   /       \   /
            ● ●         ● ●
         feature-1   feature-2
```

**Workflow**:

```bash
# 1. Create feature branch
git checkout -b feature/add-health-check

# 2. Make small changes
vim deployment.yaml

# 3. Commit frequently
git add deployment.yaml
git commit -m "feat: add liveness probe to deployment"

# 4. Push and create PR immediately
git push origin feature/add-health-check

# 5. Merge quickly (within hours)
# 6. Delete branch
git branch -d feature/add-health-check
```

### Environment Promotion

**Use Directories, Not Branches**:

```
configs/
├── dev/
│   └── app-config.yaml
├── staging/
│   └── app-config.yaml
└── prod/
    └── app-config.yaml
```

**Promotion Process**:

```bash
# 1. Test in dev
git checkout main
cd configs/dev
# make changes, test

# 2. Promote to staging
cp dev/app-config.yaml staging/app-config.yaml
# adjust environment-specific values
git add staging/
git commit -m "chore: promote dev config to staging"
git push

# 3. After validation, promote to prod
cp staging/app-config.yaml prod/app-config.yaml
# adjust environment-specific values
git add prod/
git commit -m "chore: promote staging config to prod"
git push
```

### Release Strategies

#### Git Flow (Not Recommended for GitOps)

```
main ────────────●──────────●──────────●───
                /          /          /
release ───────●──────────●──────────●─────
              /          /          /
develop ─────●──────────●──────────●───────
            / \        / \        / \
feature ───●   ●──────●   ●──────●   ●─────
```

**Why Not for GitOps**:
- Multiple long-lived branches
- Complex merge strategies
- Cherry-picking required
- Doesn't match declarative model

#### GitHub Flow (Recommended)

```
main ─────●─────●─────●─────●─────●─────
           \   /       \   /       \   /
            ● ●         ● ●         ● ●
         feature-1   feature-2   feature-3
```

**Workflow**:
1. Branch from main
2. Make changes
3. Create PR
4. Review and test
5. Merge to main
6. Delete branch

---

## CI/CD Pipeline Design

### Pipeline Stages

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│  Code   │───▶│ Build   │───▶│  Test   │───▶│ Deploy  │
└─────────┘    └─────────┘    └─────────┘    └─────────┘
```

#### 1. Code Stage

**Pre-commit Hooks**:

```bash
# .git/hooks/pre-commit
#!/bin/bash

# Run linting
npm run lint
if [ $? -ne 0 ]; then
    echo "Linting failed. Commit aborted."
    exit 1
fi

# Run tests
npm test
if [ $? -ne 0 ]; then
    echo "Tests failed. Commit aborted."
    exit 1
fi

exit 0
```

**Pre-push Hooks**:

```bash
# .git/hooks/pre-push
#!/bin/bash

# Prevent push to main
branch=$(git rev-parse --abbrev-ref HEAD)
if [ "$branch" = "main" ]; then
    echo "Direct push to main is not allowed."
    exit 1
fi

exit 0
```

#### 2. Build Stage

```yaml
build:
  stage: build
  script:
    # Build application
    - docker build -t $IMAGE:$CI_COMMIT_SHA .
    
    # Scan for vulnerabilities
    - trivy image --severity HIGH,CRITICAL $IMAGE:$CI_COMMIT_SHA
    
    # Push to registry
    - docker push $IMAGE:$CI_COMMIT_SHA
    
    # Update image tag in GitOps repo
    - cd gitops-repo
    - kustomize edit set image app=$IMAGE:$CI_COMMIT_SHA
    - git commit -am "Update image to $CI_COMMIT_SHA"
    - git push
```

#### 3. Test Stage

**Test Types**:

```yaml
test:
  parallel:
    matrix:
      - TEST_TYPE: unit
      - TEST_TYPE: integration
      - TEST_TYPE: e2e
  script:
    - npm run test:$TEST_TYPE
```

**Test Pyramid**:
```
       /\
      /  \     E2E Tests (Few)
     /____\
    /      \   Integration Tests (Some)
   /________\
  /          \ Unit Tests (Many)
 /____________\
```

#### 4. Deploy Stage

**GitOps Deploy** (Update manifest, agent applies):

```yaml
deploy:
  stage: deploy
  script:
    - git clone https://gitlab.com/org/gitops-repo.git
    - cd gitops-repo
    - yq eval ".spec.template.spec.containers[0].image = \"$IMAGE:$TAG\"" -i deployment.yaml
    - git add deployment.yaml
    - git commit -m "Deploy $TAG to production"
    - git push
```

### Pipeline Best Practices

#### 1. Fail Fast

```yaml
jobs:
  quick-checks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Lint
        run: npm run lint
      - name: Type check
        run: npm run typecheck
  
  expensive-tests:
    needs: quick-checks  # Only run if quick checks pass
    runs-on: ubuntu-latest
    steps:
      - name: Integration tests
        run: npm run test:integration
```

#### 2. Cache Dependencies

```yaml
- name: Cache node modules
  uses: actions/cache@v3
  with:
    path: ~/.npm
    key: ${{ runner.os }}-node-${{ hashFiles('**/package-lock.json') }}
    restore-keys: |
      ${{ runner.os }}-node-
```

#### 3. Parallel Execution

```yaml
test:
  strategy:
    matrix:
      suite: [unit, integration, e2e]
  runs-on: ubuntu-latest
  steps:
    - run: npm run test:${{ matrix.suite }}
```

#### 4. Conditional Execution

```yaml
deploy-staging:
  if: github.ref == 'refs/heads/develop'
  runs-on: ubuntu-latest
  steps:
    - run: ./deploy.sh staging

deploy-prod:
  if: github.event_name == 'release'
  runs-on: ubuntu-latest
  steps:
    - run: ./deploy.sh production
```

#### 5. Manual Approval Gates

```yaml
deploy-production:
  runs-on: ubuntu-latest
  environment:
    name: production
    url: https://example.com
  steps:
    - run: ./deploy.sh
  # Requires manual approval in GitHub
```

---

## Infrastructure as Code

### Terraform

**Example Configuration**:

```hcl
# main.tf
terraform {
  required_version = ">= 1.0"
  
  backend "s3" {
    bucket = "terraform-state"
    key    = "prod/terraform.tfstate"
    region = "us-east-1"
  }
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

# VPC
module "vpc" {
  source = "./modules/vpc"
  
  cidr_block = "10.0.0.0/16"
  azs        = ["us-east-1a", "us-east-1b", "us-east-1c"]
  
  tags = {
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

# EKS Cluster
module "eks" {
  source = "./modules/eks"
  
  cluster_name    = "my-cluster"
  cluster_version = "1.28"
  
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
  
  node_groups = {
    general = {
      desired_capacity = 3
      max_capacity     = 10
      min_capacity     = 2
      
      instance_types = ["t3.medium"]
      
      labels = {
        role = "general"
      }
    }
  }
}

# Outputs
output "cluster_endpoint" {
  value = module.eks.cluster_endpoint
}

output "cluster_name" {
  value = module.eks.cluster_name
}
```

**Terraform Workflow**:

```bash
# Initialize
terraform init

# Plan changes
terraform plan -out=tfplan

# Apply changes
terraform apply tfplan

# Destroy resources
terraform destroy
```

**GitOps with Terraform**:

```yaml
# .github/workflows/terraform.yml
name: Terraform

on:
  push:
    branches: [ main ]
    paths:
      - 'terraform/**'
  pull_request:
    branches: [ main ]
    paths:
      - 'terraform/**'

jobs:
  terraform:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: terraform
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: 1.6.0
      
      - name: Terraform Init
        run: terraform init
      
      - name: Terraform Format
        run: terraform fmt -check
      
      - name: Terraform Validate
        run: terraform validate
      
      - name: Terraform Plan
        run: terraform plan -no-color
        continue-on-error: true
      
      - name: Terraform Apply
        if: github.ref == 'refs/heads/main' && github.event_name == 'push'
        run: terraform apply -auto-approve
```

### Kubernetes Manifests

**Plain YAML**:

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
        version: v1.0.0
    spec:
      containers:
      - name: app
        image: myapp:v1.0.0
        ports:
        - containerPort: 8080
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: url
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 500m
            memory: 512Mi
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: my-app
  namespace: production
spec:
  selector:
    app: my-app
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer
```

### Kustomize

**Directory Structure**:

```
k8s/
├── base/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── configmap.yaml
│   └── kustomization.yaml
└── overlays/
    ├── dev/
    │   ├── kustomization.yaml
    │   ├── patch-replicas.yaml
    │   └── patch-resources.yaml
    ├── staging/
    │   ├── kustomization.yaml
    │   └── patch-replicas.yaml
    └── prod/
        ├── kustomization.yaml
        └── patch-replicas.yaml
```

**Base Configuration**:

```yaml
# base/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
  - deployment.yaml
  - service.yaml
  - configmap.yaml

commonLabels:
  app: my-app
  managedBy: kustomize
```

**Dev Overlay**:

```yaml
# overlays/dev/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

bases:
  - ../../base

namespace: dev

patches:
  - patch-replicas.yaml
  - patch-resources.yaml

images:
  - name: myapp
    newTag: dev-latest

configMapGenerator:
  - name: app-config
    behavior: merge
    literals:
      - LOG_LEVEL=debug
      - ENVIRONMENT=development
```

```yaml
# overlays/dev/patch-replicas.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
spec:
  replicas: 1
```

**Build and Apply**:

```bash
# Build kustomization
kustomize build overlays/dev

# Apply to cluster
kustomize build overlays/dev | kubectl apply -f -

# Or use kubectl directly
kubectl apply -k overlays/dev
```

### Helm

**Chart Structure**:

```
my-app/
├── Chart.yaml
├── values.yaml
├── templates/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── configmap.yaml
│   ├── secret.yaml
│   └── _helpers.tpl
└── charts/
```

**Chart.yaml**:

```yaml
apiVersion: v2
name: my-app
description: A Helm chart for my application
type: application
version: 1.0.0
appVersion: "1.0.0"
```

**values.yaml**:

```yaml
replicaCount: 3

image:
  repository: myapp
  tag: "1.0.0"
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 80

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: app.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: app-tls
      hosts:
        - app.example.com

resources:
  requests:
    cpu: 100m
    memory: 128Mi
  limits:
    cpu: 500m
    memory: 512Mi

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
```

**Template**:

```yaml
# templates/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "my-app.fullname" . }}
  labels:
    {{- include "my-app.labels" . | nindent 4 }}
spec:
  {{- if not .Values.autoscaling.enabled }}
  replicas: {{ .Values.replicaCount }}
  {{- end }}
  selector:
    matchLabels:
      {{- include "my-app.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      labels:
        {{- include "my-app.selectorLabels" . | nindent 8 }}
    spec:
      containers:
      - name: {{ .Chart.Name }}
        image: "{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}"
        imagePullPolicy: {{ .Values.image.pullPolicy }}
        ports:
        - name: http
          containerPort: 8080
          protocol: TCP
        resources:
          {{- toYaml .Values.resources | nindent 12 }}
```

**Environment-Specific Values**:

```yaml
# values-dev.yaml
replicaCount: 1

image:
  tag: dev-latest

resources:
  requests:
    cpu: 50m
    memory: 64Mi

autoscaling:
  enabled: false
```

```yaml
# values-prod.yaml
replicaCount: 5

image:
  tag: "1.0.0"

resources:
  requests:
    cpu: 200m
    memory: 256Mi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20
```

**Helm Commands**:

```bash
# Install chart
helm install my-app ./my-app -f values-prod.yaml

# Upgrade
helm upgrade my-app ./my-app -f values-prod.yaml

# Rollback
helm rollback my-app 1

# Uninstall
helm uninstall my-app

# List releases
helm list

# Get values
helm get values my-app
```

---

## Deployment Strategies

### Rolling Update

**Description**: Gradually replace old pods with new ones.

```yaml
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1        # Max pods above desired count
      maxUnavailable: 0  # Max pods that can be unavailable
```

**Pros**:
- Zero downtime
- Gradual rollout
- Easy rollback

**Cons**:
- Both versions run simultaneously
- Slower than recreate

### Blue-Green Deployment

**Description**: Run two identical environments, switch traffic between them.

```yaml
# Blue deployment (current)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app-blue
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-app
      version: blue

---
# Green deployment (new)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app-green
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-app
      version: green

---
# Service points to active version
apiVersion: v1
kind: Service
metadata:
  name: my-app
spec:
  selector:
    app: my-app
    version: blue  # Switch to 'green' to cutover
```

**Pros**:
- Instant rollback
- Zero downtime
- Full testing before cutover

**Cons**:
- Double resources required
- Database migrations complex

### Canary Deployment

**Description**: Route small percentage of traffic to new version.

```yaml
# Stable version (90% of traffic)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app-stable
spec:
  replicas: 9

---
# Canary version (10% of traffic)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app-canary
spec:
  replicas: 1
```

**Using Istio**:

```yaml
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: my-app
spec:
  hosts:
  - my-app
  http:
  - match:
    - headers:
        canary:
          exact: "true"
    route:
    - destination:
        host: my-app
        subset: canary
  - route:
    - destination:
        host: my-app
        subset: stable
      weight: 90
    - destination:
        host: my-app
        subset: canary
      weight: 10
```

**Pros**:
- Reduced risk
- Real user testing
- Gradual rollout

**Cons**:
- Complex setup
- Monitoring required
- Longer deployment time

---

## Security Best Practices

### 1. Secrets Management

**Never Commit Secrets to Git**:

```bash
# .gitignore
.env
secrets.yaml
*.pem
*.key
credentials.json
```

**Use External Secret Stores**:

```yaml
# Using External Secrets Operator
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: db-credentials
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secrets-manager
    kind: SecretStore
  target:
    name: db-credentials
    creationPolicy: Owner
  data:
  - secretKey: password
    remoteRef:
      key: prod/db/password
```

**Sealed Secrets** (Bitnami):

```bash
# Encrypt secret
kubeseal --format yaml < secret.yaml > sealed-secret.yaml

# Commit sealed secret to Git
git add sealed-secret.yaml
git commit -m "Add database credentials"
```

```yaml
# sealed-secret.yaml (safe to commit)
apiVersion: bitnami.com/v1alpha1
kind: SealedSecret
metadata:
  name: db-credentials
spec:
  encryptedData:
    password: AgBHW3N2c3RoaW5nZW5jcnlwdGVkCg==
```

### 2. RBAC (Role-Based Access Control)

```yaml
# Role
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: developer
  namespace: production
rules:
- apiGroups: ["", "apps"]
  resources: ["pods", "deployments"]
  verbs: ["get", "list", "watch"]

---
# RoleBinding
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: developer-binding
  namespace: production
subjects:
- kind: User
  name: jane.doe@example.com
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: Role
  name: developer
  apiGroup: rbac.authorization.k8s.io
```

### 3. Pod Security

**Pod Security Standards**:

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: production
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
```

**Security Context**:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: secure-pod
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 2000
    seccompProfile:
      type: RuntimeDefault
  containers:
  - name: app
    image: myapp:1.0.0
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
        - ALL
```

### 4. Network Policies

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-all
  namespace: production
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress

---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-frontend-to-backend
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: backend
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: frontend
    ports:
    - protocol: TCP
      port: 8080
```

### 5. Image Security

**Image Scanning**:

```yaml
# .github/workflows/security.yml
- name: Run Trivy scanner
  uses: aquasecurity/trivy-action@master
  with:
    image-ref: ${{ env.IMAGE }}:${{ github.sha }}
    format: 'sarif'
    output: 'trivy-results.sarif'
    severity: 'CRITICAL,HIGH'

- name: Upload to GitHub Security
  uses: github/codeql-action/upload-sarif@v2
  with:
    sarif_file: 'trivy-results.sarif'
```

**Image Signing** (Cosign):

```bash
# Sign image
cosign sign --key cosign.key $IMAGE:$TAG

# Verify signature
cosign verify --key cosign.pub $IMAGE:$TAG
```

### 6. Audit Logging

```yaml
# Enable audit logging in Kubernetes
apiVersion: audit.k8s.io/v1
kind: Policy
rules:
- level: Metadata
  resources:
  - group: ""
    resources: ["secrets", "configmaps"]
- level: RequestResponse
  resources:
  - group: "apps"
    resources: ["deployments", "statefulsets"]
```

---

## Monitoring and Observability

### Metrics

**Prometheus**:

```yaml
apiVersion: v1
kind: ServiceMonitor
metadata:
  name: my-app
spec:
  selector:
    matchLabels:
      app: my-app
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
```

**Key Metrics**:
- **Application**: Request rate, error rate, latency (RED)
- **Infrastructure**: CPU, memory, disk, network (USE)
- **GitOps**: Sync status, drift detection, reconciliation time

### Logging

**Structured Logging**:

```json
{
  "timestamp": "2025-01-15T10:30:00Z",
  "level": "info",
  "message": "Deployment successful",
  "service": "my-app",
  "version": "v1.2.3",
  "environment": "production",
  "user": "jane.doe@example.com"
}
```

**Log Aggregation**:
- ELK Stack (Elasticsearch, Logstash, Kibana)
- Loki (Grafana)
- CloudWatch Logs (AWS)

### Tracing

**OpenTelemetry**:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: otel-collector-config
data:
  config.yaml: |
    receivers:
      otlp:
        protocols:
          grpc:
          http:
    
    processors:
      batch:
    
    exporters:
      jaeger:
        endpoint: jaeger:14250
    
    service:
      pipelines:
        traces:
          receivers: [otlp]
          processors: [batch]
          exporters: [jaeger]
```

### Alerting

```yaml
# PrometheusRule
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: my-app-alerts
spec:
  groups:
  - name: my-app
    rules:
    - alert: HighErrorRate
      expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
      for: 10m
      labels:
        severity: critical
      annotations:
        summary: "High error rate detected"
        description: "Error rate is {{ $value }} requests/second"
    
    - alert: PodCrashLooping
      expr: rate(kube_pod_container_status_restarts_total[15m]) > 0
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "Pod {{ $labels.pod }} is crash looping"
```

---

## GitOps Tools

### ArgoCD

**Installation**:

```bash
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
```

**Application Definition**:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: my-app
  namespace: argocd
spec:
  project: default
  
  source:
    repoURL: https://github.com/org/gitops-repo.git
    targetRevision: HEAD
    path: apps/my-app/overlays/prod
  
  destination:
    server: https://kubernetes.default.svc
    namespace: production
  
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
    - CreateNamespace=true
  
  ignoreDifferences:
  - group: apps
    kind: Deployment
    jsonPointers:
    - /spec/replicas
```

**ArgoCD CLI**:

```bash
# Login
argocd login argocd.example.com

# List applications
argocd app list

# Get application details
argocd app get my-app

# Sync application
argocd app sync my-app

# Rollback
argocd app rollback my-app 0
```

### Flux

**Installation**:

```bash
flux bootstrap github \
  --owner=myorg \
  --repository=fleet-infra \
  --branch=main \
  --path=./clusters/production \
  --personal
```

**GitRepository**:

```yaml
apiVersion: source.toolkit.fluxcd.io/v1beta2
kind: GitRepository
metadata:
  name: my-app
  namespace: flux-system
spec:
  interval: 1m
  url: https://github.com/org/my-app
  ref:
    branch: main
```

**Kustomization**:

```yaml
apiVersion: kustomize.toolkit.fluxcd.io/v1beta2
kind: Kustomization
metadata:
  name: my-app
  namespace: flux-system
spec:
  interval: 5m
  path: ./k8s/overlays/prod
  prune: true
  sourceRef:
    kind: GitRepository
    name: my-app
  healthChecks:
  - apiVersion: apps/v1
    kind: Deployment
    name: my-app
    namespace: production
```

**Flux CLI**:

```bash
# Check Flux components
flux check

# Get kustomizations
flux get kustomizations

# Reconcile
flux reconcile kustomization my-app

# Suspend/resume
flux suspend kustomization my-app
flux resume kustomization my-app
```

### Jenkins X

**Installation**:

```bash
jx boot
```

**Pipeline Configuration**:

```yaml
# jenkins-x.yml
buildPack: none
pipelineConfig:
  pipelines:
    release:
      pipeline:
        stages:
        - name: build
          steps:
          - sh: docker build -t $DOCKER_REGISTRY/$APP_NAME:$VERSION .
        - name: test
          steps:
          - sh: make test
        - name: deploy
          steps:
          - sh: jx step helm apply
```

### Comparison

| Feature | ArgoCD | Flux | Jenkins X |
|---------|--------|------|-----------|
| **UI** | ✅ Rich Web UI | ❌ Limited | ✅ Web UI |
| **Multi-cluster** | ✅ Native | ✅ Via Git repos | ✅ Native |
| **Helm Support** | ✅ Full | ✅ Full | ✅ Native |
| **Kustomize Support** | ✅ Full | ✅ Full | ✅ Via plugin |
| **SSO** | ✅ OIDC, LDAP | ❌ | ✅ OAuth |
| **RBAC** | ✅ Fine-grained | ✅ K8s RBAC | ✅ K8s RBAC |
| **Notifications** | ✅ Slack, Email | ✅ Slack, Email | ✅ Multiple |
| **CI Integration** | ✅ Any CI | ✅ Any CI | ✅ Built-in |
| **Learning Curve** | Medium | Low | High |

---

## Best Practices

### 1. Git Repository Organization

**Separate Concerns**:
- Application code repository
- Infrastructure repository
- Configuration repository

**Benefits**:
- Different lifecycles
- Different teams
- Different security requirements
- Different approval processes

```
org/
├── app-user-service/       # Application code
├── infrastructure/         # Terraform, CloudFormation
└── gitops-configs/         # K8s manifests, Helm values
```

### 2. Environment Management

**Use Directories, Not Branches**:

```
configs/
├── base/                   # Common configuration
└── environments/
    ├── dev/
    ├── staging/
    └── prod/
```

**Environment Promotion**:

```bash
# Promote staging to prod
git diff environments/staging environments/prod
git checkout environments/staging -- app-config.yaml
mv app-config.yaml environments/prod/
git add environments/prod/
git commit -m "Promote staging config to prod"
```

### 3. Declarative Configuration

**Always Use Declarative Syntax**:

```yaml
# Good - Declarative
apiVersion: apps/v1
kind: Deployment
spec:
  replicas: 3

# Bad - Imperative
# kubectl scale deployment my-app --replicas=3
```

### 4. Version Everything

**Tag Releases**:

```bash
git tag -a v1.2.3 -m "Release version 1.2.3"
git push origin v1.2.3
```

**Semantic Versioning**:
- MAJOR.MINOR.PATCH (1.2.3)
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

### 5. Automated Testing

**Test Infrastructure Code**:

```yaml
# .github/workflows/terraform-test.yml
name: Terraform Test

on:
  pull_request:
    paths:
      - 'terraform/**'

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Terraform Format
        run: terraform fmt -check -recursive
      
      - name: Terraform Validate
        run: |
          terraform init -backend=false
          terraform validate
      
      - name: TFLint
        uses: terraform-linters/setup-tflint@v3
      
      - name: Run TFLint
        run: tflint --recursive
      
      - name: Checkov Security Scan
        uses: bridgecrewio/checkov-action@master
        with:
          directory: terraform/
```

**Test Kubernetes Manifests**:

```yaml
# .github/workflows/k8s-test.yml
name: Kubernetes Manifest Test

on:
  pull_request:
    paths:
      - 'k8s/**'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup tools
        run: |
          curl -s https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh | bash
          sudo snap install kubeconform
      
      - name: Validate with kustomize
        run: |
          kustomize build k8s/overlays/prod > output.yaml
      
      - name: Validate with kubeconform
        run: |
          kubeconform -summary -output json output.yaml
      
      - name: Policy check with OPA
        uses: open-policy-agent/opa-action@v2
        with:
          tests: policies/
```

### 6. Security Practices

**Scan for Secrets**:

```yaml
- name: Gitleaks scan
  uses: gitleaks/gitleaks-action@v2
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

**Sign Commits**:

```bash
# Configure GPG
git config --global user.signingkey YOUR_GPG_KEY
git config --global commit.gpgsign true

# Sign commits
git commit -S -m "Add deployment configuration"
```

**Verify Commits**:

```bash
git verify-commit HEAD
```

### 7. Documentation

**README Template**:

```markdown
# Project Name

## Overview
Brief description of the project and its purpose.

## Architecture
High-level architecture diagram and explanation.

## Repository Structure
```
project/
├── apps/           # Application manifests
├── infrastructure/ # Infrastructure code
└── docs/          # Documentation
```

## Prerequisites
- Kubernetes 1.25+
- kubectl
- kustomize

## Deployment
Step-by-step deployment instructions.

## Monitoring
Links to dashboards and monitoring tools.

## Troubleshooting
Common issues and solutions.

## Contributing
Contribution guidelines.
```

### 8. Rollback Strategy

**Keep Rollback Simple**:

```bash
# With ArgoCD
argocd app rollback my-app 0

# With Flux
flux reconcile kustomization my-app --with-source

# With Git
git revert HEAD
git push
```

**Test Rollback Procedures**:
- Practice rollbacks regularly
- Automate rollback triggers
- Monitor rollback success

### 9. Change Management

**Pull Request Template**:

```markdown
## Description
What does this PR do?

## Type of Change
- [ ] New feature
- [ ] Bug fix
- [ ] Configuration change
- [ ] Infrastructure change

## Impact Analysis
- [ ] Affects production
- [ ] Requires downtime
- [ ] Breaking change
- [ ] Rollback plan documented

## Testing
- [ ] Tested in dev
- [ ] Tested in staging
- [ ] Load testing completed
- [ ] Security review completed

## Deployment Plan
Step-by-step deployment instructions

## Rollback Plan
Step-by-step rollback instructions

## Checklist
- [ ] Documentation updated
- [ ] Monitoring alerts configured
- [ ] Team notified
```

### 10. Observability

**Monitor GitOps Health**:

```yaml
# Prometheus metrics
- argocd_app_sync_total
- argocd_app_health_status
- gitops_runtime_reconcile_duration_seconds
- flux_reconcile_duration_seconds
```

**Dashboard Metrics**:
- Sync success rate
- Time to sync
- Drift detection count
- Failed reconciliations
- Deployment frequency
- Lead time for changes
- Mean time to recovery (MTTR)

### 11. Disaster Recovery

**Backup Strategy**:

```bash
# Backup cluster state
kubectl get all --all-namespaces -o yaml > cluster-backup.yaml

# Backup ArgoCD applications
argocd app list -o yaml > argocd-apps-backup.yaml
```

**Recovery Plan**:
1. Restore infrastructure (Terraform)
2. Deploy GitOps operator
3. Apply application definitions
4. Verify sync status

### 12. Progressive Delivery

**Canary with Flagger**:

```yaml
apiVersion: flagger.app/v1beta1
kind: Canary
metadata:
  name: my-app
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: my-app
  service:
    port: 80
  analysis:
    interval: 1m
    threshold: 10
    maxWeight: 50
    stepWeight: 5
    metrics:
    - name: request-success-rate
      thresholdRange:
        min: 99
      interval: 1m
    - name: request-duration
      thresholdRange:
        max: 500
      interval: 1m
  webhooks:
  - name: load-test
    url: http://load-tester.test/
    timeout: 5s
```

---

## Common Pitfalls

### 1. Committing Secrets to Git

**Problem**: Secrets accidentally committed to repository.

**Solution**:
- Use `.gitignore`
- Use git-secrets or gitleaks
- Use external secret management
- Rotate exposed secrets immediately

```bash
# Install git-secrets
git secrets --install
git secrets --register-aws

# Scan repository
git secrets --scan
```

### 2. Direct Cluster Modifications

**Problem**: Manual kubectl commands bypass GitOps.

**Solution**:
- Enforce RBAC policies
- Use admission controllers
- Audit cluster changes
- Educate team on GitOps workflow

```yaml
# OPA Policy: Deny manual changes
package kubernetes.admission

deny[msg] {
  input.request.userInfo.username != "system:serviceaccount:flux-system:flux"
  msg := "Manual changes not allowed. Use GitOps."
}
```

### 3. Not Testing Before Merge

**Problem**: Broken configurations merged to main.

**Solution**:
- Require CI checks to pass
- Use branch protection
- Enable preview environments

```yaml
# Branch protection
main:
  required_status_checks:
    - validate-manifests
    - security-scan
  required_reviews: 2
```

### 4. Ignoring Drift

**Problem**: Actual state diverges from desired state.

**Solution**:
- Enable auto-sync
- Monitor drift metrics
- Set up alerts

```yaml
# ArgoCD auto-sync
syncPolicy:
  automated:
    prune: true
    selfHeal: true
```

### 5. Poor Repository Structure

**Problem**: Difficult to navigate and maintain.

**Solution**:
- Follow consistent structure
- Document organization
- Use clear naming conventions

### 6. Missing Rollback Plan

**Problem**: No clear way to revert changes.

**Solution**:
- Document rollback procedures
- Practice rollbacks
- Keep rollback simple (git revert)

### 7. Inadequate Monitoring

**Problem**: Don't know when deployments fail.

**Solution**:
- Monitor GitOps metrics
- Set up alerts
- Integrate with incident management

### 8. Over-Complicated Pipelines

**Problem**: Complex pipelines are hard to maintain.

**Solution**:
- Keep pipelines simple
- Use reusable workflows
- Document complex logic

### 9. Lack of Documentation

**Problem**: Team doesn't understand workflows.

**Solution**:
- Document processes
- Create runbooks
- Provide training

### 10. Not Using Environments Properly

**Problem**: Testing directly in production.

**Solution**:
- Use dev/staging/prod environments
- Test in lower environments first
- Automate promotion

---

## Jargon Tables

### Table 1: GitOps Lifecycle Terminology

| GitOps Term | Alternative Terms | Definition | Context |
|-------------|-------------------|------------|---------|
| **Desired State** | Target state, intended state | Configuration stored in Git | What you want |
| **Actual State** | Current state, live state, runtime state | Current configuration in cluster | What you have |
| **Reconciliation** | Sync, convergence, drift correction | Process of aligning actual with desired | Continuous process |
| **Drift** | Configuration drift, state divergence | Difference between desired and actual state | Problem detection |
| **Sync** | Synchronization, apply, deploy | Update actual state to match desired | Action |
| **Pull-based** | Agent-based, operator pattern | Agent pulls changes from Git | GitOps model |
| **Push-based** | Traditional CI/CD, pipeline deploy | Pipeline pushes to cluster | Traditional model |
| **Declarative** | Descriptive, state-based | Define what you want, not how | Configuration style |
| **Imperative** | Procedural, command-based | Define how to achieve state | Traditional approach |
| **Manifest** | Configuration file, resource definition | YAML/JSON describing resources | K8s terminology |
| **GitOps Agent** | Operator, controller, reconciler | Software monitoring and applying changes | ArgoCD, Flux |
| **Source of Truth** | Single source, canonical source | Authoritative configuration location | Git repository |
| **Auto-sync** | Automated sync, continuous deployment | Automatic application of changes | GitOps feature |
| **Self-heal** | Auto-remediation, drift correction | Automatic correction of manual changes | GitOps feature |
| **Prune** | Cleanup, deletion | Remove resources not in desired state | GitOps operation |

### Table 2: Git Operations Terminology

| Git Term | Alternative Terms | Definition | Common Commands |
|----------|-------------------|------------|-----------------|
| **Repository** | Repo, project | Directory with Git history | `git init`, `git clone` |
| **Commit** | Revision, snapshot, changeset | Saved state of repository | `git commit` |
| **Branch** | Line of development | Parallel version of code | `git branch`, `git checkout` |
| **Merge** | Integration, combine | Integrate changes from branches | `git merge` |
| **Pull Request** | PR, merge request (GitLab) | Request to merge changes | GitHub/GitLab UI |
| **Tag** | Release tag, version tag | Named reference to commit | `git tag` |
| **Push** | Upload, publish | Send commits to remote | `git push` |
| **Pull** | Download, fetch+merge | Get changes from remote | `git pull` |
| **Fetch** | Retrieve, download | Get remote changes without merge | `git fetch` |
| **Rebase** | Reapply, replay commits | Move commits to new base | `git rebase` |
| **Cherry-pick** | Select commit | Apply specific commit | `git cherry-pick` |
| **Stash** | Temporary save | Save uncommitted changes | `git stash` |
| **Reset** | Undo, rewind | Move HEAD to different commit | `git reset` |
| **Revert** | Reverse, undo commit | Create new commit undoing changes | `git revert` |
| **Remote** | Repository URL | Remote repository reference | `git remote` |

### Table 3: CI/CD Pipeline Stages

| Stage | Alternative Names | Purpose | Common Tools |
|-------|-------------------|---------|--------------|
| **Source** | Code checkout, clone | Get code from repository | Git, GitHub, GitLab |
| **Build** | Compile, package | Create deployable artifacts | Docker, Maven, npm |
| **Test** | Validation, quality check | Verify code quality | Jest, pytest, JUnit |
| **Security Scan** | SAST, vulnerability scan | Identify security issues | Trivy, Snyk, SonarQube |
| **Artifact Storage** | Registry, repository | Store build artifacts | Docker Hub, ECR, Nexus |
| **Deploy** | Release, rollout | Deploy to environment | ArgoCD, Flux, Helm |
| **Verify** | Smoke test, health check | Confirm deployment success | curl, k8s probes |
| **Promote** | Environment progression | Move between environments | Git operations |

### Table 4: Hierarchical GitOps Architecture

| Level | Component | Sub-Component | Purpose | Tools |
|-------|-----------|---------------|---------|-------|
| **1** | **Source Control** | | Version control system | Git |
| | | **Repository** | Store configurations | GitHub, GitLab |
| | | **Branch** | Parallel development | Git branches |
| | | **Pull Request** | Code review mechanism | GitHub PR, GitLab MR |
| **2** | **CI Pipeline** | | Continuous Integration | GitHub Actions, GitLab CI |
| | | **Build** | Create artifacts | Docker build |
| | | **Test** | Validation | pytest, jest |
| | | **Security** | Vulnerability scanning | Trivy, Snyk |
| **3** | **Artifact Registry** | | Store build outputs | Container registries |
| | | **Container Images** | Docker images | Docker Hub, ECR, GCR |
| | | **Helm Charts** | K8s packages | Helm registry |
| **4** | **GitOps Operator** | | Sync engine | ArgoCD, Flux |
| | | **Source Controller** | Monitor Git repos | Flux Source Controller |
| | | **Sync Controller** | Apply changes | ArgoCD Application Controller |
| | | **Health Assessment** | Check resource status | Health checks |
| **5** | **Target Environment** | | Deployment destination | Kubernetes |
| | | **Cluster** | K8s cluster | EKS, GKE, AKS |
| | | **Namespace** | Logical separation | K8s namespaces |
| | | **Workloads** | Running applications | Deployments, StatefulSets |

### Table 5: Deployment Strategy Comparison

| Strategy | Speed | Risk | Downtime | Resource Cost | Rollback Speed | Use Case |
|----------|-------|------|----------|---------------|----------------|----------|
| **Recreate** | Fast | High | Yes | Low | Slow | Dev environments |
| **Rolling Update** | Medium | Medium | No | Low | Medium | Most applications |
| **Blue-Green** | Instant | Low | No | High (2x) | Instant | Critical services |
| **Canary** | Slow | Low | No | Medium | Fast | High-risk changes |
| **A/B Testing** | Slow | Low | No | Medium | N/A | Feature testing |

### Table 6: GitOps Tool Comparison

| Feature | ArgoCD | Flux | Jenkins X | Spinnaker |
|---------|--------|------|-----------|-----------|
| **Architecture** | Controller | Operator | Platform | Pipeline |
| **UI** | ✅ Rich | ⚠️ Basic | ✅ Good | ✅ Rich |
| **Multi-tenant** | ✅ Native | ✅ Via namespaces | ✅ Native | ✅ Native |
| **Helm Support** | ✅ Full | ✅ Full | ✅ Native | ✅ Full |
| **Kustomize** | ✅ Native | ✅ Native | ✅ Plugin | ✅ Plugin |
| **SSO/OIDC** | ✅ Yes | ❌ No | ✅ Yes | ✅ Yes |
| **RBAC** | ✅ Fine-grained | ✅ K8s RBAC | ✅ K8s RBAC | ✅ Fine-grained |
| **Webhook Events** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| **Notifications** | ✅ Multiple | ✅ Multiple | ✅ Multiple | ✅ Multiple |
| **Progressive Delivery** | ⚠️ Via Argo Rollouts | ✅ Via Flagger | ❌ No | ✅ Native |
| **Learning Curve** | Medium | Low | High | High |
| **Community** | Large | Large | Medium | Large |

### Table 7: Infrastructure as Code Tools

| Tool | Language | Cloud Support | State Management | Use Case |
|------|----------|---------------|------------------|----------|
| **Terraform** | HCL | Multi-cloud | Remote backends | Universal IaC |
| **Pulumi** | TypeScript, Python, Go | Multi-cloud | Cloud storage | Code-first IaC |
| **CloudFormation** | YAML/JSON | AWS only | AWS managed | AWS native |
| **Ansible** | YAML | Multi-cloud | Stateless | Configuration management |
| **Helm** | YAML + Templates | Kubernetes | In-cluster | K8s packages |
| **Kustomize** | YAML + Overlays | Kubernetes | Stateless | K8s configuration |

### Table 8: Security Components in GitOps

| Component | Purpose | Tools | Integration Point |
|-----------|---------|-------|-------------------|
| **Secret Management** | Secure credentials | Sealed Secrets, External Secrets | Git repository |
| **Image Scanning** | Vulnerability detection | Trivy, Snyk, Clair | CI pipeline |
| **Policy Enforcement** | Compliance checks | OPA, Kyverno, Gatekeeper | Admission controller |
| **RBAC** | Access control | K8s RBAC, IAM | Cluster |
| **Network Policies** | Traffic control | Calico, Cilium | Kubernetes |
| **Audit Logging** | Change tracking | K8s audit, Git history | Multiple |
| **Signing** | Artifact verification | Cosign, Notary | Container registry |
| **SAST** | Code analysis | SonarQube, CodeQL | CI pipeline |

---

## Complete GitOps Workflow Example

### Scenario: Deploy New Application Version

**Step 1: Developer Makes Changes**

```bash
# Create feature branch
git checkout -b feature/update-version

# Update application code
vim src/app.py

# Update Docker image version
vim k8s/base/deployment.yaml

# Commit changes
git add .
git commit -m "feat: update application to v1.2.0"

# Push to remote
git push origin feature/update-version
```

**Step 2: Create Pull Request**

```yaml
# GitHub Actions runs automatically
name: CI Pipeline
on:
  pull_request:
    branches: [ main ]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Lint code
        run: npm run lint
      
      - name: Run tests
        run: npm test
      
      - name: Build Docker image
        run: docker build -t myapp:pr-${{ github.event.number }} .
      
      - name: Scan image
        run: trivy image myapp:pr-${{ github.event.number }}
      
      - name: Validate K8s manifests
        run: kustomize build k8s/overlays/prod | kubeconform -
```

**Step 3: Code Review and Approval**

```markdown
# PR Review Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass
- [ ] Security scan clean
- [ ] Documentation updated
- [ ] Approved by 2 reviewers
```

**Step 4: Merge to Main**

```bash
# After approval, merge PR
git checkout main
git merge feature/update-version
git push origin main
```

**Step 5: CI/CD Pipeline Builds and Pushes**

```yaml
name: Build and Deploy
on:
  push:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Build image
        run: docker build -t myapp:${{ github.sha }} .
      
      - name: Push to registry
        run: docker push myapp:${{ github.sha }}
      
      - name: Update GitOps repo
        run: |
          git clone https://github.com/org/gitops-repo.git
          cd gitops-repo/k8s/overlays/prod
          kustomize edit set image myapp=myapp:${{ github.sha }}
          git commit -am "Deploy myapp:${{ github.sha }}"
          git push
```

**Step 6: GitOps Agent Syncs**

```bash
# ArgoCD detects change in Git
# Reconciliation loop:
# 1. Fetch latest from Git
# 2. Compare with cluster state
# 3. Apply differences
# 4. Monitor health

# View sync status
argocd app get myapp
```

**Step 7: Verification**

```bash
# Check deployment
kubectl get deployment myapp -n production

# Check pods
kubectl get pods -n production -l app=myapp

# Check logs
kubectl logs -n production -l app=myapp --tail=100

# Verify health
curl https://myapp.example.com/health
```

**Step 8: Monitoring**

```yaml
# Prometheus alerts fire if issues detected
- alert: DeploymentFailed
  expr: kube_deployment_status_replicas_unavailable > 0
  for: 5m

- alert: HighErrorRate
  expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
  for: 10m
```

**Step 9: Rollback (if needed)**

```bash
# Option 1: Git revert
git revert HEAD
git push

# Option 2: ArgoCD rollback
argocd app rollback myapp 0

# Option 3: Manual kubectl
kubectl rollout undo deployment/myapp -n production
```

---

## References

<div style="line-height: 1.8;">

1. <a href="https://www.gitops.tech/" target="_blank">GitOps - Official Definition and Principles</a>

2. <a href="https://opengitops.dev/" target="_blank">OpenGitOps - GitOps Principles and Standards</a>

3. <a href="https://argo-cd.readthedocs.io/" target="_blank">Argo CD Documentation</a>

4. <a href="https://fluxcd.io/docs/" target="_blank">Flux Documentation</a>

5. <a href="https://docs.github.com/en/actions" target="_blank">GitHub Actions Documentation</a>

6. <a href="https://docs.gitlab.com/ee/ci/" target="_blank">GitLab CI/CD Documentation</a>

7. <a href="https://git-scm.com/doc" target="_blank">Git Official Documentation</a>

8. <a href="https://kubernetes.io/docs/home/" target="_blank">Kubernetes Documentation</a>

9. <a href="https://kustomize.io/" target="_blank">Kustomize Documentation</a>

10. <a href="https://helm.sh/docs/" target="_blank">Helm Documentation</a>

11. <a href="https://www.terraform.io/docs" target="_blank">Terraform Documentation</a>

12. <a href="https://www.weave.works/technologies/gitops/" target="_blank">Weaveworks - GitOps</a>

13. <a href="https://about.gitlab.com/topics/gitops/" target="_blank">GitLab - GitOps Guide</a>

14. <a href="https://www.cncf.io/blog/2021/03/12/introduction-to-gitops/" target="_blank">CNCF - Introduction to GitOps</a>

15. <a href="https://github.com/open-gitops/documents" target="_blank">OpenGitOps - GitOps Principles Documentation</a>

16. <a href="https://argoproj.github.io/argo-rollouts/" target="_blank">Argo Rollouts - Progressive Delivery</a>

17. <a href="https://flagger.app/" target="_blank">Flagger - Progressive Delivery Operator</a>

18. <a href="https://www.openpolicyagent.org/docs/latest/" target="_blank">Open Policy Agent Documentation</a>

19. <a href="https://external-secrets.io/" target="_blank">External Secrets Operator Documentation</a>

20. <a href="https://sealed-secrets.netlify.app/" target="_blank">Sealed Secrets Documentation</a>

</div>

---

## Summary

GitOps is a powerful operational framework that leverages Git as the single source of truth for declarative infrastructure and applications. By treating infrastructure and application configuration as code stored in Git repositories, teams can achieve:

### Key Benefits

- **Increased Velocity**: Faster deployments with automated pipelines
- **Improved Stability**: Declarative configurations reduce errors
- **Enhanced Security**: Audit trails, RBAC, and secret management
- **Better Collaboration**: Git-based workflows enable code review
- **Disaster Recovery**: Complete system state in Git enables easy restoration
- **Compliance**: Full audit trail of all changes

### Core Principles

1. **Declarative**: System's desired state described declaratively
2. **Versioned**: All configuration stored in Git with full history
3. **Automated**: Software agents automatically apply desired state
4. **Reconciled**: Continuous monitoring and drift correction

### Essential Components

- **Git**: Version control and source of truth
- **CI/CD**: Automated pipelines for building and testing
- **GitOps Agent**: ArgoCD, Flux, or similar tools
- **Kubernetes**: Target platform for deployments
- **IaC Tools**: Terraform, Helm, Kustomize

### Best Practices

- Separate application code from configuration
- Use directories, not branches, for environments
- Implement comprehensive testing
- Secure secrets with external management
- Monitor GitOps metrics and health
- Document processes and maintain runbooks
- Practice rollback procedures regularly

GitOps represents a paradigm shift in how we manage and deploy applications, bringing the best practices of software development to operations. By embracing GitOps, teams can build more reliable, secure, and scalable systems.
   What is GitOps?

GitOps treats infrastructure and application configuration as code, stored in Git repositories. All changes to infrastructure and applications are made through Git commits and pull requests, triggering automated processes that synchronize the desired state (in Git) with the actual state (in production).

### Key Characteristics

- **Declarative**: Define the desired state of your system rather than imperative instructions
- **Versioned and Immutable**: All changes are tracked in Git with complete history
- **Automatically Applied**: Automated agents continuously reconcile actual state with desired state
- **Continuously Reconciled**: Systems self-heal by detecting and correcting drift

### When to Use GitOps

✅ **Ideal For**:
- Kubernetes and container orchestration
- Cloud-native applications
- Microservices architectures
- Infrastructure as Code (IaC) deployments
- Multi-environment management
- Teams requiring audit trails and compliance

❌ **Not Ideal For**:
- Legacy monolithic applications without automation
- Simple static websites
- One-off deployments without version control needs

---

## Core Concepts

### Single Source of Truth

Git serves as the canonical source for both application code and infrastructure configuration. Every aspect of your system's desired state is stored in Git repositories.

**Benefits**:
- Complete audit trail of all changes
- Easy rollback to any previous state
- Clear separation of concerns
- Disaster recovery capabilities

```yaml
# Example: Kubernetes deployment stored in Git
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-application
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-application
  template:
    metadata:
      labels:
        app: my-application
    spec:
      containers:
      - name: app
        image: myapp:v1.2.3
        ports:
        - containerPort: 8080
```

### Declarative Configuration

Describe **what** you want, not **how** to achieve it. The system determines the necessary steps to reach the desired state.

**Imperative vs Declarative**:

```bash
# Imperative (how to do it)
kubectl create namespace production
kubectl create deployment my-app --image=myapp:1.0
kubectl scale deployment my-app --replicas=3
kubectl expose deployment my-app --port=8080

# Declarative (what you want)
kubectl apply -f production-deployment.yaml
```

### Continuous Reconciliation

Automated agents continuously monitor the actual state and compare it with the desired state in Git. Any drift is automatically corrected.

**Reconciliation Loop**:
1. **Observe**: Monitor actual state of infrastructure
2. **Compare**: Check against desired state in Git
3. **Detect Drift**: Identify differences
4. **Remediate**: Automatically apply changes to align states
5. **Repeat**: Continuously monitor

### Pull vs Push Deployment

**Traditional Push Model** (CI/CD):
- CI/CD pipeline pushes changes to production
- Requires cluster credentials in CI/CD system
- Pipeline has write access to production

**GitOps Pull Model**:
- Agent inside cluster pulls changes from Git
- No external system needs cluster access
- Improved security posture
- Self-healing capabilities

```
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│  Developer  │────────▶│   Git Repo  │◀────────│  GitOps     │
│             │  commit │             │  pull   │  Agent      │
└─────────────┘         └─────────────┘         └──────┬──────┘
                                                        │
                                                        │ apply
                                                        ▼
                                                ┌─────────────┐
                                                │  Kubernetes │
                                                │   Cluster   │
                                                └─────────────┘
```

---

## GitOps Principles

### 1. Declarative Description

The entire system's desired state is described declaratively in a format that machines can parse and understand (YAML, JSON, HCL, etc.).

**Example - Terraform Configuration**:

```hcl
resource "aws_instance" "web_server" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t3.micro"
  
  tags = {
    Name = "WebServer"
    Environment = "Production"
  }
}
```

### 2. Versioned and Immutable

All desired state is stored in a version control system that provides versioning, immutability, and audit trails.

**Git Provides**:
- Complete change history
- Author attribution
- Timestamps
- Commit messages explaining changes
- Ability to revert to any previous state

```bash
# View change history
git log --oneline --graph

# See what changed
git diff HEAD~1 deployment.yaml

# Revert to previous version
git revert abc123
```

### 3. Pulled Automatically

Software agents automatically pull the desired state declarations from Git and apply them to the infrastructure.

**GitOps Agents**:
- **ArgoCD**: Kubernetes-native continuous delivery
- **Flux**: GitOps operator for Kubernetes
- **Jenkins X**: Cloud-native CI/CD for Kubernetes
- **Terraform Cloud**: IaC automation platform

### 4. Continuously Reconciled

Software agents continuously observe actual system state and attempt to apply the desired state.

**Drift Detection and Correction**:

```yaml
# Desired state in Git: 3 replicas
spec:
  replicas: 3

# Actual state: 5 replicas (manually scaled)
# GitOps agent detects drift and corrects to 3 replicas
```

---

## Git Fundamentals

### Git Basics

Git is a distributed version control system that tracks changes in source code during software development.

#### Key Concepts

**Repository**: Directory containing your project files and Git metadata

```bash
# Initialize new repository
git init

# Clone existing repository
git clone https://github.com/username/repo.git
```

**Commit**: Snapshot of your repository at a specific point in time

```bash
# Stage files
git add filename.yaml

# Commit with message
git commit -m "Add production deployment configuration"

# View commit history
git log
```

**Branch**: Parallel version of your repository

```bash
# Create new branch
git branch feature/new-deployment

# Switch to branch
git checkout feature/new-deployment

# Create and switch in one command
git checkout -b feature/new-deployment

# List branches
git branch -a
```

**Merge**: Integrate changes from one branch into another

```bash
# Merge feature branch into main
git checkout main
git merge feature/new-deployment
```

**Tag**: Named reference to a specific commit (often used for releases)

```bash
# Create tag
git tag -a v1.0.0 -m "Release version 1.0.0"

# Push tags to remote
git push origin --tags

# List tags
git tag -l
```

### Git Workflow

```bash
# 1. Update local repository
git pull origin main

# 2. Create feature branch
git checkout -b feature/update-deployment

# 3. Make changes to files
vim deployment.yaml

# 4. Stage changes
git add deployment.yaml

# 5. Commit changes
git commit -m "Update deployment replicas to 5"

# 6. Push to remote
git push origin feature/update-deployment

# 7. Create pull request (on GitHub/GitLab)
# 8. Review and merge
# 9. Delete feature branch
git branch -d feature/update-deployment
```

### Git Configuration

```bash
# Set user information
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Set default branch name
git config --global init.defaultBranch main

# Configure editor
git config --global core.editor "vim"

# View configuration
git config --list
```

### Git Best Practices

#### Commit Messages

**Good Commit Messages**:
```
Add Kubernetes deployment for user service

- Configure 3 replicas for high availability
- Set resource limits: 500m CPU, 512Mi memory
- Add health checks on /health endpoint
```

**Bad Commit Messages**:
```
update
fix stuff
changes
```

**Conventional Commits Format**:
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

**Example**:
```
feat(deployment): add horizontal pod autoscaling

Configure HPA to scale between 3-10 replicas based on CPU
utilization target of 70%. This improves application availability
during traffic spikes.

Closes #123
```

#### Branching Hygiene

```bash
# Keep branches short-lived
# Delete merged branches
git branch -d feature/completed-feature

# Prune remote-tracking branches
git fetch --prune

# Clean up old branches
git branch --merged | grep -v "\*" | xargs -n 1 git branch -d
```

---

## GitHub and GitHub Actions

### GitHub Basics

GitHub is a web-based hosting service for Git repositories with collaboration features.

#### Repository Structure

```
my-project/
├── .github/
│   ├── workflows/          # GitHub Actions workflows
│   │   ├── ci.yml
│   │   └── deploy.yml
│   ├── CODEOWNERS          # Code review assignments
│   └── dependabot.yml      # Dependency updates
├── .gitignore              # Files to ignore
├── README.md               # Project documentation
├── LICENSE                 # License file
└── src/                    # Application code
```

#### GitHub Features for GitOps

**Pull Requests**: Code review and collaboration mechanism

```yaml
# Example: PR template (.github/pull_request_template.md)
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
```

**Protected Branches**: Enforce code quality standards

```yaml
# Branch protection rules
main:
  required_reviews: 2
  require_code_owner_review: true
  dismiss_stale_reviews: true
  require_status_checks: true
  required_status_checks:
    - ci/lint
    - ci/test
    - ci/security-scan
  enforce_admins: true
  restrict_pushes: true
```

**Code Owners**: Automatic reviewer assignment

```
# .github/CODEOWNERS
# Global owners
* @team-leads

# Infrastructure files
/terraform/ @platform-team @sre-team
/kubernetes/ @platform-team

# Application code
/src/ @dev-team

# Documentation
/docs/ @doc-team @dev-team
```

### GitHub Actions

GitHub Actions is a CI/CD platform integrated directly into GitHub.

#### Workflow Components

**Workflow**: Automated process defined in YAML

**Event**: Triggers that start workflows (push, pull_request, schedule, etc.)

**Job**: Set of steps executed on the same runner

**Step**: Individual task (run command, use action)

**Action**: Reusable unit of code

**Runner**: Server that executes workflows

#### Basic Workflow Structure

```yaml
# .github/workflows/ci.yml
name: CI Pipeline

# Events that trigger workflow
on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

# Environment variables
env:
  NODE_VERSION: '18'

# Jobs to run
jobs:
  test:
    name: Run Tests
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'
      
      - name: Install dependencies
        run: npm ci
      
      - name: Run linter
        run: npm run lint
      
      - name: Run tests
        run: npm test
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage/coverage.xml
```

#### Advanced Workflow Features

**Matrix Builds**: Test across multiple configurations

```yaml
jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        node-version: [16, 18, 20]
        exclude:
          - os: macos-latest
            node-version: 16
    
    steps:
      - uses: actions/checkout@v4
      - name: Setup Node.js ${{ matrix.node-version }}
        uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node-version }}
      - run: npm ci
      - run: npm test
```

**Conditional Execution**:

```yaml
steps:
  - name: Deploy to production
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    run: ./deploy.sh production
  
  - name: Deploy to staging
    if: github.ref == 'refs/heads/develop'
    run: ./deploy.sh staging
```

**Secrets Management**:

```yaml
steps:
  - name: Deploy application
    env:
      AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
      AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
    run: |
      aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID
      aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY
      ./deploy.sh
```

**Caching Dependencies**:

```yaml
steps:
  - uses: actions/checkout@v4
  
  - name: Cache dependencies
    uses: actions/cache@v3
    with:
      path: ~/.npm
      key: ${{ runner.os }}-node-${{ hashFiles('**/package-lock.json') }}
      restore-keys: |
        ${{ runner.os }}-node-
  
  - name: Install dependencies
    run: npm ci
```

**Reusable Workflows**:

```yaml
# .github/workflows/reusable-deploy.yml
name: Reusable Deploy Workflow

on:
  workflow_call:
    inputs:
      environment:
        required: true
        type: string
      version:
        required: true
        type: string
    secrets:
      deploy-key:
        required: true

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: ${{ inputs.environment }}
    steps:
      - uses: actions/checkout@v4
      - name: Deploy
        run: ./deploy.sh ${{ inputs.environment }} ${{ inputs.version }}
        env:
          DEPLOY_KEY: ${{ secrets.deploy-key }}
```

```yaml
# .github/workflows/main.yml
name: Deploy Application

on:
  push:
    branches: [ main ]

jobs:
  deploy-staging:
    uses: ./.github/workflows/reusable-deploy.yml
    with:
      environment: staging
      version: ${{ github.sha }}
    secrets:
      deploy-key: ${{ secrets.STAGING_DEPLOY_KEY }}
  
  deploy-production:
    needs: deploy-staging
    uses: ./.github/workflows/reusable-deploy.yml
    with:
      environment: production
      version: ${{ github.sha }}
    secrets:
      deploy-key: ${{ secrets.PROD_DEPLOY_KEY }}
```

**Parallel Jobs**:

```yaml
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm run lint
  
  test-unit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm run test:unit
  
  test-integration:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm run test:integration
  
  deploy:
    needs: [lint, test-unit, test-integration]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: ./deploy.sh
```

**Service Containers**:

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
      
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379
    
    steps:
      - uses: actions/checkout@v4
      - name: Run tests
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost:5432/testdb
          REDIS_URL: redis://localhost:6379
        run: npm test
```

**Self-Hosted Runners**:

```yaml
jobs:
  build:
    runs-on: [self-hosted, linux, x64, gpu]
    steps:
      - uses: actions/checkout@v4
      - name: Build with GPU
        run: ./build-with-cuda.sh
```

#### Complete CI/CD Workflow Example

```yaml
name: Complete CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  release:
    types: [published]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  # Code quality checks
  lint:
    name: Lint Code
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '18'
          cache: 'npm'
      
      - name: Install dependencies
        run: npm ci
      
      - name: Run ESLint
        run: npm run lint
      
      - name: Run Prettier
        run: npm run format:check
  
  # Security scanning
  security:
    name: Security Scan
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
          format: 'sarif'
          output: 'trivy-results.sarif'
      
      - name: Upload to GitHub Security
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-results.sarif'
  
  # Unit and integration tests
  test:
    name: Run Tests
    runs-on: ubuntu-latest
    strategy:
      matrix:
        node-version: [16, 18, 20]
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Node.js ${{ matrix.node-version }}
        uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node-version }}
          cache: 'npm'
      
      - name: Install dependencies
        run: npm ci
      
      - name: Run tests
        run: npm test -- --coverage
      
      - name: Upload coverage
        if: matrix.node-version == '18'
        uses: codecov/codecov-action@v3
  
  # Build and push Docker image
  build:
    name: Build Image
    needs: [lint, security, test]
    runs-on: ubuntu-latest
    if: github.event_name != 'pull_request'
    permissions:
      contents: read
      packages: write
    
    outputs:
      image-tag: ${{ steps.meta.outputs.tags }}
    
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
            type=sha
      
      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
  
  # Deploy to staging
  deploy-staging:
    name: Deploy to Staging
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'
    environment:
      name: staging
      url: https://staging.example.com
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Update Kubernetes manifests
        run: |
          cd k8s/staging
          kustomize edit set image app=${{ needs.build.outputs.image-tag }}
      
      - name: Commit changes
        run: |
          git config user.name github-actions
          git config user.email github-actions@github.com
          git add .
          git commit -m "Deploy ${{ github.sha }} to staging"
          git push
  
  # Deploy to production
  deploy-production:
    name: Deploy to Production
    needs: build
    runs-on: ubuntu-latest
    if: github.event_name == 'release'
    environment:
      name: production
      url: https://example.com
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Update Kubernetes manifests
        run: |
          cd k8s/production
          kustomize edit set image app=${{ needs.build.outputs.image-tag }}
      
      - name: Commit changes
        run: |
          git config user.name github-actions
          git config user.email github-actions@github.com
          git add .
          git commit -m "Deploy ${{ github.event.release.tag_name }} to production"
          git push
```

---

## GitLab CI/CD

### GitLab Overview

GitLab is a complete DevOps platform with built-in CI/CD capabilities.

#### GitLab CI/CD Configuration

GitLab uses `.gitlab-ci.yml` file in the repository root.

```yaml
# .gitlab-ci.yml
stages:
  - build
  - test
  - deploy

variables:
  DOCKER_REGISTRY: registry.gitlab.com
  IMAGE_NAME: $CI_REGISTRY_IMAGE

# Build stage
build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -t $IMAGE_NAME:$CI_COMMIT_SHA .
    - docker push $IMAGE_NAME:$CI_COMMIT_SHA
  only:
    - main
    - develop

# Test stage
test:
  stage: test
  image: node:18
  cache:
    paths:
      - node_modules/
  script:
    - npm ci
    - npm run lint
    - npm test
  coverage: '/Lines\s*:\s*(\d+\.\d+)%/'
  artifacts:
    reports:
      junit: junit.xml
      coverage_report:
        coverage_format: cobertura
        path: coverage/cobertura-coverage.xml

# Deploy to staging
deploy-staging:
  stage: deploy
  image: bitnami/kubectl:latest
  script:
    - kubectl set image deployment/my-app app=$IMAGE_NAME:$CI_COMMIT_SHA
    - kubectl rollout status deployment/my-app
  environment:
    name: staging
    url: https://staging.example.com
  only:
    - develop

# Deploy to production
deploy-production:
  stage: deploy
  image: bitnami/kubectl:latest
  script:
    - kubectl set image deployment/my-app app=$IMAGE_NAME:$CI_COMMIT_SHA
    - kubectl rollout status deployment/my-app
  environment:
    name: production
    url: https://example.com
  when: manual
  only:
    - main
```

#### GitLab Features

**Auto DevOps**: Automated CI/CD pipeline

**Container Registry**: Built-in Docker registry

**Kubernetes Integration**: Native K8s deployment

**Security Scanning**: SAST, DAST, dependency scanning

**Merge Requests**: Code review process

**Protected Branches**: Enforce merge requirements

---

## Repository Structure

### Separation of Concerns

**Best Practice**: Separate application code from infrastructure configuration.

**Reasons**:
1. **Different lifecycles**: Applications update frequently, infrastructure changes rarely
2. **Different release cadences**: App releases (daily/weekly) vs infrastructure (monthly/quarterly)
3. **Different teams**: Developers vs Platform/SRE teams
4. **Different approval processes**: Code review vs infrastructure approval
5. **Security and access control**: Not all developers need infrastructure access
6. **Configuration changes shouldn't trigger app rebuilds**: Avoid unnecessary CI runs

**Anti-Pattern**: Monolithic repository mixing application code, infrastructure, and configurations creates complexity and coupling.

### Repository Organization Patterns

#### Pattern 1: Monorepo for Small Projects

**Use Case**: Small teams, single application, limited environments

```
project/
├── .github/
│   └── workflows/
│       ├── ci.yml                    # Build and test
│       ├── deploy-dev.yml            # Deploy to dev
│       ├── deploy-staging.yml        # Deploy to staging
│       └── deploy-prod.yml           # Deploy to production
├── src/                              # Application source code
│   ├── frontend/
│   │   ├── components/
│   │   ├── pages/
│   │   └── package.json
│   ├── backend/
│   │   ├── api/
│   │   ├── services/
│   │   └── requirements.txt
│   └── shared/
│       └── utils/
├── tests/                            # Test suites
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── infrastructure/                   # Infrastructure as Code
│   ├── terraform/
│   │   ├── modules/
│   │   │   ├── vpc/
│   │   │   │   ├── main.tf
│   │   │   │   ├── variables.tf
│   │   │   │   └── outputs.tf
│   │   │   ├── eks/
│   │   │   │   ├── main.tf
│   │   │   │   ├── variables.tf
│   │   │   │   └── outputs.tf
│   │   │   └── rds/
│   │   │       ├── main.tf
│   │   │       ├── variables.tf
│   │   │       └── outputs.tf
│   │   └── environments/
│   │       ├── dev/
│   │       │   ├── main.tf
│   │       │   ├── variables.tf
│   │       │   ├── terraform.tfvars
│   │       │   └── backend.tf
│   │       ├── staging/
│   │       │   ├── main.tf
│   │       │   ├── variables.tf
│   │       │   ├── terraform.tfvars
│   │       │   └── backend.tf
│   │       └── prod/
│   │           ├── main.tf
│   │           ├── variables.tf
│   │           ├── terraform.tfvars
│   │           └── backend.tf
│   └── kubernetes/                   # K8s manifests
│       ├── base/
│       │   ├── deployment.yaml
│       │   ├── service.yaml
│       │   ├── configmap.yaml
│       │   ├── secrets.yaml
│       │   └── kustomization.yaml
│       └── overlays/
│           ├── dev/
│           │   ├── kustomization.yaml
│           │   ├── patch-replicas.yaml
│           │   ├── patch-resources.yaml
│           │   └── configmap-patch.yaml
│           ├── staging/
│           │   ├── kustomization.yaml
│           │   ├── patch-replicas.yaml
│           │   ├── patch-resources.yaml
│           │   └── configmap-patch.yaml
│           └── prod/
│               ├── kustomization.yaml
│               ├── patch-replicas.yaml
│               ├── patch-resources.yaml
│               ├── hpa.yaml
│               └── configmap-patch.yaml
├── docs/                             # Documentation
│   ├── architecture/
│   │   ├── diagrams/
│   │   └── decisions/
│   ├── api/
│   ├── deployment/
│   └── runbooks/
├── scripts/                          # Utility scripts
│   ├── build.sh
│   ├── test.sh
│   └── deploy.sh
├── .gitignore
├── .dockerignore
├── Dockerfile
├── docker-compose.yml
├── README.md
└── CHANGELOG.md
```

**Pros**:
- Single source of truth
- Easy to navigate
- Atomic changes across code and config
- Simpler CI/CD setup

**Cons**:
- Can become cluttered
- Mixed permissions difficult
- Code changes trigger infrastructure pipelines
- Not suitable for multiple applications

---

#### Pattern 2: Multi-Repo for Enterprise

**Use Case**: Large teams, multiple applications, strict separation of concerns

##### Application Repository

```
app-user-service/
├── .github/
│   └── workflows/
│       ├── ci.yml                    # Build, test, scan
│       ├── build-image.yml           # Build Docker image
│       └── security-scan.yml         # Security scanning
├── src/                              # Application code
│   ├── main/
│   │   └── java/
│   │       └── com/
│   │           └── example/
│   │               ├── controller/
│   │               ├── service/
│   │               ├── repository/
│   │               └── model/
│   └── test/
│       └── java/
│           └── com/
│               └── example/
├── build.gradle                      # Build configuration
├── Dockerfile                        # Container image
├── .dockerignore
├── README.md
└── CHANGELOG.md
```

**Purpose**: Contains only application source code, tests, and build configuration.

**CI/CD Responsibilities**:
1. Run unit and integration tests
2. Build Docker image
3. Push image to container registry
4. Update image tag in GitOps repository
5. Run security scans

**Example CI Workflow**:

```yaml
# .github/workflows/ci.yml
name: Build and Push Image

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up JDK
        uses: actions/setup-java@v4
        with:
          java-version: '17'
          distribution: 'temurin'
      
      - name: Run tests
        run: ./gradlew test
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
  
  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push'
    permissions:
      contents: read
      packages: write
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Log in to registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
      
      - name: Update GitOps repo
        run: |
          git clone https://${{ secrets.GITOPS_PAT }}@github.com/org/gitops-config.git
          cd gitops-config/apps/user-service/overlays/dev
          kustomize edit set image user-service=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          git config user.name "github-actions"
          git config user.email "github-actions@github.com"
          git add .
          git commit -m "Update user-service image to ${{ github.sha }}"
          git push
```

---

##### Infrastructure Repository

```
infrastructure/
├── .github/
│   └── workflows/
│       ├── terraform-plan.yml        # Plan on PR
│       ├── terraform-apply.yml       # Apply on merge
│       └── terraform-destroy.yml     # Manual destroy
├── terraform/
│   ├── modules/                      # Reusable modules
│   │   ├── vpc/
│   │   │   ├── main.tf
│   │   │   ├── variables.tf
│   │   │   ├── outputs.tf
│   │   │   └── README.md
│   │   ├── eks/
│   │   │   ├── main.tf
│   │   │   ├── variables.tf
│   │   │   ├── outputs.tf
│   │   │   └── README.md
│   │   ├── rds/
│   │   │   ├── main.tf
│   │   │   ├── variables.tf
│   │   │   ├── outputs.tf
│   │   │   └── README.md
│   │   ├── s3/
│   │   │   ├── main.tf
│   │   │   ├── variables.tf
│   │   │   ├── outputs.tf
│   │   │   └── README.md
│   │   └── monitoring/
│   │       ├── main.tf
│   │       ├── variables.tf
│   │       ├── outputs.tf
│   │       └── README.md
│   ├── environments/
│   │   ├── dev/
│   │   │   ├── main.tf
│   │   │   ├── variables.tf
│   │   │   ├── terraform.tfvars
│   │   │   ├── backend.tf
│   │   │   └── outputs.tf
│   │   ├── staging/
│   │   │   ├── main.tf
│   │   │   ├── variables.tf
│   │   │   ├── terraform.tfvars
│   │   │   ├── backend.tf
│   │   │   └── outputs.tf
│   │   └── prod/
│   │       ├── main.tf
│   │       ├── variables.tf
│   │       ├── terraform.tfvars
│   │       ├── backend.tf
│   │       └── outputs.tf
│   └── global/
│       ├── iam/
│       │   ├── main.tf
│       │   └── variables.tf
│       └── route53/
│           ├── main.tf
│           └── variables.tf
├── ansible/                          # Configuration management
│   ├── inventories/
│   │   ├── dev/
│   │   ├── staging/
│   │   └── prod/
│   ├── playbooks/
│   └── roles/
├── docs/
│   ├── architecture.md
│   ├── disaster-recovery.md
│   └── runbooks/
├── scripts/
│   ├── init-backend.sh
│   └── validate.sh
└── README.md
```

**Purpose**: Manages cloud infrastructure, networking, databases, and platform services.

**CI/CD Responsibilities**:
1. Validate Terraform syntax
2. Run security scans (Checkov, tfsec)
3. Generate and review Terraform plans
4. Apply infrastructure changes
5. Update documentation

**Example Terraform Workflow**:

```yaml
# .github/workflows/terraform.yml
name: Terraform

on:
  pull_request:
    branches: [ main ]
    paths:
      - 'terraform/**'
  push:
    branches: [ main ]
    paths:
      - 'terraform/**'

jobs:
  terraform:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        environment: [dev, staging, prod]
    
    defaults:
      run:
        working-directory: terraform/environments/${{ matrix.environment }}
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: 1.6.0
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
      
      - name: Terraform Format
        run: terraform fmt -check -recursive
      
      - name: Terraform Init
        run: terraform init
      
      - name: Terraform Validate
        run: terraform validate
      
      - name: Run Checkov
        uses: bridgecrewio/checkov-action@master
        with:
          directory: terraform/environments/${{ matrix.environment }}
          soft_fail: true
      
      - name: Terraform Plan
        id: plan
        run: terraform plan -no-color -out=tfplan
        continue-on-error: true
      
      - name: Comment PR
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          script: |
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: '#### Terraform Plan for ${{ matrix.environment }}\n```\n${{ steps.plan.outputs.stdout }}\n```'
            })
      
      - name: Terraform Apply
        if: github.ref == 'refs/heads/main' && github.event_name == 'push'
        run: terraform apply -auto-approve tfplan
```

---

##### Configuration Repository (GitOps)

```
gitops-config/
├── .github/
│   └── workflows/
│       ├── validate.yml              # Validate manifests
│       └── policy-check.yml          # OPA policy checks
├── clusters/                         # Cluster-specific configs
│   ├── dev-cluster/
│   │   ├── flux-system/
│   │   │   ├── gotk-components.yaml
│   │   │   ├── gotk-sync.yaml
│   │   │   └── kustomization.yaml
│   │   ├── infrastructure/
│   │   │   ├── ingress-nginx/
│   │   │   ├── cert-manager/
│   │   │   └── external-secrets/
│   │   └── apps/
│   │       ├── kustomization.yaml
│   │       └── app-references.yaml
│   ├── staging-cluster/
│   │   ├── flux-system/
│   │   ├── infrastructure/
│   │   └── apps/
│   └── prod-cluster/
│       ├── flux-system/
│       ├── infrastructure/
│       └── apps/
├── apps/                             # Application manifests
│   ├── user-service/
│   │   ├── base/
│   │   │   ├── namespace.yaml
│   │   │   ├── deployment.yaml
│   │   │   ├── service.yaml
│   │   │   ├── configmap.yaml
│   │   │   ├── servicemonitor.yaml
│   │   │   └── kustomization.yaml
│   │   └── overlays/
│   │       ├── dev/
│   │       │   ├── kustomization.yaml
│   │       │   ├── patch-replicas.yaml
│   │       │   ├── patch-resources.yaml
│   │       │   └── configmap-values.yaml
│   │       ├── staging/
│   │       │   ├── kustomization.yaml
│   │       │   ├── patch-replicas.yaml
│   │       │   ├── patch-resources.yaml
│   │       │   └── configmap-values.yaml
│   │       └── prod/
│   │           ├── kustomization.yaml
│   │           ├── patch-replicas.yaml
│   │           ├── patch-resources.yaml
│   │           ├── hpa.yaml
│   │           ├── pdb.yaml
│   │           └── configmap-values.yaml
│   ├── payment-service/
│   │   ├── base/
│   │   └── overlays/
│   ├── notification-service/
│   │   ├── base/
│   │   └── overlays/
│   └── frontend/
│       ├── base/
│       └── overlays/
├── infrastructure/                   # Platform services
│   ├── ingress-nginx/
│   │   ├── base/
│   │   │   ├── namespace.yaml
│   │   │   ├── helmrelease.yaml
│   │   │   └── kustomization.yaml
│   │   └── overlays/
│   │       ├── dev/
│   │       ├── staging/
│   │       └── prod/
│   ├── cert-manager/
│   │   ├── base/
│   │   └── overlays/
│   ├── external-secrets/
│   │   ├── base/
│   │   └── overlays/
│   ├── monitoring/
│   │   ├── prometheus/
│   │   ├── grafana/
│   │   └── alertmanager/
│   └── logging/
│       ├── loki/
│       └── promtail/
├── policies/                         # OPA/Kyverno policies
│   ├── pod-security/
│   │   ├── require-non-root.yaml
│   │   ├── drop-capabilities.yaml
│   │   └── readonly-root-fs.yaml
│   ├── resource-limits/
│   │   └── require-limits.yaml
│   └── network/
│       └── deny-default.yaml
├── docs/
│   ├── architecture.md
│   ├── deployment-guide.md
│   └── troubleshooting.md
└── README.md
```

**Purpose**: Contains all Kubernetes manifests and configurations for applications and infrastructure.

**Key Principles**:
1. **Declarative**: Everything defined as YAML manifests
2. **Versioned**: All changes tracked in Git
3. **Separate by environment**: Use overlays, not branches
4. **GitOps agent managed**: Flux or ArgoCD syncs from this repo

**Example Base Deployment**:

```yaml
# apps/user-service/base/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: user-service
  labels:
    app: user-service
    version: v1
spec:
  replicas: 3
  selector:
    matchLabels:
      app: user-service
  template:
    metadata:
      labels:
        app: user-service
        version: v1
    spec:
      serviceAccountName: user-service
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 2000
      containers:
      - name: user-service
        image: ghcr.io/org/user-service:latest
        imagePullPolicy: IfNotPresent
        ports:
        - name: http
          containerPort: 8080
          protocol: TCP
        env:
        - name: SPRING_PROFILES_ACTIVE
          value: production
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: user-service-db
              key: url
        envFrom:
        - configMapRef:
            name: user-service-config
        resources:
          requests:
            cpu: 100m
            memory: 256Mi
          limits:
            cpu: 500m
            memory: 512Mi
        livenessProbe:
          httpGet:
            path: /actuator/health/liveness
            port: http
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /actuator/health/readiness
            port: http
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
            - ALL
        volumeMounts:
        - name: tmp
          mountPath: /tmp
      volumes:
      - name: tmp
        emptyDir: {}
```

**Example Dev Overlay**:

```yaml
# apps/user-service/overlays/dev/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

namespace: user-service-dev

bases:
  - ../../base

patches:
  - patch-replicas.yaml
  - patch-resources.yaml

images:
  - name: ghcr.io/org/user-service
    newTag: dev-latest

configMapGenerator:
  - name: user-service-config
    behavior: merge
    literals:
      - LOG_LEVEL=DEBUG
      - ENVIRONMENT=development
      - CACHE_ENABLED=false

labels:
  - pairs:
      environment: dev
      managed-by: flux
```

```yaml
# apps/user-service/overlays/dev/patch-replicas.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: user-service
spec:
  replicas: 1
```

```yaml
# apps/user-service/overlays/dev/patch-resources.yaml
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
            cpu: 50m
            memory: 128Mi
          limits:
            cpu: 200m
            memory: 256Mi
```

**Pros of Multi-Repo**:
- Clear separation of concerns
- Independent versioning and releases
- Fine-grained access control
- Smaller, focused repositories
- Parallel development without conflicts

**Cons of Multi-Repo**:
- More complex to manage
- Requires coordination between repos
- More CI/CD pipelines to maintain
- Cross-repo changes require multiple PRs

---

#### Pattern 3: Hybrid Approach

**Use Case**: Medium-sized teams, multiple related applications

```
organization/
├── apps/
│   ├── user-service/
│   │   ├── src/
│   │   ├── tests/
│   │   ├── Dockerfile
│   │   └── .github/workflows/
│   ├── payment-service/
│   │   ├── src/
│   │   ├── tests/
│   │   ├── Dockerfile
│   │   └── .github/workflows/
│   └── frontend/
│       ├── src/
│       ├── tests/
│       ├── Dockerfile
│       └── .github/workflows/
├── platform/
│   ├── infrastructure/
│   │   └── terraform/
│   └── gitops/
│       ├── clusters/
│       ├── apps/
│       └── infrastructure/
└── shared/
    ├── libraries/
    └── configs/
```

**Pros**:
- Logical grouping
- Shared resources in one place
- Easier discovery
- Moderate complexity

**Cons**:
- Can become large
- Requires clear conventions
- Build times may increase

---

### Directory Structure Best Practices

#### 1. Use Folders, Not Branches for Environments

**❌ Anti-Pattern: Environment Branches**

```
main (production)
├── develop (staging)
│   └── feature-x (development)
└── feature-y
```

**Problems**:
- Difficult to compare environments
- Merge conflicts between environments
- No single source of truth
- Complex promotion process

**✅ Recommended: Directory Structure**

```
configs/
├── base/                     # Common configuration
│   ├── deployment.yaml
│   ├── service.yaml
│   └── kustomization.yaml
└── environments/
    ├── dev/
    │   ├── kustomization.yaml
    │   └── patches/
    ├── staging/
    │   ├── kustomization.yaml
    │   └── patches/
    └── prod/
        ├── kustomization.yaml
        └── patches/
```

**Benefits**:
- All environments visible simultaneously
- Easy to compare configurations
- Simple promotion (copy/modify files)
- Single main branch

#### 2. Organize by Environment, Not by Resource Type

**❌ Poor Organization**:

```
k8s/
├── deployments/
│   ├── app1-dev.yaml
│   ├── app1-prod.yaml
│   ├── app2-dev.yaml
│   └── app2-prod.yaml
├── services/
│   ├── app1-dev.yaml
│   └── app1-prod.yaml
└── configmaps/
    └── ...
```

**✅ Good Organization**:

```
k8s/
├── app1/
│   ├── base/
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   └── configmap.yaml
│   └── overlays/
│       ├── dev/
│       ├── staging/
│       └── prod/
└── app2/
    ├── base/
    └── overlays/
```

#### 3. Keep Configuration DRY with Base and Overlays

**Base Configuration** (shared across environments):

```yaml
# base/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
spec:
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
    spec:
      containers:
      - name: myapp
        image: myapp:latest
        ports:
        - containerPort: 8080
```

**Environment-Specific Overlays**:

```yaml
# overlays/prod/kustomization.yaml
bases:
  - ../../base

patchesStrategicMerge:
  - replica-patch.yaml
  - resource-patch.yaml

images:
  - name: myapp
    newTag: v1.2.3

configMapGenerator:
  - name: app-config
    literals:
      - ENVIRONMENT=production
```

#### 4. WET vs DRY Configuration

**DRY (Don't Repeat Yourself)**: Use templates and generators
```yaml
# Template with variables
replicas: {{ .Values.replicas }}
image: {{ .Values.image }}:{{ .Values.tag }}
```

**Pros**: Less repetition, easier bulk updates
**Cons**: Harder to review changes, requires processing

**WET (Write Everything Twice)**: Explicit configuration files
```yaml
# Explicit YAML for each environment
# dev/deployment.yaml
replicas: 1
image: myapp:dev

# prod/deployment.yaml
replicas: 5
image: myapp:v1.2.3
```

**Pros**: Easy to review, no hidden logic, GitOps-friendly
**Cons**: More files, potential inconsistencies

**Recommendation for GitOps**: Use WET approach
- Changes visible in Git diffs
- No hidden logic or transformations
- GitOps agents apply exactly what's in Git
- Better for auditing and compliance

#### 5. Namespace Organization

**Option A: Namespace per Environment**

```
namespaces/
├── dev/
│   ├── namespace.yaml
│   └── resource-quotas.yaml
├── staging/
│   ├── namespace.yaml
│   └── resource-quotas.yaml
└── prod/
    ├── namespace.yaml
    └── resource-quotas.yaml
```

**Option B: Namespace per Application per Environment**

```
namespaces/
├── user-service-dev/
├── user-service-staging/
├── user-service-prod/
├── payment-service-dev/
├── payment-service-staging/
└── payment-service-prod/
```

**Option C: Separate Clusters per Environment** (Recommended for production)

```
Each environment in its own cluster:
- dev-cluster
- staging-cluster  
- prod-cluster
```

---

### Repository Naming Conventions

**Application Repositories**:
- `app-<service-name>`: Example: `app-user-service`
- `service-<name>`: Example: `service-payment`
- `<team>-<service>`: Example: `platform-api-gateway`

**Infrastructure Repositories**:
- `infra-<platform>`: Example: `infra-aws`
- `infrastructure`: Simple, clear name
- `terraform-<env>`: Example: `terraform-prod` (if env-specific)

**Configuration Repositories**:
- `gitops-<cluster>`: Example: `gitops-prod-cluster`
- `k8s-manifests`: Clear purpose
- `fleet-config`: For multi-cluster setups
- `config-<env>`: Example: `config-production`

---

### File Naming Conventions

**Kubernetes Manifests**:
```
namespace.yaml
deployment.yaml
service.yaml
configmap.yaml
secret.yaml (or sealed-secret.yaml)
ingress.yaml
hpa.yaml (Horizontal Pod Autoscaler)
pdb.yaml (Pod Disruption Budget)
networkpolicy.yaml
serviceaccount.yaml
role.yaml / rolebinding.yaml
```

**Terraform Files**:
```
main.tf           # Primary resources
variables.tf      # Input variables
outputs.tf        # Output values
backend.tf        # Backend configuration
providers.tf      # Provider configuration
terraform.tfvars  # Variable values
versions.tf       # Version constraints
```

**Documentation**:
```
README.md
CHANGELOG.md
CONTRIBUTING.md
ARCHITECTURE.md
DEPLOYMENT.md
TROUBLESHOOTING.md
```

---

### Repository Size Management

**Keep Repositories Focused**:
- Single responsibility principle
- Clear boundaries
- Independent versioning

**Avoid**:
- Multi-thousand file repositories
- Mixing unrelated concerns
- Deep nesting (> 5 levels)

**Split When**:
- Repository > 1000 files
- Multiple independent services
- Different teams/ownership
- Different security requirements
- Build times > 10 minutes

---

