---
layout: post
title: "🌊 DuckDB: Deep Dive & Best Practices with Python"
description: "Concise, clear, and validated revision notes on DuckDB and practical best practices for beginners and practitioners."
author: technical_notes
date: 2025-12-01 00:00:00 +05:30
categories: [Notes, DuckDB]
tags: [DuckDB, Python, OLAP, SQL, Database, Analytics, Data Engineering, Best Practices]
image: /assets/img/posts/duckdb-logo.webp
toc: true
math: false
mermaid: false
---

## Table of Contents
- [Introduction](#introduction)
- [Core Concepts](#core-concepts)
- [Architecture & Query Execution](#architecture--query-execution)
- [Jargon Tables](#jargon-tables)
- [Installation & Setup](#installation--setup)
- [Data Ingestion](#data-ingestion)
- [Python API](#python-api)
- [Query Execution & Optimization](#query-execution--optimization)
- [Advanced Features](#advanced-features)
- [Performance Best Practices](#performance-best-practices)
- [Common Patterns & Use Cases](#common-patterns--use-cases)
- [Limitations & Considerations](#limitations--considerations)
- [References](#references)

---

## Introduction

**DuckDB** is an open-source, in-process OLAP (Online Analytical Processing) database management system designed for analytical workloads. Unlike traditional client-server databases, DuckDB operates as an embedded database that runs directly within the application process, eliminating network overhead and simplifying deployment.

### Key Characteristics

- **In-Process Architecture**: No separate server process required; runs within the host application
- **Column-Oriented Storage**: Optimized for analytical queries with efficient columnar data layout
- **Vectorized Query Execution**: Processes data in batches (vectors) for CPU cache efficiency
- **Zero External Dependencies**: Single binary with no configuration needed
- **ACID Compliant**: Ensures data integrity through transactions
- **MIT Licensed**: Free and open-source

---

## Core Concepts

### OLAP vs OLTP

DuckDB is explicitly designed for **OLAP** workloads, which differ fundamentally from OLTP (Online Transaction Processing):

| Characteristic | OLAP (DuckDB) | OLTP (e.g., PostgreSQL) |
|---|---|---|
| **Workload Type** | Analytical, read-heavy | Transactional, write-heavy |
| **Query Pattern** | Complex aggregations, scans | Simple lookups, updates |
| **Data Volume** | Large datasets (GB to TB) | Smaller, normalized data |
| **Concurrency** | Few concurrent queries | Many concurrent transactions |
| **Storage Layout** | Column-oriented | Row-oriented |
| **Optimization For** | Throughput | Latency |

### In-Process Database

An **in-process database** runs within the same memory space as the application, contrasting with client-server databases:

- **No Network Latency**: Direct memory access to data
- **Simplified Deployment**: Single executable/library
- **Zero Configuration**: No server setup, ports, or authentication
- **Memory Efficiency**: Shared memory space with application
- **Process Lifecycle**: Database lifetime tied to application process

### Vectorized Execution

**Vectorization** is a technique where operations are performed on arrays (vectors) of values rather than individual values:

- **Batch Processing**: Operates on chunks of 2048 rows (default) called **DataChunks**
- **CPU Cache Efficiency**: Better cache locality and reduced instruction overhead
- **SIMD Utilization**: Leverages Single Instruction Multiple Data CPU instructions
- **Reduced Interpretation**: Fewer function calls and branches per row

---

## Architecture & Query Execution

### Query Processing Pipeline

DuckDB processes queries through several distinct phases:

```
SQL String → Parser → Binder → Planner → Optimizer → Physical Plan → Execution → Results
```

#### 1. **Parser Phase**

The parser converts SQL text into an Abstract Syntax Tree (AST):

- **Tokenization**: Breaks SQL into tokens (keywords, identifiers, operators)
- **Grammar Validation**: Ensures syntactic correctness
- **AST Generation**: Creates ParsedExpression, TableRef, QueryNode objects
- **Schema-Agnostic**: Does not validate table/column existence yet

**Key Classes**:
- `ParsedExpression`: Represents expressions (column refs, operators, constants)
- `TableRef`: Represents table sources (base tables, joins, subqueries)
- `QueryNode`: Represents SELECT statements or set operations
- `SQLStatement`: Complete SQL statement with statement type

#### 2. **Binder Phase**

The binder resolves references using the catalog:

- **Catalog Resolution**: Validates tables and columns exist
- **Type Resolution**: Determines data types for expressions
- **Bound Nodes**: Converts parsed nodes to BoundColumnRefExpression
- **Semantic Validation**: Ensures logical correctness

#### 3. **Logical Planner**

Creates a logical operator tree describing **what** to do:

- **Logical Operators**: Scan, Filter, Join, Aggregate, Project
- **Query Tree**: Hierarchical representation of operations
- **No Implementation Details**: Abstract, implementation-agnostic

Example logical operators:
- `LogicalGet`: Table scan
- `LogicalFilter`: WHERE clause
- `LogicalJoin`: Join operations
- `LogicalAggregate`: GROUP BY aggregations
- `LogicalProjection`: SELECT expressions

#### 4. **Query Optimizer**

Transforms the logical plan into an optimized equivalent:

**Rule-Based Optimizations**:
- **Filter Pushdown**: Moves filters closer to data sources
- **Projection Pushdown**: Eliminates unnecessary columns early
- **Common Subexpression Elimination**: Reuses computed expressions
- **In Clause Rewriter**: Converts large IN clauses to joins

**Cost-Based Optimizations**:
- **Join Order Optimization**: Uses dynamic programming (DPccp algorithm)
- **Cardinality Estimation**: Estimates result sizes using statistics
- **Cost Models**: Evaluates different physical implementations

#### 5. **Physical Planner**

Converts logical operators to physical operators with implementation details:

- **Physical Operators**: Hash Join, Index Scan, Sequential Scan
- **Pipeline Construction**: Groups operators into execution pipelines
- **Memory Management**: Plans buffer allocation and spilling

#### 6. **Execution Phase**

DuckDB uses a **push-based vectorized execution model**:

```
Source → Pipeline 1 → Sink (blocking operator)
                ↓
        Pipeline 2 → Sink (blocking operator)
                ↓
        Pipeline 3 → Results
```

**Key Concepts**:
- **Push-Based**: Data flows from source to sink
- **DataChunks**: Batches of columnar data (default 2048 rows)
- **Pipeline Breakers**: Operators requiring full input (e.g., Hash Join BUILD, GROUP BY)
- **Morsel-Driven Parallelism**: Work units distributed across threads

**Execution Flow**:
1. Source operator fetches data
2. Data pushed through pipeline operators
3. Blocking operators accumulate data
4. Next pipeline processes accumulated results
5. Final pipeline produces output

### Pipeline Architecture

A **pipeline** is a sequence of operators that can execute in a streaming fashion without materializing intermediate results.

**Pipeline Breakers** (operators that stop streaming):
- Hash Join BUILD phase
- GROUP BY aggregation
- Window functions
- ORDER BY (sort)
- DISTINCT

**Example Query**:
```sql
SELECT c.name, SUM(s.revenue)
FROM sales s
JOIN customers c ON s.customer_id = c.id
GROUP BY c.name;
```

**Pipeline Breakdown**:
1. **Pipeline 1**: Scan customers → Build hash table (BREAKER)
2. **Pipeline 2**: Scan sales → Probe hash table → Partial aggregation (BREAKER)
3. **Pipeline 3**: Final aggregation → Output results

### Morsel-Driven Parallelism

DuckDB implements parallel execution using **morsels** (work units):

- **Morsel**: Subset of data (typically 122,880 rows)
- **Task Queue**: Shared queue of morsel tasks
- **Thread Pool**: Worker threads pull tasks from queue
- **Producer Token**: Identifies which executor owns tasks
- **Parallel Execution**: Multiple threads process different morsels simultaneously

**Parallelization Strategy**:
- Data sources split into morsels
- Each thread processes independent morsels
- Synchronization at pipeline breakers
- Load balancing through work stealing

---

## Jargon Tables

### Table 1: Query Processing Phases and Terminology

| **DuckDB Term** | **Alternative Terms** | **Description** | **Phase** |
|---|---|---|---|
| **Parser** | Lexer/Tokenizer, Syntax Analyzer | Converts SQL text into AST | Phase 1 |
| **Binder** | Semantic Analyzer, Name Resolver | Resolves schema references | Phase 2 |
| **Logical Planner** | Query Planner, Logical Optimizer | Creates logical operator tree | Phase 3 |
| **Optimizer** | Query Optimizer, Rewriter | Optimizes logical plan | Phase 4 |
| **Physical Planner** | Code Generator, Executor Builder | Generates physical execution plan | Phase 5 |
| **Execution Engine** | Runtime, Executor | Executes physical plan | Phase 6 |
| **Pipeline** | Operator Chain, Execution Stage | Streamable operator sequence | Execution |
| **Pipeline Breaker** | Blocking Operator, Materializer | Operator requiring full input | Execution |
| **DataChunk** | Vector, Batch, Row Block | Columnar batch of rows | Execution |
| **Morsel** | Work Unit, Chunk, Partition | Parallelization unit | Execution |

### Table 2: Hierarchical Jargon Classification

| **Category** | **Level 1 (Broad)** | **Level 2 (Specific)** | **Level 3 (Implementation)** |
|---|---|---|---|
| **Data Organization** | Database | Table | DataChunk/Vector |
| | | Column | Segment |
| | | Row | Tuple |
| **Query Processing** | SQL Query | Logical Plan | Logical Operators |
| | | Physical Plan | Physical Operators |
| | | Execution | Pipelines |
| **Operators** | Scan | Sequential Scan | Parquet Scan |
| | | Index Scan | (Limited in DuckDB) |
| | Join | Hash Join | Symmetric Hash Join |
| | | Merge Join | Sort-Merge Join |
| | | Nested Loop | Piecewise Merge Join |
| | Aggregation | Hash Aggregate | Partitioned Aggregate |
| | | Sort Aggregate | Streaming Aggregate |
| **Optimization** | Rule-Based | Filter Pushdown | Predicate Pushdown |
| | | Projection Pushdown | Column Pruning |
| | Cost-Based | Join Ordering | DPccp Algorithm |
| | | Cardinality Estimation | HyperLogLog Statistics |
| **Storage** | File Format | CSV | Text-based rows |
| | | Parquet | Columnar binary |
| | | JSON | Nested documents |
| | Database Format | DuckDB File | Native columnar |
| **Parallelism** | Thread-Level | Morsel-Driven | Work Stealing |
| | | Task Queue | Producer Tokens |
| | Pipeline-Level | Inter-Pipeline | Sequential stages |
| | | Intra-Pipeline | Parallel morsels |
| **Memory Management** | Buffer Manager | Memory Pool | Allocator |
| | | Spill to Disk | Out-of-Memory Processing |
| | | Statistics | Zonemaps, HyperLogLog |

### Table 3: Data Lifecycle Terms

| **Stage** | **DuckDB Term** | **Alternative Names** | **Description** |
|---|---|---|---|
| **Ingestion** | Load | Import, Read, Ingest | Bringing data into DuckDB |
| | read_csv/read_parquet | Scan Function, Reader | File reading functions |
| | COPY | Bulk Load, Import Command | SQL-based loading |
| | Appender | Bulk Inserter, Fast Loader | API-based bulk insertion |
| **Storage** | Table | Relation, Dataset | Logical data container |
| | Persistent Database | Disk-Backed, File-Based | Data saved to file |
| | In-Memory Database | Transient, RAM-Only | Data only in memory |
| **Transformation** | Query | SQL Statement, Command | Data manipulation |
| | View | Virtual Table, Query Alias | Stored query definition |
| | CTE | Common Table Expression, WITH clause | Temporary named result |
| **Output** | Export | Write, Save, Output | Writing results externally |
| | write_csv/write_parquet | Export Function, Writer | File writing functions |
| | COPY TO | Export Command | SQL-based export |
| | fetch/fetchall | Result Retrieval | Getting query results |

---

## Installation & Setup

### Python Installation

```bash
# Install via pip
pip install duckdb

# Verify installation
python -c "import duckdb; print(duckdb.__version__)"
```

**Python Version Requirement**: Python 3.9 or newer

### Alternative Installation Methods

```bash
# Using conda
conda install python-duckdb -c conda-forge

# Install specific version
pip install duckdb==1.2.0
```

### Basic Setup Patterns

#### In-Memory Database

```python
import duckdb

# Method 1: Global in-memory database
duckdb.sql("SELECT 42 AS answer").show()

# Method 2: Explicit connection
con = duckdb.connect()  # or duckdb.connect(':memory:')
result = con.sql("SELECT 42").fetchall()
con.close()
```

#### Persistent Database

```python
import duckdb

# Create/open persistent database
con = duckdb.connect('analytics.db')

# Create table (persisted to disk)
con.execute("""
    CREATE TABLE sales AS
    SELECT * FROM read_csv('sales.csv')
""")

# Data persists across sessions
con.close()

# Reconnect later
con = duckdb.connect('analytics.db')
result = con.execute("SELECT COUNT(*) FROM sales").fetchone()
```

---

## Data Ingestion

### File Format Support

DuckDB natively supports multiple file formats with optimized readers:

| Format | Function | Auto-Detection | Streaming | Compression |
|---|---|---|---|---|
| **CSV** | `read_csv_auto()` | Yes | Yes | gz, zstd |
| **Parquet** | `read_parquet()` | N/A | Yes | Built-in |
| **JSON** | `read_json_auto()` | Yes | Yes | gz, zstd |
| **Excel** | `read_excel()` | Limited | No | N/A |
| **Delta Lake** | Extension | Yes | Yes | Various |
| **Iceberg** | Extension | Yes | Yes | Various |

### CSV Ingestion

#### Basic CSV Reading

```python
import duckdb

# Auto-detect everything
df = duckdb.read_csv('data.csv')

# Query directly from file
result = duckdb.sql("SELECT * FROM 'data.csv' WHERE age > 25")

# Multiple files with glob patterns
duckdb.sql("SELECT * FROM 'data/*.csv'")

# Read all CSVs in a folder
duckdb.sql("SELECT * FROM read_csv('folder/*.csv')")
```

#### Advanced CSV Options

```python
# Specify delimiter and options
duckdb.read_csv('data.csv', 
    header=False,
    sep='|',
    dtype={'id': 'INTEGER', 'name': 'VARCHAR'},
    null_padding=True
)

# Parallel CSV reading (experimental)
duckdb.read_csv('large_file.csv', parallel=True)

# Skip bad rows
duckdb.read_csv('messy.csv', ignore_errors=True)
```

#### CSV Performance Tips

- **Compressed Files**: DuckDB decompresses on-the-fly (often faster than pre-decompression)
- **Disable Sniffer**: For many small files, turn off auto-detection
  ```python
  duckdb.read_csv('files/*.csv', auto_detect=False, header=True, sep=',')
  ```
- **Type Specification**: Provide explicit types to skip inference
- **Parallel Reading**: Enable for large single files

### Parquet Ingestion

#### Basic Parquet Reading

```python
import duckdb

# Single file
duckdb.read_parquet('data.parquet')

# Multiple files
duckdb.read_parquet(['file1.parquet', 'file2.parquet'])

# Glob patterns
duckdb.sql("SELECT * FROM 'data/**/*.parquet'")

# Direct query
duckdb.sql("SELECT col1, SUM(col2) FROM 'data.parquet' GROUP BY col1")
```

#### Parquet Metadata

```python
# Inspect metadata
metadata = duckdb.sql("""
    SELECT * FROM parquet_metadata('data.parquet')
""").fetchall()

# Check row group sizes
duckdb.sql("""
    SELECT 
        row_group_id, 
        num_rows,
        total_byte_size
    FROM parquet_metadata('data.parquet')
""")
```

#### Parquet Best Practices

**Row Group Sizing**:
- Optimal: 100K - 1M rows per row group
- DuckDB parallelizes over row groups
- Single giant row group = single-threaded processing

```python
# Write with optimal row group size
duckdb.sql("""
    COPY (SELECT * FROM large_table)
    TO 'output.parquet' (
        FORMAT PARQUET,
        ROW_GROUP_SIZE 500000
    )
""")
```

**When to Load vs. Query Directly**:

Load into DuckDB if:
- Running many repeated queries
- Join-heavy workload
- Need HyperLogLog statistics
- Query plans are suboptimal

Query directly if:
- Limited disk space
- One-off analysis
- Data already in optimal format
- Cloud storage scenarios

### JSON Ingestion

#### Basic JSON Reading

```python
import duckdb

# Auto-detect format (regular or newline-delimited)
duckdb.read_json('data.json')

# Newline-delimited JSON (NDJSON)
duckdb.read_json('events.ndjson')

# Direct query
duckdb.sql("SELECT * FROM 'data.json'")

# Multiple JSON files
duckdb.sql("SELECT * FROM 'logs/*.json'")
```

#### Nested JSON Handling

```python
# Flatten nested structure
duckdb.sql("""
    SELECT 
        id,
        user.name,          -- Dot notation
        user.email,
        UNNEST(tags) AS tag -- Unnest arrays
    FROM 'nested_data.json'
""")

# Extract specific nested fields
duckdb.sql("""
    SELECT 
        event_id,
        actor.login AS username,
        actor.id AS user_id,
        payload.action
    FROM 'github_events.json'
""")
```

### In-Memory Data Ingestion

#### From Pandas

```python
import duckdb
import pandas as pd

# Create pandas DataFrame
df = pd.DataFrame({
    'id': [1, 2, 3],
    'value': [10, 20, 30]
})

# Query directly (automatic registration)
result = duckdb.sql("SELECT * FROM df WHERE value > 15")

# Explicit registration
duckdb.register('my_table', df)
duckdb.sql("SELECT * FROM my_table")

# Create persistent table from DataFrame
con = duckdb.connect('database.db')
con.execute("CREATE TABLE persistent_data AS SELECT * FROM df")
```

#### From Polars

```python
import duckdb
import polars as pl

# Create polars DataFrame
df = pl.DataFrame({
    'id': [1, 2, 3],
    'value': [10, 20, 30]
})

# Query directly
result = duckdb.sql("SELECT * FROM df WHERE value > 15")
```

#### From Arrow

```python
import duckdb
import pyarrow as pa

# Arrow table
table = pa.table({
    'id': [1, 2, 3],
    'value': [10, 20, 30]
})

# Query directly
result = duckdb.sql("SELECT * FROM table")
```

### Remote Data Ingestion

#### HTTP/S3 Access

```python
import duckdb

# Install httpfs extension
con = duckdb.connect()
con.execute("INSTALL httpfs")
con.execute("LOAD httpfs")

# Read from HTTP
duckdb.sql("""
    SELECT * FROM 'https://example.com/data.parquet'
""")

# Read from S3
con.execute("SET s3_region='us-east-1'")
duckdb.sql("""
    SELECT * FROM 's3://bucket-name/data.parquet'
""")

# With credentials
con.execute("SET s3_access_key_id='...'")
con.execute("SET s3_secret_access_key='...'")
```

#### Cloud Storage Tips

- Use Parquet for cloud data (column pruning reduces I/O)
- Enable httpfs extension for all HTTP/S3/R2 access
- Consider data locality (close to compute)
- Use Hive partitioning for large datasets

---

## Python API

### Connection Management

#### Global In-Memory Database

```python
import duckdb

# Uses shared global in-memory database
duckdb.sql("CREATE TABLE test (x INTEGER)")
duckdb.sql("INSERT INTO test VALUES (42)")

# All duckdb.sql() calls share same database
result = duckdb.sql("SELECT * FROM test")
```

#### Explicit Connections

```python
import duckdb

# In-memory connection
con1 = duckdb.connect(':memory:')  # or duckdb.connect()

# Persistent connection
con2 = duckdb.connect('analytics.db')

# Isolate databases
con1.execute("CREATE TABLE test1 (x INT)")
con2.execute("CREATE TABLE test2 (y INT)")

# Close connections
con1.close()
con2.close()

# Context manager (automatic cleanup)
with duckdb.connect('temp.db') as con:
    con.execute("SELECT 42")
# Connection automatically closed
```

### Query Execution

#### execute() Method

```python
import duckdb

con = duckdb.connect()

# Execute without returning results
con.execute("CREATE TABLE sales (id INT, amount DECIMAL)")

# Execute with placeholders
con.execute("""
    INSERT INTO sales VALUES (?, ?)
""", [1, 99.99])

# Fetch results after execute
result = con.execute("SELECT * FROM sales").fetchall()
```

#### sql() Method

```python
import duckdb

# Direct execution (global database)
relation = duckdb.sql("SELECT * FROM 'data.csv'")

# With explicit connection
con = duckdb.connect('db.duckdb')
relation = con.sql("SELECT * FROM table")

# Chain operations
duckdb.sql("SELECT * FROM 'data.csv'").show()
```

### Result Retrieval

#### Fetching Methods

```python
import duckdb

con = duckdb.connect()
con.execute("SELECT * FROM generate_series(1, 100) t(x)")

# Fetch all rows as list of tuples
all_rows = con.fetchall()

# Fetch one row
single_row = con.fetchone()

# Fetch specific number of rows
ten_rows = con.fetchmany(10)

# Fetch as pandas DataFrame
df = con.fetchdf()

# Fetch as Arrow table
arrow_table = con.fetch_arrow_table()

# Fetch as NumPy arrays
numpy_result = con.fetchnumpy()
```

#### Conversion Methods

```python
import duckdb

# Query to DataFrame
df = duckdb.sql("SELECT * FROM 'data.parquet'").df()

# Query to Arrow
arrow = duckdb.sql("SELECT * FROM 'data.csv'").arrow()

# Query to Polars
pl_df = duckdb.sql("SELECT * FROM 'data.json'").pl()

# Query and show (print preview)
duckdb.sql("SELECT * FROM 'data.csv'").show()
```

### Relational API

The **Relational API** provides a Pythonic way to build queries:

```python
import duckdb

# Load data
rel = duckdb.read_csv('sales.csv')

# Chain operations
result = (rel
    .filter("amount > 100")
    .aggregate("region, SUM(amount) as total")
    .order("total DESC")
    .limit(10)
)

# Convert to DataFrame
df = result.df()

# Generate SQL
sql = result.sql_query()
print(sql)
```

#### Relational Methods

```python
import duckdb

rel = duckdb.read_parquet('data.parquet')

# Filtering
rel.filter("age > 30 AND city = 'NYC'")

# Projection
rel.project("name, age, salary")

# Aggregation
rel.aggregate("department, AVG(salary) as avg_sal, COUNT(*) as cnt")

# Joining
rel2 = duckdb.read_csv('other.csv')
rel.join(rel2, "rel.id = rel2.user_id")

# Ordering
rel.order("salary DESC")

# Limiting
rel.limit(100)

# Distinct
rel.distinct()
```

### Prepared Statements

Use prepared statements for repeated queries with different parameters:

```python
import duckdb

con = duckdb.connect()
con.execute("CREATE TABLE products (id INT, name VARCHAR, price DECIMAL)")

# Prepare statement
stmt = con.prepare("INSERT INTO products VALUES (?, ?, ?)")

# Execute multiple times
stmt.execute([1, 'Widget', 9.99])
stmt.execute([2, 'Gadget', 14.99])
stmt.execute([3, 'Doohickey', 7.49])

# Prepared SELECT
select_stmt = con.prepare("SELECT * FROM products WHERE price > ?")
result = select_stmt.execute([10.00]).fetchall()
```

### Configuration

#### PRAGMA Settings

```python
import duckdb

con = duckdb.connect()

# Set number of threads
con.execute("PRAGMA threads=4")

# Set memory limit
con.execute("PRAGMA memory_limit='8GB'")

# Enable/disable progress bar
con.execute("PRAGMA enable_progress_bar")

# View all settings
settings = con.execute("SELECT * FROM duckdb_settings()").fetchdf()
```

#### Python Configuration

```python
import duckdb

# Configure at connection time
config = {
    'threads': 8,
    'memory_limit': '16GB',
    'default_null_order': 'NULLS LAST'
}

con = duckdb.connect(database=':memory:', config=config)
```

---

## Query Execution & Optimization

### Understanding Query Plans

#### EXPLAIN Statement

```python
import duckdb

# Logical plan
duckdb.sql("""
    EXPLAIN 
    SELECT region, SUM(amount)
    FROM sales
    WHERE date >= '2024-01-01'
    GROUP BY region
""").show()
```

**Output** (simplified):
```
┌─────────────────────────────┐
│┌───────────────────────────┐│
││    Physical Plan          ││
│└───────────────────────────┘│
└─────────────────────────────┘
┌───────────────────────────────┐
│       PROJECTION              │
│    ─ ─ ─ ─ ─ ─ ─ ─ ─ ─       │
│          region               │
│       SUM(amount)             │
└─────────────┬─────────────────┘
┌─────────────┴─────────────────┐
│      HASH_GROUP_BY            │
│    ─ ─ ─ ─ ─ ─ ─ ─ ─ ─       │
│         #Groups: 5            │
│      region                   │
│      SUM(amount)              │
└─────────────┬─────────────────┘
┌─────────────┴─────────────────┐
│         FILTER                │
│    ─ ─ ─ ─ ─ ─ ─ ─ ─ ─       │
│  date >= '2024-01-01'         │
└─────────────┬─────────────────┘
┌─────────────┴─────────────────┐
│       SEQ_SCAN                │
│    ─ ─ ─ ─ ─ ─ ─ ─ ─ ─       │
│         sales                 │
└───────────────────────────────┘
```

#### EXPLAIN ANALYZE

Profile actual execution:

```python
import duckdb

duckdb.sql("""
    EXPLAIN ANALYZE
    SELECT region, SUM(amount)
    FROM sales
    WHERE date >= '2024-01-01'
    GROUP BY region
""").show()
```

**Adds timing information**:
- Execution time per operator
- Cardinality (rows processed)
- Memory usage

### Optimization Techniques

#### Filter Pushdown

**Bad** (filters after scan):
```python
# Don't do this
df = duckdb.sql("SELECT * FROM 'large_file.parquet'").df()
filtered = df[df['date'] >= '2024-01-01']
```

**Good** (filter pushed to scan):
```python
# Do this
result = duckdb.sql("""
    SELECT * FROM 'large_file.parquet'
    WHERE date >= '2024-01-01'
""")
```

#### Projection Pushdown

**Bad** (reads all columns):
```python
# Don't do this
duckdb.sql("SELECT name FROM 'data.parquet'")  # Reads all columns first
```

**Good** (reads only needed columns):
```python
# This automatically pushes projection
# DuckDB only reads 'name' column from Parquet
duckdb.sql("SELECT name FROM 'data.parquet'")
```

#### Join Optimization

```python
import duckdb

# Let optimizer choose join order
result = duckdb.sql("""
    SELECT *
    FROM large_table l
    JOIN small_table s ON l.id = s.id
    JOIN medium_table m ON l.id = m.id
""")

# Manual join order control (if needed)
duckdb.execute("PRAGMA disable_optimizer")
```

**Join Tips**:
- Smaller table on the right for hash joins
- Enable optimizer to reorder joins
- Check EXPLAIN ANALYZE for join type (hash vs. merge)
- Consider loading Parquet files if join order is suboptimal

### Statistics and Cardinality

#### Built-in Statistics

DuckDB maintains statistics automatically:

- **Zonemaps**: Min/max values per column segment
- **HyperLogLog**: Approximate distinct counts
- **Null Counts**: Number of NULL values

```python
import duckdb

# View table statistics
stats = duckdb.sql("""
    SELECT * FROM duckdb_tables 
    WHERE table_name = 'my_table'
""").df()

# Column statistics
col_stats = duckdb.sql("""
    SELECT * FROM duckdb_columns
    WHERE table_name = 'my_table'
""").df()
```

#### ANALYZE for Better Statistics

```python
import duckdb

con = duckdb.connect('analytics.db')

# Analyze table (samples data for statistics)
con.execute("ANALYZE my_table")

# Better query plans after ANALYZE
result = con.sql("SELECT * FROM my_table WHERE...")
```

### Performance Profiling

#### Timing Queries

```python
import duckdb
import time

con = duckdb.connect()

# Manual timing
start = time.time()
result = con.execute("SELECT * FROM large_table WHERE...").fetchall()
elapsed = time.time() - start
print(f"Query took {elapsed:.2f} seconds")

# Use EXPLAIN ANALYZE for detailed profiling
con.sql("EXPLAIN ANALYZE SELECT ...").show()
```

#### Profiling Output

```python
import duckdb

# Enable profiling
con = duckdb.connect()
con.execute("PRAGMA enable_profiling='json'")
con.execute("PRAGMA profiling_output='profile.json'")

# Run query
con.execute("SELECT * FROM large_table GROUP BY category")

# Profiling data saved to profile.json
```

---

## Advanced Features

### Extensions

DuckDB uses extensions to add functionality without bloating the core:

#### Installing Extensions

```python
import duckdb

con = duckdb.connect()

# Install extension
con.execute("INSTALL httpfs")
con.execute("LOAD httpfs")

# Install multiple extensions
extensions = ['parquet', 'json', 'httpfs', 'spatial']
for ext in extensions:
    con.execute(f"INSTALL {ext}")
    con.execute(f"LOAD {ext}")
```

#### Common Extensions

| Extension | Purpose | Key Features |
|---|---|---|
| **httpfs** | HTTP/S3 access | Read remote files, S3 credentials |
| **parquet** | Parquet support | Included by default |
| **json** | JSON support | Included by default |
| **spatial** | GIS functions | PostGIS-like spatial queries |
| **postgres_scanner** | Query PostgreSQL | Read from Postgres databases |
| **sqlite_scanner** | Query SQLite | Read from SQLite databases |
| **excel** | Excel files | Read .xlsx files |
| **iceberg** | Apache Iceberg | Query Iceberg tables |
| **delta** | Delta Lake | Query Delta Lake tables |
| **mysql_scanner** | Query MySQL | Read from MySQL databases |

#### Extension Examples

**Spatial Extension**:
```python
import duckdb

con = duckdb.connect()
con.execute("INSTALL spatial")
con.execute("LOAD spatial")

# Create geometry
con.execute("""
    CREATE TABLE locations AS 
    SELECT 
        'Store A' as name,
        ST_Point(40.7128, -74.0060) as location
""")

# Spatial query
result = con.execute("""
    SELECT name,
           ST_AsText(location) as coords,
           ST_Distance(
               location, 
               ST_Point(40.7589, -73.9851)
           ) as distance
    FROM locations
""").fetchall()
```

**PostgreSQL Scanner**:
```python
import duckdb

con = duckdb.connect()
con.execute("INSTALL postgres_scanner")
con.execute("LOAD postgres_scanner")

# Attach PostgreSQL database
con.execute("""
    ATTACH 'host=localhost user=postgres password=secret dbname=mydb' 
    AS postgres_db (TYPE POSTGRES)
""")

# Query PostgreSQL tables directly
result = con.execute("""
    SELECT * FROM postgres_db.public.users
    WHERE created_at > '2024-01-01'
""").fetchall()

# Join local DuckDB data with remote PostgreSQL
con.execute("""
    SELECT 
        l.order_id,
        r.customer_name
    FROM local_orders l
    JOIN postgres_db.public.customers r ON l.customer_id = r.id
""")
```

### Window Functions

DuckDB supports comprehensive window functions for analytical queries:

```python
import duckdb

# Running totals
result = duckdb.sql("""
    SELECT 
        date,
        amount,
        SUM(amount) OVER (
            ORDER BY date 
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) as running_total
    FROM sales
    ORDER BY date
""")

# Ranking
duckdb.sql("""
    SELECT 
        product,
        revenue,
        RANK() OVER (ORDER BY revenue DESC) as rank,
        DENSE_RANK() OVER (ORDER BY revenue DESC) as dense_rank,
        ROW_NUMBER() OVER (ORDER BY revenue DESC) as row_num
    FROM product_sales
""")

# Partitioned windows
duckdb.sql("""
    SELECT 
        region,
        product,
        sales,
        AVG(sales) OVER (PARTITION BY region) as region_avg,
        sales - AVG(sales) OVER (PARTITION BY region) as diff_from_avg
    FROM regional_sales
""")

# Lead and lag
duckdb.sql("""
    SELECT 
        date,
        price,
        LAG(price, 1) OVER (ORDER BY date) as prev_price,
        LEAD(price, 1) OVER (ORDER BY date) as next_price,
        price - LAG(price, 1) OVER (ORDER BY date) as price_change
    FROM stock_prices
    ORDER BY date
""")
```

### Common Table Expressions (CTEs)

```python
import duckdb

# Simple CTE
result = duckdb.sql("""
    WITH high_value_customers AS (
        SELECT customer_id, SUM(amount) as total
        FROM orders
        GROUP BY customer_id
        HAVING SUM(amount) > 10000
    )
    SELECT 
        c.name,
        h.total
    FROM high_value_customers h
    JOIN customers c ON h.customer_id = c.id
    ORDER BY h.total DESC
""")

# Multiple CTEs
duckdb.sql("""
    WITH 
    monthly_sales AS (
        SELECT 
            DATE_TRUNC('month', date) as month,
            SUM(amount) as total
        FROM sales
        GROUP BY month
    ),
    avg_monthly AS (
        SELECT AVG(total) as avg_total
        FROM monthly_sales
    )
    SELECT 
        m.month,
        m.total,
        a.avg_total,
        m.total - a.avg_total as diff
    FROM monthly_sales m
    CROSS JOIN avg_monthly a
    ORDER BY m.month
""")

# Recursive CTE
duckdb.sql("""
    WITH RECURSIVE date_series AS (
        SELECT DATE '2024-01-01' as date
        UNION ALL
        SELECT date + INTERVAL 1 DAY
        FROM date_series
        WHERE date < DATE '2024-12-31'
    )
    SELECT * FROM date_series
""")
```

### Transactions

DuckDB is ACID-compliant with full transaction support:

```python
import duckdb

con = duckdb.connect('transactional.db')

# Automatic transaction (single statement)
con.execute("INSERT INTO accounts VALUES (1, 1000)")

# Explicit transaction
try:
    con.execute("BEGIN TRANSACTION")
    
    # Debit account
    con.execute("""
        UPDATE accounts 
        SET balance = balance - 100 
        WHERE id = 1
    """)
    
    # Credit account
    con.execute("""
        UPDATE accounts 
        SET balance = balance + 100 
        WHERE id = 2
    """)
    
    con.execute("COMMIT")
except Exception as e:
    con.execute("ROLLBACK")
    raise e
```

**Transaction Isolation**:
- DuckDB uses **snapshot isolation** (MVCC - Multi-Version Concurrency Control)
- Read queries see consistent snapshot
- Write conflicts result in transaction failure

**Concurrency**:
```python
import duckdb
from concurrent.futures import ThreadPoolExecutor

# Multiple readers (concurrent)
def read_query(query_id):
    con = duckdb.connect('shared.db', read_only=True)
    result = con.execute("SELECT COUNT(*) FROM large_table").fetchone()
    con.close()
    return result

# Parallel reads work fine
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(read_query, range(4)))

# Single writer at a time
con = duckdb.connect('shared.db')
con.execute("INSERT INTO large_table VALUES (...)")  # Blocks other writers
```

### User-Defined Functions (UDFs)

#### Python UDFs

```python
import duckdb

con = duckdb.connect()

# Simple scalar function
def double_value(x):
    return x * 2

# Register function
con.create_function("double", double_value)

# Use in query
result = con.execute("""
    SELECT x, double(x) as doubled
    FROM generate_series(1, 5) t(x)
""").fetchall()

# Function with multiple parameters
def calculate_discount(price, discount_pct):
    return price * (1 - discount_pct / 100)

con.create_function("apply_discount", calculate_discount)

con.execute("""
    SELECT 
        product,
        price,
        apply_discount(price, 10) as discounted_price
    FROM products
""")
```

#### Type Annotations

```python
import duckdb
from typing import List

con = duckdb.connect()

# Explicit type specification
def sum_values(values: List[int]) -> int:
    return sum(values)

con.create_function("sum_list", sum_values)

result = con.execute("""
    SELECT sum_list([1, 2, 3, 4, 5])
""").fetchone()
```

### Appender API (Bulk Loading)

The **Appender** provides the fastest way to bulk-insert data:

```python
import duckdb

con = duckdb.connect('fast_load.db')
con.execute("CREATE TABLE fast_table (id INTEGER, value VARCHAR)")

# Create appender
appender = con.appender("fast_table")

# Append rows (very fast)
for i in range(1000000):
    appender.append_row([i, f"value_{i}"])

# Flush and close
appender.close()

# Verify
count = con.execute("SELECT COUNT(*) FROM fast_table").fetchone()[0]
print(f"Inserted {count} rows")
```

**Appender vs INSERT**:
- Appender: 10-100x faster for bulk loads
- Bypasses query parsing and optimization
- No transaction overhead per row
- Must match schema exactly

**Best Practices**:
- Use for initial bulk loads (millions of rows)
- Not for incremental updates
- Batch size: 10,000+ rows for efficiency

### Sampling

```python
import duckdb

# Random sample (percentage)
duckdb.sql("""
    SELECT * FROM large_table 
    USING SAMPLE 10%
""")

# Fixed number of rows
duckdb.sql("""
    SELECT * FROM large_table 
    USING SAMPLE 10000 ROWS
""")

# Reservoir sampling (deterministic)
duckdb.sql("""
    SELECT * FROM large_table 
    USING SAMPLE reservoir(10000 ROWS) REPEATABLE(42)
""")

# System sampling (block-level)
duckdb.sql("""
    SELECT * FROM large_table 
    USING SAMPLE SYSTEM(5%)
""")
```

**Sampling Use Cases**:
- Data exploration on large datasets
- Testing queries before full execution
- Statistical analysis
- Quick profiling

---

## Performance Best Practices

### Memory Management

#### Setting Memory Limits

```python
import duckdb

con = duckdb.connect()

# Set memory limit (prevents OOM)
con.execute("PRAGMA memory_limit='8GB'")

# Query memory limit
limit = con.execute("PRAGMA memory_limit").fetchone()
print(f"Memory limit: {limit}")

# Set at connection time
config = {'memory_limit': '16GB'}
con = duckdb.connect(database=':memory:', config=config)
```

#### Out-of-Memory Processing

DuckDB automatically spills to disk when memory is exceeded:

```python
import duckdb

con = duckdb.connect()
con.execute("PRAGMA memory_limit='2GB'")

# Query larger than memory works (spills to disk)
result = con.execute("""
    SELECT x, COUNT(*) 
    FROM generate_series(1, 100000000) t(x)
    GROUP BY x % 1000
""").fetchall()
```

**Spilling Behavior**:
- Hash tables spill partitions to disk
- Sorts use external merge sort
- Temporary files created in system temp directory
- Automatic cleanup after query completion

#### Monitoring Memory Usage

```python
import duckdb

con = duckdb.connect()

# Check current memory usage
mem_info = con.execute("""
    SELECT * FROM pragma_database_size()
""").fetchdf()

print(mem_info)
```

### Thread Configuration

```python
import duckdb

con = duckdb.connect()

# Set thread count
con.execute("PRAGMA threads=8")

# Auto-detect (use all cores)
con.execute("PRAGMA threads=0")  # 0 = auto

# Query current setting
threads = con.execute("PRAGMA threads").fetchone()
print(f"Using {threads} threads")
```

**Threading Guidelines**:
- Default: Number of CPU cores
- I/O bound: More threads may help
- CPU bound: Match core count
- Small data: Fewer threads (less overhead)

### Parallelization Tips

```python
import duckdb

# Parallel CSV reading
duckdb.read_csv('large.csv', parallel=True)

# Parallel Parquet (automatic)
duckdb.sql("SELECT * FROM '*.parquet'")  # Parallelizes over files

# Disable parallelism (debugging)
con = duckdb.connect()
con.execute("PRAGMA threads=1")
```

**When Parallelism Helps**:
- Large file scans (CSV, Parquet)
- Aggregations over many groups
- Multi-file queries
- Hash joins on large tables

**When It Doesn't**:
- Small datasets (overhead dominates)
- Single small file
- Limited CPU cores
- I/O bottlenecked queries

### Query Optimization Checklist

1. **Use appropriate file formats**
   - Parquet > CSV for analytics
   - Compress with zstd or gzip
   - Partition large datasets

2. **Leverage column pruning**
   ```python
   # Good: Only read needed columns
   duckdb.sql("SELECT id, name FROM 'data.parquet'")
   
   # Bad: Read all columns
   duckdb.sql("SELECT * FROM 'data.parquet'")
   ```

3. **Push filters down**
   ```python
   # Good: Filter in SQL
   duckdb.sql("SELECT * FROM 'data.parquet' WHERE date > '2024-01-01'")
   
   # Bad: Filter in Python
   df = duckdb.sql("SELECT * FROM 'data.parquet'").df()
   df = df[df['date'] > '2024-01-01']
   ```

4. **Use appropriate data types**
   ```python
   # Use smallest type that fits
   # INT32 vs INT64
   # FLOAT vs DOUBLE
   # VARCHAR vs TEXT
   ```

5. **Create indexes (limited support)**
   ```python
   # DuckDB has limited index support
   # Instead, rely on:
   # - Column statistics (zonemaps)
   # - Sorted data
   # - Partitioning
   ```

6. **Batch operations**
   ```python
   # Good: Bulk insert
   con.appender("table").append_rows(large_list)
   
   # Bad: Row-by-row insert
   for row in large_list:
       con.execute("INSERT INTO table VALUES (?)", row)
   ```

7. **Use EXPLAIN ANALYZE**
   ```python
   # Profile queries to find bottlenecks
   duckdb.sql("EXPLAIN ANALYZE SELECT ...").show()
   ```

### Persistence vs In-Memory

**Use In-Memory When**:
- Data fits in RAM
- Temporary analysis
- Prototype/exploration
- Speed is critical

```python
# In-memory
con = duckdb.connect(':memory:')
```

**Use Persistent When**:
- Data larger than RAM
- Need durability
- Share across sessions
- Production workloads

```python
# Persistent
con = duckdb.connect('analytics.db')
```

**Hybrid Approach**:
```python
import duckdb

# Persistent database
con = duckdb.connect('main.db')

# Create tables on disk
con.execute("CREATE TABLE persistent_data AS SELECT * FROM 'large.parquet'")

# Temporary in-memory tables for intermediates
con.execute("CREATE TEMP TABLE temp_results AS SELECT ...")

# Temp tables dropped on disconnect
```

---

## Common Patterns & Use Cases

### ETL Pipelines

```python
import duckdb

con = duckdb.connect('warehouse.db')

# Extract
raw_data = con.execute("""
    SELECT * FROM read_csv('raw/*.csv')
""")

# Transform
con.execute("""
    CREATE TABLE clean_data AS
    SELECT 
        CAST(id AS INTEGER) as id,
        UPPER(TRIM(name)) as name,
        CAST(amount AS DECIMAL(10,2)) as amount,
        TRY_CAST(date AS DATE) as date
    FROM raw_data
    WHERE amount > 0 
      AND date IS NOT NULL
""")

# Load to Parquet
con.execute("""
    COPY clean_data 
    TO 'output/clean_data.parquet' 
    (FORMAT PARQUET, COMPRESSION 'zstd')
""")
```

### Data Analysis Workflows

```python
import duckdb
import pandas as pd

# Quick exploratory analysis
con = duckdb.connect()

# Load data
con.execute("CREATE TABLE sales AS SELECT * FROM 'sales.parquet'")

# Summary statistics
summary = con.execute("""
    SELECT 
        COUNT(*) as total_records,
        SUM(amount) as total_revenue,
        AVG(amount) as avg_order,
        MIN(date) as first_date,
        MAX(date) as last_date
    FROM sales
""").fetchdf()

# Group analysis
by_region = con.execute("""
    SELECT 
        region,
        COUNT(*) as orders,
        SUM(amount) as revenue,
        AVG(amount) as avg_order
    FROM sales
    GROUP BY region
    ORDER BY revenue DESC
""").fetchdf()

# Time series analysis
monthly = con.execute("""
    SELECT 
        DATE_TRUNC('month', date) as month,
        SUM(amount) as revenue,
        COUNT(*) as orders
    FROM sales
    GROUP BY month
    ORDER BY month
""").fetchdf()

# Export to pandas for visualization
import matplotlib.pyplot as plt
monthly.plot(x='month', y='revenue', kind='line')
plt.show()
```

### Joining DataFrames

```python
import duckdb
import pandas as pd

# Multiple pandas DataFrames
customers = pd.DataFrame({
    'customer_id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Carol']
})

orders = pd.DataFrame({
    'order_id': [101, 102, 103],
    'customer_id': [1, 2, 1],
    'amount': [100, 200, 150]
})

# Join using DuckDB
result = duckdb.sql("""
    SELECT 
        c.name,
        o.order_id,
        o.amount
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
""").df()

print(result)
```

### Working with Nested Data

```python
import duckdb

# Nested JSON data
con = duckdb.connect()
con.execute("""
    CREATE TABLE events AS 
    SELECT * FROM read_json('events.json')
""")

# Unnest arrays
con.execute("""
    SELECT 
        event_id,
        UNNEST(participants) as participant
    FROM events
""")

# Access nested fields
con.execute("""
    SELECT 
        event_id,
        metadata.location as location,
        metadata.timestamp as timestamp,
        UNNEST(tags) as tag
    FROM events
""")

# Create nested structures
con.execute("""
    SELECT 
        region,
        LIST(product) as products,
        STRUCT_PACK(
            total := SUM(amount),
            count := COUNT(*)
        ) as aggregates
    FROM sales
    GROUP BY region
""")
```

### Incremental Updates

```python
import duckdb
from datetime import datetime

con = duckdb.connect('incremental.db')

# Initial load
con.execute("""
    CREATE TABLE IF NOT EXISTS sales (
        id INTEGER PRIMARY KEY,
        date DATE,
        amount DECIMAL,
        processed_at TIMESTAMP
    )
""")

# Track last update
last_update = con.execute("""
    SELECT MAX(processed_at) FROM sales
""").fetchone()[0]

# Incremental insert
now = datetime.now()
con.execute("""
    INSERT INTO sales
    SELECT 
        id,
        date,
        amount,
        ? as processed_at
    FROM read_csv('new_data.csv')
    WHERE date > ?
""", [now, last_update])
```

### Query Caching Pattern

```python
import duckdb
import os
from hashlib import md5

def cached_query(query, cache_file):
    """Execute query and cache results"""
    
    # Generate cache key
    cache_key = md5(query.encode()).hexdigest()
    cache_path = f"{cache_file}_{cache_key}.parquet"
    
    # Check cache
    if os.path.exists(cache_path):
        return duckdb.read_parquet(cache_path)
    
    # Execute and cache
    result = duckdb.sql(query)
    duckdb.sql(f"""
        COPY (SELECT * FROM result) 
        TO '{cache_path}' (FORMAT PARQUET)
    """)
    
    return result

# Usage
expensive_query = """
    SELECT region, SUM(amount) as total
    FROM read_parquet('huge_dataset.parquet')
    GROUP BY region
"""

result = cached_query(expensive_query, 'cache/regional_totals')
```

### Database Versioning

```python
import duckdb
import shutil
from datetime import datetime

def create_snapshot(db_path):
    """Create timestamped database snapshot"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    snapshot_path = f"{db_path}.snapshot_{timestamp}"
    
    # Close any connections first
    shutil.copy2(db_path, snapshot_path)
    print(f"Snapshot created: {snapshot_path}")
    
    return snapshot_path

# Usage
con = duckdb.connect('production.db')
# ... make changes ...
con.close()

# Create backup before major changes
create_snapshot('production.db')
```

---

## Limitations & Considerations

### What DuckDB Is Not

**Not a Replacement For**:

1. **OLTP Databases** (PostgreSQL, MySQL)
   - High concurrent writes
   - Row-level locking
   - Many simultaneous small transactions

2. **Distributed Systems** (Spark, Presto)
   - Multi-node clusters
   - Petabyte-scale data
   - Network-based computation

3. **Real-time Streaming** (Kafka, Flink)
   - Continuous ingestion
   - Stream processing
   - Sub-second latency requirements

### Known Limitations

#### Concurrency

```python
# Single writer at a time
# Multiple readers okay (with read_only=True)

# This blocks other writers:
con1 = duckdb.connect('db.duckdb')
con1.execute("INSERT INTO table VALUES (...)")  # Holds write lock

# This waits:
con2 = duckdb.connect('db.duckdb')
con2.execute("INSERT INTO table VALUES (...)")  # Blocked
```

**Workaround**: Use in-memory database for write-heavy workloads, periodically flush to disk.

#### Index Support

DuckDB has limited traditional index support:

- No B-tree indexes
- Relies on column statistics (zonemaps)
- Adaptive radix tree (ART) for joins (internal)

**Workaround**: 
- Sort data by query patterns
- Use partitioning
- Leverage Parquet row group statistics

#### Update Performance

```python
# Updates are slower than inserts
con.execute("UPDATE large_table SET x = x + 1")  # Can be slow

# Better: Create new table
con.execute("""
    CREATE TABLE new_table AS 
    SELECT *, x + 1 as x 
    FROM large_table
""")
con.execute("DROP TABLE large_table")
con.execute("ALTER TABLE new_table RENAME TO large_table")
```

#### String Operations

- String comparisons slower than numeric
- Large VARCHAR columns increase memory usage
- No full-text search (use extensions)

**Tip**: Use ENUM for repetitive string values:

```python
# Instead of VARCHAR
con.execute("CREATE TABLE data (status VARCHAR)")

# Use ENUM (more efficient)
con.execute("""
    CREATE TYPE status_enum AS ENUM ('pending', 'active', 'completed');
    CREATE TABLE data (status status_enum);
""")
```

### Memory Constraints

```python
# Be careful with:
# - Large GROUP BY (many groups)
# - Hash joins on large tables
# - Window functions over entire table

# Monitor memory:
con.execute("PRAGMA memory_limit='8GB'")

# For very large operations, increase temp_directory space
con.execute("PRAGMA temp_directory='/path/to/large/disk'")
```

### Platform Differences

- **Windows**: Path separators differ (`\` vs `/`)
- **macOS ARM**: Some performance differences vs x86
- **Linux**: Generally best performance

```python
import os
import duckdb

# Cross-platform paths
data_path = os.path.join('data', 'file.parquet')
duckdb.read_parquet(data_path)
```

### When to Use Alternatives

**Use PostgreSQL if**:
- Need ACID transactions with high concurrency
- Application database (CRUD operations)
- Require row-level security
- Need replication/high availability

**Use Pandas if**:
- Data fits comfortably in memory
- Heavy use of Python-specific operations
- Integration with scikit-learn/ML libraries
- Simple transformations

**Use Polars if**:
- Similar to DuckDB use case
- Prefer DataFrame API over SQL
- Need lazy evaluation

**Use Spark if**:
- Multi-node cluster available
- Data >> single machine RAM
- Need fault tolerance
- Stream processing required

---

## References

<div class="references" markdown="1">

1. <a href="https://duckdb.org/docs/" target="_blank">DuckDB Official Documentation</a>
2. <a href="https://duckdb.org/docs/api/python/overview" target="_blank">DuckDB Python API Reference</a>
3. <a href="https://duckdb.org/docs/guides/performance/overview" target="_blank">DuckDB Performance Guide</a>
4. <a href="https://duckdb.org/docs/extensions/overview" target="_blank">DuckDB Extensions Documentation</a>
5. <a href="https://duckdb.org/docs/sql/query_syntax/select" target="_blank">DuckDB SQL Reference</a>
6. <a href="https://duckdb.org/docs/internals/overview" target="_blank">DuckDB Internals Documentation</a>
7. <a href="https://github.com/duckdb/duckdb" target="_blank">DuckDB GitHub Repository</a>
8. <a href="https://duckdb.org/docs/guides/python/jupyter" target="_blank">DuckDB with Jupyter Notebooks</a>
9. <a href="https://duckdb.org/docs/data/overview" target="_blank">DuckDB Data Import/Export Guide</a>
10. <a href="https://duckdb.org/docs/guides/python/relational_api" target="_blank">DuckDB Relational API Guide</a>

</div>

<style>
.references a {
  display: block;
  margin: 0.5em 0;
  color: #0366d6;
  text-decoration: none;
}

.references a:hover {
  text-decoration: underline;
}

table {
  width: 100%;
  margin: 1em 0;
  border-collapse: collapse;
}

th, td {
  border: 1px solid #ddd;
  padding: 8px;
  text-align: left;
}

th {
  background-color: #f6f8fa;
  font-weight: bold;
}

code {
  background-color: #f6f8fa;
  padding: 0.2em 0.4em;
  border-radius: 3px;
  font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
  font-size: 85%;
}

pre {
  background-color: #f6f8fa;
  padding: 16px;
  overflow: auto;
  border-radius: 6px;
  margin: 1em 0;
}

pre code {
  background-color: transparent;
  padding: 0;
}
</style>
