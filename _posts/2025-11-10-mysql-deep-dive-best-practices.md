---
title: "🌊 MySQL: Deep Dive & Best Practices"
layout: post
author: Kalyan Narayana
date: 2025-11-10 2:30:00 +0530
categories: [Notes, MySQL]
tags: [MySQL, Database, Lifecycle, Best-practices, Normalization, Query, Transactions, ACID]
description: "Concise, clear, and validated revision notes on MySQL — structured for beginners and practitioners."
toc: true
math: true
mermaid: true
---

# MySQL: Deep Dive & Best Practices

## Introduction

Understanding how MySQL processes SQL queries is fundamental to writing efficient, optimized database applications. This comprehensive guide explores the complete lifecycle of SQL queries in MySQL, from submission to result delivery, along with industry best practices for optimal database design and performance.

---

## Table 1: SQL Lifecycle Phase Terminology Mapping

Different sources, database systems, and industries use varying terminology for the same lifecycle phases. This table maps equivalent terms:

| Standard Term | Alternative Names | Context/Usage |
|--------------|-------------------|---------------|
| **Connection** | Connection Establishment, Client Connection, Session Initialization | All DBMS contexts |
| **Query Cache** | Result Cache, Shared Pool (Oracle), Query Result Cache | MySQL ≤5.7, Deprecated in 8.0+ |
| **Parsing** | Lexical Analysis, Syntax Checking, Query Parsing, Tokenization | Universal across DBMS |
| **Optimization** | Query Optimization, Execution Planning, Cost-Based Optimization, Query Rewriting | All DBMS contexts |
| **Compilation** | Hard Parse, Query Compilation, Plan Generation | Oracle, SQL Server terminology |
| **Execution** | Query Execution, Plan Execution, Data Retrieval, Result Generation | Universal |
| **Transaction Begin** | START TRANSACTION, BEGIN, BEGIN WORK | MySQL, PostgreSQL, SQL Server |
| **Commit** | COMMIT TRANSACTION, Transaction Commit, Savepoint Commit | Universal |
| **Rollback** | ROLLBACK TRANSACTION, Transaction Abort, Undo | Universal |
| **Normalization** | Data Normalization, Schema Normalization, Relational Normalization | Database Design |
| **Denormalization** | Schema Denormalization, Controlled Redundancy, Performance Optimization | Data Warehousing, OLAP |

---

## Table 2: Hierarchical Jargon Structure

This table organizes SQL and database concepts hierarchically from high-level abstractions to specific implementations:

| Level | Category | Term | Parent Concept | Description |
|-------|----------|------|----------------|-------------|
| **L1** | Architecture | MySQL Server | - | Top-level database management system |
| **L2** | Architecture | Connection Layer | MySQL Server | Handles client-server communication |
| **L2** | Architecture | SQL Layer | MySQL Server | Processes and optimizes queries |
| **L2** | Architecture | Storage Engine Layer | MySQL Server | Manages data storage and retrieval |
| **L3** | Connection | Connection Manager | Connection Layer | Establishes and manages connections |
| **L3** | Connection | Authentication | Connection Layer | Validates user credentials |
| **L3** | SQL Processing | Parser | SQL Layer | Performs syntax and semantic checks |
| **L3** | SQL Processing | Optimizer | SQL Layer | Generates optimal execution plans |
| **L3** | SQL Processing | Executor | SQL Layer | Executes the query plan |
| **L3** | Storage | InnoDB | Storage Engine Layer | Default transactional storage engine |
| **L3** | Storage | MyISAM | Storage Engine Layer | Non-transactional storage engine |
| **L4** | Parsing | Lexical Analysis | Parser | Breaks query into tokens |
| **L4** | Parsing | Syntax Analysis | Parser | Validates SQL grammar |
| **L4** | Parsing | Semantic Analysis | Parser | Validates table/column existence |
| **L4** | Parsing | Parse Tree Generation | Parser | Creates internal query representation |
| **L4** | Optimization | Cost Estimation | Optimizer | Calculates execution costs |
| **L4** | Optimization | Join Reordering | Optimizer | Optimizes join sequences |
| **L4** | Optimization | Index Selection | Optimizer | Chooses optimal indexes |
| **L4** | Optimization | Execution Plan | Optimizer | Final query execution strategy |
| **L4** | Execution | Access Methods | Executor | Table scan, index scan, etc. |
| **L4** | Execution | Result Set Generation | Executor | Constructs final output |
| **L5** | Transaction | ACID Properties | InnoDB | Atomicity, Consistency, Isolation, Durability |
| **L5** | Transaction | Write-Ahead Logging | InnoDB | WAL mechanism for durability |
| **L5** | Transaction | Redo Log | InnoDB | Physical change log |
| **L5** | Transaction | Binlog | MySQL Server | Logical replication log |
| **L5** | Transaction | Two-Phase Commit | InnoDB | Ensures redo/binlog consistency |
| **L5** | Design | Normalization | Schema Design | Reduces redundancy |
| **L5** | Design | Denormalization | Schema Design | Optimizes read performance |
| **L6** | Normal Forms | First Normal Form (1NF) | Normalization | Atomic values, no repeating groups |
| **L6** | Normal Forms | Second Normal Form (2NF) | Normalization | 1NF + no partial dependencies |
| **L6** | Normal Forms | Third Normal Form (3NF) | Normalization | 2NF + no transitive dependencies |
| **L6** | Normal Forms | Boyce-Codd Normal Form (BCNF) | Normalization | Stricter version of 3NF |
| **L6** | Optimization | Query Rewriting | Query Optimization | Restructures queries for efficiency |
| **L6** | Optimization | Prepared Statements | Query Optimization | Pre-compiled query execution |
| **L6** | Optimization | Query Hints | Query Optimization | Manual optimizer instructions |

---

## SQL Query Lifecycle in MySQL

### Phase 1: Connection and Authentication

When a client initiates a database operation, the first step is establishing a connection with the MySQL server.

**Process Flow:**
1. **Connection Request**: Client sends connection request to MySQL server
2. **Connection Manager**: Receives and processes the connection request
3. **Authentication**: Validates credentials against user privileges
4. **Session Establishment**: Creates a session with allocated resources
5. **Connection Pooling**: Reuses existing connections for efficiency

**Key Concepts:**
- **Connection Timeout**: Maximum time a connection can remain idle
- **Max Connections**: Server-level limit on concurrent connections
- **Connection Pooling**: Maintains reusable connection pool to reduce overhead

```sql
-- View current connections
SHOW PROCESSLIST;

-- View connection limits
SHOW VARIABLES LIKE 'max_connections';

-- Set connection timeout
SET SESSION wait_timeout = 28800;
```

---

### Phase 2: Query Cache (MySQL ≤5.7 Only)

**Important Note**: Query cache was deprecated in MySQL 5.7 and completely removed in MySQL 8.0 due to performance issues in high-concurrency environments.

**Historical Context:**
- Query cache stored complete result sets of SELECT queries
- If an identical query was executed, results returned from cache
- Any table modification invalidated all cached queries using that table
- Performance degradation in write-heavy workloads led to its removal

**Modern Alternatives:**
- Application-level caching (Redis, Memcached)
- Result set caching in application layer
- ProxySQL query caching as middleware solution

---

### Phase 3: Parsing

The parser is responsible for converting the SQL string into an internal representation that MySQL can understand and process.

#### 3.1 Lexical Analysis (Tokenization)

Breaking the SQL statement into meaningful tokens.

**Example:**
```sql
SELECT user_id, email FROM users WHERE status = 'active';
```

**Tokenization Output:**
- Keywords: `SELECT`, `FROM`, `WHERE`
- Identifiers: `user_id`, `email`, `users`, `status`
- Operators: `=`
- Literals: `'active'`
- Delimiters: `,`

#### 3.2 Syntax Analysis

Validates that tokens follow SQL grammar rules.

**Checks Performed:**
- Proper keyword placement
- Balanced parentheses
- Correct clause ordering
- Valid SQL statement structure

**Common Syntax Errors:**
```sql
-- Missing FROM clause
SELECT user_id WHERE status = 'active'; -- ERROR

-- Incorrect clause order
SELECT user_id FROM users LIMIT 10 WHERE status = 'active'; -- ERROR
```

#### 3.3 Semantic Analysis

Validates logical correctness and object existence.

**Checks Performed:**
- Table existence in database
- Column existence in specified tables
- Data type compatibility
- Permission validation
- Referential integrity

**Example Semantic Error:**
```sql
-- Non-existent column
SELECT user_id, non_existent_column FROM users; 
-- ERROR: Unknown column 'non_existent_column'
```

#### 3.4 Parse Tree Generation

Creates an internal tree structure representing the query logic.

**Parse Tree Example:**
```
        SELECT
          |
       PROJECT
       /     \
   COLUMNS   FROM
  /     \      |
user_id email TABLE
              |
            users
              |
            WHERE
              |
          EQUALS
          /    \
      status  'active'
```

---

### Phase 4: Query Optimization

The optimizer's goal is to find the most efficient execution plan for the query.

#### 4.1 Statistical Analysis

**Data Gathered:**
- Table cardinality (number of rows)
- Column selectivity (unique value distribution)
- Index statistics
- Data distribution patterns
- Table size and page counts

```sql
-- Update table statistics
ANALYZE TABLE users;

-- View index statistics
SHOW INDEX FROM users;
```

#### 4.2 Cost-Based Optimization

MySQL assigns a cost to different execution strategies and selects the one with lowest estimated cost.

**Cost Factors:**
- **Disk I/O Cost**: Reading data from storage
- **CPU Cost**: Processing operations (joins, sorts, aggregations)
- **Memory Cost**: Buffer usage and temporary tables
- **Network Cost**: Data transfer in distributed queries

**Cost Formula (Simplified):**
$$
\text{Total Cost} = (\text{I/O Cost} \times \text{Pages Read}) + (\text{CPU Cost} \times \text{Rows Processed})
$$

#### 4.3 Access Path Selection

**Available Access Methods:**

1. **Full Table Scan**
   - Reads every row in the table
   - Used when no suitable index exists
   - Efficient for small tables or high selectivity queries

2. **Index Scan**
   - Uses B-tree index to locate rows
   - Efficient for selective queries
   - Avoids reading entire table

3. **Index-Only Scan** (Covering Index)
   - All required columns present in index
   - No table access needed
   - Fastest possible access method

4. **Range Scan**
   - Uses index for range conditions
   - `BETWEEN`, `>`, `<`, `IN` operations

```sql
-- Analyze query execution plan
EXPLAIN SELECT * FROM orders WHERE order_date BETWEEN '2024-01-01' AND '2024-12-31';
```

#### 4.4 Join Optimization

**Join Algorithms:**

1. **Nested Loop Join**
   - Simple algorithm: for each row in table A, scan table B
   - Efficient for small result sets
   - Time complexity: O(n × m)

2. **Hash Join** (MySQL 8.0+)
   - Builds hash table for smaller table
   - Probes hash table for matches
   - Efficient for large result sets
   - Time complexity: O(n + m)

3. **Block Nested Loop**
   - Buffers rows from outer table
   - Reduces inner table scans
   - Default when hash join unavailable

**Join Reordering:**

Optimizer may reorder joins for better performance.

```sql
-- Original query
SELECT * FROM orders o
JOIN customers c ON o.customer_id = c.id
JOIN products p ON o.product_id = p.id
WHERE c.country = 'USA';

-- Optimizer may reorder to:
-- 1. Filter customers by country first (smallest result set)
-- 2. Join with orders
-- 3. Join with products
```

#### 4.5 Execution Plan Generation

Final output: detailed step-by-step plan for query execution.

**EXPLAIN Output:**
```sql
EXPLAIN FORMAT=JSON 
SELECT u.name, o.total 
FROM users u 
JOIN orders o ON u.id = o.user_id 
WHERE u.status = 'active';
```

**Key EXPLAIN Columns:**
- `select_type`: Query type (SIMPLE, SUBQUERY, UNION)
- `table`: Table being accessed
- `type`: Join type (const, eq_ref, ref, range, index, ALL)
- `possible_keys`: Indexes that could be used
- `key`: Actually used index
- `rows`: Estimated rows to examine
- `Extra`: Additional execution details

---

### Phase 5: Query Execution

The executor takes the optimized plan and retrieves actual data from the storage engine.

#### 5.1 Storage Engine Interaction

MySQL uses a pluggable storage engine architecture.

**InnoDB (Default Engine):**
- ACID-compliant transactions
- Row-level locking
- Foreign key support
- Crash recovery
- MVCC (Multi-Version Concurrency Control)

**MyISAM:**
- Table-level locking
- No transaction support
- Full-text indexing
- Faster for read-heavy workloads
- No foreign keys

```sql
-- Check table storage engine
SHOW TABLE STATUS WHERE Name = 'users';

-- Change storage engine
ALTER TABLE users ENGINE = InnoDB;
```

#### 5.2 Data Retrieval Process

**Steps:**
1. **Buffer Pool Check**: Look for data in InnoDB buffer pool
2. **Disk Read**: If not cached, read from disk
3. **Lock Acquisition**: Apply appropriate locks (shared/exclusive)
4. **Row Filtering**: Apply WHERE conditions
5. **Sorting**: Perform ORDER BY operations
6. **Aggregation**: Execute GROUP BY and aggregate functions
7. **Result Set Construction**: Build final result set

#### 5.3 Memory Management

**Key Memory Areas:**

1. **InnoDB Buffer Pool**
   - Caches table and index data
   - Default: 128MB
   - Recommended: 70-80% of available RAM

```sql
-- View buffer pool size
SHOW VARIABLES LIKE 'innodb_buffer_pool_size';

-- Set buffer pool size (requires restart)
SET GLOBAL innodb_buffer_pool_size = 2147483648; -- 2GB
```

2. **Query Cache** (Removed in 8.0)

3. **Sort Buffer**
   - Used for ORDER BY and GROUP BY operations
   - Per-connection allocation

```sql
SHOW VARIABLES LIKE 'sort_buffer_size';
```

4. **Join Buffer**
   - Used for join operations without indexes

```sql
SHOW VARIABLES LIKE 'join_buffer_size';
```

---

### Phase 6: Transaction Management

Transactions ensure data consistency and integrity through ACID properties.

#### 6.1 ACID Properties

**Atomicity:**
All operations in a transaction succeed or all fail—no partial execution.

```sql
START TRANSACTION;

UPDATE accounts SET balance = balance - 500 WHERE account_id = 1;
UPDATE accounts SET balance = balance + 500 WHERE account_id = 2;

-- Either both updates succeed or both are rolled back
COMMIT;
```

**Consistency:**
Database transitions from one valid state to another, maintaining all constraints.

```sql
START TRANSACTION;

-- This maintains referential integrity
INSERT INTO orders (customer_id, product_id, quantity) 
VALUES (101, 501, 2);

-- If customer_id 101 doesn't exist, transaction fails
COMMIT;
```

**Isolation:**
Concurrent transactions don't interfere with each other.

**Isolation Levels:**

| Level | Dirty Read | Non-Repeatable Read | Phantom Read | Performance |
|-------|-----------|---------------------|--------------|-------------|
| READ UNCOMMITTED | Yes | Yes | Yes | Fastest |
| READ COMMITTED | No | Yes | Yes | Fast |
| REPEATABLE READ | No | No | Yes | Moderate |
| SERIALIZABLE | No | No | No | Slowest |

```sql
-- Set isolation level
SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ;

-- View current isolation level
SELECT @@transaction_isolation;
```

**Durability:**
Once committed, changes persist even after system failure.

**Implementation:**
- Write-Ahead Logging (WAL)
- Redo logs
- Binary logs (binlog)
- Two-phase commit protocol

#### 6.2 Transaction Lifecycle

```sql
-- Explicit transaction
START TRANSACTION;

-- Multiple DML operations
INSERT INTO users (name, email) VALUES ('John', 'john@example.com');
UPDATE users SET status = 'active' WHERE id = LAST_INSERT_ID();

-- Success: make changes permanent
COMMIT;

-- OR Failure: undo all changes
ROLLBACK;
```

#### 6.3 Savepoints

Allow partial rollback within a transaction.

```sql
START TRANSACTION;

INSERT INTO orders (customer_id, total) VALUES (1, 100);
SAVEPOINT order_inserted;

INSERT INTO order_items (order_id, product_id) VALUES (LAST_INSERT_ID(), 501);
SAVEPOINT item_inserted;

-- Error occurred, rollback to savepoint
ROLLBACK TO SAVEPOINT order_inserted;

-- Continue with different operation
INSERT INTO order_items (order_id, product_id) VALUES (LAST_INSERT_ID(), 502);

COMMIT;
```

#### 6.4 Write-Ahead Logging (WAL)

Ensures durability and crash recovery.

**Process:**
1. Changes written to redo log (in-memory buffer)
2. Redo log flushed to disk (WAL)
3. Data pages eventually written to disk
4. Binary log updated for replication

**Two-Phase Commit:**
```
Phase 1 - Prepare:
  Write to redo log → Mark as PREPARE
  
Phase 2 - Commit:
  Write to binlog → Commit binlog
  Update redo log → Mark as COMMIT
```

---

## Logical SQL Clause Execution Order

Although SQL is written in a specific order, execution follows a different logical sequence.

### Standard SQL Clause Order

| Written Order | Clause | Executed Order | Purpose |
|--------------|--------|----------------|---------|
| 1 | SELECT | 5 | Choose columns to return |
| 2 | FROM | 1 | Specify source tables |
| 3 | WHERE | 2 | Filter rows before grouping |
| 4 | GROUP BY | 3 | Group rows by columns |
| 5 | HAVING | 4 | Filter groups after aggregation |
| 6 | ORDER BY | 6 | Sort final result set |
| 7 | LIMIT | 7 | Restrict number of rows |

### Detailed Execution Flow

**Step 1: FROM Clause**
```sql
FROM orders o
JOIN customers c ON o.customer_id = c.id
```
- Identifies source tables
- Performs JOIN operations
- Creates Cartesian product or filtered join result
- May create temporary tables

**Step 2: WHERE Clause**
```sql
WHERE c.country = 'USA' AND o.status = 'shipped'
```
- Filters rows from FROM result
- Applied before any grouping
- Cannot use aggregate functions (use HAVING instead)
- Row-level filtering

**Common Error:**
```sql
-- WRONG: Cannot use aggregate in WHERE
SELECT customer_id, SUM(total) 
FROM orders 
WHERE SUM(total) > 1000  -- ERROR
GROUP BY customer_id;

-- CORRECT: Use HAVING for aggregates
SELECT customer_id, SUM(total) 
FROM orders 
GROUP BY customer_id
HAVING SUM(total) > 1000;
```

**Step 3: GROUP BY Clause**
```sql
GROUP BY customer_id
```
- Groups rows with same values
- Reduces result set to unique group values
- Prepares for aggregate functions
- Creates groups in memory or temporary tables

**Step 4: HAVING Clause**
```sql
HAVING SUM(total) > 1000
```
- Filters groups created by GROUP BY
- Can use aggregate functions
- Group-level filtering
- Applied after aggregation

**Step 5: SELECT Clause**
```sql
SELECT customer_id, SUM(total) AS total_spent, COUNT(*) AS order_count
```
- Computes expressions and aggregates
- Applies column aliases
- Projects final columns
- Window functions evaluated here

**Step 6: ORDER BY Clause**
```sql
ORDER BY total_spent DESC
```
- Sorts final result set
- Can reference SELECT aliases
- May use indexes or temporary table
- Costly for large result sets

**Step 7: LIMIT Clause**
```sql
LIMIT 10 OFFSET 20
```
- Restricts number of rows returned
- Applied after sorting
- Efficient for pagination

### Complete Example

```sql
-- Written query
SELECT 
    customer_id,
    COUNT(*) AS order_count,
    SUM(total) AS total_spent,
    AVG(total) AS avg_order_value
FROM orders
WHERE order_date >= '2024-01-01'
GROUP BY customer_id
HAVING SUM(total) > 5000
ORDER BY total_spent DESC
LIMIT 10;

-- Execution order:
-- 1. FROM orders
-- 2. WHERE order_date >= '2024-01-01'
-- 3. GROUP BY customer_id
-- 4. HAVING SUM(total) > 5000
-- 5. SELECT customer_id, COUNT(*), SUM(total), AVG(total)
-- 6. ORDER BY total_spent DESC
-- 7. LIMIT 10
```

---

## MySQL Best Practices

### 1. Indexing Best Practices

#### 1.1 When to Create Indexes

**Good Candidates:**
- Columns in WHERE clauses
- Columns in JOIN conditions
- Columns in ORDER BY clauses
- Foreign key columns
- Columns with high selectivity (many unique values)

```sql
-- Create single-column index
CREATE INDEX idx_email ON users(email);

-- Create composite index
CREATE INDEX idx_user_date ON orders(user_id, order_date);

-- Create unique index
CREATE UNIQUE INDEX idx_username ON users(username);

-- Create full-text index
CREATE FULLTEXT INDEX idx_content ON articles(title, body);
```

#### 1.2 Index Types

**B-Tree Index (Default):**
- Most common index type
- Good for equality and range queries
- Supports ASC/DESC ordering

**Hash Index:**
- Only in MEMORY storage engine
- Fast for equality lookups
- No range queries support

**Full-Text Index:**
- For text search operations
- Supports MATCH AGAINST queries

```sql
-- Full-text search
SELECT * FROM articles 
WHERE MATCH(title, body) AGAINST('mysql optimization' IN NATURAL LANGUAGE MODE);
```

**Spatial Index:**
- For geometric data types
- GIS applications

#### 1.3 Composite Index Best Practices

**Column Order Matters:**

```sql
-- Index on (a, b, c) can be used for:
WHERE a = 1                    -- Yes
WHERE a = 1 AND b = 2          -- Yes
WHERE a = 1 AND b = 2 AND c = 3 -- Yes
WHERE b = 2                    -- No
WHERE a = 1 AND c = 3          -- Partially (only 'a' used)
```

**Leftmost Prefix Rule:**
Create index with most selective column first.

```sql
-- If querying by date range frequently
CREATE INDEX idx_date_user ON orders(order_date, user_id);

-- If querying specific users frequently
CREATE INDEX idx_user_date ON orders(user_id, order_date);
```

#### 1.4 Index Maintenance

```sql
-- Analyze table to update statistics
ANALYZE TABLE orders;

-- Optimize table to defragment and rebuild indexes
OPTIMIZE TABLE orders;

-- Check index usage
SELECT * FROM sys.schema_unused_indexes WHERE object_schema = 'mydb';

-- Remove unused indexes
DROP INDEX idx_unused ON table_name;
```

---

### 2. Query Optimization Techniques

#### 2.1 Avoid SELECT *

```sql
-- Bad: Fetches all columns
SELECT * FROM users WHERE id = 1;

-- Good: Fetch only needed columns
SELECT id, name, email FROM users WHERE id = 1;
```

**Why?**
- Reduces I/O operations
- Decreases network traffic
- Enables covering indexes
- Improves cache efficiency

#### 2.2 Use LIMIT for Large Result Sets

```sql
-- Limit results for testing
SELECT * FROM large_table LIMIT 100;

-- Pagination
SELECT * FROM products 
ORDER BY created_at DESC 
LIMIT 20 OFFSET 40;  -- Page 3 (20 items per page)
```

#### 2.3 Optimize Joins

**Use Appropriate Join Types:**

```sql
-- INNER JOIN: Only matching rows
SELECT o.*, c.name 
FROM orders o
INNER JOIN customers c ON o.customer_id = c.id;

-- LEFT JOIN: All left table rows + matches
SELECT c.*, o.order_date
FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id;

-- Avoid: Cartesian product
SELECT * FROM table1, table2;  -- No join condition
```

**Index Join Columns:**

```sql
-- Both columns should be indexed
CREATE INDEX idx_customer_id ON orders(customer_id);
CREATE INDEX idx_id ON customers(id);  -- Primary key automatically indexed
```

#### 2.4 Use EXISTS Instead of IN for Subqueries

```sql
-- Slower: IN with subquery
SELECT * FROM customers c
WHERE c.id IN (SELECT customer_id FROM orders WHERE total > 1000);

-- Faster: EXISTS
SELECT * FROM customers c
WHERE EXISTS (
    SELECT 1 FROM orders o 
    WHERE o.customer_id = c.id AND o.total > 1000
);
```

#### 2.5 Avoid Functions on Indexed Columns

```sql
-- Bad: Function prevents index usage
SELECT * FROM orders WHERE YEAR(order_date) = 2024;

-- Good: Range condition uses index
SELECT * FROM orders 
WHERE order_date BETWEEN '2024-01-01' AND '2024-12-31';
```

#### 2.6 Use UNION ALL Instead of UNION

```sql
-- Slower: UNION removes duplicates (requires sorting)
SELECT id, name FROM customers WHERE country = 'USA'
UNION
SELECT id, name FROM customers WHERE country = 'Canada';

-- Faster: UNION ALL keeps duplicates
SELECT id, name FROM customers WHERE country = 'USA'
UNION ALL
SELECT id, name FROM customers WHERE country = 'Canada';
```

#### 2.7 Batch Operations

```sql
-- Slow: Multiple inserts
INSERT INTO users (name) VALUES ('John');
INSERT INTO users (name) VALUES ('Jane');
INSERT INTO users (name) VALUES ('Bob');

-- Fast: Batch insert
INSERT INTO users (name) VALUES 
('John'), ('Jane'), ('Bob');
```

#### 2.8 Use Prepared Statements

```sql
-- Prepare statement once
PREPARE stmt FROM 'SELECT * FROM users WHERE id = ?';

-- Execute multiple times with different parameters
SET @id = 1;
EXECUTE stmt USING @id;

SET @id = 2;
EXECUTE stmt USING @id;

-- Deallocate when done
DEALLOCATE PREPARE stmt;
```

**Benefits:**
- Query parsed and optimized once
- Prevents SQL injection
- Reduces compilation overhead
- Better performance for repeated queries

---

### 3. Schema Design Best Practices

#### 3.1 Normalization

**Definition:** Organizing data to reduce redundancy and improve integrity.

**Normal Forms:**

**First Normal Form (1NF):**
- Atomic values (no repeating groups)
- Each column contains single value
- Each row is unique

```sql
-- Violation of 1NF: Multiple values in one column
CREATE TABLE users_bad (
    id INT,
    name VARCHAR(100),
    phone_numbers VARCHAR(255)  -- '555-1234, 555-5678, 555-9012'
);

-- Complies with 1NF: Separate table for phone numbers
CREATE TABLE users (
    id INT PRIMARY KEY,
    name VARCHAR(100)
);

CREATE TABLE user_phones (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT,
    phone_number VARCHAR(20),
    FOREIGN KEY (user_id) REFERENCES users(id)
);
```

**Second Normal Form (2NF):**
- Must be in 1NF
- No partial dependencies (all non-key attributes depend on entire primary key)

```sql
-- Violation of 2NF: product_name depends only on product_id, not full key
CREATE TABLE order_items_bad (
    order_id INT,
    product_id INT,
    product_name VARCHAR(100),  -- Dependent only on product_id
    quantity INT,
    PRIMARY KEY (order_id, product_id)
);

-- Complies with 2NF: Separate product details
CREATE TABLE products (
    product_id INT PRIMARY KEY,
    product_name VARCHAR(100)
);

CREATE TABLE order_items (
    order_id INT,
    product_id INT,
    quantity INT,
    PRIMARY KEY (order_id, product_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);
```

**Third Normal Form (3NF):**
- Must be in 2NF
- No transitive dependencies (non-key attributes don't depend on other non-key attributes)

```sql
-- Violation of 3NF: city depends on zip_code (transitive dependency)
CREATE TABLE customers_bad (
    customer_id INT PRIMARY KEY,
    name VARCHAR(100),
    zip_code VARCHAR(10),
    city VARCHAR(100)  -- Depends on zip_code, not customer_id
);

-- Complies with 3NF: Separate location table
CREATE TABLE customers (
    customer_id INT PRIMARY KEY,
    name VARCHAR(100),
    zip_code VARCHAR(10),
    FOREIGN KEY (zip_code) REFERENCES zip_codes(code)
);

CREATE TABLE zip_codes (
    code VARCHAR(10) PRIMARY KEY,
    city VARCHAR(100)
);
```

**Boyce-Codd Normal Form (BCNF):**
- Stricter version of 3NF
- Every determinant must be a candidate key
- Removes all anomalies related to functional dependencies

#### 3.2 Denormalization

**When to Denormalize:**
- Read-heavy applications (OLAP, reporting, analytics)
- Complex joins causing performance issues
- Data warehouse and star schema designs
- Caching aggregate values

```sql
-- Normalized: Requires join to get order total
CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    customer_id INT,
    order_date DATE
);

CREATE TABLE order_items (
    item_id INT PRIMARY KEY,
    order_id INT,
    product_id INT,
    quantity INT,
    price DECIMAL(10,2)
);

-- Denormalized: Store calculated total
CREATE TABLE orders_denorm (
    order_id INT PRIMARY KEY,
    customer_id INT,
    order_date DATE,
    total_amount DECIMAL(10,2),  -- Denormalized
    item_count INT               -- Denormalized
);
```

**Denormalization Strategies:**

1. **Add Redundant Columns:**
```sql
ALTER TABLE orders ADD COLUMN customer_name VARCHAR(100);
-- Avoids join with customers table
```

2. **Materialized Aggregates:**
```sql
CREATE TABLE daily_sales_summary (
    sale_date DATE PRIMARY KEY,
    total_orders INT,
    total_revenue DECIMAL(12,2),
    avg_order_value DECIMAL(10,2)
);

-- Update daily via scheduled job
INSERT INTO daily_sales_summary
SELECT 
    DATE(order_date),
    COUNT(*),
    SUM(total),
    AVG(total)
FROM orders
WHERE DATE(order_date) = CURDATE()
GROUP BY DATE(order_date)
ON DUPLICATE KEY UPDATE
    total_orders = VALUES(total_orders),
    total_revenue = VALUES(total_revenue),
    avg_order_value = VALUES(avg_order_value);
```

3. **Summary Tables:**
```sql
-- For faster dashboard queries
CREATE TABLE customer_summary (
    customer_id INT PRIMARY KEY,
    total_orders INT,
    total_spent DECIMAL(12,2),
    last_order_date DATE,
    avg_order_value DECIMAL(10,2),
    INDEX idx_total_spent (total_spent)
);
```

#### 3.3 Data Types Selection

**Choose Appropriate Data Types:**

```sql
-- Efficient data types
CREATE TABLE users (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,  -- Use UNSIGNED for positive values
    username VARCHAR(50) NOT NULL,               -- VARCHAR for variable length
    email VARCHAR(100) NOT NULL,
    age TINYINT UNSIGNED,                        -- TINYINT for 0-255 range
    is_active BOOLEAN DEFAULT TRUE,              -- BOOLEAN for true/false
    balance DECIMAL(10,2),                       -- DECIMAL for exact monetary values
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    birth_date DATE,                             -- DATE for dates without time
    last_login DATETIME,                         -- DATETIME for date with time
    profile_data JSON                            -- JSON for flexible data
);
```

**Data Type Best Practices:**

| Use Case | Recommended Type | Reason |
|----------|-----------------|--------|
| ID columns | INT UNSIGNED or BIGINT UNSIGNED | Auto-increment, positive values only |
| Money amounts | DECIMAL(p,s) | Exact precision, no floating errors |
| True/False flags | BOOLEAN or TINYINT(1) | Space efficient |
| Timestamps | TIMESTAMP or DATETIME | TIMESTAMP auto-updates, timezone aware |
| Small integers (0-255) | TINYINT | 1 byte storage |
| Medium integers | INT | 4 bytes storage |
| Large integers | BIGINT | 8 bytes storage |
| Fixed length strings | CHAR(n) | Faster for fixed-width data |
| Variable strings | VARCHAR(n) | Space efficient for variable length |
| Large text | TEXT, MEDIUMTEXT, LONGTEXT | For content exceeding VARCHAR limits |
| Binary data | BLOB, MEDIUMBLOB, LONGBLOB | For images, files |
| Enum values | ENUM or VARCHAR | ENUM faster but less flexible |

**Avoid Common Mistakes:**

```sql
-- Bad: VARCHAR too large wastes space
username VARCHAR(1000)  -- Overkill for usernames

-- Good: Appropriate size
username VARCHAR(50)

-- Bad: Using VARCHAR for fixed-length data
country_code VARCHAR(2)

-- Good: Use CHAR for fixed length
country_code CHAR(2)

-- Bad: FLOAT for money (precision errors)
price FLOAT

-- Good: DECIMAL for exact values
price DECIMAL(10,2)
```

#### 3.4 Primary and Foreign Keys

**Primary Key Guidelines:**

```sql
-- Auto-incrementing integer (most common)
CREATE TABLE orders (
    order_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    customer_id INT UNSIGNED NOT NULL,
    order_date DATE NOT NULL
);

-- Composite primary key
CREATE TABLE order_items (
    order_id INT UNSIGNED,
    product_id INT UNSIGNED,
    quantity INT NOT NULL,
    PRIMARY KEY (order_id, product_id)
);

-- Natural key (when appropriate)
CREATE TABLE countries (
    country_code CHAR(2) PRIMARY KEY,  -- ISO code
    country_name VARCHAR(100) NOT NULL
);
```

**Foreign Key Constraints:**

```sql
CREATE TABLE orders (
    order_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    customer_id INT UNSIGNED NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        ON DELETE RESTRICT    -- Prevent deletion of referenced customer
        ON UPDATE CASCADE     -- Update order if customer_id changes
);

-- Referential actions:
-- CASCADE: Propagate changes to child records
-- SET NULL: Set foreign key to NULL
-- RESTRICT: Prevent operation if references exist
-- NO ACTION: Same as RESTRICT
-- SET DEFAULT: Set to default value (not supported in InnoDB)
```

---

### 4. Transaction Best Practices

#### 4.1 Keep Transactions Short

```sql
-- Bad: Long-running transaction
START TRANSACTION;
SELECT * FROM large_table;  -- Time-consuming query
-- ... more operations ...
-- ... user input/processing ...
COMMIT;  -- Holds locks for extended period

-- Good: Minimal transaction scope
-- Perform reads outside transaction
SELECT * FROM large_table;  
-- Process data in application

-- Only transactional writes
START TRANSACTION;
UPDATE account SET balance = balance - 100 WHERE id = 1;
UPDATE account SET balance = balance + 100 WHERE id = 2;
COMMIT;
```

#### 4.2 Use Appropriate Isolation Levels

```sql
-- For financial transactions: SERIALIZABLE
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;
START TRANSACTION;
-- Critical operations
COMMIT;

-- For reporting: READ COMMITTED
SET TRANSACTION ISOLATION LEVEL READ COMMITTED;
-- Generate reports

-- Default: REPEATABLE READ (balanced)
```

#### 4.3 Error Handling in Transactions

```sql
-- Using stored procedure with error handling
DELIMITER $
CREATE PROCEDURE transfer_money(
    IN from_account INT,
    IN to_account INT,
    IN amount DECIMAL(10,2)
)
BEGIN
    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        ROLLBACK;
        -- Log error or re-throw
    END;
    
    START TRANSACTION;
    
    UPDATE accounts SET balance = balance - amount 
    WHERE account_id = from_account AND balance >= amount;
    
    IF ROW_COUNT() = 0 THEN
        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'Insufficient funds';
    END IF;
    
    UPDATE accounts SET balance = balance + amount 
    WHERE account_id = to_account;
    
    COMMIT;
END$
DELIMITER ;
```

#### 4.4 Avoid Deadlocks

**Deadlock Prevention Strategies:**

1. **Access tables in consistent order:**
```sql
-- Thread 1 and Thread 2 both do:
START TRANSACTION;
UPDATE accounts WHERE id = 1;  -- Always update lower ID first
UPDATE accounts WHERE id = 2;
COMMIT;
```

2. **Use shorter transactions**

3. **Use lower isolation levels when possible**

4. **Add timeouts:**
```sql
SET innodb_lock_wait_timeout = 50;  -- Wait max 50 seconds for lock
```

**Deadlock Detection:**
```sql
-- View recent deadlocks
SHOW ENGINE INNODB STATUS;
```

---

### 5. Performance Monitoring and Analysis

#### 5.1 Using EXPLAIN

```sql
-- Analyze query execution plan
EXPLAIN SELECT * FROM orders WHERE customer_id = 100;

-- Detailed analysis
EXPLAIN FORMAT=JSON SELECT o.*, c.name
FROM orders o
JOIN customers c ON o.customer_id = c.id
WHERE o.order_date >= '2024-01-01';

-- Extended information
EXPLAIN EXTENDED SELECT * FROM users WHERE email LIKE '%example.com';
SHOW WARNINGS;  -- Shows optimizer notes
```

**EXPLAIN Output Interpretation:**

| Type | Description | Performance |
|------|-------------|-------------|
| system | Single row (system table) | Best |
| const | At most one row (PRIMARY KEY or UNIQUE) | Excellent |
| eq_ref | One row per row from previous table | Very Good |
| ref | Multiple rows with matching index value | Good |
| range | Index range scan (BETWEEN, >, <) | Acceptable |
| index | Full index scan | Poor |
| ALL | Full table scan | Worst |

#### 5.2 Slow Query Log

```sql
-- Enable slow query log
SET GLOBAL slow_query_log = 'ON';
SET GLOBAL long_query_time = 2;  -- Log queries taking >2 seconds
SET GLOBAL log_queries_not_using_indexes = 'ON';

-- View settings
SHOW VARIABLES LIKE 'slow_query%';
SHOW VARIABLES LIKE 'long_query_time';
```

**Analyze slow queries:**
```bash
# Using mysqldumpslow (command line tool)
mysqldumpslow -s t -t 10 /var/log/mysql/slow-query.log
# Shows top 10 slowest queries
```

#### 5.3 Performance Schema

```sql
-- Enable performance schema
SET GLOBAL performance_schema = ON;

-- Find expensive queries
SELECT 
    DIGEST_TEXT,
    COUNT_STAR AS execution_count,
    AVG_TIMER_WAIT/1000000000 AS avg_time_ms,
    SUM_TIMER_WAIT/1000000000 AS total_time_ms
FROM performance_schema.events_statements_summary_by_digest
ORDER BY SUM_TIMER_WAIT DESC
LIMIT 10;

-- Monitor table access
SELECT 
    OBJECT_NAME,
    COUNT_READ,
    COUNT_WRITE,
    COUNT_FETCH,
    SUM_TIMER_WAIT/1000000000 AS total_time_ms
FROM performance_schema.table_io_waits_summary_by_table
WHERE OBJECT_SCHEMA = 'mydb'
ORDER BY SUM_TIMER_WAIT DESC;
```

#### 5.4 Key Metrics to Monitor

```sql
-- Buffer pool hit ratio (should be >95%)
SHOW STATUS LIKE 'Innodb_buffer_pool_read%';

-- Connection statistics
SHOW STATUS LIKE 'Threads_connected';
SHOW STATUS LIKE 'Max_used_connections';

-- Query cache hit rate (MySQL ≤5.7)
SHOW STATUS LIKE 'Qcache%';

-- Table locks
SHOW STATUS LIKE 'Table_locks%';

-- Temporary tables
SHOW STATUS LIKE 'Created_tmp%';

-- Sort operations
SHOW STATUS LIKE 'Sort%';
```

---

### 6. Backup and Recovery Best Practices

#### 6.1 Backup Strategies

**Logical Backups (mysqldump):**

```bash
# Full database backup
mysqldump -u root -p --all-databases > full_backup.sql

# Single database
mysqldump -u root -p mydb > mydb_backup.sql

# Specific tables
mysqldump -u root -p mydb users orders > tables_backup.sql

# With compression
mysqldump -u root -p mydb | gzip > mydb_backup.sql.gz

# Exclude certain tables
mysqldump -u root -p mydb --ignore-table=mydb.logs > backup.sql
```

**Physical Backups:**

```bash
# Using MySQL Enterprise Backup or Percona XtraBackup
xtrabackup --backup --target-dir=/backup/full

# Incremental backup
xtrabackup --backup --target-dir=/backup/inc1 --incremental-basedir=/backup/full
```

#### 6.2 Point-in-Time Recovery

```sql
-- Enable binary logging
SET GLOBAL log_bin = ON;
SET GLOBAL binlog_format = 'ROW';  -- Recommended for recovery

-- View binary logs
SHOW BINARY LOGS;

-- View binary log events
SHOW BINLOG EVENTS IN 'mysql-bin.000001';
```

**Recovery Process:**

```bash
# 1. Restore from full backup
mysql -u root -p < full_backup.sql

# 2. Apply binary logs from backup time to desired point
mysqlbinlog --start-datetime="2024-11-01 10:00:00" \
            --stop-datetime="2024-11-10 14:30:00" \
            mysql-bin.000001 mysql-bin.000002 | mysql -u root -p
```

#### 6.3 Backup Best Practices

1. **Regular automated backups** (daily full, hourly incremental)
2. **Test restore procedures** regularly
3. **Store backups offsite** (cloud storage, remote location)
4. **Encrypt sensitive backups**
5. **Monitor backup success/failure**
6. **Document recovery procedures**
7. **Keep multiple backup generations**

---

### 7. Security Best Practices

#### 7.1 User Management

```sql
-- Create user with specific privileges
CREATE USER 'appuser'@'localhost' IDENTIFIED BY 'strong_password';

-- Grant minimal required privileges
GRANT SELECT, INSERT, UPDATE ON mydb.* TO 'appuser'@'localhost';

-- Read-only user
GRANT SELECT ON mydb.* TO 'readonly'@'%';

-- Admin user (use sparingly)
GRANT ALL PRIVILEGES ON *.* TO 'admin'@'localhost' WITH GRANT OPTION;

-- View user privileges
SHOW GRANTS FOR 'appuser'@'localhost';

-- Revoke privileges
REVOKE INSERT ON mydb.* FROM 'appuser'@'localhost';

-- Remove user
DROP USER 'olduser'@'localhost';

-- Require SSL connection
CREATE USER 'secureuser'@'%' IDENTIFIED BY 'password' REQUIRE SSL;
```

#### 7.2 SQL Injection Prevention

**Never concatenate user input:**

```sql
-- VULNERABLE to SQL injection
query = "SELECT * FROM users WHERE username = '" + userInput + "'"

-- SAFE: Use prepared statements
PREPARE stmt FROM 'SELECT * FROM users WHERE username = ?';
SET @username = 'user_input';
EXECUTE stmt USING @username;
```

**Application-level (example in Python):**

```python
# VULNERABLE
cursor.execute("SELECT * FROM users WHERE id = " + user_id)

# SAFE: Parameterized query
cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
```

#### 7.3 Password and Authentication

```sql
-- Use strong password validation plugin
INSTALL PLUGIN validate_password SONAME 'validate_password.so';

-- Set password policy
SET GLOBAL validate_password.policy = STRONG;
SET GLOBAL validate_password.length = 12;

-- Password expiration
ALTER USER 'appuser'@'localhost' PASSWORD EXPIRE INTERVAL 90 DAY;

-- Force password change
ALTER USER 'appuser'@'localhost' PASSWORD EXPIRE;

-- Account locking after failed attempts
CREATE USER 'user'@'localhost' 
    IDENTIFIED BY 'password'
    FAILED_LOGIN_ATTEMPTS 3 
    PASSWORD_LOCK_TIME 1;  -- Lock for 1 day
```

#### 7.4 Network Security

```sql
-- Bind to specific network interface
-- In my.cnf:
[mysqld]
bind-address = 127.0.0.1  -- Localhost only

-- Restrict user by host
CREATE USER 'appuser'@'192.168.1.100' IDENTIFIED BY 'password';
CREATE USER 'appuser'@'192.168.1.%' IDENTIFIED BY 'password';  -- Subnet

-- Disable remote root login
DELETE FROM mysql.user WHERE User='root' AND Host NOT IN ('localhost', '127.0.0.1', '::1');
FLUSH PRIVILEGES;
```

#### 7.5 Audit Logging

```sql
-- Enable general query log (development only - performance impact)
SET GLOBAL general_log = 'ON';
SET GLOBAL general_log_file = '/var/log/mysql/general.log';

-- Audit plugin for production
INSTALL PLUGIN audit_log SONAME 'audit_log.so';

-- Configure audit log
SET GLOBAL audit_log_policy = 'QUERIES';
SET GLOBAL audit_log_format = 'JSON';
```

---

### 8. Configuration Optimization

#### 8.1 InnoDB Configuration

```sql
-- Buffer pool size (most important)
-- Set to 70-80% of available RAM for dedicated DB server
SET GLOBAL innodb_buffer_pool_size = 8589934592;  -- 8GB

-- Log file size (larger = better for write-heavy workloads)
-- In my.cnf:
[mysqld]
innodb_log_file_size = 512M

-- Buffer pool instances (for systems with >1GB buffer pool)
innodb_buffer_pool_instances = 8

-- I/O capacity (based on storage speed)
innodb_io_capacity = 2000  -- SSD
innodb_io_capacity_max = 4000

-- Flush method (Linux)
innodb_flush_method = O_DIRECT  -- Bypass OS cache

-- Thread concurrency
innodb_thread_concurrency = 0  -- Auto (recommended)
```

#### 8.2 Connection Management

```sql
-- Maximum connections
SET GLOBAL max_connections = 500;

-- Connection timeout
SET GLOBAL wait_timeout = 28800;  -- 8 hours
SET GLOBAL interactive_timeout = 28800;

-- Thread cache
SET GLOBAL thread_cache_size = 64;

-- View connection statistics
SHOW STATUS LIKE 'Threads%';
SHOW STATUS LIKE 'Connections';
```

#### 8.3 Query Cache Configuration (MySQL ≤5.7)

```sql
-- Query cache size
SET GLOBAL query_cache_size = 268435456;  -- 256MB

-- Query cache type
SET GLOBAL query_cache_type = 1;  -- ON

-- Minimum result size to cache
SET GLOBAL query_cache_min_res_unit = 4096;

-- View cache statistics
SHOW STATUS LIKE 'Qcache%';
```

#### 8.4 Temporary Tables and Sorting

```sql
-- Maximum size of in-memory temporary table
SET GLOBAL max_heap_table_size = 67108864;  -- 64MB
SET GLOBAL tmp_table_size = 67108864;

-- Sort buffer (per connection)
SET SESSION sort_buffer_size = 2097152;  -- 2MB

-- Read buffer
SET SESSION read_buffer_size = 131072;  -- 128KB
```

---

### 9. Replication and High Availability

#### 9.1 Master-Slave Replication

**Master Configuration:**

```sql
-- Enable binary logging
[mysqld]
server-id = 1
log-bin = mysql-bin
binlog_format = ROW

-- Create replication user
CREATE USER 'replicator'@'%' IDENTIFIED BY 'password';
GRANT REPLICATION SLAVE ON *.* TO 'replicator'@'%';
FLUSH PRIVILEGES;

-- Get master status
SHOW MASTER STATUS;
-- Note: File and Position
```

**Slave Configuration:**

```sql
-- Configure slave
[mysqld]
server-id = 2
relay-log = relay-bin
read_only = 1

-- Connect to master
CHANGE MASTER TO
    MASTER_HOST='master_ip',
    MASTER_USER='replicator',
    MASTER_PASSWORD='password',
    MASTER_LOG_FILE='mysql-bin.000001',
    MASTER_LOG_POS=12345;

-- Start replication
START SLAVE;

-- Check status
SHOW SLAVE STATUS\G
```

#### 9.2 Monitoring Replication

```sql
-- Check replication lag
SHOW SLAVE STATUS\G
-- Look at: Seconds_Behind_Master

-- Skip replication errors (use with caution)
SET GLOBAL sql_slave_skip_counter = 1;
START SLAVE;

-- Stop replication
STOP SLAVE;
```

---

### 10. Common Pitfalls and How to Avoid Them

#### 10.1 N+1 Query Problem

```sql
-- Bad: Separate query for each user's orders
SELECT * FROM users;
-- Then for each user:
SELECT * FROM orders WHERE user_id = ?;  -- Executed N times

-- Good: Single join query
SELECT u.*, o.* 
FROM users u
LEFT JOIN orders o ON u.id = o.user_id;

-- Or: Two queries with IN clause
SELECT * FROM users;
SELECT * FROM orders WHERE user_id IN (1,2,3,4,5);
```

#### 10.2 Using Wrong Data Types

```sql
-- Bad: Storing dates as strings
CREATE TABLE events (
    event_date VARCHAR(10)  -- '2024-11-10'
);

-- Good: Use proper date type
CREATE TABLE events (
    event_date DATE
);

-- Bad: Storing booleans as strings
is_active VARCHAR(5)  -- 'true' or 'false'

-- Good: Use BOOLEAN
is_active BOOLEAN
```

#### 10.3 Not Using Transactions

```sql
-- Bad: Separate queries without transaction
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
-- System crash here = inconsistent state
UPDATE accounts SET balance = balance + 100 WHERE id = 2;

-- Good: Atomic transaction
START TRANSACTION;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;
```

#### 10.4 Over-Indexing

```sql
-- Bad: Too many indexes slow down writes
CREATE INDEX idx1 ON users(email);
CREATE INDEX idx2 ON users(username);
CREATE INDEX idx3 ON users(created_at);
CREATE INDEX idx4 ON users(last_login);
CREATE INDEX idx5 ON users(status);
-- Every INSERT/UPDATE must update all indexes

-- Good: Only necessary indexes
CREATE INDEX idx_email ON users(email);  -- Frequent lookups
CREATE INDEX idx_username ON users(username);  -- Unique lookups
-- Remove rarely used indexes
```

#### 10.5 Ignoring Query Execution Plans

```sql
-- Always analyze performance-critical queries
EXPLAIN SELECT * FROM large_table WHERE column = 'value';

-- If showing "ALL" (full table scan), add index:
CREATE INDEX idx_column ON large_table(column);

-- Verify improvement
EXPLAIN SELECT * FROM large_table WHERE column = 'value';
-- Should now show "ref" or better
```

---

## Performance Optimization Checklist

### Database Design
- [ ] Normalize to 3NF, denormalize only when necessary
- [ ] Use appropriate data types
- [ ] Define primary keys on all tables
- [ ] Implement foreign key constraints
- [ ] Avoid NULL values where possible (use defaults)

### Indexing
- [ ] Index columns used in WHERE clauses
- [ ] Index foreign key columns
- [ ] Index columns used in JOIN conditions
- [ ] Index columns used in ORDER BY
- [ ] Create covering indexes for frequent queries
- [ ] Remove unused indexes
- [ ] Keep index column order optimal

### Query Optimization
- [ ] Avoid SELECT *
- [ ] Use LIMIT for large result sets
- [ ] Use EXISTS instead of IN for subqueries
- [ ] Avoid functions on indexed columns in WHERE
- [ ] Use UNION ALL instead of UNION when duplicates acceptable
- [ ] Batch INSERT/UPDATE operations
- [ ] Use prepared statements for repeated queries

### Transaction Management
- [ ] Keep transactions short
- [ ] Use appropriate isolation levels
- [ ] Implement proper error handling
- [ ] Avoid deadlocks through consistent ordering
- [ ] Commit or rollback promptly

### Configuration
- [ ] Set innodb_buffer_pool_size appropriately
- [ ] Configure connection pool size
- [ ] Adjust query cache (if MySQL ≤5.7)
- [ ] Tune temporary table sizes
- [ ] Configure appropriate timeouts

### Monitoring
- [ ] Enable slow query log
- [ ] Monitor buffer pool hit ratio
- [ ] Track connection usage
- [ ] Monitor replication lag (if applicable)
- [ ] Regularly analyze query performance
- [ ] Review and optimize slow queries

### Security
- [ ] Use strong passwords
- [ ] Grant minimum required privileges
- [ ] Use prepared statements (prevent SQL injection)
- [ ] Enable SSL for connections
- [ ] Implement password expiration policies
- [ ] Regular security audits

### Maintenance
- [ ] Regular backups (full + incremental)
- [ ] Test restore procedures
- [ ] ANALYZE tables regularly
- [ ] OPTIMIZE tables periodically
- [ ] Monitor disk space
- [ ] Update statistics
- [ ] Review and archive old data

---

## Quick Reference: Common Commands

### Database Operations
```sql
-- Create database
CREATE DATABASE mydb CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- Use database
USE mydb;

-- Drop database
DROP DATABASE mydb;

-- Show databases
SHOW DATABASES;

-- Show tables
SHOW TABLES;

-- Describe table structure
DESCRIBE users;
-- or
SHOW CREATE TABLE users;
```

### Table Operations
```sql
-- Create table
CREATE TABLE users (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_email (email)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Alter table
ALTER TABLE users ADD COLUMN phone VARCHAR(20);
ALTER TABLE users MODIFY COLUMN phone VARCHAR(25);
ALTER TABLE users DROP COLUMN phone;
ALTER TABLE users ADD INDEX idx_username (username);

-- Rename table
RENAME TABLE old_name TO new_name;

-- Drop table
DROP TABLE users;

-- Truncate table (faster than DELETE)
TRUNCATE TABLE users;
```

### Data Operations
```sql
-- Insert
INSERT INTO users (username, email) VALUES ('john', 'john@example.com');

-- Insert multiple
INSERT INTO users (username, email) VALUES 
    ('jane', 'jane@example.com'),
    ('bob', 'bob@example.com');

-- Update
UPDATE users SET email = 'newemail@example.com' WHERE id = 1;

-- Delete
DELETE FROM users WHERE id = 1;

-- Select
SELECT * FROM users WHERE status = 'active' ORDER BY created_at DESC LIMIT 10;
```

### Index Operations
```sql
-- Create index
CREATE INDEX idx_name ON users(name);
CREATE UNIQUE INDEX idx_email ON users(email);
CREATE INDEX idx_composite ON users(last_name, first_name);

-- Show indexes
SHOW INDEX FROM users;

-- Drop index
DROP INDEX idx_name ON users;

-- Analyze index usage
SELECT * FROM sys.schema_unused_indexes WHERE object_schema = 'mydb';
```

### Transaction Commands
```sql
START TRANSACTION;
-- or
BEGIN;

COMMIT;
ROLLBACK;

SAVEPOINT sp1;
ROLLBACK TO SAVEPOINT sp1;
RELEASE SAVEPOINT sp1;
```

### Performance Analysis
```sql
-- Explain query
EXPLAIN SELECT * FROM users WHERE email = 'test@example.com';
EXPLAIN FORMAT=JSON SELECT ...;

-- Show server status
SHOW STATUS;
SHOW STATUS LIKE 'Threads%';

-- Show variables
SHOW VARIABLES;
SHOW VARIABLES LIKE 'innodb%';

-- Process list
SHOW PROCESSLIST;
SHOW FULL PROCESSLIST;

-- Kill query
KILL QUERY 123;
KILL CONNECTION 123;
```

---

## Mathematical Formulas in Database Context

### Index Selectivity

Index selectivity measures how unique values are in an indexed column:

$
\text{Selectivity} = \frac{\text{Number of Distinct Values}}{\text{Total Number of Rows}}
$

- Selectivity close to 1 = High uniqueness (excellent for indexing)
- Selectivity close to 0 = Low uniqueness (poor for indexing)

**Example:**
```sql
SELECT 
    COUNT(DISTINCT email) / COUNT(*) AS selectivity
FROM users;
-- Result: 0.98 (high selectivity, good for indexing)
```

### Query Cost Estimation

Simplified cost formula:

$
\text{Cost} = (C_{io} \times \text{Pages}) + (C_{cpu} \times \text{Rows})
$

Where:
- $C_{io}$ = I/O cost coefficient
- $C_{cpu}$ = CPU processing cost coefficient

### Buffer Pool Hit Ratio

$
\text{Hit Ratio} = \frac{\text{Innodb\_buffer\_pool\_read\_requests}}{\text{Innodb\_buffer\_pool\_read\_requests} + \text{Innodb\_buffer\_pool\_reads}} \times 100\%
$

Target: > 95%

```sql
SHOW STATUS LIKE 'Innodb_buffer_pool_read%';
```

### Replication Lag

$
\text{Replication Lag (seconds)} = \text{Seconds\_Behind\_Master}
$

```sql
SHOW SLAVE STATUS\G
```

---

## Conclusion

Understanding the SQL lifecycle and implementing best practices are essential for building performant, reliable database applications. Key takeaways:

1. **Query Lifecycle**: Connection → Parsing → Optimization → Execution → Result Delivery
2. **Optimization**: Proper indexing, query structure, and execution plan analysis
3. **Design**: Normalize first, denormalize strategically for performance
4. **Transactions**: Keep short, use appropriate isolation, handle errors properly
5. **Monitoring**: Regularly analyze slow queries, buffer pool usage, and system metrics
6. **Security**: Principle of least privilege, prepared statements, strong authentication
7. **Maintenance**: Regular backups, statistics updates, and performance reviews

Mastering these concepts enables you to design efficient schemas, write optimized queries, and maintain high-performance MySQL databases that scale with your application needs.

---

## References

<a href="https://dev.mysql.com/doc/refman/8.0/en/" target="_blank">MySQL 8.0 Reference Manual - Official Documentation</a>

<a href="https://dev.mysql.com/doc/refman/8.0/en/optimization.html" target="_blank">MySQL 8.0 Optimization Guide</a>

<a href="https://dev.mysql.com/doc/refman/8.0/en/innodb-storage-engine.html" target="_blank">InnoDB Storage Engine Documentation</a>

<a href="https://dev.mysql.com/doc/refman/8.0/en/sql-statements.html" target="_blank">MySQL SQL Statement Syntax</a>

<a href="https://dev.mysql.com/doc/refman/8.0/en/explain.html" target="_blank">MySQL EXPLAIN Statement</a>

<a href="https://dev.mysql.com/doc/refman/8.0/en/replication.html" target="_blank">MySQL Replication</a>

<a href="https://dev.mysql.com/doc/refman/8.0/en/backup-and-recovery.html" target="_blank">MySQL Backup and Recovery</a>

<a href="https://dev.mysql.com/doc/refman/8.0/en/security.html" target="_blank">MySQL Security Guidelines</a>

<a href="https://www.percona.com/blog/" target="_blank">Percona Database Performance Blog</a>

<a href="https://planet.mysql.com/" target="_blank">Planet MySQL - Community Blog Aggregator</a>

---

*Last Updated: November 10, 2025*
