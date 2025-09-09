# GraphRAG with Neo4j Integration - Installation & Usage Guide

This guide explains how to install and use the enhanced GraphRAG with Neo4j graph database backend support.

## Overview

This enhanced GraphRAG package includes:
- ✅ **Full Neo4j backend support** for all search types
- ✅ **CLI integration** with environment variables and command-line options
- ✅ **Backward compatibility** with existing parquet-based workflows
- ✅ **AWS Bedrock** model integration
- ✅ **Production-ready packaging** with pip installer

## Prerequisites

Before installing GraphRAG with Neo4j support, ensure you have:

### System Requirements
- **Python**: 3.10 or higher
- **Memory**: Minimum 8GB RAM (16GB+ recommended for large datasets)
- **Storage**: At least 10GB free disk space
- **Operating System**: Linux, macOS, or Windows

### Required Software
- **Neo4j Database**: Version 4.4+ or Neo4j Aura (see setup instructions below)
- **Git**: For cloning the repository
- **Poetry** or **pip**: For package management

### Optional but Recommended
- **Docker**: For easy Neo4j deployment
- **Neo4j Desktop**: For local development and visualization
- **APOC Plugin**: For advanced graph operations (see setup instructions)

## Neo4j Database Setup

### Option 1: Docker (Recommended for Development)

```bash
# Pull and run Neo4j with APOC plugin
docker run -d \
  --name graphrag-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your-secure-password \
  -e NEO4J_PLUGINS='["apoc"]' \
  -e NEO4J_apoc_export_file_enabled=true \
  -e NEO4J_apoc_import_file_enabled=true \
  -e NEO4J_apoc_import_file_use__neo4j__config=true \
  -v neo4j_data:/data \
  -v neo4j_logs:/logs \
  neo4j:5.15

# Verify Neo4j is running
curl http://localhost:7474
```

### Option 2: Neo4j Desktop (Local Development)

1. Download Neo4j Desktop from https://neo4j.com/download/
2. Create a new project and database
3. Set password for the database
4. Install APOC plugin:
   - Go to the database settings
   - Click "Plugins" tab
   - Install "APOC" plugin
5. Start the database

### Option 3: Neo4j Aura (Cloud - Production)

1. Sign up at https://neo4j.com/aura/
2. Create a new database instance
3. Note the connection URI, username, and password
4. APOC Core is included by default

### Option 4: Manual Installation

```bash
# Download and install Neo4j
wget https://neo4j.com/artifact.php?name=neo4j-community-5.15.0-unix.tar.gz
tar -xzf neo4j-community-5.15.0-unix.tar.gz
cd neo4j-community-5.15.0

# Configure Neo4j (edit conf/neo4j.conf)
echo "dbms.security.auth_enabled=true" >> conf/neo4j.conf
echo "dbms.connector.bolt.listen_address=0.0.0.0:7687" >> conf/neo4j.conf
echo "dbms.connector.http.listen_address=0.0.0.0:7474" >> conf/neo4j.conf

# Install APOC plugin
wget https://github.com/neo4j/apoc/releases/download/5.15.0/apoc-5.15.0-core.jar -P plugins/

# Start Neo4j
bin/neo4j start

# Set initial password
bin/neo4j-admin dbms set-initial-password your-secure-password
```

## Installation

### Step 1: Install GraphRAG with Neo4j Integration

**Option A: From Official GraphRAG (Future - when Neo4j integration is merged)**
```bash
# Install GraphRAG with Neo4j support from PyPI
pip install graphrag[neo4j]

# Or install with multiple extras
pip install graphrag[neo4j,bedrock]
```

**Option B: From Development Version (Current)**
```bash
# Navigate to the enhanced GraphRAG directory
cd /path/to/your/graphrag

# Install in development mode with Neo4j dependencies using Poetry
poetry install --extras neo4j

# Or using pip in development mode
pip install -e .[neo4j]
```

### Step 2: Verify Installation

```bash
# Test the standard GraphRAG CLI
graphrag --help

# Test the Neo4j CLI wrapper
./graphrag_neo4j --help

# Test Neo4j integration components
python -c "from graphrag.neo4j_integration import is_neo4j_available; print('✅ Neo4j integration available!' if is_neo4j_available() else '❌ Neo4j integration not available')"
```

## Complete End-to-End Setup Workflow

This section provides a complete workflow from raw documents to Neo4j-powered queries.

### Step 1: Prepare Your Documents

```bash
# Create project directory
mkdir my-graphrag-project
cd my-graphrag-project

# Create input directory and add your documents
mkdir input
# Copy your .txt files to the input/ directory
cp /path/to/your/documents/*.txt input/
```

### Step 2: Initialize GraphRAG Configuration

```bash
# Initialize GraphRAG configuration
graphrag init

# This creates:
# - settings.yaml (with Neo4j configuration template)
# - .env file (for API keys)
# - prompts/ directory (for custom prompts)
```

### Step 3: Configure Your Models

Edit the generated `settings.yaml`:

```yaml
# Configure your LLM (choose one option)

# Option A: OpenAI (Default)
models:
  default_chat_model:
    type: openai_chat
    api_key: ${GRAPHRAG_API_KEY}
    model: gpt-4-turbo-preview

  default_embedding_model:
    type: openai_embedding
    api_key: ${GRAPHRAG_API_KEY}
    model: text-embedding-3-small

# Option B: Azure OpenAI
models:
  default_chat_model:
    type: azure_openai_chat
    api_base: https://your-instance.openai.azure.com
    api_version: 2024-05-01-preview
    api_key: ${AZURE_OPENAI_API_KEY}
    deployment_name: your-gpt4-deployment

  default_embedding_model:
    type: azure_openai_embedding
    api_base: https://your-instance.openai.azure.com
    api_version: 2024-05-01-preview
    api_key: ${AZURE_OPENAI_API_KEY}
    deployment_name: your-embedding-deployment

# Option C: AWS Bedrock (Recommended for this integration)
models:
  default_chat_model:
    type: bedrock_chat
    model: us.anthropic.claude-3-7-sonnet-20250219-v1:0
    aws_region: us-east-1
    temperature: 0.1
    max_tokens: 4000

  default_embedding_model:
    type: bedrock_embedding
    model: amazon.titan-embed-text-v2:0
    aws_region: us-east-1
```

### Step 4: Set API Keys

Edit the `.env` file:

```bash
# For OpenAI
GRAPHRAG_API_KEY=your-openai-api-key

# For Azure OpenAI
AZURE_OPENAI_API_KEY=your-azure-api-key

# For AWS Bedrock (configure AWS credentials instead)
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_DEFAULT_REGION=us-east-1
```

### Step 5: Run GraphRAG Indexing

```bash
# Process your documents and create the knowledge graph
graphrag index

# This creates parquet files in the output/ directory:
# - output/create_final_entities.parquet
# - output/create_final_relationships.parquet
# - output/create_final_text_units.parquet
# - output/create_final_communities.parquet
# - output/create_final_community_reports.parquet
# - output/lancedb/ (vector embeddings)
```

### Step 6: Import Data into Neo4j

**Option A: Using the Enhanced Import Script (Recommended)**

```bash
# Set Neo4j connection environment variables
export NEO4J_URI="neo4j://localhost:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="your-password"

# Copy the enhanced import script from the GraphRAG integration
cp graphrag/neo4j_integration/neo4j_import_script_enhanced.py ./

# Run the import (this may take several minutes for large datasets)
python neo4j_import_script_enhanced.py

# You should see output like:
# ✅ Connected to Neo4j successfully
# 📊 Importing 1,250 entities...
# 📊 Importing 2,100 relationships...
# 📊 Importing 450 text units...
# 📊 Importing 85 communities...
# ✅ Import completed successfully!
```

**Option B: Using Jupyter Notebook (Interactive)**

```bash
# Copy the import notebook
cp graphrag/neo4j_integration/neo4j_import_notebook.ipynb ./

# Start Jupyter
jupyter lab neo4j_import_notebook.ipynb

# Follow the notebook instructions to import data interactively
```

### Step 7: Configure Neo4j Backend

Edit your `settings.yaml` to enable Neo4j:

```yaml
# Uncomment and configure the neo4j section
neo4j:
  enabled: true
  uri: "neo4j://localhost:7687"
  username: "neo4j"
  password: "your-password"
  database: "neo4j"
  
  # Optional: Tune performance parameters
  global_search:
    community_level: 2
    max_data_tokens: 8000
  local_search:
    top_k_entities: 10
    top_k_text_units: 3
    top_k_relationships: 10
    top_k_communities: 3
    max_context_tokens: 8000
  basic_search:
    top_k_text_units: 20
    max_context_tokens: 8000
  drift_search:
    n_depth: 3
    drift_k_followups: 5
    local_search_max_data_tokens: 8000
```

### Step 8: Verify Neo4j Integration

```bash
# Test the connection
export GRAPHRAG_GRAPH_DB=neo4j
graphrag query --method local --query "How many documents were processed?"

# You should see output using Neo4j backend instead of parquet files
```

### Step 9: Optimize Neo4j (Optional but Recommended)

```cypher
// Connect to Neo4j Browser (http://localhost:7474)
// Run these commands to create performance indexes:

// Entity indexes
CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:__Entity__) ON (e.name);
CREATE INDEX entity_description_idx IF NOT EXISTS FOR (e:__Entity__) ON (e.description);
CREATE INDEX entity_degree_idx IF NOT EXISTS FOR (e:__Entity__) ON (e.degree);

// Community indexes  
CREATE INDEX community_level_idx IF NOT EXISTS FOR (c:__Community__) ON (c.level);
CREATE INDEX community_title_idx IF NOT EXISTS FOR (c:__Community__) ON (c.title);

// Text unit indexes
CREATE INDEX text_unit_id_idx IF NOT EXISTS FOR (t:__TextUnit__) ON (t.id);

// Relationship indexes
CREATE INDEX relationship_weight_idx IF NOT EXISTS FOR ()-[r:RELATED]->() ON (r.weight);
CREATE INDEX relationship_description_idx IF NOT EXISTS FOR ()-[r:RELATED]->() ON (r.description);

// Verify indexes
SHOW INDEXES;
```

## Configuration

### Option 1: Environment Variable (Recommended)

```bash
# Set environment variable to enable Neo4j
export GRAPHRAG_GRAPH_DB=neo4j

# Use standard GraphRAG CLI - it will automatically detect Neo4j
graphrag query --method local --query "Tell me about the deployment process"
```

### Option 2: CLI Option

```bash
# Use CLI option to specify Neo4j backend
graphrag query --method local --query "Tell me about the deployment process" --graphrag-graph-db neo4j
```

### Option 3: Configuration File

Add Neo4j configuration to your `settings.yaml`:

```yaml
# Your existing GraphRAG configuration
llm:
  type: bedrock_chat
  model: us.anthropic.claude-3-7-sonnet-20250219-v1:0
  aws_region: us-east-1

embeddings:
  llm:
    type: bedrock_embedding  
    model: amazon.titan-embed-text-v2:0
    aws_region: us-east-1

# Configure vector store (for LanceDB paths) 
vector_store:
  default_vector_store:
    type: lancedb
    db_uri: "output/lancedb"              # Path to your LanceDB vector stores
    container_name: default
    overwrite: true

# Add Neo4j configuration
neo4j:
  enabled: true  # Enable Neo4j backend
  uri: "neo4j://your-neo4j-server:7687"    # Replace with your Neo4j URI
  username: "your-username"                # Replace with your username
  password: "your-password"                # Replace with your password
  database: "neo4j"                        # Replace with your database name
  
  # Optional: Customize search parameters
  global_search:
    community_level: 2
    max_data_tokens: 8000
    
  local_search:
    top_k_entities: 10
    top_k_text_units: 3
    max_context_tokens: 8000
```

## Usage Examples

### 1. Local Search with Neo4j

```bash
# Using environment variable
export GRAPHRAG_GRAPH_DB=neo4j
graphrag query --method local --query "Explain in detail about the build and deploy process"

# Using CLI option
graphrag query --method local --query "Explain in detail about the build and deploy process" --graphrag-graph-db neo4j

# Using the Neo4j CLI wrapper
./graphrag_neo4j query --method local --query "Explain in detail about the build and deploy process"
```

### 2. Global Search with Neo4j

```bash
export GRAPHRAG_GRAPH_DB=neo4j
graphrag query --method global --query "What are the main components of our software architecture?"
```

### 3. Basic Search with Neo4j

```bash
export GRAPHRAG_GRAPH_DB=neo4j
graphrag query --method basic --query "How does the change management process work?"
```

### 4. DRIFT Search with Neo4j

```bash
export GRAPHRAG_GRAPH_DB=neo4j  
graphrag query --method drift --query "Analyze the software development workflow"
```

## Key Features

### 1. Automatic Backend Detection

The CLI automatically detects which backend to use:

1. **CLI option**: `--graphrag-graph-db neo4j` (highest priority)
2. **Environment variable**: `GRAPHRAG_GRAPH_DB=neo4j` 
3. **Configuration file**: `neo4j.enabled: true` in settings.yaml
4. **Default**: Falls back to parquet files

### 2. Performance Benefits

- **Faster startup**: No loading of large parquet files
- **Real-time queries**: Direct graph traversal
- **Memory efficient**: Only load needed data
- **Scalable**: Neo4j cluster support

### 3. Backward Compatibility

```bash
# Disable Neo4j to use parquet files
unset GRAPHRAG_GRAPH_DB
graphrag query --method local --query "Your query"

# Or explicitly specify default backend
graphrag query --method local --query "Your query" --graphrag-graph-db default
```

## Configuration Template

The Neo4j configuration is included in the standard GraphRAG template:

```bash
# Initialize GraphRAG with Neo4j configuration included
graphrag init

# The generated settings.yaml will include Neo4j configuration (commented out)
# Simply uncomment and customize the neo4j section:
nano settings.yaml
```

## Troubleshooting

### 1. Installation and Import Issues

#### Python Import Errors
```bash
# If you see "ModuleNotFoundError: No module named 'neo4j'"
pip install neo4j>=5.0.0

# If you see "ModuleNotFoundError: No module named 'lancedb'"
pip install lancedb>=0.3.0

# If you see "ModuleNotFoundError: No module named 'boto3'" (for Bedrock)
pip install boto3>=1.34.0

# Complete reinstall with all dependencies
pip uninstall graphrag
pip install -e .[neo4j]
```

#### GraphRAG Integration Import Errors
```bash
# Test if Neo4j integration is properly installed
python -c "
try:
    from graphrag.neo4j_integration import is_neo4j_available
    print('✅ Neo4j integration available!')
except ImportError as e:
    print(f'❌ Neo4j integration not available: {e}')
    print('💡 Try: pip install -e .[neo4j]')
"
```

### 2. Neo4j Connection Issues

#### Connection Refused
```bash
# Check if Neo4j is running
netstat -tlnp | grep 7687  # Bolt protocol port
netstat -tlnp | grep 7474  # HTTP port

# For Docker installations
docker ps | grep neo4j
docker logs graphrag-neo4j

# Test connection manually
python -c "
from neo4j import GraphDatabase
try:
    driver = GraphDatabase.driver('neo4j://localhost:7687', auth=('neo4j', 'your-password'))
    with driver.session() as session:
        result = session.run('RETURN 1 as test')
        print('✅ Neo4j connection successful!')
        print(f'Test result: {result.single()[0]}')
    driver.close()
except Exception as e:
    print(f'❌ Neo4j connection failed: {e}')
"
```

#### Authentication Issues
```bash
# Reset Neo4j password (for local installations)
docker exec -it graphrag-neo4j neo4j-admin dbms set-initial-password new-password

# Or for manual installations
bin/neo4j-admin dbms set-initial-password new-password

# Test with cypher-shell
echo "RETURN 'Hello Neo4j'" | cypher-shell -u neo4j -p your-password
```

#### Network and Firewall Issues
```bash
# Check if ports are accessible
telnet localhost 7687
telnet localhost 7474

# For cloud deployments, ensure security groups/firewalls allow:
# - Port 7687 (Bolt protocol)
# - Port 7474 (HTTP interface)
```

### 3. Data Import Issues

#### Missing Parquet Files
```bash
# Verify GraphRAG output files exist
ls -la output/
# Should contain:
# - create_final_entities.parquet
# - create_final_relationships.parquet  
# - create_final_text_units.parquet
# - create_final_communities.parquet
# - create_final_community_reports.parquet

# If files are missing, re-run indexing
graphrag index
```

#### Import Script Failures
```bash
# Check for common issues in import logs
python neo4j_import_script_enhanced.py 2>&1 | tee import.log

# Common solutions:
# 1. Memory issues - reduce batch size in script
# 2. Data type errors - check for null values
# 3. Constraint violations - clear database first

# Clear Neo4j database if needed
cypher-shell -u neo4j -p your-password "MATCH (n) DETACH DELETE n"
```

#### Large Dataset Import Issues
```bash
# For large datasets, increase Neo4j memory settings
# Edit neo4j.conf:
echo "dbms.memory.heap.initial_size=4G" >> conf/neo4j.conf
echo "dbms.memory.heap.max_size=8G" >> conf/neo4j.conf
echo "dbms.memory.pagecache.size=2G" >> conf/neo4j.conf

# Or for Docker:
docker run -d \
  --name graphrag-neo4j \
  -e NEO4J_dbms_memory_heap_initial__size=4G \
  -e NEO4J_dbms_memory_heap_max__size=8G \
  -e NEO4J_dbms_memory_pagecache_size=2G \
  # ... other parameters
```

### 4. Configuration Issues

#### GraphRAG Configuration Not Found
```bash
# Verify settings.yaml structure
python -c "
import yaml
with open('settings.yaml', 'r') as f:
    config = yaml.safe_load(f)
    
if 'neo4j' in config:
    print('✅ Neo4j configuration found')
    neo4j_config = config['neo4j']
    required = ['uri', 'username', 'password']
    missing = [field for field in required if field not in neo4j_config]
    if missing:
        print(f'❌ Missing required fields: {missing}')
    else:
        print('✅ All required Neo4j fields present')
else:
    print('❌ Neo4j configuration not found in settings.yaml')
    print('💡 Run: graphrag init')
    print('💡 Then uncomment the neo4j section in settings.yaml')
"
```

#### Environment Variable Issues
```bash
# Check environment variables
echo "GRAPHRAG_GRAPH_DB: $GRAPHRAG_GRAPH_DB"
echo "NEO4J_URI: $NEO4J_URI"
echo "NEO4J_USERNAME: $NEO4J_USERNAME"
echo "NEO4J_PASSWORD: $NEO4J_PASSWORD"

# Set missing variables
export GRAPHRAG_GRAPH_DB=neo4j
export NEO4J_URI="neo4j://localhost:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="your-password"
```

### 5. Performance Issues

#### Slow Query Performance
```cypher
// Check query performance in Neo4j Browser
PROFILE MATCH (e:__Entity__) RETURN count(e);

// Essential indexes for performance
CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:__Entity__) ON (e.name);
CREATE INDEX entity_degree_idx IF NOT EXISTS FOR (e:__Entity__) ON (e.degree);
CREATE INDEX community_level_idx IF NOT EXISTS FOR (c:__Community__) ON (c.level);
CREATE INDEX text_unit_idx IF NOT EXISTS FOR (t:__TextUnit__) ON (t.id);

// For vector similarity (if using embeddings in Neo4j)
CREATE VECTOR INDEX entity_embedding_idx IF NOT EXISTS
FOR (e:__Entity__) ON (e.description_embedding)
OPTIONS {indexConfig: {
  `vector.dimensions`: 1536,
  `vector.similarity_function`: 'cosine'
}};
```

#### Memory Issues
```bash
# Monitor Neo4j memory usage
docker stats graphrag-neo4j

# Or check Neo4j logs for memory warnings
docker logs graphrag-neo4j | grep -i memory

# Tune GraphRAG search parameters for less memory usage
# In settings.yaml:
neo4j:
  local_search:
    top_k_entities: 5      # Reduce from 10
    max_context_tokens: 4000  # Reduce from 8000
  global_search:
    max_data_tokens: 4000     # Reduce from 8000
```

#### Network Latency (for Remote Neo4j)
```yaml
# Optimize for remote Neo4j instances
neo4j:
  uri: "neo4j+s://your-aura-instance.databases.neo4j.io:7687"
  # Use connection pooling and retry settings
  connection_acquisition_timeout: 60
  max_connection_lifetime: 3600
  max_connection_pool_size: 50
```

### 6. Data Quality Issues

#### Verify Data Import Completeness
```cypher
// Check imported data counts
MATCH (e:__Entity__) RETURN count(e) as entities;
MATCH (r:__Relationship__) RETURN count(r) as relationships;
MATCH (t:__TextUnit__) RETURN count(t) as text_units;
MATCH (c:__Community__) RETURN count(c) as communities;

// Compare with parquet file row counts
// python -c "import pandas as pd; print(f'Entities: {len(pd.read_parquet(\"output/create_final_entities.parquet\"))}')"
```

#### Check for Missing Relationships
```cypher
// Find entities without relationships
MATCH (e:__Entity__)
WHERE NOT (e)-[:RELATED]-()
RETURN count(e) as isolated_entities;

// Should be minimal for good graph connectivity
```

### 7. Version Compatibility Issues

```bash
# Check versions for compatibility
python -c "
import sys
import neo4j
import pandas as pd
import lancedb

print(f'Python: {sys.version}')
print(f'Neo4j Driver: {neo4j.__version__}')
print(f'Pandas: {pd.__version__}')
print(f'LanceDB: {lancedb.__version__}')

# Compatible versions:
# - Python: 3.10+
# - Neo4j Driver: 5.0+
# - Pandas: 2.0+
# - LanceDB: 0.3+
"

# Update if needed
pip install --upgrade neo4j pandas lancedb
```

### 8. Getting Help

If you continue to experience issues:

1. **Check Neo4j Browser**: http://localhost:7474
2. **Review Neo4j logs**: `docker logs graphrag-neo4j`
3. **Enable debug logging** in GraphRAG:
   ```yaml
   # In settings.yaml
   logging:
     level: DEBUG
   ```
4. **Test with minimal data**: Try with a small document set first
5. **Community support**: 
   - GraphRAG GitHub Issues
   - Neo4j Community Forum
   - Stack Overflow with tags: `graphrag`, `neo4j`

## Development

### Running Tests

```bash
# Install development dependencies
pip install -e .[dev]

# Run tests
pytest tests/

# Run specific Neo4j integration tests
pytest tests/test_neo4j_integration.py -v
```

### Building Package

```bash
# GraphRAG uses Poetry for package management
poetry build

# Install from built package
pip install dist/graphrag-*.whl

# Or install directly from source with Poetry
poetry install --extras neo4j
```

## Production Deployment

### 1. Environment Setup

```bash
# Set production environment variables
export GRAPHRAG_GRAPH_DB=neo4j
export AWS_DEFAULT_REGION=us-east-1
export NEO4J_URI="neo4j://your-production-neo4j:7687"
export NEO4J_USERNAME="production_user"
export NEO4J_PASSWORD="secure_password"
```

### 2. Configuration Management

Use environment variable substitution in settings.yaml:

```yaml
neo4j:
  enabled: true
  uri: ${NEO4J_URI}
  username: ${NEO4J_USERNAME}
  password: ${NEO4J_PASSWORD}
  database: ${NEO4J_DATABASE:-neo4j}

# LanceDB path is configured separately in vector_store section
vector_store:
  default_vector_store:
    type: lancedb
    db_uri: ${LANCEDB_PATH:-output/lancedb}
    container_name: default
    overwrite: true
```

### 3. Security Best Practices

#### Secure Neo4j Configuration
```bash
# Use strong passwords
NEO4J_PASSWORD=$(openssl rand -base64 32)

# Enable SSL/TLS for production
# In neo4j.conf:
echo "dbms.connector.bolt.tls_level=REQUIRED" >> conf/neo4j.conf
echo "dbms.connector.https.enabled=true" >> conf/neo4j.conf

# Restrict network access
echo "dbms.connector.bolt.listen_address=127.0.0.1:7687" >> conf/neo4j.conf
```

#### Secure Credentials Management
```yaml
# Use environment variable substitution in settings.yaml
neo4j:
  enabled: true
  uri: ${NEO4J_URI}
  username: ${NEO4J_USERNAME}
  password: ${NEO4J_PASSWORD}
  database: ${NEO4J_DATABASE:-neo4j}

# Store credentials in secure environment variables or secrets manager
# Never commit passwords to version control
```

#### AWS Secrets Manager Integration (Production)
```bash
# Store Neo4j credentials in AWS Secrets Manager
aws secretsmanager create-secret \
  --name graphrag/neo4j/credentials \
  --description "Neo4j credentials for GraphRAG" \
  --secret-string '{
    "username": "neo4j",
    "password": "your-secure-password",
    "uri": "neo4j+s://your-aura-instance.databases.neo4j.io:7687"
  }'

# Retrieve in application
python -c "
import boto3
import json
client = boto3.client('secretsmanager')
secret = client.get_secret_value(SecretId='graphrag/neo4j/credentials')
credentials = json.loads(secret['SecretString'])
print(f'URI: {credentials[\"uri\"]}')
"
```

#### Network Security
```bash
# For production Neo4j deployments:
# 1. Use VPC/private networks
# 2. Configure security groups to allow only necessary traffic
# 3. Use Neo4j Aura for managed security
# 4. Enable audit logging

# Neo4j Aura provides:
# - Automatic encryption in transit and at rest
# - Network isolation
# - Regular security updates
# - Compliance certifications
```

### 4. Monitoring and Observability

#### Neo4j Performance Monitoring
```cypher
// Check active queries
SHOW TRANSACTIONS YIELD transactionId, status, database, currentQuery;

// Monitor memory usage  
CALL dbms.memory();

// Check database statistics
CALL apoc.meta.stats();

// Query performance metrics
CALL dbms.queryManagement.listQueries();

// Database size and growth
CALL apoc.meta.graph();
```

#### Application Monitoring
```yaml
# Enable GraphRAG logging in settings.yaml
logging:
  level: INFO
  handlers:
    - type: file
      filename: graphrag.log
    - type: console

# Monitor key metrics:
# - Query response times
# - Neo4j connection pool usage  
# - Memory usage
# - Error rates
```

#### Health Checks
```bash
# Neo4j health check endpoint
curl http://localhost:7474/db/neo4j/ping

# Custom health check script
python -c "
from graphrag.neo4j_integration import is_neo4j_available
from neo4j import GraphDatabase
import os

# Test connection
try:
    driver = GraphDatabase.driver(
        os.getenv('NEO4J_URI', 'neo4j://localhost:7687'),
        auth=(os.getenv('NEO4J_USERNAME', 'neo4j'), os.getenv('NEO4J_PASSWORD'))
    )
    with driver.session() as session:
        result = session.run('RETURN count(*) as node_count')
        count = result.single()['node_count']
        print(f'✅ Neo4j healthy - {count} nodes in database')
    driver.close()
except Exception as e:
    print(f'❌ Neo4j health check failed: {e}')
    exit(1)
"
```

### 5. Backup and Disaster Recovery

#### Neo4j Backup Strategy
```bash
# For Neo4j Enterprise (automated backups)
neo4j-admin database backup --database=neo4j --to-path=/backups/

# For Neo4j Community (manual export)
cypher-shell -u neo4j -p password \
  "CALL apoc.export.cypher.all('backup.cypher', {format: 'cypher-shell'})"

# For Neo4j Aura (automated backups included)
# Backups are handled automatically by Neo4j Aura
```

#### Restore Procedures
```bash
# Restore from backup (Neo4j Enterprise)
neo4j-admin database restore --from-path=/backups/neo4j

# Restore from Cypher export
cypher-shell -u neo4j -p password < backup.cypher

# Test restore in staging environment first
```

## Quick Reference

### Essential Commands
```bash
# Installation
pip install -e .[neo4j]

# Setup Neo4j (Docker)
docker run -d --name graphrag-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your-password \
  -e NEO4J_PLUGINS='["apoc"]' \
  neo4j:5.15

# Initialize GraphRAG project
graphrag init

# Run indexing
graphrag index

# Import to Neo4j
export NEO4J_URI="neo4j://localhost:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="your-password"
python neo4j_import_script_enhanced.py

# Query with Neo4j
export GRAPHRAG_GRAPH_DB=neo4j
graphrag query --method local --query "Your question"
```

### Essential Configuration (settings.yaml)
```yaml
# Models (choose one)
models:
  default_chat_model:
    type: bedrock_chat  # or openai_chat, azure_openai_chat
    model: us.anthropic.claude-3-7-sonnet-20250219-v1:0

  default_embedding_model:
    type: bedrock_embedding  # or openai_embedding, azure_openai_embedding
    model: amazon.titan-embed-text-v2:0

# Vector store
vector_store:
  default_vector_store:
    type: lancedb
    db_uri: "output/lancedb"

# Neo4j backend
neo4j:
  enabled: true
  uri: "neo4j://localhost:7687"
  username: "neo4j"
  password: "your-password"
  database: "neo4j"
```

### Essential Neo4j Indexes
```cypher
CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:__Entity__) ON (e.name);
CREATE INDEX entity_degree_idx IF NOT EXISTS FOR (e:__Entity__) ON (e.degree);
CREATE INDEX community_level_idx IF NOT EXISTS FOR (c:__Community__) ON (c.level);
CREATE INDEX text_unit_idx IF NOT EXISTS FOR (t:__TextUnit__) ON (t.id);
```

### Backend Selection Priority
1. CLI option: `--graphrag-graph-db neo4j`
2. Environment variable: `GRAPHRAG_GRAPH_DB=neo4j`
3. Configuration: `neo4j.enabled: true` in settings.yaml
4. Default: Parquet files

### Key URLs and Resources
- **Neo4j Browser**: http://localhost:7474
- **GraphRAG Documentation**: https://microsoft.github.io/graphrag/
- **Neo4j Documentation**: https://neo4j.com/docs/
- **APOC Documentation**: https://neo4j.com/docs/apoc/

## Summary

With this enhanced GraphRAG package, you can:

1. **Install easily**: `pip install -e .[neo4j]`
2. **Configure simply**: Set `GRAPHRAG_GRAPH_DB=neo4j`
3. **Use immediately**: Same CLI commands with better performance
4. **Scale efficiently**: Neo4j graph database backend
5. **Maintain compatibility**: Falls back to parquet files when needed

The integration is designed to be **seamless**, **performant**, and **production-ready** while maintaining full compatibility with existing GraphRAG workflows.

### Next Steps
1. Follow the [Complete End-to-End Setup Workflow](#complete-end-to-end-setup-workflow)
2. Review [Security Best Practices](#security-best-practices) for production
3. Optimize performance with [Neo4j indexes](#essential-neo4j-indexes)
4. Monitor your deployment using [health checks](#health-checks)
5. Join the community for support and updates
