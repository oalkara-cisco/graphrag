# AWS Bedrock Provider for GraphRAG

This provider adds support for AWS Bedrock language models to GraphRAG, including both chat and embedding models.

## Installation

Install GraphRAG with the Bedrock extra:

```bash
pip install graphrag[bedrock]
```

## Supported Models

### Chat Models
- **Claude**: `anthropic.claude-3-sonnet-*`, `anthropic.claude-3-haiku-*`, `anthropic.claude-instant-*`
- **Claude Cross-Region**: `us.anthropic.claude-sonnet-4-*`, `eu.anthropic.claude-*` (Cross-region inference profiles)
- **Llama**: `meta.llama2-*`, `meta.llama3-*`, `us.meta.llama-*` (Cross-region inference profiles)
- **Mistral**: `mistral.mistral-*`, `mistral.mixtral-*`, `us.mistral.*` (Cross-region inference profiles)
- **Cohere Command**: `cohere.command-*`, `us.cohere.command-*` (Cross-region inference profiles)
- **Amazon Titan**: `amazon.titan-text-*`, `us.amazon.titan-text-*` (Cross-region inference profiles)
- **AI21**: `ai21.j2-*`, `ai21.jamba-*`, `us.ai21.*` (Cross-region inference profiles)

### Embedding Models
- **Amazon Titan**: `amazon.titan-embed-text-v1`, `amazon.titan-embed-text-v2:0`
- **Cohere**: `cohere.embed-english-v3`, `cohere.embed-multilingual-v3`

### Cross-Region Inference Profiles

Cross-region inference profiles allow you to use models hosted in different regions for better performance, availability, and cost optimization. These models have region prefixes like `us.`, `eu.`, `ap.` etc.

**Benefits:**
- **Improved availability** - fallback to different regions
- **Better performance** - route to closest/fastest region  
- **Cost optimization** - use different regional pricing
- **Latest models** - access to newest model versions

**Note:** Cross-region inference profiles may use different API formats than regular models. For example, Claude cross-region profiles use `max_tokens` while regular Claude models use `max_tokens_to_sample`. GraphRAG automatically handles these differences.

## Configuration

Configure Bedrock models in your `settings.yml`:

```yaml
models:
  # Regular Bedrock model
  bedrock_chat:
    type: bedrock_chat
    model: anthropic.claude-3-sonnet-20240229-v1:0
    aws_region: us-east-1
    # aws_profile: default  # Optional - omit to use default credential chain
    max_tokens: 4096
    temperature: 0.7
    
  # Cross-region inference profile model
  bedrock_cross_region:
    type: bedrock_chat
    model: us.anthropic.claude-sonnet-4-20250514-v1:0  # Cross-region inference profile
    aws_region: us-east-1
    max_tokens: 4096
    temperature: 0.7
    
  bedrock_embedding:
    type: bedrock_embedding
    model: amazon.titan-embed-text-v2:0
    aws_region: us-east-1
    # aws_profile: my-profile  # Optional - omit for EC2 instance profiles
```

## Authentication

The provider uses boto3's standard credential chain in the following order:

### Option 1: AWS Profile (Explicit)
```yaml
models:
  bedrock_chat:
    aws_profile: my-profile  # Uses specific AWS profile
```

### Option 2: Default Credential Chain (Recommended for EC2)
```yaml
models:
  bedrock_chat:
    # aws_profile: omitted  # Falls back to credential chain
```

**Default credential chain includes:**
1. **Environment Variables** (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
2. **AWS credentials file** (`~/.aws/credentials`)
3. **EC2 Instance Profile** (automatic when running on EC2)
4. **ECS Task Role** (when running in containers)
5. **Other AWS credential sources**

### For EC2 Deployment
When deploying GraphRAG on EC2 instances, **omit the `aws_profile` setting** to automatically use the instance's IAM role. This is the most secure approach as it doesn't require storing credentials in configuration files.

## Required Permissions

Your AWS credentials need these IAM permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": "arn:aws:bedrock:*:*:model/*"
    }
  ]
}
```

## Features

- **Automatic parameter mapping** for different model families
- **Streaming support** for chat models
- **Batch optimization** for embeddings (Titan: 25/batch, Cohere: 96/batch)
- **Response caching** to minimize API calls
- **Retry logic** with exponential backoff
- **Full async/await support**

## Implementation Details

The provider implements GraphRAG's `ChatModel` and `EmbeddingModel` protocols, ensuring compatibility with all GraphRAG workflows. Model-specific parameter names and response formats are handled automatically, providing a consistent interface regardless of the underlying Bedrock model.
