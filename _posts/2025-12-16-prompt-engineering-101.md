---
layout: post
title: "💡 Prompt Engineering 101: Deep Dive & Best Practices"
description: "CO-STAR framework prompting technique"
author: technical_notes
date: 2025-12-16 00:00:00 +0530
categories: [Notes, Prompting]
tags: [Prompting, Techniques, Prompt Engineering, LLM. Chain-of-thought, Few-shot, CO-STAR, Best Practices]
image: /assets/img/posts/prompt_engineering.webp
toc: true
math: true
mermaid: true
---

## Table of Contents
- [Introduction](#introduction)
- [What is Prompt Engineering?](#what-is-prompt-engineering)
- [Why Prompt Engineering Matters](#why-prompt-engineering-matters)
- [Core Concepts and Terminology](#core-concepts-and-terminology)
- [Prompt Types and Techniques](#prompt-types-and-techniques)
- [CO-STAR Framework](#co-star-framework)
- [Advanced Prompting Techniques](#advanced-prompting-techniques)
- [Comparison of Prompting Frameworks](#comparison-of-prompting-frameworks)
- [Best Practices](#best-practices)
- [Common Pitfalls and How to Avoid Them](#common-pitfalls-and-how-to-avoid-them)
- [References](#references)

---

## Introduction

Prompt engineering has emerged as one of the most critical skills in the age of large language models (LLMs). As artificial intelligence systems like GPT-4, Claude, and Gemini become increasingly integrated into everyday workflows, the ability to communicate effectively with these models has transformed from a novelty into a necessity.

This comprehensive guide explores prompt engineering from foundational principles to advanced techniques, with special emphasis on the CO-STAR framework and comparative analysis of various prompting methods. Whether you're a beginner looking to understand the basics or an experienced practitioner seeking to refine your skills, this guide provides actionable insights validated across multiple trusted sources.

---

## What is Prompt Engineering?

**Prompt engineering** is the practice of designing and refining inputs (prompts) to guide large language models toward producing desired outputs. It serves as the interface between human intent and machine understanding, transforming vague requests into precise, actionable instructions.

### Definition

At its core, prompt engineering involves:

- **Crafting clear instructions** that minimize ambiguity
- **Providing relevant context** to guide model behavior
- **Structuring inputs** to optimize output quality
- **Iterating and refining** based on results

### The Evolution

Prompt engineering evolved from simple query-based interactions to sophisticated frameworks:

1. **Early NLP Era** (1950s-2000s): Rule-based systems with rigid structures
2. **Statistical NLP** (2000s-2017): Machine learning with limited contextual understanding
3. **Transformer Era** (2017-present): Self-attention mechanisms enabling nuanced language processing
4. **Modern Prompt Engineering** (2020-present): Systematic approaches to guide increasingly capable models

The introduction of models like GPT-3 and GPT-4 marked a paradigm shift—suddenly, the quality of outputs became heavily dependent on how questions were asked rather than just what was asked.

---

## Why Prompt Engineering Matters

### Performance Optimization

Well-crafted prompts can dramatically improve model performance without requiring fine-tuning or additional training. This makes prompt engineering:

- **Cost-effective**: No infrastructure changes needed
- **Accessible**: Requires language skills, not necessarily technical expertise
- **Fast**: Immediate iteration and testing possible

### Bridging Intent and Understanding

LLMs are powerful but not omniscient. Prompt engineering helps:

- Align model outputs with human expectations
- Reduce hallucinations and factual errors
- Control tone, style, and structure
- Ensure safety and compliance

### Business Impact

Organizations leveraging prompt engineering report:

- **17% to 91% accuracy improvements** in classification tasks
- **Reduced review time** in legal and compliance workflows
- **Enhanced triage accuracy** in customer support
- **Better diagnostic precision** in healthcare applications

---

## Core Concepts and Terminology

### Key Terms

| Term | Definition |
|------|------------|
| **Prompt** | The input text or instruction given to an LLM |
| **Context** | Background information provided to help the model understand the scenario |
| **Token** | The basic unit of text processing (roughly 4 characters or ¾ of a word) |
| **Temperature** | Controls randomness in outputs (0 = deterministic, higher = more creative) |
| **System Message** | Instructions that define the model's behavior or role |
| **Few-shot Learning** | Providing examples to demonstrate desired behavior |
| **Zero-shot Learning** | Making requests without examples |
| **Hallucination** | When models generate plausible but false information |

### Prompt Components

Effective prompts typically include:

1. **Instruction**: Clear directive about what to do
2. **Context**: Background information or scenario
3. **Input Data**: The specific information to process
4. **Output Format**: Desired structure or style of response
5. **Examples** (optional): Demonstrations of expected behavior
6. **Constraints**: Limitations or boundaries for the response

---

## Prompt Types and Techniques

### 1. Zero-Shot Prompting

**Definition**: Providing direct instructions without examples.

**When to Use**: 
- Simple, well-defined tasks
- When the model has strong baseline knowledge
- Time-constrained scenarios

**Example**:
```
Summarize the following customer support chat in three bullet points, 
focusing on the issue, customer sentiment, and resolution.
```

**Strengths**: Fast, simple, requires minimal setup
**Weaknesses**: Less control over format, may miss nuances

---

### 2. Few-Shot Prompting

**Definition**: Providing 2-5 examples to demonstrate the desired pattern.

**When to Use**:
- Teaching specific formats or styles
- Domain-specific tasks
- When consistency matters

**Example**:
```
Classify sentiment:

Review: "The product exceeded my expectations!"
Sentiment: Positive

Review: "Terrible quality, broke after one day."
Sentiment: Negative

Review: "It works okay, nothing special."
Sentiment: Neutral

Review: "Amazing customer service and fast delivery."
Sentiment: [Model completes]
```

**Strengths**: High accuracy, format control, style consistency
**Weaknesses**: Requires example preparation, uses more tokens

---

### 3. Chain-of-Thought (CoT) Prompting

**Definition**: Guiding the model through step-by-step reasoning.

**When to Use**:
- Complex problem-solving
- Mathematical calculations
- Logic-based tasks
- Troubleshooting

**Example**:
```
Let's solve this step by step:

Question: If a train travels at 60 mph for 2.5 hours, 
then 75 mph for 1.5 hours, what's the total distance?

Step 1: Calculate distance for first segment
Distance = Speed × Time = 60 × 2.5 = 150 miles

Step 2: Calculate distance for second segment
Distance = 75 × 1.5 = 112.5 miles

Step 3: Add both segments
Total = 150 + 112.5 = 262.5 miles

Answer: 262.5 miles
```

**Strengths**: Improved accuracy, transparent reasoning, easier to debug
**Weaknesses**: Verbose, slower generation

---

### 4. Self-Consistency

**Definition**: Generating multiple reasoning paths and selecting the most consistent answer.

**When to Use**:
- High-stakes decisions
- Verification of complex reasoning
- Tasks with multiple valid approaches

**How It Works**:
1. Generate multiple CoT responses (3-5)
2. Extract final answers from each
3. Select the most frequent answer

**Performance Gains**:
- GSM8K: +17.9% accuracy
- SVAMP: +11.0% accuracy
- AQuA: +12.2% accuracy

---

### 5. Tree of Thoughts (ToT)

**Definition**: Exploring multiple reasoning branches and pruning unpromising paths.

**When to Use**:
- Complex planning tasks
- Creative problem-solving
- Tasks requiring exploration of alternatives

**Core Questions**:
1. How to decompose into thought steps?
2. How to generate potential thoughts?
3. How to evaluate states?
4. What search algorithm to use?

**Example Application**: Game of 24
- Standard prompting: 4% success
- Chain-of-Thought: 4% success
- Tree of Thoughts (b=5): 74% success

---

### 6. ReAct (Reasoning + Acting)

**Definition**: Combining reasoning traces with action generation for interactive tasks.

**When to Use**:
- Information retrieval tasks
- Multi-step research
- Decision-making with external data

**Structure**:
```
Thought 1: I need to find information about X
Action 1: Search[X]
Observation 1: [Search results]
Thought 2: Based on the results, I should...
Action 2: [Next action]
```

**Strengths**: Reduces hallucination, enables self-correction, interpretable
**Weaknesses**: Requires external tools/APIs, slower than direct generation

---

### 7. Metacognitive Prompting

**Definition**: Mimicking human introspective reasoning through structured self-evaluation.

**Five Stages**:
1. **Understanding**: Comprehend the input text
2. **Preliminary Judgment**: Form initial interpretation
3. **Critical Evaluation**: Self-scrutinize the judgment
4. **Final Decision**: Conclude with reasoning
5. **Confidence Assessment**: Gauge certainty

**Performance**: Consistently outperforms standard prompting and CoT variants across NLU tasks.

---

## CO-STAR Framework

The **CO-STAR framework**, developed by GovTech Singapore's Data Science & AI Division, provides a structured template for creating effective prompts that consider all key aspects influencing LLM responses.

### Framework Components

| Component | Abbreviation | Purpose | Example |
|-----------|--------------|---------|---------|
| **Context** | C | Background information to help the model understand the scenario | "I am a social media manager for a tech startup launching a new product" |
| **Objective** | O | Clearly defined task or goal | "Create a Facebook post to drive product page visits and purchases" |
| **Style** | S | Desired writing style or format | "Follow the writing style of successful tech companies like Apple—concise, benefit-focused, aspirational" |
| **Tone** | T | Emotional context and manner | "Professional yet approachable, enthusiastic but not overly promotional" |
| **Audience** | A | Target demographic or reader profile | "Tech-savvy millennials and Gen Z, aged 25-40, interested in productivity tools" |
| **Response** | R | Expected output format or structure | "A 2-3 sentence post with an engaging hook, key benefit, and clear call-to-action. Include 3 relevant hashtags." |

### Why CO-STAR Works

1. **Comprehensive Coverage**: Addresses all dimensions affecting output quality
2. **Structured Approach**: Reduces ambiguity and increases consistency
3. **Versatile**: Applicable across domains and use cases
4. **Iterative**: Easy to refine individual components
5. **Model-Agnostic**: Works across GPT-4, Claude, Gemini, and others

### CO-STAR vs. Basic Prompting

**Without CO-STAR**:
```
Write a Facebook post for my new product.
```

Result: Generic, lacks specificity, misses audience targeting

**With CO-STAR**:
```
# CONTEXT #
I am a social media manager for Alpha Tech. We're launching Beta, 
an ultra-fast hairdryer targeting environmentally conscious consumers.

# OBJECTIVE #
Create a Facebook post that drives clicks to our product page and 
highlights our sustainability features.

# STYLE #
Follow Dyson's approach: technical sophistication made accessible, 
emphasis on innovation.

# TONE #
Professional but warm, informative yet exciting, eco-conscious.

# AUDIENCE #
Eco-aware consumers aged 25-45, interested in sustainable 
premium appliances.

# RESPONSE #
A 3-sentence post with: (1) attention-grabbing opening about 
environmental impact, (2) key product benefit, (3) call-to-action. 
Include 2-3 hashtags focused on sustainability and innovation.
```

Result: Targeted, on-brand, actionable content that resonates with the specific audience

### CO-STAR Implementation Tips

#### For Different Models

**GPT-4**:
- Use markdown headers (# CONTEXT #, # OBJECTIVE #)
- Clear section separation improves adherence
- Works well with explicit formatting instructions

**Claude**:
- Highly responsive to role-based context
- Benefits from XML-style tags (<context>, <objective>)
- Excellent at following detailed stylistic guidelines

**Gemini**:
- Prefers hierarchical structure
- Strong with explicit audience definitions
- Best results with concrete output examples

#### Optimization Techniques

1. **Start Minimal, Add Complexity**: Begin with C, O, R; add S, T, A as needed
2. **Use Examples**: In the Response section, show format examples
3. **Iterate Components**: Refine one element at a time based on outputs
4. **Combine with Other Techniques**: Layer CoT or few-shot with CO-STAR
5. **Document Successful Patterns**: Build a library of effective CO-STAR prompts

---

## Advanced Prompting Techniques

### Prompt Chaining

**Definition**: Breaking complex tasks into sequential prompts, where each output feeds into the next.

**When to Use**:
- Multi-stage workflows
- Quality control pipelines
- Complex content generation

**Example**:
```
Prompt 1: "Extract key themes from this research paper."
→ Output 1: [List of themes]

Prompt 2: "For each theme in [Output 1], identify supporting evidence 
from the paper."
→ Output 2: [Themes with evidence]

Prompt 3: "Create a 200-word executive summary using [Output 2]."
→ Final Output
```

**Benefits**: Better quality control, easier debugging, modular design

---

### Retrieval-Augmented Generation (RAG)

**Definition**: Combining LLM generation with external knowledge retrieval.

**Architecture**:
1. User query → Vector database search
2. Retrieve relevant documents
3. Inject documents into prompt context
4. Generate response using retrieved information

**Advantages**:
- Reduces hallucinations
- Access to current information
- Domain-specific knowledge integration

---

### Directional Stimulus Prompting

**Definition**: Using hints or cues to guide the model toward desired reasoning directions.

**Example**:
```
Instead of: "Solve this equation"

Use: "Hint: Consider isolating the variable first. 
Now solve: 3x + 7 = 22"
```

**Results**: Improved accuracy on reasoning tasks without full CoT overhead

---

### Active-Prompt

**Definition**: Automatically selecting the most effective examples for few-shot prompting.

**Process**:
1. Generate multiple candidate prompts
2. Evaluate uncertainty/confidence
3. Select examples that maximize model confidence
4. Use selected examples in final prompt

---

### Automatic Prompt Engineer (APE)

**Definition**: Using LLMs to generate and optimize prompts automatically.

**Workflow**:
1. Describe the task
2. LLM generates candidate prompts
3. Test prompts on validation set
4. Select best-performing prompt
5. Optionally iterate for refinement

---

## Comparison of Prompting Frameworks

### Technique Comparison Table

| Technique | Complexity | Token Usage | Accuracy Gain | Best For | Limitations |
|-----------|------------|-------------|---------------|----------|-------------|
| **Zero-Shot** | Low | Low | Baseline | Simple tasks, broad knowledge | Limited control, inconsistent |
| **Few-Shot** | Medium | Medium | +20-40% | Format control, style matching | Requires examples, token cost |
| **Chain-of-Thought** | Medium | High | +15-30% | Math, logic, reasoning | Verbose, slower |
| **Self-Consistency** | High | Very High | +10-20% (over CoT) | High-stakes decisions | Expensive, slow |
| **Tree of Thoughts** | Very High | Very High | +50-70% (planning tasks) | Complex planning, exploration | Computationally intensive |
| **ReAct** | High | High | +15-25% | Research, fact-checking | Requires external tools |
| **Metacognitive** | High | High | +10-15% (over CoT) | Understanding tasks | Complex implementation |
| **CO-STAR** | Medium | Medium | +30-50% | All tasks | Requires upfront planning |

### When to Use Each Framework

#### Use CO-STAR When:
- Building production applications
- Need consistent, predictable outputs
- Working across multiple models
- Onboarding non-technical users
- Require clear documentation

#### Use Chain-of-Thought When:
- Solving mathematical problems
- Debugging code
- Making logical deductions
- Explaining complex concepts

#### Use Tree of Thoughts When:
- Creative problem-solving
- Game-like scenarios
- Multi-path decision-making
- Strategic planning

#### Use ReAct When:
- Researching factual information
- Building AI agents
- Verifying claims
- Multi-step information gathering

#### Use Metacognitive When:
- Natural language understanding tasks
- Sentiment analysis
- Question answering
- Paraphrase detection

### Hierarchical Framework Comparison

```
Level 1: Basic Prompting
├── Zero-Shot (simplest)
└── Few-Shot (adds examples)

Level 2: Structured Frameworks
├── CO-STAR (comprehensive structure)
└── Role-Based Prompting (persona assignment)

Level 3: Reasoning Enhancement
├── Chain-of-Thought (step-by-step)
├── Self-Consistency (multiple paths)
└── Metacognitive (introspective)

Level 4: Advanced Exploration
├── Tree of Thoughts (multi-branch)
└── ReAct (reasoning + action)

Level 5: Automated/Hybrid
├── Prompt Chaining (sequential)
├── RAG (retrieval-augmented)
└── Automatic Prompt Engineering
```

### Technique Synergy

Many advanced applications combine multiple techniques:

**Example 1: Enterprise Q&A System**
- RAG (retrieval) + CO-STAR (structure) + Few-Shot (examples)

**Example 2: AI Research Assistant**
- ReAct (tool use) + Chain-of-Thought (reasoning) + Self-Consistency (verification)

**Example 3: Creative Writing Tool**
- Tree of Thoughts (exploration) + CO-STAR (output control) + Few-Shot (style)

---

## Best Practices

### 1. Clarity and Specificity

**DO**:
- Use precise language
- Specify desired output format
- Set clear constraints (length, style, tone)
- Define success criteria

**DON'T**:
- Use vague instructions ("make it better")
- Assume the model understands context
- Mix multiple unrelated tasks in one prompt

**Example**:
```
❌ "Write about climate change"

✅ "Write a 300-word article explaining the top 3 impacts of 
climate change on coastal cities for a high school science 
magazine. Use accessible language and include one real-world example 
for each impact."
```

### 2. Context Provision

**Effective Context Includes**:
- Relevant background information
- Domain-specific terminology
- User or audience characteristics
- Constraints or requirements

**Tip**: Use delimiters to separate context from instruction
```
### Context ###
[Background information]

### Task ###
[Actual request]
```

### 3. Example Quality

When using few-shot prompting:

- **Diversity**: Show variation in inputs and edge cases
- **Consistency**: Maintain format across all examples
- **Clarity**: Use clean, unambiguous examples
- **Relevance**: Match examples to the actual task

### 4. Iteration Strategy

1. **Start Simple**: Begin with zero-shot
2. **Add Structure**: Implement frameworks like CO-STAR if needed
3. **Provide Examples**: Include few-shot examples for complex tasks
4. **Enable Reasoning**: Add CoT for logic-heavy tasks
5. **Test and Refine**: Iterate based on outputs

### 5. Model-Specific Optimization

**GPT-4 / GPT-4o**:
- Excellent with system messages
- Strong few-shot learning
- Benefits from explicit formatting

**Claude (Anthropic)**:
- Highly steerable with detailed instructions
- Prefers markdown and XML tags
- Strong constitutional AI alignment

**Gemini (Google)**:
- Large context windows (up to 1M tokens)
- Excellent for long-document tasks
- Prefers hierarchical structure

### 6. Safety and Ethical Considerations

- **Bias Mitigation**: Test prompts across diverse scenarios
- **Harmful Content**: Use explicit safety constraints
- **Privacy**: Never include sensitive personal information
- **Transparency**: Document prompt engineering decisions

### 7. Testing and Evaluation

**Create Test Suites**:
- Define success metrics (accuracy, relevance, style)
- Build diverse test cases
- Track performance across iterations
- Document failure modes

**A/B Testing**:
- Compare prompt variants
- Measure quantitative improvements
- Consider cost vs. performance trade-offs

---

## Common Pitfalls and How to Avoid Them

### 1. Vague Instructions

**Problem**: Model generates generic or off-target responses

**Solution**: Use CO-STAR framework, specify format, provide constraints

### 2. Insufficient Context

**Problem**: Model lacks necessary background to respond appropriately

**Solution**: 
- Provide relevant domain information
- Define key terms
- Set the scenario clearly

### 3. Over-Reliance on Examples

**Problem**: Few-shot examples that don't generalize

**Solution**:
- Test with edge cases
- Ensure example diversity
- Validate on unseen inputs

### 4. Ignoring Token Limits

**Problem**: Prompts exceed model context windows

**Solution**:
- Compress prompts without losing meaning
- Use prompt chaining for long tasks
- Summarize lengthy context

### 5. Not Testing Edge Cases

**Problem**: Prompts fail with unusual inputs

**Solution**:
- Create adversarial test cases
- Test with empty, malformed, or extreme inputs
- Use stress testing

### 6. Hallucination Acceptance

**Problem**: Model generates plausible but false information

**Solution**:
- Use RAG for factual tasks
- Request citations/sources
- Implement verification steps
- Apply temperature controls

### 7. Prompt Injection Vulnerabilities

**Problem**: Users can manipulate prompts to bypass safety measures

**Solution**:
- Use delimiter escaping
- Implement output filtering
- Apply security-focused prompt scaffolding
- Regular security audits

---

## Jargon Equivalency Tables

### Table 1: Lifecycle Phase Terminology

| General Term | Prompt Engineering | Traditional ML | Software Development |
|--------------|-------------------|----------------|---------------------|
| **Planning** | Prompt Design | Problem Formulation | Requirements Gathering |
| **Creation** | Prompt Writing | Model Selection | Coding |
| **Testing** | Prompt Validation | Model Evaluation | Unit Testing |
| **Refinement** | Prompt Iteration | Hyperparameter Tuning | Debugging |
| **Deployment** | Prompt Production | Model Deployment | Release |
| **Monitoring** | Output Analysis | Performance Monitoring | Production Monitoring |

### Table 2: Hierarchical Differentiation

```
Prompt Engineering Hierarchy
│
├── Fundamental Concepts (Entry Level)
│   ├── Prompt: The input text
│   ├── Context: Background information
│   ├── Instruction: What to do
│   └── Output: Model response
│
├── Basic Techniques (Intermediate)
│   ├── Zero-Shot: Direct instruction
│   ├── Few-Shot: Learning from examples
│   ├── Role Prompting: Persona assignment
│   └── Format Control: Structure specification
│
├── Advanced Frameworks (Advanced)
│   ├── CO-STAR: Structured prompting
│   ├── Chain-of-Thought: Step-by-step reasoning
│   ├── Self-Consistency: Multiple reasoning paths
│   └── Metacognitive: Introspective reasoning
│
├── Complex Techniques (Expert)
│   ├── Tree of Thoughts: Multi-path exploration
│   ├── ReAct: Reasoning + Action
│   ├── Prompt Chaining: Sequential workflows
│   └── RAG: External knowledge integration
│
└── Specialized Applications (Mastery)
    ├── Adversarial Testing: Security validation
    ├── Automated Prompt Engineering: APE
    ├── Multi-Modal Prompting: Image + Text
    └── Agent Orchestration: Complex systems
```

### Table 3: Technique Maturity Levels

| Maturity Level | Characteristics | Techniques | Skill Requirement |
|----------------|-----------------|------------|-------------------|
| **Level 0** | Ad-hoc queries | Basic questions | None |
| **Level 1** | Structured prompts | Zero-shot, role-based | Basic understanding |
| **Level 2** | Example-driven | Few-shot, format control | Intermediate |
| **Level 3** | Framework-based | CO-STAR, CoT | Advanced planning |
| **Level 4** | Multi-technique | ToT, ReAct, Self-Consistency | Expert knowledge |
| **Level 5** | Automated/Adaptive | APE, Dynamic optimization | Mastery + Programming |

---

## References

1. <a href="https://www.promptingguide.ai/techniques" target="_blank">Prompt Engineering Guide - Techniques</a>
2. <a href="https://cloud.google.com/discover/what-is-prompt-engineering" target="_blank">Google Cloud - What is Prompt Engineering</a>
3. <a href="https://www.datacamp.com/blog/what-is-prompt-engineering-the-future-of-ai-communication" target="_blank">DataCamp - What is Prompt Engineering: The Future of AI Communication</a>
4. <a href="https://www.lakera.ai/blog/prompt-engineering-guide" target="_blank">Lakera - The Ultimate Guide to Prompt Engineering</a>
5. <a href="https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview" target="_blank">Anthropic - Prompt Engineering with Claude</a>
6. <a href="https://ai.google.dev/gemini-api/docs/prompting-strategies" target="_blank">Google AI - Prompt Design Strategies for Gemini</a>
7. <a href="https://help.openai.com/en/articles/10032626-prompt-engineering-best-practices-for-chatgpt" target="_blank">OpenAI - Prompt Engineering Best Practices</a>
8. <a href="https://aclanthology.org/2024.naacl-long.106/" target="_blank">Wang & Zhao (2024) - Metacognitive Prompting Improves Understanding in Large Language Models</a>
9. <a href="https://arxiv.org/abs/2510.12637" target="_blank">Ohalete et al. (2025) - COSTAR-A: A Prompting Framework for Enhanced LLM Performance</a>
10. <a href="https://docs.datastax.com/en/ragstack/default-architecture/generation.html" target="_blank">DataStax - CO-STAR Framework for RAG Applications</a>
11. <a href="https://www.promptingguide.ai/techniques/tot" target="_blank">Tree of Thoughts (ToT) - Prompt Engineering Guide</a>
12. <a href="https://www.promptingguide.ai/techniques/react" target="_blank">ReAct Prompting - Prompt Engineering Guide</a>
13. <a href="https://zerotomastery.io/blog/tree-of-thought-prompting/" target="_blank">Zero to Mastery - Tree of Thoughts Prompting Guide</a>
14. <a href="https://www.mercity.ai/blog-post/advanced-prompt-engineering-techniques" target="_blank">Mercity AI - Advanced Prompt Engineering Techniques</a>
15. <a href="https://aws.amazon.com/blogs/machine-learning/implementing-advanced-prompt-engineering-with-amazon-bedrock/" target="_blank">AWS - Implementing Advanced Prompt Engineering with Amazon Bedrock</a>

---

*Last Updated: December 16, 2024*

*This guide is maintained as a living document and will be updated as new techniques and frameworks emerge.*
