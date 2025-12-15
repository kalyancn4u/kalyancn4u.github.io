---
layout: post
title: "💡 Prompting: CO-STAR Framework"
description: "CO-STAR framework prompting technique"
author: technical_notes
date: 2025-12-09 00:00:00 +0530
categories: [Notes, CO-STAR]
tags: [CO-STAR, Prompting, Technique, Prompt Engineering]
image:
  path: /assets/img/posts/costar_framework.webp
  alt: "CO-STAR Framework Diagram"
  class: img-center
css: [ "/assets/css/custom.css" ]
toc: true
math: false
mermaid: false
---

<style>
  /* Base message box */
.msg {
    padding: .7rem 1rem;
    border-left: 4px solid;
    border-radius: 4px;
    margin: 1rem 0;
    font-size: .95rem;
}

/* Chirpy-aligned colors (using theme neutrals + accent colors) */

/* INFO – uses Chirpy link/accent blue */
/* SUCCESS – uses GitHub-style green used in Chirpy buttons */
/* WARNING – soft amber matching Chirpy alert tone */
/* DANGER – GitHub/Chirpy red */
/* NOTE – aligns to Chirpy blockquote border color */
/* TIP – teal accent (Chirpy supports cyan/teal utilities) */
/* DEBUG – Chirpy neutral gray family */
/* QUOTE – aligned with Chirpy blockquote styling */

.msg-info {
    background: #e8f1fc;
    border-color: #1a73e8;
    color: #0b3d91;
}

/* Chirpy link color */
.msg-success {
    background: #e8f6ec;
    border-color: #2da44e;
    color: #0f5227;
}

/* GitHub green */
.msg-warning {
    background: #fff8e6;
    border-color: #d97706;
    color: #8a5300;
}

/* warm amber */
.msg-danger {
    background: #fcebea;
    border-color: #cf222e;
    color: #8a1c1f;
}

/* GitHub danger red */
.msg-note {
    background: #f5f7fa;
    border-color: #6cb6ff;
    color: #244466;
}

/* soft blue */
.msg-tip {
    background: #e6f7f6;
    border-color: #0d9488;
    color: #065f5b;
}

/* teal-600 */
.msg-debug {
    background: #f3f4f6;
    border-color: #9ca3af;
    color: #374151;
    font-family: monospace;
}

/* gray-400 */
/* gray-700 */
.msg-quote {
    background: #fafafa;
    border-color: #d1d5db;
    color: #4b5563;
    font-style: italic;
}

/* gray-300 */
/* gray-600 */
</style>
## 🌟 What is CO-STAR?

CO-STAR is a **structured prompting framework** that helps you guide LLMs with clarity and intent.

- **C** — Context  
- **O** — Objective  
- **S** — Style  
- **T** — Tone  
- **A** — Audience  
- **R** — Response format  

---

![CO-STAR Framework](/assets/img/posts/costar_framework.svg){: width="800" }
_CO-STAR Framework — visual overview_

---

<!-- MESSAGE BLOCKS (HTML ONLY, SAFE ZONE) -->

<div class="msg msg-info">
<strong>Info:</strong> CO-STAR improves consistency and reduces ambiguity in prompts.
</div>

<div class="msg msg-success">
<strong>Success:</strong> Structured prompts yield more predictable outputs.
</div>

<div class="msg msg-warning">
<strong>Warning:</strong> Over-constraining can reduce creativity.
</div>

<div class="msg msg-danger">
<strong>Error:</strong> Missing context often leads to hallucinations.
</div>

<div class="msg msg-note">
<strong>Note:</strong> CO-STAR works best with iterative refinement.
</div>

<div class="msg msg-tip">
<strong>Tip:</strong> Start minimal, then layer constraints gradually.
</div>

<div class="msg msg-debug">
<strong>Debug:</strong> context="insufficient", tone="neutral"
</div>

<div class="msg msg-quote">
“Good prompts don’t command — they guide.”
</div>

---

## 🧠 When to Use CO-STAR

- Prompt engineering
- Documentation generation
- AI tutoring
- System instruction design
- Agent workflows

---

⚑ **Takeaway:**  
CO-STAR brings *discipline* to creativity — structure without suffocation.

---

![Co-star framework](/assets/img/posts/costar_structured_prompting.webp){: width="500" }
_Prompting: Co-star framework Illustrated!_

---

<pre>
image: /assets/img/posts/costar_framework.svg

---
layout: post
title: "💡 Prompting: CO-STAR Framework"
description: "CO-STAR framework prompting technique"
author: technical_notes
date: 2025-12-09 00:00:00 +0530
categories: [Notes, CO-STAR]
tags: [CO-STAR, Prompting, Technique, Prompt Engineering]
image:
  path: /assets/img/posts/costar_framework.webp
  alt: "CO-STAR Framework Diagram"
  class: img-center
css: [ "/assets/css/custom.css" ]
toc: true
math: false
mermaid: false
---

![Co-star framework](/assets/img/posts/costar_framework.svg){: w="400" h="200" }
_Prompting: Co-star framework Illustrated!_
</pre>

---

⚑ **Why Log Levels Matter (in Simple Words)**

Log levels help organize application messages by **importance**, so developers can quickly understand *what’s happening* without drowning in noise.

Think of them like volume controls for information 🔊 —
you turn up details when debugging, and turn them down in production.

---

🧭 **Common Log Levels (Most → Least Severe)**

- **CRITICAL / FATAL** – Something went terribly wrong. The app may not continue.
- **ERROR** – A serious problem affecting functionality that needs fixing.
- **WARNING (WARN)** – Something looks off; not broken yet, but could become a problem.
- **INFO** – Normal, useful updates (app started, user logged in, task completed).
- **DEBUG** – Detailed information for developers to investigate issues.
- **TRACE** – Extremely fine-grained, step-by-step execution details.
- **OFF** – Turns logging completely off.

<div class="msg msg-danger">
<strong>CRITICAL / FATAL - </strong> Something went terribly wrong. The app may not continue.
</div>

<div class="msg msg-danger">
<strong>ERROR – </strong> A serious problem affecting functionality that needs fixing.
</div>

<div class="msg msg-warning">
<strong>WARNING (WARN) - </strong> Something looks off; not broken yet, but could become a problem.
</div>

<div class="msg msg-info">
<strong>INFO - </strong> Normal, useful updates (app started, user logged in, task completed).
</div>

<div class="msg msg-debug">
<strong>DEBUG - </strong> Detailed information for developers to investigate issues.
</div>

<div class="msg msg-success">
<strong>SUCCESS - </strong> Everything is okay.
</div>

<div class="msg msg-note">
<strong>NOTE / TRACE - </strong> Extremely fine-grained, step-by-step execution details.
</div>

<div class="msg msg-tip">
<strong>TIP / OFF - </strong> Turns logging completely off / just highlight.
</div>

<div class="msg msg-quote">
<strong>QUOTE - </strong> Just highlight with visual demarcation for distinction for Reading.
</div>

---

✨ **Why Developers Use Log Levels**

- **Less Noise** – See only what matters in production.
- **Better Debugging** – Enable DEBUG or TRACE when chasing bugs.
- **Faster Alerts** – Critical errors can trigger emails or notifications.
- **Clarity** – Clean logs make systems easier to understand and maintain.

<div class="msg msg-info">Info: Tailwind simplifies styling.</div>
<div class="msg msg-success">Success: Build completed.</div>
<div class="msg msg-warning">Warning: Token expiry soon.</div>
<div class="msg msg-danger">Error: Deployment failed.</div>
<div class="msg msg-note">Note: Jekyll regenerates on file changes.</div>
<div class="msg msg-tip">Tip: Prefer Markdown includes for reuse.</div>
<div class="msg msg-debug">Debug: x=42, mode="test"</div>
<div class="msg msg-quote">“Good code is simple code.”</div>

---

🌱 **In short:**
Log levels keep logs **useful, readable, and purposeful** —
quiet when everything is fine, loud when something breaks.

---
