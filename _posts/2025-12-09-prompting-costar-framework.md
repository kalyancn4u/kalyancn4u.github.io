---
layout: post
title: "💡 Prompting: CO-STAR Framework"
description: "CO-START framework prompting technique"
author: technical_notes
date: 2025-12-09 00:00:00 +0530
categories: [Notes, CO-STAR]
tags: [CO-STAR, Prompting, Technique, Prompt Engineering]
image:
  path: /assets/img/posts/costar_framework.svg
  alt: "CO-STAR Framework Diagram"
  class: img-center
css: ["/assets/css/custom.css", "/assets/css/msg-types.css"]
toc: true
math: false
mermaid: false
---

<pre>
image: /assets/img/posts/costar_framework.svg

image:
  path: /assets/img/posts/costar_framework.svg
  alt: "CO-STAR Framework Diagram"
  class: img-center
css: ["/assets/css/custom.css", "/assets/css/msg-types.css"]
</pre>

![Co-star framework](/assets/img/posts/costar_framework.svg){: width="800" height="400" }
_Prompting: Co-star framework_

![Co-star framework](/assets/img/posts/costar_framework.svg){: w="700" h="400" }
_Prompting: Co-star framework Illustrated!_

<div class="msg msg-info">Info: Tailwind simplifies styling.</div>
<div class="msg msg-success">Success: Build completed.</div>
<div class="msg msg-warning">Warning: Token expiry soon.</div>
<div class="msg msg-danger">Error: Deployment failed.</div>
<div class="msg msg-note">Note: Jekyll regenerates on file changes.</div>
<div class="msg msg-tip">Tip: Prefer Markdown includes for reuse.</div>
<div class="msg msg-debug">Debug: x=42, mode="test"</div>
<div class="msg msg-quote">“Good code is simple code.”</div>

---

⚑ **Why Log Levels Matter (in Simple Words)**

Log levels help organize application messages by **importance**, so developers can quickly understand *what’s happening* without drowning in noise.

Think of them like volume controls for information 🔊 —
you turn up details when debugging, and turn them down in production.

---

🧭 **Common Log Levels (Most → Least Severe)**

• **CRITICAL / FATAL** – Something went terribly wrong. The app may not continue.
• **ERROR** – A serious problem affecting functionality that needs fixing.
• **WARNING (WARN)** – Something looks off; not broken yet, but could become a problem.
• **INFO** – Normal, useful updates (app started, user logged in, task completed).
• **DEBUG** – Detailed information for developers to investigate issues.
• **TRACE** – Extremely fine-grained, step-by-step execution details.
• **OFF** – Turns logging completely off.

---

✨ **Why Developers Use Log Levels**

• **Less Noise** – See only what matters in production.
• **Better Debugging** – Enable DEBUG or TRACE when chasing bugs.
• **Faster Alerts** – Critical errors can trigger emails or notifications.
• **Clarity** – Clean logs make systems easier to understand and maintain.

---

🌱 **In short:**
Log levels keep logs **useful, readable, and purposeful** —
quiet when everything is fine, loud when something breaks.

---
