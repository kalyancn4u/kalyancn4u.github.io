---
layout: post
title: "Prompting: CO-STAR Framework"
description: "CO-START framework prompting technique"
author: technical_notes
date: 2025-12-09 00:00:00 +0530
categories: [Notes, CO-STAR]
tags: [CO-STAR, Prompting, Technique, Prompt Engineering]
image:
  path: /assets/img/posts/costar_framework.svg
  alt: "CO-STAR Framework Diagram"
  class: img-center
css: ["/assets/css/custom.css"]
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
css: ["/assets/css/custom.css"]
</pre>

![Co-star framework](/assets/img/posts/costar_framework.svg){: width="800" height="400" }
_Prompting: Co-star framework_

![Co-star framework](/assets/img/posts/costar_framework.svg){: w="700" h="400" }
_Prompting: Co-star framework Illustrated!_

<style>
.msg {
  padding: .7rem 1rem;
  border-left: 4px solid;
  border-radius: 4px;
  margin: 1rem 0;
  font-size: .95rem;
}

/* Types (Bootstrap-aligned colors) */
.msg-info    { background:#e7f3ff; border-color:#0d6efd; color:#084298; }
.msg-success { background:#e8f5e9; border-color:#198754; color:#0f5132; }
.msg-warning { background:#fff3cd; border-color:#ffc107; color:#664d03; }
.msg-danger  { background:#f8d7da; border-color:#dc3545; color:#842029; }
.msg-note    { background:#f1f5f9; border-color:#0ea5e9; color:#0c4a6e; }
.msg-tip     { background:#e0f2f1; border-color:#20c997; color:#0f766e; }
.msg-debug   { background:#f3f3f3; border-color:#6c757d; color:#343a40; font-family:monospace; }
.msg-quote   { background:#fafafa; border-color:#adb5bd; color:#495057; font-style:italic; }
</style>

<div class="msg msg-info">Info: Tailwind simplifies styling.</div>
<div class="msg msg-success">Success: Build completed.</div>
<div class="msg msg-warning">Warning: Token expiry soon.</div>
<div class="msg msg-danger">Error: Deployment failed.</div>
<div class="msg msg-note">Note: Jekyll regenerates on file changes.</div>
<div class="msg msg-tip">Tip: Prefer Markdown includes for reuse.</div>
<div class="msg msg-debug">Debug: x=42, mode="test"</div>
<div class="msg msg-quote">“Good code is simple code.”</div>

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
                  
  .msg-info     { background: #e8f1fc; border-color: #1a73e8; color: #0b3d91; }    /* Chirpy link color */
  .msg-success  { background: #e8f6ec; border-color: #2da44e; color: #0f5227; }    /* GitHub green */
  .msg-warning  { background: #fff8e6; border-color: #d97706; color: #8a5300; }    /* warm amber */
  .msg-danger   { background: #fcebea; border-color: #cf222e; color: #8a1c1f; }    /* GitHub danger red */
  .msg-note     { background: #f5f7fa; border-color: #6cb6ff; color: #244466; }    /* soft blue */
  .msg-tip      { background: #e6f7f6; border-color: #0d9488; color: #065f5b; }    /* teal-600 */                          
  .msg-debug    { background: #f3f4f6; border-color: #9ca3af; color: #374151; font-family: monospace; }   /* gray-400 */    /* gray-700 */
  .msg-quote    { background: #fafafa; border-color: #d1d5db; color: #4b5563; font-style: italic; }       /* gray-300 */    /* gray-600 */
</style>

<div class="msg msg-info">Info: Tailwind simplifies styling.</div>
<div class="msg msg-success">Success: Build completed.</div>
<div class="msg msg-warning">Warning: Token expiry soon.</div>
<div class="msg msg-danger">Error: Deployment failed.</div>
<div class="msg msg-note">Note: Jekyll regenerates on file changes.</div>
<div class="msg msg-tip">Tip: Prefer Markdown includes for reuse.</div>
<div class="msg msg-debug">Debug: x=42, mode="test"</div>
<div class="msg msg-quote">“Good code is simple code.”</div>
