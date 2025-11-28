---
layout: post
title: "🌊 CSS3: Deep Dive & Best Practices"
description: "Concise, clear, and validated revision notes on Docker technologies — containers, Dockerfile patterns, docker-compose, and practical best practices for beginners and practitioners."
author: technical_notes
date: 2025-11-25 14:30:00 +0530
categories: [Notes, CSS3]
tags: [CSS, CSS3, Front-end, UI/UX Design, Web Development, Best Practices]
toc: true
comments: false
math: true
mermaid: true
image: /assets/img/posts/css3-banner.jpg
  path: /assets/img/
  alt: CSS3 Deep Dive & Best Practices
---

## Introduction

Cascading Style Sheets Level 3 (CSS3) represents a transformative evolution in web styling technology. Unlike its predecessors, CSS3 is modularized into independent specifications, allowing features to progress at different rates and enabling browsers to implement capabilities incrementally. This modular architecture has revolutionized how developers approach web design, providing powerful tools for creating responsive, animated, and visually sophisticated interfaces.

CSS3 introduces advanced layout systems, animation capabilities, transformation functions, and enhanced styling options that fundamentally change how we build modern web applications. Understanding these features and their optimal usage patterns is essential for creating performant, maintainable, and accessible web experiences.

## CSS3 Architecture & Specification Maturity

### Modular Structure

CSS3 departed from the monolithic CSS2.1 specification by dividing functionality into independent modules. Each module evolves through distinct maturity stages, enabling incremental browser implementation and specification refinement.

### W3C Specification Maturity Levels

The World Wide Web Consortium (W3C) uses a standardized progression system for CSS specifications:

| **Stage** | **Description** | **Implementation Status** |
|-----------|----------------|---------------------------|
| **Working Draft (WD)** | Initial public draft for community review; subject to significant changes | Experimental; use with caution |
| **Candidate Recommendation (CR)** | Feature-complete specification ready for implementation testing | Stable enough for production with vendor prefixes |
| **Proposed Recommendation (PR)** | Final review stage before becoming a standard | Safe for production use |
| **Recommendation (REC)** | Official W3C standard | Fully production-ready |
| **Living Standard** | Continuously updated specification (used by WHATWG) | Production-ready with ongoing enhancements |

### CSS Jargon Hierarchy

Understanding CSS terminology requires knowledge of how different concepts relate hierarchically:

| **Level** | **Term** | **Definition** | **Example** |
|-----------|----------|----------------|-------------|
| **1. Document** | Stylesheet | Complete CSS document containing all rules | `styles.css` |
| **2. Structure** | Rule Set / Rule | Complete styling instruction with selector and declarations | `.button { color: blue; }` |
| **3. Targeting** | Selector | Pattern identifying which HTML elements to style | `.button`, `#header`, `div > p` |
| **4. Styling** | Declaration Block | Collection of property-value pairs within curly braces | `{ color: blue; padding: 10px; }` |
| **5. Property** | Declaration | Single property-value pair | `color: blue;` |
| **6. Components** | Property | CSS attribute being styled | `color`, `margin`, `font-size` |
| **7. Components** | Value | Setting assigned to a property | `blue`, `10px`, `center` |
| **8. Functional** | Function | Computed value using function notation | `calc()`, `rgb()`, `var()` |
| **9. Advanced** | At-rule | Special instruction starting with `@` | `@media`, `@keyframes`, `@import` |
| **10. Specificity** | Cascade Layer | Named layer controlling specificity precedence | `@layer utilities { }` |

### Lifecycle Terminology Comparison

Different contexts use varied terminology for CSS development stages:

| **W3C Standards** | **Browser Implementation** | **Developer Workflow** | **Framework Context** |
|-------------------|---------------------------|------------------------|----------------------|
| Working Draft | Experimental | Prototyping | Alpha |
| Candidate Recommendation | Prefixed Support | Testing | Beta |
| Proposed Recommendation | Unprefixed Support | Staging | Release Candidate |
| Recommendation | Full Support | Production | Stable Release |
| Living Standard | Continuous Updates | Maintenance | Long-term Support |

## Core CSS3 Selectors

### Basic Selectors

CSS3 enhanced selector capabilities while maintaining backward compatibility:

```css
/* Universal Selector - Specificity: (0,0,0,0) */
* {
  box-sizing: border-box;
}

/* Type Selector - Specificity: (0,0,0,1) */
p {
  line-height: 1.6;
}

/* Class Selector - Specificity: (0,0,1,0) */
.container {
  max-width: 1200px;
  margin: 0 auto;
}

/* ID Selector - Specificity: (0,1,0,0) */
#header {
  position: fixed;
  top: 0;
  width: 100%;
}

/* Attribute Selector - Specificity: (0,0,1,0) */
input[type="email"] {
  border: 2px solid #007bff;
}

/* Pseudo-class - Specificity: (0,0,1,0) */
a:hover {
  text-decoration: underline;
}

/* Pseudo-element - Specificity: (0,0,0,1) */
p::first-line {
  font-weight: bold;
}
```

### Advanced Attribute Selectors

CSS3 introduced powerful pattern-matching capabilities:

```css
/* Exact match */
[data-state="active"] {
  background: green;
}

/* Contains word (space-separated) */
[class~="featured"] {
  border: 2px solid gold;
}

/* Starts with */
[href^="https"] {
  padding-left: 20px;
}

/* Ends with */
[src$=".pdf"] {
  background: url('pdf-icon.png') no-repeat left;
}

/* Contains substring */
[class*="button"] {
  display: inline-block;
  padding: 10px 20px;
}

/* Starts with or is followed by hyphen */
[lang|="en"] {
  quotes: '"' '"';
}

/* Case-insensitive matching */
[href$=".PDF" i] {
  color: red;
}
```

### Structural Pseudo-classes

These selectors enable dynamic targeting based on document structure:

```css
/* First and last child */
li:first-child { margin-top: 0; }
li:last-child { margin-bottom: 0; }

/* Nth patterns */
tr:nth-child(odd) { background: #f5f5f5; }
tr:nth-child(even) { background: white; }
tr:nth-child(3n) { font-weight: bold; }
tr:nth-child(3n+1) { color: blue; }

/* Type-based nth patterns */
p:nth-of-type(2) { font-size: 1.2em; }

/* From the end */
li:nth-last-child(2) { border-bottom: 2px solid red; }

/* Only child */
p:only-child { text-align: center; }

/* Empty elements */
div:empty { display: none; }
```

### Relational Selectors (Combinators)

CSS3 provides precise element relationships:

```css
/* Descendant (space) - Any nested level */
article p {
  color: #333;
}

/* Child (>) - Direct children only */
nav > ul {
  list-style: none;
}

/* Adjacent sibling (+) - Immediately following */
h2 + p {
  font-size: 1.1em;
  margin-top: 0;
}

/* General sibling (~) - All following siblings */
h2 ~ p {
  text-indent: 20px;
}
```

### Modern Functional Pseudo-classes

CSS3 introduced functional selectors for complex matching:

```css
/* :not() - Negation (Specificity: content-based) */
input:not([type="submit"]) {
  border: 1px solid #ccc;
}

/* :is() - Matches any selector in list (Specificity: highest in list) */
:is(h1, h2, h3) a {
  text-decoration: none;
}

/* :where() - Same as :is() but zero specificity */
:where(article, section) p {
  line-height: 1.8;
}

/* :has() - Parent selector (Specificity: content-based) */
article:has(> img) {
  display: grid;
  grid-template-columns: 1fr 2fr;
}

/* Combining functional selectors */
li:not(:last-child):has(> a) {
  border-bottom: 1px solid #eee;
}
```

## Specificity Deep Dive

### Specificity Calculation

CSS specificity determines which style rules apply when multiple selectors target the same element. Specificity is calculated as a four-part value: `(a, b, c, d)`

**Calculation Formula:**

$$
\text{Specificity} = (a \times 1000) + (b \times 100) + (c \times 10) + (d \times 1)
$$

Where:
- **a**: Inline styles (1 if present, 0 otherwise)
- **b**: ID selectors count
- **c**: Class selectors, attribute selectors, and pseudo-classes count
- **d**: Element selectors and pseudo-elements count

```css
/* Examples with specificity values */

/* (0,0,0,0) - Inherited or default */
body { color: black; }

/* (0,0,0,1) - Element selector */
p { color: blue; }

/* (0,0,1,0) - Class selector */
.text { color: green; }

/* (0,0,1,1) - Element + Class */
p.text { color: purple; }

/* (0,1,0,0) - ID selector */
#main { color: red; }

/* (0,1,1,2) - ID + Class + 2 Elements */
#main .content p span { color: orange; }

/* (1,0,0,0) - Inline style */
<p style="color: pink;">Text</p>

/* (∞) - !important flag (overrides everything except later !important) */
p { color: yellow !important; }
```

### Specificity Hierarchy

The following hierarchy determines style precedence (highest to lowest):

1. **!important declarations** - Overrides all normal declarations
2. **Inline styles** - `style` attribute in HTML
3. **ID selectors** - `#identifier`
4. **Class selectors, attribute selectors, pseudo-classes** - `.class`, `[attr]`, `:hover`
5. **Element selectors, pseudo-elements** - `div`, `::before`
6. **Universal selector** - `*` (zero specificity)
7. **Inherited values** - From parent elements

### Specificity Best Practices

```css
/* ❌ AVOID: Over-specific selectors */
#header nav ul li a.nav-link {
  color: blue; /* (0,1,1,4) - Difficult to override */
}

/* ✅ PREFER: Simpler, class-based approach */
.nav-link {
  color: blue; /* (0,0,1,0) - Easy to manage */
}

/* ❌ AVOID: Excessive !important usage */
.button {
  background: blue !important;
  color: white !important;
}

/* ✅ PREFER: Proper specificity management */
.button {
  background: blue;
  color: white;
}

.button.button--primary {
  background: darkblue; /* Overrides naturally */
}

/* ❌ AVOID: ID selectors for styling */
#sidebar {
  width: 300px;
}

/* ✅ PREFER: Class selectors for flexibility */
.sidebar {
  width: 300px;
}
```

### Cascade Layers

CSS Cascade Layers provide controlled specificity management:

```css
/* Define layer order (highest precedence last) */
@layer reset, base, components, utilities;

/* Reset layer - lowest precedence */
@layer reset {
  * {
    margin: 0;
    padding: 0;
  }
}

/* Base styles */
@layer base {
  body {
    font-family: sans-serif;
    line-height: 1.6;
  }
}

/* Component styles */
@layer components {
  .button {
    padding: 10px 20px;
    background: blue;
  }
}

/* Utility classes - highest precedence */
@layer utilities {
  .mt-0 { margin-top: 0 !important; }
}

/* Unlayered styles have highest precedence */
.special-button {
  background: red; /* Overrides .button in components layer */
}
```

## Modern Layout Systems

### Flexbox

Flexbox provides one-dimensional layout control (row or column):

```css
/* Basic flex container */
.flex-container {
  display: flex;
  flex-direction: row; /* row | row-reverse | column | column-reverse */
  flex-wrap: wrap; /* nowrap | wrap | wrap-reverse */
  justify-content: space-between; /* flex-start | flex-end | center | space-around | space-evenly */
  align-items: center; /* flex-start | flex-end | center | stretch | baseline */
  align-content: flex-start; /* Aligns multiple lines */
  gap: 20px; /* Spacing between items */
}

/* Flex items */
.flex-item {
  flex-grow: 1; /* Growth factor */
  flex-shrink: 1; /* Shrink factor */
  flex-basis: 200px; /* Initial size */
  
  /* Shorthand */
  flex: 1 1 200px; /* grow shrink basis */
  
  align-self: flex-start; /* Override container's align-items */
  order: 2; /* Change visual order */
}

/* Common flex patterns */

/* Equal-width columns */
.column {
  flex: 1;
}

/* Fixed sidebar with flexible content */
.sidebar {
  flex: 0 0 250px; /* Don't grow, don't shrink, 250px wide */
}

.content {
  flex: 1; /* Take remaining space */
}

/* Center content horizontally and vertically */
.centered {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 100vh;
}

/* Responsive navigation */
.nav {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.nav-item {
  flex: 1 1 auto;
  min-width: 100px;
}
```

### CSS Grid

Grid provides two-dimensional layout control:

```css
/* Grid container basics */
.grid-container {
  display: grid;
  
  /* Define columns */
  grid-template-columns: 200px 1fr 2fr; /* Fixed + flexible */
  grid-template-columns: repeat(3, 1fr); /* Equal columns */
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); /* Responsive */
  
  /* Define rows */
  grid-template-rows: auto 1fr auto; /* Header, content, footer */
  
  /* Gaps */
  gap: 20px; /* row-gap and column-gap */
  
  /* Named grid areas */
  grid-template-areas:
    "header header header"
    "sidebar content content"
    "footer footer footer";
}

/* Grid items */
.grid-item {
  /* Spanning */
  grid-column: 1 / 3; /* Start line / End line */
  grid-row: 1 / span 2; /* Start / Span count */
  
  /* Shorthand */
  grid-area: 2 / 1 / 4 / 3; /* row-start / col-start / row-end / col-end */
  
  /* Named areas */
  grid-area: header;
  
  /* Alignment */
  justify-self: start; /* Horizontal */
  align-self: center; /* Vertical */
}

/* Advanced Grid Patterns */

/* Holy Grail Layout */
.page-grid {
  display: grid;
  grid-template-columns: minmax(150px, 200px) 1fr minmax(150px, 200px);
  grid-template-rows: auto 1fr auto;
  min-height: 100vh;
  gap: 20px;
}

.header {
  grid-column: 1 / -1; /* Span all columns */
}

.footer {
  grid-column: 1 / -1;
}

/* Responsive card grid */
.card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 30px;
  padding: 20px;
}

/* Complex magazine layout */
.magazine-grid {
  display: grid;
  grid-template-columns: repeat(12, 1fr);
  grid-auto-rows: 100px;
  gap: 15px;
}

.featured-article {
  grid-column: 1 / 7;
  grid-row: 1 / 4;
}

.sidebar-article {
  grid-column: 7 / 13;
  grid-row: 1 / 2;
}

/* Subgrid (for nested grids) */
.nested-grid {
  display: grid;
  grid-template-columns: subgrid;
  grid-column: 1 / 4;
}
```

### Grid vs Flexbox Decision Matrix

| **Use Flexbox When** | **Use Grid When** |
|-----------------------|-------------------|
| Single-dimension layouts (row or column) | Two-dimensional layouts (rows and columns) |
| Content-driven sizing | Layout-driven sizing |
| Navigation bars | Page layouts |
| Toolbars | Image galleries |
| Simple card arrangements | Complex magazine-style layouts |
| Aligning items in a line | Creating overlapping elements |
| Dynamic content order | Fixed grid structures |

## Box Model & Sizing

### Box Model Components

Every CSS element consists of nested rectangular boxes:

```css
/* Traditional box model */
.traditional {
  width: 300px; /* Content width only */
  padding: 20px; /* Adds to total width */
  border: 5px solid black; /* Adds to total width */
  margin: 10px; /* Outside the box, creates spacing */
  
  /* Total width: 300 + (20×2) + (5×2) = 350px */
  /* Total space: 350 + (10×2) = 370px */
}

/* Border-box model (recommended) */
.border-box {
  box-sizing: border-box;
  width: 300px; /* Includes padding and border */
  padding: 20px;
  border: 5px solid black;
  margin: 10px;
  
  /* Total width: 300px (padding and border included) */
  /* Content width: 300 - (20×2) - (5×2) = 250px */
}

/* Global border-box reset */
*,
*::before,
*::after {
  box-sizing: border-box;
}
```

### Modern Sizing Units

```css
/* Absolute units */
.absolute {
  width: 300px; /* Pixels */
  height: 2in; /* Inches */
  margin: 1cm; /* Centimeters */
}

/* Relative to font size */
.relative-font {
  font-size: 16px;
  padding: 1em; /* 16px (relative to element's font-size) */
  margin: 2rem; /* 32px (relative to root font-size) */
  width: 20ch; /* Width of 20 "0" characters */
  height: 3ex; /* Height of 3 "x" characters */
}

/* Viewport units */
.viewport {
  width: 100vw; /* 100% of viewport width */
  height: 100vh; /* 100% of viewport height */
  font-size: 5vmin; /* 5% of smaller viewport dimension */
  padding: 2vmax; /* 2% of larger viewport dimension */
}

/* Container query units (modern) */
.container-relative {
  font-size: 5cqw; /* 5% of container width */
  padding: 2cqh; /* 2% of container height */
}

/* Percentage units */
.percentage {
  width: 50%; /* 50% of parent width */
  height: 100%; /* 100% of parent height */
}

/* Fractional units (Grid) */
.grid-fraction {
  display: grid;
  grid-template-columns: 1fr 2fr 1fr; /* Proportional fractions */
}

/* Calculation */
.calculated {
  width: calc(100% - 40px);
  height: calc(100vh - 100px);
  font-size: calc(16px + 0.5vw);
  padding: clamp(10px, 2vw, 30px); /* min, preferred, max */
}
```

### Logical Properties

Logical properties adapt to writing direction:

```css
/* Physical properties (traditional) */
.physical {
  margin-top: 20px;
  margin-right: 15px;
  margin-bottom: 20px;
  margin-left: 15px;
  border-left: 2px solid blue;
}

/* Logical properties (modern) */
.logical {
  margin-block-start: 20px; /* Top in LTR, bottom in RTL */
  margin-inline-end: 15px; /* Right in LTR, left in RTL */
  margin-block-end: 20px;
  margin-inline-start: 15px;
  border-inline-start: 2px solid blue;
}

/* Shorthand logical properties */
.logical-short {
  margin-block: 20px; /* block-start and block-end */
  margin-inline: 15px; /* inline-start and inline-end */
  padding-block: 10px 15px;
  padding-inline: 20px 25px;
}

/* Size properties */
.logical-size {
  inline-size: 300px; /* Width in horizontal writing */
  block-size: 200px; /* Height in horizontal writing */
  min-inline-size: 200px;
  max-block-size: 500px;
}
```

## Colors & Gradients

### Color Formats

```css
/* Named colors */
.named {
  color: red;
  background: transparent;
}

/* Hexadecimal */
.hex {
  color: #ff0000; /* Red */
  background: #0080ff; /* Blue */
  border-color: #f00; /* Shorthand */
}

/* RGB/RGBA */
.rgb {
  color: rgb(255, 0, 0); /* Red */
  background: rgba(0, 128, 255, 0.5); /* Blue with 50% opacity */
}

/* HSL/HSLA (Hue, Saturation, Lightness) */
.hsl {
  color: hsl(0, 100%, 50%); /* Red */
  background: hsla(210, 100%, 50%, 0.5); /* Blue with 50% opacity */
}

/* Modern color functions */
.modern-colors {
  /* HWB (Hue, Whiteness, Blackness) */
  color: hwb(0 0% 0%); /* Red */
  
  /* LAB (Lightness, A-axis, B-axis) */
  background: lab(50% 40 59); /* More perceptually uniform */
  
  /* LCH (Lightness, Chroma, Hue) */
  border-color: lch(50% 50 180); /* Easier to work with than LAB */
  
  /* Color-mix */
  background: color-mix(in srgb, blue 60%, white);
  
  /* Relative colors */
  background: rgb(from blue r g 200); /* Modify blue component */
}

/* CSS Custom Properties for color systems */
:root {
  --primary-h: 210;
  --primary-s: 100%;
  --primary-l: 50%;
  
  --primary: hsl(var(--primary-h), var(--primary-s), var(--primary-l));
  --primary-light: hsl(var(--primary-h), var(--primary-s), calc(var(--primary-l) + 20%));
  --primary-dark: hsl(var(--primary-h), var(--primary-s), calc(var(--primary-l) - 20%));
}
```

### Gradients

```css
/* Linear gradients */
.linear-gradient {
  /* Basic gradient */
  background: linear-gradient(red, blue);
  
  /* Angled gradient */
  background: linear-gradient(45deg, red, blue);
  
  /* Multiple color stops */
  background: linear-gradient(to right, red 0%, yellow 50%, blue 100%);
  
  /* Hard color stops */
  background: linear-gradient(to right, red 50%, blue 50%);
  
  /* Transparent gradients */
  background: linear-gradient(to bottom, rgba(0,0,0,0) 0%, rgba(0,0,0,0.8) 100%);
}

/* Radial gradients */
.radial-gradient {
  /* Centered circle */
  background: radial-gradient(circle, red, blue);
  
  /* Ellipse gradient */
  background: radial-gradient(ellipse at top, red, blue);
  
  /* Positioned gradient */
  background: radial-gradient(circle at 30% 40%, red, blue);
  
  /* Sized gradient */
  background: radial-gradient(circle 100px at center, red, blue);
  
  /* Multiple color stops */
  background: radial-gradient(circle, red 0%, yellow 30%, blue 60%, green 100%);
}

/* Conic gradients */
.conic-gradient {
  /* Basic conic */
  background: conic-gradient(red, yellow, green, blue, red);
  
  /* Angled start */
  background: conic-gradient(from 45deg, red, blue);
  
  /* Positioned center */
  background: conic-gradient(at 60% 45%, red, blue);
  
  /* Color wheel */
  background: conic-gradient(
    hsl(0, 100%, 50%),
    hsl(60, 100%, 50%),
    hsl(120, 100%, 50%),
    hsl(180, 100%, 50%),
    hsl(240, 100%, 50%),
    hsl(300, 100%, 50%),
    hsl(360, 100%, 50%)
  );
}

/* Repeating gradients */
.repeating-gradients {
  /* Repeating linear */
  background: repeating-linear-gradient(
    45deg,
    red 0px,
    red 10px,
    blue 10px,
    blue 20px
  );
  
  /* Repeating radial */
  background: repeating-radial-gradient(
    circle,
    red 0px,
    red 10px,
    blue 10px,
    blue 20px
  );
  
  /* Striped pattern */
  background: repeating-linear-gradient(
    90deg,
    #fff 0px,
    #fff 20px,
    #f0f0f0 20px,
    #f0f0f0 40px
  );
}

/* Multiple backgrounds */
.multiple-backgrounds {
  background:
    linear-gradient(to right, rgba(255,255,255,0.3), transparent),
    radial-gradient(circle at 20% 50%, rgba(0,0,255,0.2), transparent),
    url('texture.png'),
    #f0f0f0;
}
```

## Transforms & Transitions

### 2D Transforms

```css
.transform-2d {
  /* Translation */
  transform: translate(50px, 100px); /* x, y */
  transform: translateX(50px);
  transform: translateY(100px);
  
  /* Rotation */
  transform: rotate(45deg);
  transform: rotate(-0.5turn); /* Negative rotation */
  
  /* Scaling */
  transform: scale(1.5); /* Uniform scaling */
  transform: scale(1.5, 0.8); /* x, y */
  transform: scaleX(2);
  transform: scaleY(0.5);
  
  /* Skewing */
  transform: skew(20deg, 10deg); /* x, y */
  transform: skewX(20deg);
  transform: skewY(10deg);
  
  /* Combined transforms */
  transform: translate(50px, 100px) rotate(45deg) scale(1.5);
  
  /* Transform origin */
  transform-origin: center center; /* Default */
  transform-origin: top left;
  transform-origin: 50% 50%;
  transform-origin: 20px 30px;
}
```

### 3D Transforms

```css
.transform-3d {
  /* Perspective (required for 3D) */
  perspective: 1000px;
  
  /* 3D Translation */
  transform: translate3d(50px, 100px, 200px); /* x, y, z */
  transform: translateZ(100px);
  
  /* 3D Rotation */
  transform: rotateX(45deg); /* Rotate around X-axis */
  transform: rotateY(45deg); /* Rotate around Y-axis */
  transform: rotateZ(45deg); /* Rotate around Z-axis */
  transform: rotate3d(1, 1, 1, 45deg); /* Rotate around arbitrary vector */
  
  /* 3D Scaling */
  transform: scale3d(1.5, 1.5, 1.5);
  transform: scaleZ(2);
  
  /* Combined 3D transforms */
  transform: perspective(1000px) rotateY(45deg) translateZ(100px);
  
  /* Transform style */
  transform-style: preserve-3d; /* Preserve 3D for children */
  transform-style: flat; /* Flatten children */
  
  /* Backface visibility */
  backface-visibility: hidden; /* Hide back face */
  backface-visibility: visible;
}

/* 3D Card Flip Example */
.card-container {
  perspective: 1000px;
}

.card {
  width: 300px;
  height: 200px;
  position: relative;
  transform-style: preserve-3d;
  transition: transform 0.6s;
}

.card:hover {
  transform: rotateY(180deg);
}

.card-front,
.card-back {
  position: absolute;
  width: 100%;
  height: 100%;
  backface-visibility: hidden;
}

.card-back {
  transform: rotateY(180deg);
}
```

### Transitions

```css
.transition-basics {
  /* Single property */
  transition: opacity 0.3s ease-in-out;
  
  /* Multiple properties */
  transition: opacity 0.3s ease-in-out,
              transform 0.5s cubic-bezier(0.68, -0.55, 0.265, 1.55);
  
  /* All properties */
  transition: all 0.3s ease;
  
  /* Individual properties */
  transition-property: opacity, transform;
  transition-duration: 0.3s, 0.5s;
  transition-timing-function: ease-in-out, ease;
  transition-delay: 0s, 0.1s;
}

/* Timing functions */
.timing-functions {
  /* Predefined */
  transition-timing-function: linear; /* Constant speed */
  transition-timing-function: ease; /* Slow start, fast middle, slow end */
  transition-timing-function: ease-in; /* Slow start */
  transition-timing-function: ease-out; /* Slow end */
  transition-timing-function: ease-in-out; /* Slow start and end */
  
  /* Cubic Bezier (custom) */
  transition-timing-function: cubic-bezier(0.42, 0, 0.58, 1);
  transition-timing-function: cubic-bezier(0.68, -0.55, 0.265, 1.55); /* Bounce */
  
  /* Steps */
  transition-timing-function: steps(5); /* 5 discrete steps */
  transition-timing-function: steps(3, jump-start);
  transition-timing-function: step-start;
  transition-timing-function: step-end;
}

/* Practical transition examples */
.button {
  background: #007bff;
  color: white;
  padding: 10px 20px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: background 0.3s ease, transform 0.2s ease;
}

.button:hover {
  background: #0056b3;
  transform: translateY(-2px);
}

.button:active {
  transform: translateY(0);
}

.dropdown {
  max-height: 0;
  overflow: hidden;
  opacity: 0;
  transition: max-height 0.3s ease, opacity 0.3s ease;
}

.dropdown.open {
  max-height: 500px;
  opacity: 1;
}
```

## Animations & Keyframes

### Basic Animations

```css
/* Define keyframes */
@keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}

@keyframes slideIn {
  0% {
    transform: translateX(-100%);
    opacity: 0;
  }
  100% {
    transform: translateX(0);
    opacity: 1;
  }
}

@keyframes pulse {
  0%, 100% {
    transform: scale(1);
  }
  50% {
    transform: scale(1.1);
  }
}

/* Apply animations */
.animated {
  /* Shorthand */
  animation: fadeIn 1s ease-in-out 0.5s infinite alternate forwards;
  
  /* Individual properties */
  animation-name: fadeIn;
  animation-duration: 1s;
  animation-timing-function: ease-in-out;
  animation-delay: 0.5s;
  animation-iteration-count: infinite; /* or number */
  animation-direction: alternate; /* normal | reverse | alternate | alternate-reverse */
  animation-fill-mode: forwards; /* none | forwards | backwards | both */
  animation-play-state: running; /* running | paused */
}

/* Multiple animations */
.multi-animated {
  animation: 
    fadeIn 1s ease-in-out,
    slideIn 0.8s ease-out,
    pulse 2s ease-in-out infinite;
}
```

### Advanced Animation Techniques

```css
/* Complex multi-step animation */
@keyframes complexMove {
  0% {
    transform: translateX(0) rotate(0deg);
    background: red;
    border-radius: 0%;
  }
  25% {
    transform: translateX(100px) rotate(90deg);
    background: yellow;
    border-radius: 50%;
  }
  50% {
    transform: translateX(100px) translateY(100px) rotate(180deg);
    background: green;
    border-radius: 0%;
  }
  75% {
    transform: translateX(0) translateY(100px) rotate(270deg);
    background: blue;
    border-radius: 50%;
  }
  100% {
    transform: translateX(0) rotate(360deg);
    background: red;
    border-radius: 0%;
  }
}

/* Loading spinner */
@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

.spinner {
  width: 50px;
  height: 50px;
  border: 5px solid #f3f3f3;
  border-top: 5px solid #3498db;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

/* Bouncing animation */
@keyframes bounce {
  0%, 20%, 50%, 80%, 100% {
    transform: translateY(0);
  }
  40% {
    transform: translateY(-30px);
  }
  60% {
    transform: translateY(-15px);
  }
}

.bouncing {
  animation: bounce 2s ease-in-out infinite;
}

/* Typing effect */
@keyframes typing {
  from {
    width: 0;
  }
  to {
    width: 100%;
  }
}

@keyframes blink {
  50% {
    border-color: transparent;
  }
}

.typewriter {
  overflow: hidden;
  border-right: 2px solid;
  white-space: nowrap;
  animation: 
    typing 3.5s steps(40) 1s 1 normal both,
    blink 0.75s step-end infinite;
}

/* Shake animation */
@keyframes shake {
  0%, 100% {
    transform: translateX(0);
  }
  10%, 30%, 50%, 70%, 90% {
    transform: translateX(-10px);
  }
  20%, 40%, 60%, 80% {
    transform: translateX(10px);
  }
}

.shake-on-error {
  animation: shake 0.5s ease-in-out;
}

/* Parallax scroll effect with animation */
@keyframes parallax {
  to {
    transform: translateY(-50%);
  }
}

.parallax-layer {
  animation: parallax linear;
  animation-timeline: scroll();
}
```

### Animation Performance

```css
/* ✅ GOOD: Hardware-accelerated properties */
.optimized {
  /* These trigger GPU acceleration */
  animation: moveOptimized 1s ease;
}

@keyframes moveOptimized {
  from {
    transform: translateX(0);
    opacity: 1;
  }
  to {
    transform: translateX(100px);
    opacity: 0;
  }
}

/* ❌ AVOID: Properties that trigger reflow/repaint */
.non-optimized {
  animation: moveSlow 1s ease;
}

@keyframes moveSlow {
  from {
    left: 0; /* Triggers layout */
    width: 100px; /* Triggers layout */
  }
  to {
    left: 100px;
    width: 200px;
  }
}

/* Force GPU acceleration */
.gpu-accelerated {
  transform: translateZ(0); /* Creates new layer */
  will-change: transform, opacity; /* Hint to browser */
  backface-visibility: hidden; /* Prevents flickering */
}

/* will-change usage (use sparingly) */
.hover-card {
  transition: transform 0.3s;
}

.hover-card:hover {
  will-change: transform;
  transform: scale(1.05);
}

.hover-card:not(:hover) {
  will-change: auto; /* Remove hint after use */
}
```

## Responsive Design

### Media Queries

```css
/* Basic media query structure */
@media media-type and (media-feature) {
  /* CSS rules */
}

/* Width-based breakpoints */
@media (max-width: 767px) {
  /* Mobile styles */
  .container {
    width: 100%;
    padding: 15px;
  }
}

@media (min-width: 768px) and (max-width: 1023px) {
  /* Tablet styles */
  .container {
    width: 750px;
  }
}

@media (min-width: 1024px) {
  /* Desktop styles */
  .container {
    width: 1000px;
  }
}

/* Mobile-first approach (recommended) */
.element {
  /* Mobile styles (default) */
  width: 100%;
  padding: 10px;
}

@media (min-width: 768px) {
  .element {
    width: 50%;
    padding: 20px;
  }
}

@media (min-width: 1024px) {
  .element {
    width: 33.333%;
    padding: 30px;
  }
}

/* Common breakpoint system */
:root {
  --breakpoint-sm: 576px;
  --breakpoint-md: 768px;
  --breakpoint-lg: 992px;
  --breakpoint-xl: 1200px;
  --breakpoint-xxl: 1400px;
}

/* Orientation queries */
@media (orientation: portrait) {
  .image {
    width: 100%;
    height: auto;
  }
}

@media (orientation: landscape) {
  .image {
    width: auto;
    height: 100%;
  }
}

/* Resolution queries */
@media (min-resolution: 192dpi),
       (-webkit-min-device-pixel-ratio: 2) {
  /* Retina display styles */
  .logo {
    background-image: url('logo@2x.png');
    background-size: 100px 50px;
  }
}

/* Hover capability */
@media (hover: hover) and (pointer: fine) {
  /* Desktop with mouse */
  .button:hover {
    background: #0056b3;
  }
}

@media (hover: none) and (pointer: coarse) {
  /* Touch devices */
  .button:active {
    background: #0056b3;
  }
}

/* Prefers color scheme */
@media (prefers-color-scheme: dark) {
  :root {
    --bg-color: #1a1a1a;
    --text-color: #ffffff;
  }
}

@media (prefers-color-scheme: light) {
  :root {
    --bg-color: #ffffff;
    --text-color: #000000;
  }
}

/* Reduced motion */
@media (prefers-reduced-motion: reduce) {
  * {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}

/* Print styles */
@media print {
  .no-print {
    display: none;
  }
  
  body {
    font-size: 12pt;
    color: black;
    background: white;
  }
  
  a[href]::after {
    content: " (" attr(href) ")";
  }
}

/* Combined queries */
@media screen and (min-width: 768px) and (max-width: 1023px) and (orientation: landscape) {
  /* Tablet landscape styles */
}
```

### Container Queries

```css
/* Container query setup */
.card-container {
  container-type: inline-size; /* or size, normal */
  container-name: card;
}

/* Query the container */
@container card (min-width: 400px) {
  .card-title {
    font-size: 2rem;
  }
  
  .card-layout {
    display: grid;
    grid-template-columns: 1fr 2fr;
  }
}

@container card (max-width: 399px) {
  .card-title {
    font-size: 1.25rem;
  }
  
  .card-layout {
    display: block;
  }
}

/* Container query units */
.responsive-text {
  font-size: clamp(1rem, 5cqw, 3rem); /* 5% of container width */
  padding: 2cqh; /* 2% of container height */
}

/* Nested containers */
.outer-container {
  container-type: inline-size;
  container-name: outer;
}

.inner-container {
  container-type: inline-size;
  container-name: inner;
}

@container outer (min-width: 800px) {
  .outer-content {
    padding: 40px;
  }
}

@container inner (min-width: 400px) {
  .inner-content {
    display: flex;
  }
}
```

### Responsive Typography

```css
/* Fluid typography */
:root {
  /* Formula: min + (max - min) * ((100vw - min-width) / (max-width - min-width)) */
  --fluid-type-min: 1rem;
  --fluid-type-max: 1.5rem;
  --fluid-type-min-width: 320px;
  --fluid-type-max-width: 1200px;
}

body {
  font-size: clamp(1rem, 0.875rem + 0.5vw, 1.5rem);
}

h1 {
  font-size: clamp(2rem, 1.5rem + 2vw, 4rem);
}

/* Responsive line length */
.text-container {
  max-width: 65ch; /* Optimal reading length */
  width: 100%;
}

/* Scale system */
.scale-system {
  --ratio: 1.25; /* Major third */
  
  font-size: 1rem;
}

h1 { font-size: calc(1rem * var(--ratio) * var(--ratio) * var(--ratio) * var(--ratio)); }
h2 { font-size: calc(1rem * var(--ratio) * var(--ratio) * var(--ratio)); }
h3 { font-size: calc(1rem * var(--ratio) * var(--ratio)); }
h4 { font-size: calc(1rem * var(--ratio)); }

/* Responsive based on viewport */
@media (min-width: 768px) {
  .scale-system {
    --ratio: 1.333; /* Perfect fourth */
  }
}

@media (min-width: 1024px) {
  .scale-system {
    --ratio: 1.5; /* Perfect fifth */
  }
}
```

## Custom Properties (CSS Variables)

### Basic Usage

```css
/* Define variables */
:root {
  /* Colors */
  --primary-color: #007bff;
  --secondary-color: #6c757d;
  --success-color: #28a745;
  --danger-color: #dc3545;
  
  /* Spacing */
  --spacing-xs: 0.25rem;
  --spacing-sm: 0.5rem;
  --spacing-md: 1rem;
  --spacing-lg: 1.5rem;
  --spacing-xl: 2rem;
  
  /* Typography */
  --font-family-base: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  --font-family-mono: "Courier New", monospace;
  --font-size-base: 1rem;
  --line-height-base: 1.5;
  
  /* Layout */
  --container-max-width: 1200px;
  --border-radius: 0.25rem;
  --box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

/* Use variables */
.button {
  background: var(--primary-color);
  padding: var(--spacing-sm) var(--spacing-md);
  border-radius: var(--border-radius);
  font-family: var(--font-family-base);
}

/* Fallback values */
.element {
  color: var(--text-color, #333); /* Use #333 if --text-color not defined */
}

/* Nested variables */
:root {
  --primary-h: 210;
  --primary-s: 100%;
  --primary-l: 50%;
  --primary: hsl(var(--primary-h), var(--primary-s), var(--primary-l));
  --primary-light: hsl(var(--primary-h), var(--primary-s), calc(var(--primary-l) + 10%));
}
```

### Theming with Custom Properties

```css
/* Light theme (default) */
:root {
  --bg-primary: #ffffff;
  --bg-secondary: #f8f9fa;
  --text-primary: #212529;
  --text-secondary: #6c757d;
  --border-color: #dee2e6;
}

/* Dark theme */
[data-theme="dark"] {
  --bg-primary: #212529;
  --bg-secondary: #343a40;
  --text-primary: #f8f9fa;
  --text-secondary: #adb5bd;
  --border-color: #495057;
}

/* Apply theme colors */
body {
  background: var(--bg-primary);
  color: var(--text-primary);
}

.card {
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
}

/* JavaScript theme switching */
/* 
document.documentElement.setAttribute('data-theme', 'dark');
*/

/* Prefers color scheme integration */
@media (prefers-color-scheme: dark) {
  :root:not([data-theme]) {
    --bg-primary: #212529;
    --bg-secondary: #343a40;
    --text-primary: #f8f9fa;
    --text-secondary: #adb5bd;
  }
}
```

### Dynamic Variables

```css
/* Scoped variables */
.component {
  --local-spacing: 20px;
  padding: var(--local-spacing);
}

.component .child {
  margin: calc(var(--local-spacing) / 2);
}

/* Responsive variables */
:root {
  --container-padding: 15px;
}

@media (min-width: 768px) {
  :root {
    --container-padding: 30px;
  }
}

@media (min-width: 1024px) {
  :root {
    --container-padding: 50px;
  }
}

.container {
  padding: var(--container-padding);
}

/* Context-aware variables */
.button {
  --button-bg: var(--primary-color);
  --button-color: white;
  
  background: var(--button-bg);
  color: var(--button-color);
}

.button.button--secondary {
  --button-bg: var(--secondary-color);
}

.button.button--danger {
  --button-bg: var(--danger-color);
}

/* Computed values */
:root {
  --base-size: 16px;
  --scale: 1.25;
  
  --size-1: calc(var(--base-size) * var(--scale));
  --size-2: calc(var(--size-1) * var(--scale));
  --size-3: calc(var(--size-2) * var(--scale));
}
```

## CSS Methodologies

### BEM (Block Element Modifier)

```css
/* Block: Standalone component */
.card { }

/* Element: Part of block, no standalone meaning */
.card__header { }
.card__body { }
.card__footer { }
.card__title { }
.card__image { }

/* Modifier: Different state or version */
.card--featured { }
.card--large { }
.card__title--bold { }

/* Complete example */
.button {
  padding: 10px 20px;
  border: none;
  border-radius: 4px;
}

.button__icon {
  margin-right: 8px;
}

.button__text {
  font-weight: 500;
}

.button--primary {
  background: #007bff;
  color: white;
}

.button--secondary {
  background: #6c757d;
  color: white;
}

.button--large {
  padding: 15px 30px;
  font-size: 1.25rem;
}

.button--disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
```

### OOCSS (Object-Oriented CSS)

```css
/* Separate structure and skin */

/* Structure (layout) */
.box {
  padding: 20px;
  margin-bottom: 20px;
}

.box-large {
  padding: 40px;
}

/* Skin (visual appearance) */
.box-primary {
  background: #007bff;
  color: white;
}

.box-secondary {
  background: #6c757d;
  color: white;
}

/* Separate container and content */

/* Container */
.media {
  display: flex;
  align-items: flex-start;
}

.media__img {
  margin-right: 15px;
  flex-shrink: 0;
}

.media__body {
  flex: 1;
}

/* Content (reusable) */
.heading {
  font-size: 1.5rem;
  font-weight: bold;
  margin: 0 0 10px 0;
}

.text {
  line-height: 1.6;
  color: #333;
}
```

### SMACSS (Scalable and Modular Architecture)

```css
/* Base rules (element defaults) */
html {
  box-sizing: border-box;
}

body {
  font-family: sans-serif;
  line-height: 1.6;
}

/* Layout rules (major page sections) */
.l-header {
  position: fixed;
  top: 0;
  width: 100%;
}

.l-sidebar {
  width: 250px;
  float: left;
}

.l-main {
  margin-left: 270px;
}

/* Module rules (reusable components) */
.button {
  padding: 10px 20px;
}

.card {
  background: white;
  border: 1px solid #ddd;
  border-radius: 4px;
}

/* State rules (how things look in different states) */
.is-active {
  font-weight: bold;
}

.is-hidden {
  display: none;
}

.is-disabled {
  opacity: 0.5;
  pointer-events: none;
}

/* Theme rules (color schemes) */
.theme-dark .card {
  background: #333;
  color: white;
  border-color: #555;
}
```

### Utility-First (Tailwind-like)

```css
/* Spacing utilities */
.m-0 { margin: 0; }
.m-1 { margin: 0.25rem; }
.m-2 { margin: 0.5rem; }
.m-3 { margin: 1rem; }
.m-4 { margin: 1.5rem; }

.mt-1 { margin-top: 0.25rem; }
.mr-1 { margin-right: 0.25rem; }
.mb-1 { margin-bottom: 0.25rem; }
.ml-1 { margin-left: 0.25rem; }

.p-0 { padding: 0; }
.p-1 { padding: 0.25rem; }
.p-2 { padding: 0.5rem; }
.p-3 { padding: 1rem; }

/* Display utilities */
.block { display: block; }
.inline-block { display: inline-block; }
.inline { display: inline; }
.flex { display: flex; }
.grid { display: grid; }
.hidden { display: none; }

/* Flexbox utilities */
.flex-row { flex-direction: row; }
.flex-col { flex-direction: column; }
.items-center { align-items: center; }
.items-start { align-items: flex-start; }
.justify-center { justify-content: center; }
.justify-between { justify-content: space-between; }

/* Typography utilities */
.text-xs { font-size: 0.75rem; }
.text-sm { font-size: 0.875rem; }
.text-base { font-size: 1rem; }
.text-lg { font-size: 1.125rem; }
.text-xl { font-size: 1.25rem; }

.font-normal { font-weight: 400; }
.font-medium { font-weight: 500; }
.font-bold { font-weight: 700; }

.text-left { text-align: left; }
.text-center { text-align: center; }
.text-right { text-align: right; }

/* Color utilities */
.text-primary { color: var(--primary-color); }
.text-secondary { color: var(--secondary-color); }
.text-white { color: white; }
.text-black { color: black; }

.bg-primary { background-color: var(--primary-color); }
.bg-secondary { background-color: var(--secondary-color); }
.bg-white { background-color: white; }

/* Border utilities */
.border { border: 1px solid; }
.border-t { border-top: 1px solid; }
.border-2 { border-width: 2px; }
.rounded { border-radius: 0.25rem; }
.rounded-full { border-radius: 9999px; }

/* Position utilities */
.relative { position: relative; }
.absolute { position: absolute; }
.fixed { position: fixed; }
.sticky { position: sticky; }

/* Responsive utilities */
@media (min-width: 768px) {
  .md\:flex { display: flex; }
  .md\:hidden { display: none; }
  .md\:text-xl { font-size: 1.25rem; }
}
```

## Advanced Selectors & Pseudo-classes

### UI State Pseudo-classes

```css
/* Form states */
input:focus {
  outline: 2px solid #007bff;
  outline-offset: 2px;
}

input:focus-visible {
  /* Only when focused via keyboard */
  outline: 2px solid #007bff;
}

input:focus-within {
  /* Has focused descendant */
  border-color: #007bff;
}

input:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

input:enabled {
  cursor: text;
}

input:checked {
  background: #007bff;
}

input:indeterminate {
  background: #6c757d;
}

input:valid {
  border-color: #28a745;
}

input:invalid {
  border-color: #dc3545;
}

input:required {
  border-left: 3px solid #007bff;
}

input:optional {
  border-left: 3px solid #6c757d;
}

input:in-range {
  border-color: #28a745;
}

input:out-of-range {
  border-color: #dc3545;
}

input:read-only {
  background: #f8f9fa;
}

input:read-write {
  background: white;
}

input:placeholder-shown {
  border-color: #ced4da;
}
```

### Interactive Pseudo-classes

```css
/* Link states */
a:link {
  color: #007bff;
  text-decoration: none;
}

a:visited {
  color: #6610f2;
}

a:hover {
  color: #0056b3;
  text-decoration: underline;
}

a:active {
  color: #004085;
}

/* Target pseudo-class */
:target {
  background: #fff3cd;
  padding: 20px;
  border: 2px solid #ffc107;
}

/* Any-link (matches :link and :visited) */
:any-link {
  color: #007bff;
}
```

### Linguistic Pseudo-classes

```css
/* Language */
:lang(en) {
  quotes: '"' '"';
}

:lang(fr) {
  quotes: '«' '»';
}

/* Direction */
:dir(ltr) {
  text-align: left;
}

:dir(rtl) {
  text-align: right;
}
```

### Resource State Pseudo-classes

```css
/* Playing media */
video:playing {
  border: 2px solid green;
}

video:paused {
  border: 2px solid orange;
}

/* Picture-in-picture */
video:picture-in-picture {
  box-shadow: 0 0 20px rgba(0,0,0,0.5);
}
```

## Performance Optimization

### CSS Performance Best Practices

```css
/* ✅ EFFICIENT SELECTORS */

/* Good: Class selector (fast) */
.button {
  padding: 10px 20px;
}

/* Good: Direct child (efficient) */
.nav > li {
  display: inline-block;
}

/* ❌ INEFFICIENT SELECTORS */

/* Avoid: Universal selector with descendant */
.container * {
  box-sizing: border-box; /* Checks every element */
}

/* Avoid: Over-qualified selectors */
div.button#submit {
  /* Unnecessarily specific */
}

/* Avoid: Deep descendant selectors */
.header .nav ul li a span {
  /* Browser reads right-to-left, checks many elements */
}

/* Avoid: Attribute selectors on non-unique attributes */
[type="text"] {
  /* Must check every element's type attribute */
}

/* ✅ OPTIMIZED RENDERING */

/* Good: Use transform and opacity for animations */
.optimized-animation {
  transition: transform 0.3s, opacity 0.3s;
}

.optimized-animation:hover {
  transform: scale(1.05);
  opacity: 0.9;
}

/* Avoid: Properties that trigger layout recalculation */
.avoid-animation {
  transition: width 0.3s, height 0.3s, top 0.3s, left 0.3s;
}

/* ✅ CONTAINMENT */

/* Isolate expensive operations */
.card {
  contain: layout style paint; /* Limits scope of changes */
}

.isolated-component {
  contain: layout; /* Prevents layout thrashing */
}

/* ✅ CONTENT-VISIBILITY */

/* Lazy render off-screen content */
.article-list-item {
  content-visibility: auto;
  contain-intrinsic-size: 0 500px; /* Reserve space */
}

/* ✅ WILL-CHANGE (use sparingly) */

.hover-card:hover {
  will-change: transform;
  transform: translateY(-5px);
}

/* Remove will-change after animation */
.animated-element {
  animation: slideIn 0.5s;
}

.animated-element.animation-complete {
  will-change: auto;
}
```

### Critical CSS Strategy

```css
/* Inline in <head> - Above-the-fold critical styles */
/* Keep under 14KB for first TCP packet */

/* Critical CSS example */
body {
  margin: 0;
  font-family: sans-serif;
}

.header {
  position: fixed;
  top: 0;
  width: 100%;
  background: white;
  z-index: 100;
}

.hero {
  height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
}

/* Non-critical CSS loaded asynchronously */
/* <link rel="preload" href="styles.css" as="style" onload="this.onload=null;this.rel='stylesheet'"> */
```

### CSS Loading Performance

```html
<!-- Optimal CSS loading strategies -->

<!-- 1. Preload critical CSS -->
<link rel="preload" href="critical.css" as="style">
<link rel="stylesheet" href="critical.css">

<!-- 2. Async load non-critical CSS -->
<link rel="preload" href="styles.css" as="style" onload="this.onload=null;this.rel='stylesheet'">
<noscript><link rel="stylesheet" href="styles.css"></noscript>

<!-- 3. Media-specific loading -->
<link rel="stylesheet" href="print.css" media="print">
<link rel="stylesheet" href="desktop.css" media="(min-width: 1024px)">

<!-- 4. DNS prefetch for external resources -->
<link rel="dns-prefetch" href="//fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
```

## Modern CSS Features

### CSS Grid Advanced Techniques

```css
/* Auto-fit vs Auto-fill */

/* Auto-fill: Creates as many tracks as fit, even if empty */
.grid-auto-fill {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 20px;
}

/* Auto-fit: Creates tracks only for content, stretches to fill */
.grid-auto-fit {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 20px;
}

/* Masonry-like layout */
.masonry {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  grid-auto-rows: 20px;
  gap: 10px;
}

.masonry-item {
  grid-row: span var(--row-span);
}

/* Overlapping grid items */
.overlap-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  grid-template-rows: repeat(3, 100px);
}

.overlap-item-1 {
  grid-area: 1 / 1 / 3 / 3;
  z-index: 1;
}

.overlap-item-2 {
  grid-area: 2 / 2 / 4 / 4;
  z-index: 2;
}

/* Dense packing */
.dense-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  grid-auto-flow: dense; /* Fills gaps with smaller items */
  gap: 10px;
}
```

### Aspect Ratio

```css
/* Native aspect ratio (modern) */
.aspect-box {
  aspect-ratio: 16 / 9; /* Width / Height */
}

.square {
  aspect-ratio: 1; /* Equal width and height */
}

.portrait {
  aspect-ratio: 3 / 4;
}

/* With other properties */
.responsive-video {
  width: 100%;
  aspect-ratio: 16 / 9;
  object-fit: cover;
}

/* Fallback for older browsers */
.aspect-box-legacy {
  position: relative;
  width: 100%;
  padding-bottom: 56.25%; /* 16:9 = 9/16 = 0.5625 */
}

.aspect-box-legacy > * {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
}
```

### Object Fit & Position

```css
/* Object-fit controls how content fits container */
img {
  width: 300px;
  height: 200px;
}

.fit-fill {
  object-fit: fill; /* Stretch to fill (default) */
}

.fit-contain {
  object-fit: contain; /* Scale to fit, maintain aspect ratio */
}

.fit-cover {
  object-fit: cover; /* Fill container, maintain aspect ratio, crop if needed */
}

.fit-none {
  object-fit: none; /* No resizing */
}

.fit-scale-down {
  object-fit: scale-down; /* Smaller of none or contain */
}

/* Object-position controls alignment */
.positioned {
  object-fit: cover;
  object-position: center; /* Default */
}

.positioned-top {
  object-fit: cover;
  object-position: top; /* Align to top */
}

.positioned-custom {
  object-fit: cover;
  object-position: 25% 75%; /* Custom x y position */
}
```

### Clamp, Min, Max Functions

```css
/* clamp(min, preferred, max) */
.responsive-text {
  /* Font size between 1rem and 3rem, preferring 5vw */
  font-size: clamp(1rem, 5vw, 3rem);
  
  /* Padding between 10px and 50px */
  padding: clamp(10px, 3vw, 50px);
}

/* min() - picks smallest value */
.constrained-width {
  /* Never larger than 800px or 100% of parent */
  width: min(800px, 100%);
  
  /* Whichever is smaller */
  padding: min(5vw, 30px);
}

/* max() - picks largest value */
.minimum-size {
  /* Never smaller than 300px */
  width: max(300px, 50%);
  
  /* Ensure minimum font size */
  font-size: max(16px, 1rem);
}

/* Combining functions */
.complex-sizing {
  width: clamp(300px, 50%, 1200px);
  padding: clamp(
    10px,
    calc(2vw + 5px),
    40px
  );
  font-size: clamp(
    1rem,
    calc(1rem + 0.5vw),
    2rem
  );
}

/* Fluid spacing system */
.fluid-container {
  --min-space: 1rem;
  --max-space: 4rem;
  --min-width: 320px;
  --max-width: 1200px;
  
  padding: clamp(
    var(--min-space),
    calc(var(--min-space) + (var(--max-space) - var(--min-space)) * ((100vw - var(--min-width)) / (var(--max-width) - var(--min-width)))),
    var(--max-space)
  );
}
```

### CSS Filters & Backdrop Filters

```css
/* Filter effects */
.filtered-image {
  /* Blur */
  filter: blur(5px);
  
  /* Brightness */
  filter: brightness(1.5); /* 150% brightness */
  
  /* Contrast */
  filter: contrast(200%);
  
  /* Grayscale */
  filter: grayscale(100%);
  
  /* Hue rotation */
  filter: hue-rotate(90deg);
  
  /* Invert */
  filter: invert(100%);
  
  /* Opacity */
  filter: opacity(50%);
  
  /* Saturate */
  filter: saturate(200%);
  
  /* Sepia */
  filter: sepia(100%);
  
  /* Drop shadow */
  filter: drop-shadow(2px 2px 4px rgba(0,0,0,0.5));
  
  /* Multiple filters */
  filter: grayscale(50%) brightness(1.2) contrast(1.1);
}

/* Backdrop filter (blur background) */
.glass-effect {
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px) saturate(180%);
  border: 1px solid rgba(255, 255, 255, 0.2);
}

.frosted-glass {
  backdrop-filter: blur(20px) brightness(1.1);
  background: rgba(255, 255, 255, 0.8);
}

/* Dark mode with backdrop filter */
@media (prefers-color-scheme: dark) {
  .glass-effect {
    background: rgba(0, 0, 0, 0.3);
    backdrop-filter: blur(10px) saturate(180%);
  }
}
```

### Blend Modes

```css
/* Mix-blend-mode (blend element with background) */
.blend-element {
  mix-blend-mode: normal; /* Default */
  mix-blend-mode: multiply;
  mix-blend-mode: screen;
  mix-blend-mode: overlay;
  mix-blend-mode: darken;
  mix-blend-mode: lighten;
  mix-blend-mode: color-dodge;
  mix-blend-mode: color-burn;
  mix-blend-mode: hard-light;
  mix-blend-mode: soft-light;
  mix-blend-mode: difference;
  mix-blend-mode: exclusion;
  mix-blend-mode: hue;
  mix-blend-mode: saturation;
  mix-blend-mode: color;
  mix-blend-mode: luminosity;
}

/* Background-blend-mode (blend background layers) */
.blended-background {
  background: 
    linear-gradient(rgba(255, 0, 0, 0.5), rgba(0, 0, 255, 0.5)),
    url('texture.jpg');
  background-blend-mode: multiply;
}

/* Practical examples */
.duotone-effect {
  position: relative;
}

.duotone-effect::after {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: linear-gradient(45deg, #f00, #00f);
  mix-blend-mode: color;
}

.text-blend {
  font-size: 5rem;
  font-weight: bold;
  color: white;
  mix-blend-mode: difference; /* Inverts against background */
}
```

## Accessibility Best Practices

### Focus Management

```css
/* Visible focus indicators */
:focus {
  outline: 2px solid #007bff;
  outline-offset: 2px;
}

/* Keyboard-only focus (no mouse) */
:focus-visible {
  outline: 2px solid #007bff;
  outline-offset: 2px;
}

:focus:not(:focus-visible) {
  outline: none; /* Remove for mouse clicks */
}

/* Skip links */
.skip-link {
  position: absolute;
  top: -40px;
  left: 0;
  background: #000;
  color: #fff;
  padding: 8px;
  z-index: 100;
}

.skip-link:focus {
  top: 0;
}

/* Focus within (container has focused child) */
.form-group:focus-within {
  background: #f8f9fa;
  border-color: #007bff;
}
```

### Color Contrast

```css
/* Ensure sufficient contrast ratios */
/* WCAG AA: 4.5:1 for normal text, 3:1 for large text */
/* WCAG AAA: 7:1 for normal text, 4.5:1 for large text */

.good-contrast {
  background: #ffffff;
  color: #333333; /* Contrast ratio: 12.63:1 ✓ */
}

.insufficient-contrast {
  background: #ffffff;
  color: #cccccc; /* Contrast ratio: 1.61:1 ✗ */
}

/* Use contrast-checking tools during development */
```

### Screen Reader Support

```css
/* Visually hidden but accessible to screen readers */
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border-width: 0;
}

/* Focusable version */
.sr-only-focusable:focus {
  position: static;
  width: auto;
  height: auto;
  padding: inherit;
  margin: inherit;
  overflow: visible;
  clip: auto;
  white-space: normal;
}

/* Hide from screen readers but visible */
.aria-hidden {
  /* Use aria-hidden="true" attribute instead */
}
```

### Reduced Motion

```css
/* Respect user's motion preferences */
@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
    scroll-behavior: auto !important;
  }
}

/* Provide alternative animations */
.animated-element {
  animation: slideIn 0.5s ease-out;
}

@media (prefers-reduced-motion: reduce) {
  .animated-element {
    animation: fadeIn 0.01ms;
  }
}

@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}
```

### High Contrast Mode

```css
/* Windows High Contrast Mode */
@media (prefers-contrast: high) {
  .button {
    border: 2px solid;
  }
  
  .card {
    border: 1px solid;
  }
}

/* Forced colors mode */
@media (forced-colors: active) {
  .custom-button {
    border: 1px solid;
    forced-color-adjust: none; /* Opt out if needed */
  }
}
```

## CSS Architecture Patterns

### Component-Based Architecture

```css
/* Component structure */

/* 1. Component base */
.component {
  /* Base styles */
}

/* 2. Component variants */
.component--variant {
  /* Variant styles */
}

/* 3. Component states */
.component.is-active {
  /* State styles */
}

/* 4. Component elements */
.component__element {
  /* Element styles */
}

/* Complete example: Alert component */
.alert {
  padding: 15px 20px;
  border-radius: 4px;
  border: 1px solid transparent;
}

.alert--success {
  background: #d4edda;
  border-color: #c3e6cb;
  color: #155724;
}

.alert--danger {
  background: #f8d7da;
  border-color: #f5c6cb;
  color: #721c24;
}

.alert--warning {
  background: #fff3cd;
  border-color: #ffeaa7;
  color: #856404;
}

.alert__icon {
  float: left;
  margin-right: 10px;
}

.alert__title {
  font-weight: bold;
  margin-bottom: 5px;
}

.alert__message {
  margin: 0;
}

.alert.is-dismissible {
  padding-right: 40px;
  position: relative;
}

.alert__close {
  position: absolute;
  top: 15px;
  right: 15px;
  background: none;
  border: none;
  cursor: pointer;
}
```

### File Organization

```
styles/
├── base/
│   ├── _reset.css          # Browser resets
│   ├── _typography.css     # Font definitions
│   └── _variables.css      # CSS custom properties
├── components/
│   ├── _buttons.css        # Button styles
│   ├── _cards.css          # Card components
│   ├── _forms.css          # Form elements
│   └── _navigation.css     # Navigation components
├── layout/
│   ├── _grid.css           # Grid system
│   ├── _header.css         # Header layout
│   ├── _footer.css         # Footer layout
│   └── _sidebar.css        # Sidebar layout
├── pages/
│   ├── _home.css           # Home page specific
│   └── _about.css          # About page specific
├── utilities/
│   ├── _spacing.css        # Margin/padding utilities
│   ├── _text.css           # Text utilities
│   └── _display.css        # Display utilities
├── vendors/
│   └── _normalize.css      # Third-party CSS
└── main.css                # Main import file
```

```css
/* main.css - Import order matters */

/* 1. External dependencies */
@import 'vendors/normalize.css';

/* 2. Base styles (lowest specificity) */
@import 'base/variables.css';
@import 'base/reset.css';
@import 'base/typography.css';

/* 3. Layout */
@import 'layout/grid.css';
@import 'layout/header.css';
@import 'layout/footer.css';
@import 'layout/sidebar.css';

/* 4. Components */
@import 'components/buttons.css';
@import 'components/cards.css';
@import 'components/forms.css';
@import 'components/navigation.css';

/* 5. Page-specific */
@import 'pages/home.css';
@import 'pages/about.css';

/* 6. Utilities (highest specificity) */
@import 'utilities/spacing.css';
@import 'utilities/text.css';
@import 'utilities/display.css';
```

## CSS Debugging Techniques

### Visual Debugging

```css
/* Outline all elements */
* {
  outline: 1px solid red;
}

/* Outline by type */
div { outline: 1px solid red; }
span { outline: 1px solid blue; }
section { outline: 1px solid green; }

/* Debug grid/flexbox */
.debug-grid {
  background: repeating-linear-gradient(
    to right,
    rgba(255, 0, 0, 0.1) 0,
    rgba(255, 0, 0, 0.1) 1px,
    transparent 1px,
    transparent 10px
  );
}

/* Show empty elements */
:empty {
  outline: 2px solid red;
}

/* Highlight elements without alt text */
img:not([alt]),
img[alt=""] {
  border: 5px solid red;
}

/* Find broken links */
a[href=""],
a[href="#"],
a:not([href]) {
  outline: 2px solid orange;
}
```

### Logging and Comments

```css
/* Document complex calculations */
.element {
  /* 
   * Width calculation:
   * Container: 1200px
   * Padding: 20px × 2 = 40px
   * Gap: 15px × 3 = 45px
   * Available: 1200 - 40 - 45 = 1115px
   * Per item: 1115 ÷ 4 = 278.75px
   */
  width: calc((100% - 85px) / 4);
}

/* Mark browser-specific fixes */
.element {
  display: flex;
  
  /* Safari flexbox bug fix */
  min-height: 0;
  min-width: 0;
}

/* Temporary debugging styles */
/* DEBUG: Remove before production */
.debug-visible {
  opacity: 1 !important;
  display: block !important;
}
```

## Print Styles

```css
/* Print-specific styles */
@media print {
  /* Reset for print */
  *,
  *::before,
  *::after {
    background: transparent !important;
    color: black !important;
    box-shadow: none !important;
    text-shadow: none !important;
  }
  
  /* Page margins */
  @page {
    margin: 2cm;
  }
  
  /* Headers and footers */
  @page {
    @top-center {
      content: "My Document";
    }
    @bottom-right {
      content: "Page " counter(page) " of " counter(pages);
    }
  }
  
  /* Typography adjustments */
  body {
    font-size: 12pt;
    line-height: 1.5;
  }
  
  h1 {
    font-size: 24pt;
    page-break-after: avoid;
  }
  
  h2, h3 {
    page-break-after: avoid;
  }
  
  /* Avoid breaks inside elements */
  p, blockquote, pre {
    page-break-inside: avoid;
  }
  
  /* Images */
  img {
    max-width: 100% !important;
    page-break-inside: avoid;
  }
  
  /* Links */
  a[href]::after {
    content: " (" attr(href) ")";
  }
  
  /* Abbreviations */
  abbr[title]::after {
    content: " (" attr(title) ")";
  }
  
  /* Hide unnecessary elements */
  nav,
  aside,
  .no-print,
  .advertisement,
  .social-share {
    display: none !important;
  }
  
  /* Show hidden content */
  .print-only {
    display: block !important;
  }
  
  /* Tables */
  table {
    border-collapse: collapse;
  }
  
  thead {
    display: table-header-group;
  }
  
  tr {
    page-break-inside: avoid;
  }
  
  /* Orphans and widows */
  p {
    orphans: 3;
    widows: 3;
  }
}
```

## Browser Compatibility

### Vendor Prefixes

```css
/* Autoprefixer handles this automatically, but for reference: */

.prefixed {
  /* Webkit (Chrome, Safari, newer Edge) */
  -webkit-transform: rotate(45deg);
  -webkit-transition: all 0.3s;
  
  /* Mozilla (Firefox) */
  -moz-transform: rotate(45deg);
  -moz-transition: all 0.3s;
  
  /* Microsoft (IE, old Edge) */
  -ms-transform: rotate(45deg);
  -ms-transition: all 0.3s;
  
  /* Opera */
  -o-transform: rotate(45deg);
  -o-transition: all 0.3s;
  
  /* Standard (should be last) */
  transform: rotate(45deg);
  transition: all 0.3s;
}

/* Modern properties that still need prefixes */
.modern-prefixed {
  -webkit-backdrop-filter: blur(10px);
  backdrop-filter: blur(10px);
  
  -webkit-mask-image: linear-gradient(black, transparent);
  mask-image: linear-gradient(black, transparent);
  
  -webkit-line-clamp: 3;
  line-clamp: 3;
}
```

### Feature Queries

```css
/* Check if browser supports a feature */
@supports (display: grid) {
  .grid-layout {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
  }
}

/* Fallback for browsers that don't support grid */
@supports not (display: grid) {
  .grid-layout {
    display: flex;
    flex-wrap: wrap;
  }
  
  .grid-layout > * {
    flex: 1 1 calc(33.333% - 20px);
  }
}

/* Multiple conditions */
@supports (display: grid) and (gap: 20px) {
  .modern-grid {
    display: grid;
    gap: 20px;
  }
}

/* OR condition */
@supports (transform: rotate(45deg)) or (-webkit-transform: rotate(45deg)) {
  .rotated {
    transform: rotate(45deg);
  }
}

/* Practical examples */

/* Backdrop filter with fallback */
@supports (backdrop-filter: blur(10px)) or (-webkit-backdrop-filter: blur(10px)) {
  .glass-effect {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
  }
}

@supports not (backdrop-filter: blur(10px)) {
  .glass-effect {
    background: rgba(255, 255, 255, 0.8);
  }
}

/* CSS Grid with Flexbox fallback */
.layout {
  display: flex;
  flex-wrap: wrap;
}

@supports (display: grid) {
  .layout {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 20px;
  }
}

/* Aspect ratio with padding fallback */
.video-container {
  position: relative;
  padding-bottom: 56.25%; /* 16:9 */
}

@supports (aspect-ratio: 16 / 9) {
  .video-container {
    padding-bottom: 0;
    aspect-ratio: 16 / 9;
  }
}
```

## CSS Best Practices Summary

### Naming Conventions

```css
/* ✅ GOOD: Descriptive, semantic names */
.user-profile { }
.navigation-menu { }
.article-header { }
.submit-button { }

/* ❌ AVOID: Presentational names */
.red-text { }
.big-box { }
.float-left { }

/* ✅ GOOD: Consistent naming pattern */
.btn { }
.btn--primary { }
.btn--secondary { }
.btn__icon { }
.btn__text { }

/* ❌ AVOID: Inconsistent naming */
.button { }
.primaryBtn { }
.button-secondary { }
.buttonIcon { }
```

### Organization Principles

```css
/* 1. Group related properties */
.organized {
  /* Positioning */
  position: absolute;
  top: 0;
  left: 0;
  z-index: 10;
  
  /* Box model */
  display: block;
  width: 100%;
  height: 200px;
  padding: 20px;
  margin: 10px;
  
  /* Typography */
  font-family: sans-serif;
  font-size: 16px;
  line-height: 1.5;
  color: #333;
  
  /* Visual */
  background: white;
  border: 1px solid #ddd;
  border-radius: 4px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  
  /* Animation */
  transition: all 0.3s ease;
}

/* 2. Use logical groupings */
.button {
  /* State must be easily modifiable */
  background: var(--button-bg, #007bff);
  color: var(--button-color, white);
  
  /* Structure rarely changes */
  padding: var(--button-padding, 10px 20px);
  border-radius: var(--button-radius, 4px);
}
```

### DRY Principle (Don't Repeat Yourself)

```css
/* ❌ AVOID: Repetition */
.button-primary {
  padding: 10px 20px;
  border-radius: 4px;
  font-size: 16px;
  font-weight: 500;
  background: blue;
}

.button-secondary {
  padding: 10px 20px;
  border-radius: 4px;
  font-size: 16px;
  font-weight: 500;
  background: gray;
}

/* ✅ GOOD: Shared base class */
.button {
  padding: 10px 20px;
  border-radius: 4px;
  font-size: 16px;
  font-weight: 500;
}

.button-primary {
  background: blue;
}

.button-secondary {
  background: gray;
}

/* ✅ BETTER: CSS Custom Properties */
.button {
  padding: var(--button-padding, 10px 20px);
  border-radius: var(--button-radius, 4px);
  font-size: var(--button-font-size, 16px);
  font-weight: var(--button-font-weight, 500);
  background: var(--button-bg);
}

.button-primary {
  --button-bg: blue;
}

.button-secondary {
  --button-bg: gray;
}
```

### Maintainability Guidelines

```css
/* 1. Avoid magic numbers - use variables */

/* ❌ AVOID */
.header {
  height: 73px; /* Why 73? */
  margin-top: -12px; /* Why negative? */
}

/* ✅ GOOD */
:root {
  --header-height: 73px; /* Based on logo size + padding */
  --header-offset: -12px; /* Compensates for container margin */
}

.header {
  height: var(--header-height);
  margin-top: var(--header-offset);
}

/* 2. Comment complex logic */
.complex-layout {
  /* 
   * Using negative margin to pull element outside container
   * while maintaining grid alignment. This creates a full-width
   * effect without breaking the grid structure.
   */
  margin-left: calc(-1 * var(--container-padding));
  margin-right: calc(-1 * var(--container-padding));
}

/* 3. Document browser-specific fixes */
.safari-fix {
  /* Safari doesn't support gap in flexbox before version 14.1 */
  gap: 20px;
}

.safari-fix > * {
  margin: 10px; /* Fallback for older Safari */
}

/* 4. Keep specificity low */

/* ❌ AVOID: High specificity */
div#container .content ul li a.link {
  color: blue;
}

/* ✅ GOOD: Low specificity */
.nav-link {
  color: blue;
}

/* 5. Avoid !important unless absolutely necessary */

/* ❌ AVOID */
.text {
  color: blue !important;
}

/* ✅ GOOD: Increase specificity properly */
.container .text {
  color: blue;
}

/* ✅ ACCEPTABLE: Utility classes */
.u-hidden {
  display: none !important; /* Utility should always work */
}
```

## CSS Testing Strategies

### Visual Regression Testing

```css
/* Establish baseline styles for testing */
.test-container {
  /* Fixed dimensions for consistent screenshots */
  width: 1200px;
  height: 800px;
  
  /* Remove animations for testing */
  animation: none !important;
  transition: none !important;
}

/* Test different states */
.button {
  /* Default state */
}

.button:hover {
  /* Hover state */
}

.button:active {
  /* Active state */
}

.button:focus {
  /* Focus state */
}

.button:disabled {
  /* Disabled state */
}

.button.is-loading {
  /* Loading state */
}
```

### Cross-browser Testing Checklist

```css
/* Properties that commonly need testing across browsers */

/* 1. Flexbox - especially in IE11 */
.flex-test {
  display: flex;
  flex-wrap: wrap;
  gap: 20px; /* Not supported in older browsers */
}

/* 2. Grid - especially in older browsers */
.grid-test {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
}

/* 3. Custom properties - not supported in IE */
:root {
  --primary: #007bff;
}

/* 4. Sticky positioning - partial support in older browsers */
.sticky-test {
  position: sticky;
  top: 0;
}

/* 5. Backdrop filter - limited support */
.backdrop-test {
  backdrop-filter: blur(10px);
}

/* 6. Aspect ratio - newer property */
.aspect-test {
  aspect-ratio: 16 / 9;
}

/* 7. Container queries - very new */
@container (min-width: 400px) {
  .container-test {
    display: grid;
  }
}

/* 8. :has() selector - newer */
.parent:has(.child) {
  background: gray;
}
```

## Advanced CSS Patterns

### CSS Shapes

```css
/* Basic shapes */
.triangle {
  width: 0;
  height: 0;
  border-left: 50px solid transparent;
  border-right: 50px solid transparent;
  border-bottom: 100px solid blue;
}

.circle {
  width: 100px;
  height: 100px;
  border-radius: 50%;
  background: blue;
}

/* CSS clip-path */
.clipped {
  clip-path: polygon(50% 0%, 100% 50%, 50% 100%, 0% 50%); /* Diamond */
  clip-path: circle(50% at 50% 50%); /* Circle */
  clip-path: ellipse(25% 40% at 50% 50%); /* Ellipse */
  clip-path: inset(10px 20px 30px 40px); /* Rectangle with insets */
}

/* Shape-outside for text wrapping */
.shaped-float {
  width: 200px;
  height: 200px;
  float: left;
  shape-outside: circle(50%);
  clip-path: circle(50%);
  background: url('image.jpg');
}

/* Complex polygon */
.hexagon {
  clip-path: polygon(
    50% 0%,
    100% 25%,
    100% 75%,
    50% 100%,
    0% 75%,
    0% 25%
  );
}
```

### CSS Counters

```css
/* Automatic numbering */
body {
  counter-reset: section;
}

h2 {
  counter-reset: subsection;
}

h2::before {
  counter-increment: section;
  content: "Section " counter(section) ": ";
}

h3::before {
  counter-increment: subsection;
  content: counter(section) "." counter(subsection) " ";
}

/* Styled lists */
.custom-list {
  list-style: none;
  counter-reset: item;
}

.custom-list li {
  counter-increment: item;
  position: relative;
  padding-left: 40px;
}

.custom-list li::before {
  content: counter(item);
  position: absolute;
  left: 0;
  width: 30px;
  height: 30px;
  background: #007bff;
  color: white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: bold;
}

/* Nested counters */
.nested-list {
  counter-reset: section;
}

.nested-list > li {
  counter-increment: section;
}

.nested-list > li::before {
  content: counters(section, ".") " ";
}
```

### CSS Math Functions

```css
/* Calculations */
.calculated {
  /* Basic arithmetic */
  width: calc(100% - 50px);
  height: calc(100vh - 100px);
  
  /* Complex calculations */
  font-size: calc(16px + (24 - 16) * ((100vw - 320px) / (1200 - 320)));
  
  /* With CSS variables */
  --spacing: 20px;
  margin: calc(var(--spacing) * 2);
  padding: calc(var(--spacing) / 2);
}

/* Min/Max */
.constrained {
  width: min(90%, 1200px); /* Whichever is smaller */
  font-size: max(16px, 1vw); /* Whichever is larger */
}

/* Clamp (min, preferred, max) */
.fluid {
  /* Responsive typography */
  font-size: clamp(1rem, 2.5vw, 2rem);
  
  /* Responsive spacing */
  padding: clamp(1rem, 5vw, 3rem);
  
  /* Responsive width */
  width: clamp(300px, 50%, 800px);
}

/* Trigonometric functions (newer) */
.trig {
  /* Sine, cosine, tangent */
  width: calc(100px * sin(45deg));
  height: calc(100px * cos(45deg));
  transform: rotate(calc(tan(45deg) * 1rad));
  
  /* Arc functions */
  transform: rotate(asin(0.5));
  transform: rotate(acos(0.5));
  transform: rotate(atan(1));
}

/* Exponential functions */
.exponential {
  opacity: calc(pow(0.5, 2)); /* 0.5^2 = 0.25 */
  transform: scale(calc(sqrt(4))); /* √4 = 2 */
}
```

## References

- [W3C CSS Specifications](https://www.w3.org/Style/CSS/specs.en.html){:target="_blank"}
- [MDN Web Docs - CSS](https://developer.mozilla.org/en-US/docs/Web/CSS){:target="_blank"}
- [CSS Tricks](https://css-tricks.com/){:target="_blank"}
- [Can I Use - Browser Support Tables](https://caniuse.com/){:target="_blank"}
- [Web.dev - CSS](https://web.dev/learn/css/){:target="_blank"}
- [CSSWG Drafts (Editor's Drafts)](https://drafts.csswg.org/){:target="_blank"}
- [Smashing Magazine - CSS](https://www.smashingmagazine.com/category/css){:target="_blank"}
- [A Complete Guide to Flexbox](https://css-tricks.com/snippets/css/a-guide-to-flexbox/){:target="_blank"}
- [A Complete Guide to CSS Grid](https://css-tricks.com/snippets/css/complete-guide-grid/){:target="_blank"}
- [WebAIM - Contrast Checker](https://webaim.org/resources/contrastchecker/){:target="_blank"}
- [CSS Selectors Reference](https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_Selectors){:target="_blank"}
- [CSS Animation Performance](https://web.dev/animations-guide/){:target="_blank"}

---

*This comprehensive guide covers CSS3 fundamentals through advanced techniques. Regular practice and experimentation with these concepts will develop mastery over modern CSS development. Always test across multiple browsers and devices, prioritize accessibility, and follow performance best practices for production applications.*
