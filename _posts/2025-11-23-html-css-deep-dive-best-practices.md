---
layout: post
title: "🌊 HTML5 / CSS3: Deep Dive & Best Practices"
description: "Concise, clear, and validated revision notes on HTML5 / CSS3 — practical best practices for beginners and practitioners."
author: technical_notes
date: 2025-11-23 05:00:00 +0530
categories: [Notes, HTML5 CSS3]
tags: [HTML5, CSS3, RWD, Flexbox, Grid, Accessibility, Performance, Web Development, Frontend]
author: technical_notes
image: /assets/img/posts/html5_css3_js.webp
toc: true
comments: false
math: false
mermaid: true
---

# HTML5 / CSS3: Deep Dive & Best Practices

> # **[HTML / CSS / JS Cheatsheet](https://htmlcheatsheet.com/){: target="_blank" rel="noopener noreferrer" }**

## Table of Contents
1. [HTML5 Fundamentals](#html5-fundamentals)
2. [Semantic HTML5 Elements](#semantic-html5-elements)
3. [CSS3 Fundamentals](#css3-fundamentals)
4. [Responsive Web Design (RWD)](#responsive-web-design-rwd)
5. [CSS Flexbox](#css-flexbox)
6. [CSS Grid](#css-grid)
7. [Accessibility & ARIA](#accessibility--aria)
8. [Performance Optimization](#performance-optimization)
9. [Terminology Tables](#terminology-tables)

---

## HTML5 Fundamentals

### What is HTML5?

HTML5 (HyperText Markup Language version 5) represents the latest evolution of the standard markup language for creating web pages. Introduced officially in 2014 by the W3C, HTML5 brought significant enhancements including support for modern multimedia, improved semantic elements, and better integration with web applications.

**Key Features:**
- Native support for audio and video without plugins
- Canvas and SVG for graphics
- Geolocation API
- Local storage capabilities
- New form input types
- Enhanced semantic structure

### Document Structure

Every HTML5 document follows a standard structure:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Page Title</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <!-- Content goes here -->
</body>
</html>
```

**Essential Meta Tags:**
- `charset="UTF-8"`: Specifies character encoding
- `viewport`: Controls responsive behavior on mobile devices
- `description`: SEO metadata
- `keywords`: Search engine keywords

---

## Semantic HTML5 Elements

### Understanding Semantics

Semantic HTML refers to using HTML elements that clearly describe their meaning both to browsers and developers. These elements convey the purpose and structure of content, improving accessibility, SEO, and code maintainability.

### Core Structural Elements

#### `<header>`
Represents introductory content or navigational aids. Can contain logos, navigation menus, search forms, and heading elements.

```html
<header>
    <h1>Website Name</h1>
    <nav>
        <ul>
            <li><a href="#home">Home</a></li>
            <li><a href="#about">About</a></li>
        </ul>
    </nav>
</header>
```

#### `<nav>`
Defines a section containing navigation links, either for site navigation or within-page navigation.

```html
<nav role="navigation">
    <ul>
        <li><a href="/">Home</a></li>
        <li><a href="/products">Products</a></li>
        <li><a href="/contact">Contact</a></li>
    </ul>
</nav>
```

#### `<main>`
Represents the dominant content of the document. Use only once per page and not nested within `<article>`, `<aside>`, `<header>`, `<footer>`, or `<nav>`.

```html
<main>
    <h1>Main Content Title</h1>
    <p>This is the primary content of the page.</p>
</main>
```

#### `<article>`
Represents a self-contained composition that could be independently distributed or reused. Examples include blog posts, news articles, forum posts, or user comments.

```html
<article>
    <header>
        <h2>Article Title</h2>
        <time datetime="2025-11-23">November 23, 2025</time>
    </header>
    <p>Article content goes here...</p>
    <footer>
        <p>Author: Jane Doe</p>
    </footer>
</article>
```

#### `<section>`
Defines a thematic grouping of content, typically with a heading. Use when content represents a distinct section of a document.

```html
<section>
    <h2>Services</h2>
    <p>Description of services offered...</p>
</section>
```

#### `<aside>`
Contains content tangentially related to the main content, such as sidebars, pull quotes, or related links.

```html
<aside>
    <h3>Related Articles</h3>
    <ul>
        <li><a href="#">Article 1</a></li>
        <li><a href="#">Article 2</a></li>
    </ul>
</aside>
```

#### `<footer>`
Represents footer content for its nearest sectioning content or sectioning root element. Typically contains copyright information, links to privacy policies, or contact information.

```html
<footer>
    <p>&copy; 2025 Company Name. All rights reserved.</p>
    <nav>
        <a href="/privacy">Privacy Policy</a>
        <a href="/terms">Terms of Service</a>
    </nav>
</footer>
```

### Content Elements

#### `<figure>` and `<figcaption>`
Used to group media content with its caption, providing semantic meaning to images, diagrams, or code snippets.

```html
<figure>
    <img src="chart.png" alt="Sales data visualization">
    <figcaption>Q4 2025 Sales Performance</figcaption>
</figure>
```

#### `<time>`
Represents a specific time or date, enabling machine-readable datetime values.

```html
<time datetime="2025-11-23T14:30:00">November 23, 2025 at 2:30 PM</time>
```

#### `<mark>`
Highlights text for reference or notation purposes, typically rendered with a yellow background.

```html
<p>This is <mark>important information</mark> to remember.</p>
```

### Best Practices for Semantic HTML

1. **Use semantic elements over `<div>` and `<span>` when appropriate**
   - Provides better accessibility
   - Improves SEO performance
   - Enhances code readability

2. **Maintain proper nesting hierarchy**
   - Headings inside `<section>` or `<article>`
   - `<footer>` within parent context
   - Single `<main>` element per page

3. **Avoid redundant ARIA roles**
   - Native semantic elements have built-in roles
   - Only add ARIA when necessary for enhanced accessibility

4. **Test with accessibility tools**
   - Use screen readers (NVDA, JAWS, VoiceOver)
   - Validate with WAVE, Lighthouse, or axe DevTools

---

## CSS3 Fundamentals

### What is CSS3?

CSS3 (Cascading Style Sheets Level 3) is the latest evolution of CSS, introducing modular specifications that enable advanced styling, animations, transitions, and responsive design capabilities.

**Key Features:**
- Flexible box layout (Flexbox)
- Grid layout system
- Custom properties (CSS variables)
- Advanced selectors
- Animations and transitions
- Media queries for responsive design
- Transform and filter effects

### CSS Syntax

```css
selector {
    property: value;
}
```

**Example:**
```css
.container {
    display: flex;
    justify-content: center;
    align-items: center;
    background-color: #f0f0f0;
    padding: 20px;
}
```

### Box Model

The CSS box model describes the rectangular boxes generated for elements, consisting of:

1. **Content**: The actual content of the element
2. **Padding**: Space between content and border
3. **Border**: Edge surrounding the padding
4. **Margin**: Space outside the border

```css
.box {
    width: 300px;
    padding: 20px;
    border: 2px solid #333;
    margin: 10px;
    box-sizing: border-box; /* Includes padding and border in width */
}
```

### Selectors

CSS3 introduced powerful selector capabilities:

```css
/* Universal selector */
* { margin: 0; }

/* Element selector */
p { color: black; }

/* Class selector */
.highlight { background: yellow; }

/* ID selector */
#header { font-size: 24px; }

/* Attribute selector */
input[type="email"] { border: 1px solid blue; }

/* Pseudo-class */
a:hover { color: red; }

/* Pseudo-element */
p::first-line { font-weight: bold; }

/* Child combinator */
div > p { margin-bottom: 10px; }

/* Adjacent sibling */
h1 + p { font-size: 18px; }

/* Descendant selector */
article p { line-height: 1.6; }
```

### CSS Variables (Custom Properties)

CSS variables enable reusable values throughout stylesheets:

```css
:root {
    --primary-color: #3498db;
    --secondary-color: #2ecc71;
    --font-stack: 'Helvetica Neue', sans-serif;
    --spacing-unit: 8px;
}

.button {
    background-color: var(--primary-color);
    font-family: var(--font-stack);
    padding: calc(var(--spacing-unit) * 2);
}
```

---

## Responsive Web Design (RWD)

### Core Principles

Responsive Web Design ensures websites adapt seamlessly across devices with different screen sizes and resolutions. The approach emphasizes flexibility, fluid grids, and user-centric design.

**Three Foundational Concepts:**

1. **Fluid Grids**: Use relative units (percentages, em, rem) instead of fixed pixels
2. **Flexible Images**: Images scale within their containers
3. **Media Queries**: Apply styles based on device characteristics

### Mobile-First Approach

Design for the smallest screen first, then progressively enhance for larger screens. This strategy prioritizes essential content and improves performance on mobile devices.

```css
/* Base styles for mobile */
.container {
    width: 100%;
    padding: 10px;
}

/* Tablet styles */
@media (min-width: 768px) {
    .container {
        width: 750px;
        margin: 0 auto;
        padding: 20px;
    }
}

/* Desktop styles */
@media (min-width: 1024px) {
    .container {
        width: 980px;
        padding: 30px;
    }
}
```

### Viewport Meta Tag

Essential for responsive design, the viewport meta tag controls how content scales on mobile devices:

```html
<meta name="viewport" content="width=device-width, initial-scale=1.0">
```

### Media Queries

Media queries enable conditional CSS application based on device features:

```css
/* Syntax */
@media media-type and (condition) {
    /* CSS rules */
}

/* Common breakpoints */
@media (max-width: 575px) {
    /* Extra small devices */
}

@media (min-width: 576px) and (max-width: 767px) {
    /* Small devices */
}

@media (min-width: 768px) and (max-width: 991px) {
    /* Medium devices */
}

@media (min-width: 992px) and (max-width: 1199px) {
    /* Large devices */
}

@media (min-width: 1200px) {
    /* Extra large devices */
}

/* Orientation queries */
@media (orientation: landscape) {
    /* Landscape mode */
}

@media (orientation: portrait) {
    /* Portrait mode */
}

/* High-DPI displays */
@media (-webkit-min-device-pixel-ratio: 2),
       (min-resolution: 192dpi) {
    /* Retina displays */
}
```

### Flexible Images

Images should never overflow their containers:

```css
img {
    max-width: 100%;
    height: auto;
    display: block;
}

/* Responsive background images */
.hero {
    background-image: url('image-mobile.jpg');
    background-size: cover;
    background-position: center;
}

@media (min-width: 768px) {
    .hero {
        background-image: url('image-desktop.jpg');
    }
}
```

### Responsive Typography

Use relative units for scalable text:

```css
/* Base font size */
html {
    font-size: 16px;
}

body {
    font-size: 1rem; /* 16px */
    line-height: 1.6;
}

h1 {
    font-size: 2.5rem; /* 40px */
}

/* Fluid typography with clamp() */
h1 {
    font-size: clamp(1.5rem, 5vw, 3rem);
}

p {
    font-size: clamp(1rem, 2vw, 1.125rem);
}
```

### Container Queries (Modern CSS)

Container queries enable components to respond to their container's size, not just the viewport:

```css
.card-container {
    container-type: inline-size;
    container-name: card;
}

@container card (min-width: 400px) {
    .card {
        display: flex;
        gap: 20px;
    }
}
```

### Best Practices

1. **Design mobile-first**: Start with smallest screen, enhance for larger
2. **Use relative units**: Prefer %, em, rem over px
3. **Test on real devices**: Simulators don't capture all behaviors
4. **Optimize images**: Use responsive images with `srcset` and `<picture>`
5. **Touch-friendly targets**: Minimum 48x48px tap targets
6. **Minimize animations on mobile**: Reduce complexity for performance
7. **Avoid fixed widths**: Use flexible layouts that adapt
8. **Test breakpoints thoroughly**: Ensure smooth transitions between sizes

---

## CSS Flexbox

### Introduction to Flexbox

Flexbox (Flexible Box Layout Module) is a one-dimensional layout method for arranging items in rows or columns. It provides powerful alignment and distribution capabilities for building flexible, responsive interfaces.

**When to Use Flexbox:**
- Arranging items in a single direction (row or column)
- Centering elements vertically and horizontally
- Creating equal-height columns
- Distributing space between elements
- Reordering elements visually

### Flex Container Properties

Activate Flexbox by setting `display: flex` on the parent container:

```css
.container {
    display: flex; /* or inline-flex */
}
```

#### `flex-direction`
Defines the main axis direction:

```css
.container {
    flex-direction: row; /* default */
    /* row | row-reverse | column | column-reverse */
}
```

#### `flex-wrap`
Controls whether items wrap to new lines:

```css
.container {
    flex-wrap: nowrap; /* default */
    /* nowrap | wrap | wrap-reverse */
}
```

#### `flex-flow`
Shorthand for `flex-direction` and `flex-wrap`:

```css
.container {
    flex-flow: row wrap;
}
```

#### `justify-content`
Aligns items along the main axis:

```css
.container {
    justify-content: flex-start; /* default */
    /* flex-start | flex-end | center | space-between | 
       space-around | space-evenly */
}
```

**Examples:**
```css
/* Center items horizontally */
.center-h {
    display: flex;
    justify-content: center;
}

/* Space items evenly */
.spaced {
    display: flex;
    justify-content: space-between;
}
```

#### `align-items`
Aligns items along the cross axis:

```css
.container {
    align-items: stretch; /* default */
    /* stretch | flex-start | flex-end | center | baseline */
}
```

**Example:**
```css
/* Center items vertically */
.center-v {
    display: flex;
    align-items: center;
}

/* Perfect centering */
.center-both {
    display: flex;
    justify-content: center;
    align-items: center;
}
```

#### `align-content`
Aligns multiple lines of flex items (only works with `flex-wrap: wrap`):

```css
.container {
    flex-wrap: wrap;
    align-content: flex-start;
    /* flex-start | flex-end | center | space-between | 
       space-around | stretch */
}
```

#### `gap`
Creates space between flex items:

```css
.container {
    display: flex;
    gap: 20px; /* row and column gap */
    /* or */
    row-gap: 20px;
    column-gap: 10px;
}
```

### Flex Item Properties

#### `order`
Controls the order of flex items:

```css
.item {
    order: 0; /* default */
}

.item-first {
    order: -1; /* Appears first */
}

.item-last {
    order: 1; /* Appears last */
}
```

#### `flex-grow`
Defines ability to grow and take available space:

```css
.item {
    flex-grow: 0; /* default */
}

.item-grow {
    flex-grow: 1; /* Takes available space */
}

.item-grow-double {
    flex-grow: 2; /* Takes twice as much space */
}
```

#### `flex-shrink`
Defines ability to shrink if necessary:

```css
.item {
    flex-shrink: 1; /* default */
}

.no-shrink {
    flex-shrink: 0; /* Won't shrink */
}
```

#### `flex-basis`
Sets initial size of flex item before growing or shrinking:

```css
.item {
    flex-basis: auto; /* default */
    /* or specific size */
    flex-basis: 200px;
    flex-basis: 30%;
}
```

#### `flex`
Shorthand for `flex-grow`, `flex-shrink`, and `flex-basis`:

```css
.item {
    flex: 0 1 auto; /* default */
    /* or common values */
    flex: 1; /* flex-grow: 1, flex-shrink: 1, flex-basis: 0% */
    flex: none; /* flex: 0 0 auto */
    flex: auto; /* flex: 1 1 auto */
}
```

#### `align-self`
Overrides container's `align-items` for individual items:

```css
.item {
    align-self: auto; /* default */
    /* auto | flex-start | flex-end | center | baseline | stretch */
}

.special-item {
    align-self: flex-end; /* Aligns to bottom */
}
```

### Practical Examples

#### Navigation Bar
```css
.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 2rem;
    background: #333;
}

.nav-logo {
    font-size: 1.5rem;
    color: white;
}

.nav-links {
    display: flex;
    gap: 2rem;
    list-style: none;
}
```

#### Card Layout
```css
.card-container {
    display: flex;
    flex-wrap: wrap;
    gap: 20px;
}

.card {
    flex: 1 1 300px; /* Grow, shrink, basis 300px */
    display: flex;
    flex-direction: column;
    padding: 20px;
    border: 1px solid #ddd;
    border-radius: 8px;
}

.card-content {
    flex-grow: 1; /* Takes available space */
}

.card-footer {
    margin-top: auto; /* Pushes to bottom */
}
```

#### Holy Grail Layout
```css
.container {
    display: flex;
    flex-direction: column;
    min-height: 100vh;
}

.header, .footer {
    flex-shrink: 0;
    background: #333;
    color: white;
    padding: 1rem;
}

.main {
    display: flex;
    flex: 1;
}

.content {
    flex: 1;
    padding: 2rem;
}

.sidebar {
    flex: 0 0 250px;
    background: #f0f0f0;
    padding: 1rem;
}

@media (max-width: 768px) {
    .main {
        flex-direction: column;
    }
    
    .sidebar {
        flex-basis: auto;
    }
}
```

---

## CSS Grid

### Introduction to Grid

CSS Grid Layout is a two-dimensional layout system for creating complex layouts with rows and columns simultaneously. Grid excels at creating structured, responsive layouts with precise control over placement.

**When to Use Grid:**
- Creating complex two-dimensional layouts
- Precise positioning of elements
- Magazine-style layouts
- Dashboard interfaces
- Any layout requiring both row and column control

### Grid Container Properties

Activate Grid by setting `display: grid`:

```css
.container {
    display: grid; /* or inline-grid */
}
```

#### `grid-template-columns` and `grid-template-rows`
Define the structure of the grid:

```css
/* Fixed sizes */
.grid {
    display: grid;
    grid-template-columns: 200px 200px 200px;
    grid-template-rows: 100px 100px;
}

/* Flexible with fr unit */
.grid {
    grid-template-columns: 1fr 2fr 1fr; /* Proportional sizing */
}

/* repeat() function */
.grid {
    grid-template-columns: repeat(3, 1fr); /* Three equal columns */
}

/* Mixed units */
.grid {
    grid-template-columns: 200px 1fr 2fr;
    grid-template-rows: auto 100px auto;
}

/* minmax() function */
.grid {
    grid-template-columns: repeat(3, minmax(200px, 1fr));
}
```

#### `grid-template-areas`
Define named grid areas for semantic layouts:

```css
.container {
    display: grid;
    grid-template-columns: 1fr 3fr 1fr;
    grid-template-rows: auto 1fr auto;
    grid-template-areas:
        "header header header"
        "sidebar content aside"
        "footer footer footer";
}

.header { grid-area: header; }
.sidebar { grid-area: sidebar; }
.content { grid-area: content; }
.aside { grid-area: aside; }
.footer { grid-area: footer; }
```

#### `gap`
Creates space between grid items:

```css
.grid {
    display: grid;
    gap: 20px; /* Both row and column gap */
    /* or */
    row-gap: 20px;
    column-gap: 10px;
}
```

#### `justify-items` and `align-items`
Align items within their grid cells:

```css
.grid {
    justify-items: start; /* start | end | center | stretch */
    align-items: start; /* start | end | center | stretch | baseline */
}
```

#### `justify-content` and `align-content`
Align the entire grid within the container:

```css
.grid {
    justify-content: center; /* start | end | center | space-between | space-around | space-evenly */
    align-content: center;
}
```

#### `grid-auto-columns` and `grid-auto-rows`
Define size of implicitly created tracks:

```css
.grid {
    grid-auto-rows: 100px;
    grid-auto-columns: 1fr;
}
```

#### `grid-auto-flow`
Controls how auto-placed items flow into the grid:

```css
.grid {
    grid-auto-flow: row; /* row | column | dense | row dense | column dense */
}
```

### Grid Item Properties

#### `grid-column` and `grid-row`
Position items using line numbers:

```css
.item {
    grid-column: 1 / 3; /* Spans from line 1 to 3 */
    grid-row: 1 / 2;
}

/* Shorthand */
.item {
    grid-column-start: 1;
    grid-column-end: 3;
    grid-row-start: 1;
    grid-row-end: 2;
}

/* Span keyword */
.item {
    grid-column: span 2; /* Spans 2 columns */
    grid-row: span 3; /* Spans 3 rows */
}
```

#### `grid-area`
Position item in named area or by line numbers:

```css
/* Named area */
.item {
    grid-area: header;
}

/* Line numbers (row-start / column-start / row-end / column-end) */
.item {
    grid-area: 1 / 1 / 2 / 4;
}
```

#### `justify-self` and `align-self`
Align individual item within its cell:

```css
.item {
    justify-self: center; /* start | end | center | stretch */
    align-self: center;
}
```

### Practical Examples

#### Responsive Card Grid
```css
.card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 2rem;
}

.card {
    border: 1px solid #ddd;
    padding: 1.5rem;
    border-radius: 8px;
}
```

**Explanation:**
- `auto-fit`: Automatically fits columns within container
- `minmax(300px, 1fr)`: Cards never smaller than 300px, can grow to fill space

#### Magazine Layout
```css
.magazine {
    display: grid;
    grid-template-columns: repeat(12, 1fr);
    gap: 20px;
}

.hero {
    grid-column: 1 / 13; /* Full width */
    grid-row: 1 / 3;
}

.feature-1 {
    grid-column: 1 / 7;
    grid-row: 3 / 5;
}

.feature-2 {
    grid-column: 7 / 13;
    grid-row: 3 / 4;
}

.sidebar {
    grid-column: 7 / 13;
    grid-row: 4 / 5;
}
```

#### Dashboard Layout
```css
.dashboard {
    display: grid;
    grid-template-columns: 250px 1fr;
    grid-template-rows: 80px 1fr 60px;
    grid-template-areas:
        "sidebar header"
        "sidebar main"
        "sidebar footer";
    height: 100vh;
    gap: 0;
}

.sidebar {
    grid-area: sidebar;
    background: #2c3e50;
}

.header {
    grid-area: header;
    background: #ecf0f1;
}

.main {
    grid-area: main;
    padding: 2rem;
    overflow-y: auto;
}

.footer {
    grid-area: footer;
    background: #95a5a6;
}

@media (max-width: 768px) {
    .dashboard {
        grid-template-columns: 1fr;
        grid-template-areas:
            "header"
            "main"
            "footer";
    }
    
    .sidebar {
        display: none;
    }
}
```

#### Image Gallery with Masonry Effect
```css
.gallery {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
    grid-auto-rows: 20px;
    gap: 10px;
}

.gallery-item {
    border-radius: 8px;
    overflow: hidden;
}

.gallery-item img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

/* Varying heights */
.gallery-item:nth-child(1) { grid-row: span 10; }
.gallery-item:nth-child(2) { grid-row: span 15; }
.gallery-item:nth-child(3) { grid-row: span 12; }
/* etc. */
```

### Grid vs Flexbox

| Feature | Grid | Flexbox |
|---------|------|---------|
| Dimensions | Two-dimensional (rows & columns) | One-dimensional (row or column) |
| Best For | Complex layouts, precise positioning | Simple layouts, alignment |
| Content Direction | Can overlap items | Items flow in sequence |
| Browser Support | Modern browsers (IE11 with prefix) | Excellent support |
| Use Case | Page layouts, dashboards | Navigation bars, cards |

**General Rule:** Use Flexbox for components, Grid for layouts. However, they complement each other and can be combined.

---

## Accessibility & ARIA

### Understanding Web Accessibility

Web accessibility ensures that websites are usable by people with disabilities, including visual, auditory, motor, and cognitive impairments. Accessible design benefits all users through improved usability and SEO.

### ARIA Overview

ARIA (Accessible Rich Internet Applications) is a set of attributes that define ways to make web content and applications more accessible to assistive technologies like screen readers.

**The First Rule of ARIA:** If you can use a native HTML element with the semantics and behavior you require, then do so. Use ARIA only when HTML alone is insufficient.

### ARIA Components

#### 1. Roles
Define the type of user interface element:

```html
<!-- Landmark roles -->
<div role="navigation">
<div role="main">
<div role="banner">
<div role="contentinfo">
<div role="complementary">
<div role="search">

<!-- Widget roles -->
<div role="button">
<div role="tab">
<div role="tabpanel">
<div role="dialog">
<div role="progressbar">
<div role="slider">
```

**Note:** HTML5 semantic elements have implicit roles, so avoid redundancy:

```html
<!-- Bad: Redundant -->
<nav role="navigation">

<!-- Good: Implicit role -->
<nav>

<!-- When needed: Custom widget -->
<div role="dialog" aria-labelledby="dialog-title">
```

#### 2. States and Properties

**States** (dynamic, change with user interaction):
```html
<button aria-pressed="false">Toggle</button>
<div aria-expanded="false">Collapsible content</div>
<input aria-invalid="true" aria-errormessage="error-1">
<input aria-checked="true" type="checkbox">
```

**Properties** (describe characteristics):
```html
<button aria-label="Close dialog">×</button>
<input aria-describedby="help-text">
<div aria-live="polite">Notification area</div>
<div aria-hidden="true">Decorative element</div>
```

### Essential ARIA Attributes

#### `aria-label`
Provides accessible name when no visible label exists:

```html
<button aria-label="Close">
    <svg><!-- close icon --></svg>
</button>

<nav aria-label="Main navigation">
    <!-- navigation links -->
</nav>
```

#### `aria-labelledby`
References visible label element(s):

```html
<h2 id="section-title">Account Settings</h2>
<div role="region" aria-labelledby="section-title">
    <!-- content -->
</div>

<!-- Multiple references -->
<span id="first-name">First Name:</span>
<span id="required">Required</span>
<input aria-labelledby="first-name required">
```

#### `aria-describedby`
Provides additional descriptive information:

```html
<label for="password">Password:</label>
<input 
    id="password" 
    type="password"
    aria-describedby="password-hint">
<span id="password-hint">Must be at least 8 characters</span>

<button aria-describedby="delete-confirm">Delete Account</button>
<div id="delete-confirm">This action cannot be undone</div>
```

#### `aria-live`
Announces dynamic content changes to screen readers:

```html
<!-- Polite: Announces when current speech finishes -->
<div aria-live="polite" aria-atomic="true">
    Items added to cart: 3
</div>

<!-- Assertive: Interrupts current speech -->
<div aria-live="assertive" role="alert">
    Error: Payment failed
</div>

<!-- Off: No announcement -->
<div aria-live="off">
    Background updates
</div>
```

#### `aria-hidden`
Hides content from assistive technologies:

```html
<!-- Decorative icon hidden from screen readers -->
<span aria-hidden="true" class="icon">★</span>
<span class="sr-only">Featured item</span>

<!-- Don't hide focusable elements -->
<!-- Bad -->
<button aria-hidden="true">Click me</button>

<!-- Good -->
<button>
    <span aria-hidden="true">→</span>
    Next
</button>
```

#### `aria-expanded`
Indicates if collapsible content is expanded:

```html
<button 
    aria-expanded="false"
    aria-controls="menu-dropdown"
    id="menu-button">
    Menu
</button>
<ul id="menu-dropdown" hidden>
    <li><a href="#">Item 1</a></li>
    <li><a href="#">Item 2</a></li>
</ul>

<script>
document.getElementById('menu-button').addEventListener('click', function() {
    const isExpanded = this.getAttribute('aria-expanded') === 'true';
    this.setAttribute('aria-expanded', !isExpanded);
    document.getElementById('menu-dropdown').hidden = isExpanded;
});
</script>
```

### Accessible Form Design

```html
<form>
    <!-- Proper label association -->
    <label for="username">Username:</label>
    <input 
        id="username" 
        type="text"
        required
        aria-required="true"
        aria-describedby="username-help">
    <small id="username-help">Choose a unique username</small>

    <!-- Fieldset for radio groups -->
    <fieldset>
        <legend>Select your plan:</legend>
        <label>
            <input type="radio" name="plan" value="basic">
            Basic
        </label>
        <label>
            <input type="radio" name="plan" value="premium">
            Premium
        </label>
    </fieldset>

    <!-- Error messaging -->
    <label for="email">Email:</label>
    <input 
        id="email"
        type="email"
        aria-invalid="true"
        aria-describedby="email-error">
    <span id="email-error" role="alert">
        Please enter a valid email address
    </span>

    <button type="submit">Submit</button>
</form>
```

### Focus Management

```css
/* Never remove focus indicators without replacement */
/* Bad */
*:focus {
    outline: none;
}

/* Good: Custom focus style */
:focus {
    outline: 2px solid #4A90E2;
    outline-offset: 2px;
}

/* Even better: Focus-visible for keyboard only */
:focus-visible {
    outline: 2px solid #4A90E2;
    outline-offset: 2px;
}

:focus:not(:focus-visible) {
    outline: none;
}

/* Skip link for keyboard navigation */
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
```

```html
<a href="#main-content" class="skip-link">Skip to main content</a>
```

### Accessible Modal Dialog

```html
<button id="open-modal" aria-haspopup="dialog">
    Open Settings
</button>

<div 
    id="modal"
    role="dialog"
    aria-modal="true"
    aria-labelledby="modal-title"
    hidden>
    <div class="modal-overlay"></div>
    <div class="modal-content">
        <h2 id="modal-title">Settings</h2>
        <p>Configure your preferences</p>
        <button id="close-modal" aria-label="Close dialog">
            ×
        </button>
    </div>
</div>

<script>
const modal = document.getElementById('modal');
const openBtn = document.getElementById('open-modal');
const closeBtn = document.getElementById('close-modal');
let previousFocus;

function openModal() {
    previousFocus = document.activeElement;
    modal.hidden = false;
    closeBtn.focus();
    document.body.style.overflow = 'hidden';
}

function closeModal() {
    modal.hidden = true;
    document.body.style.overflow = '';
    previousFocus.focus();
}

openBtn.addEventListener('click', openModal);
closeBtn.addEventListener('click', closeModal);

// Trap focus within modal
modal.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
        closeModal();
    }
});
</script>
```

### Screen Reader Only Content

```css
.sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
}

.sr-only-focusable:focus {
    position: static;
    width: auto;
    height: auto;
    overflow: visible;
    clip: auto;
    white-space: normal;
}
```

```html
<button>
    <span aria-hidden="true">🔍</span>
    <span class="sr-only">Search</span>
</button>
```

### Accessibility Checklist

1. **Semantic HTML**: Use appropriate elements (`<button>`, `<nav>`, `<main>`)
2. **Keyboard Navigation**: All interactive elements accessible via keyboard
3. **Focus Indicators**: Visible focus styles for all focusable elements
4. **Alt Text**: Descriptive alt text for images
5. **Color Contrast**: Minimum 4.5:1 ratio for normal text, 3:1 for large text
6. **Form Labels**: Every input has an associated label
7. **Heading Hierarchy**: Logical heading structure (h1 → h2 → h3)
8. **ARIA**: Use appropriately, not excessively
9. **Dynamic Content**: Announce updates with aria-live
10. **Test**: Use screen readers (NVDA, JAWS, VoiceOver) and automated tools

---

## Performance Optimization

### CSS Performance Best Practices

#### 1. Minimize CSS File Size

**Minification**: Remove whitespace, comments, and unnecessary characters:

```css
/* Before minification */
.container {
    display: flex;
    justify-content: center;
    align-items: center;
}

/* After minification */
.container{display:flex;justify-content:center;align-items:center}
```

**Remove Unused CSS**: Use tools like PurgeCSS to eliminate dead code:

```javascript
// PostCSS configuration
module.exports = {
    plugins: [
        require('@fullhuman/postcss-purgecss')({
            content: ['./**/*.html'],
            defaultExtractor: content => content.match(/[\w-/:]+(?<!:)/g) || []
        })
    ]
}
```

#### 2. Optimize Selectors

**Avoid Universal Selectors**:
```css
/* Bad: Matches every element */
* {
    box-sizing: border-box;
}

/* Good: Target specific elements */
html {
    box-sizing: border-box;
}

*, *::before, *::after {
    box-sizing: inherit;
}
```

**Keep Specificity Low**:
```css
/* Bad: High specificity */
body div.container ul li a.link {
    color: blue;
}

/* Good: Low specificity */
.nav-link {
    color: blue;
}
```

**Avoid Over-Qualification**:
```css
/* Bad */
div.container {
    width: 100%;
}

/* Good */
.container {
    width: 100%;
}
```

#### 3. Reduce Repaints and Reflows

**Use `transform` and `opacity` for Animations** (GPU-accelerated):

```css
/* Bad: Triggers layout */
.box {
    transition: left 0.3s, top 0.3s;
}

.box:hover {
    left: 100px;
    top: 100px;
}

/* Good: GPU-accelerated */
.box {
    transition: transform 0.3s;
}

.box:hover {
    transform: translate(100px, 100px);
}
```

**Batch DOM Changes**:
```javascript
// Bad: Multiple reflows
element.style.width = '100px';
element.style.height = '100px';
element.style.border = '1px solid black';

// Good: Single reflow
element.style.cssText = 'width: 100px; height: 100px; border: 1px solid black;';

// Better: Use classes
element.classList.add('styled-box');
```

**Use `will-change` Sparingly**:
```css
/* Hint browser to optimize */
.animated-box {
    will-change: transform, opacity;
}

/* Remove after animation */
.animated-box.finished {
    will-change: auto;
}
```

#### 4. Optimize Font Loading

```css
/* Preload critical fonts */
<link rel="preload" href="font.woff2" as="font" type="font/woff2" crossorigin>

/* font-display for FOUT control */
@font-face {
    font-family: 'CustomFont';
    src: url('font.woff2') format('woff2');
    font-display: swap; /* or optional, fallback, block */
}
```

**Font-display values**:
- `swap`: Show fallback immediately, swap when custom font loads
- `optional`: Use custom font if available quickly, otherwise use fallback
- `fallback`: Brief block period, then fallback, swap if loaded in time
- `block`: Block rendering until font loads (avoid unless necessary)

#### 5. Critical CSS

Inline critical above-the-fold CSS to eliminate render-blocking:

```html
<head>
    <style>
        /* Critical CSS inlined */
        body { margin: 0; font-family: sans-serif; }
        .hero { min-height: 100vh; display: flex; }
    </style>
    
    <!-- Non-critical CSS loaded asynchronously -->
    <link rel="preload" href="styles.css" as="style" onload="this.onload=null;this.rel='stylesheet'">
    <noscript><link rel="stylesheet" href="styles.css"></noscript>
</head>
```

#### 6. Use CSS Containment

Limit scope of browser's layout and paint work:

```css
.widget {
    contain: layout style; /* Isolate layout and style changes */
}

.card {
    contain: layout paint; /* Isolate layout and paint */
}

.isolated-component {
    contain: strict; /* Strongest containment */
}
```

#### 7. Optimize Images and Background Images

```css
/* Responsive background images */
.hero {
    background-image: url('hero-small.jpg');
}

@media (min-width: 768px) {
    .hero {
        background-image: url('hero-medium.jpg');
    }
}

@media (min-width: 1200px) {
    .hero {
        background-image: url('hero-large.jpg');
    }
}

/* Modern image formats with fallback */
.hero {
    background-image: url('hero.jpg');
    background-image: image-set(
        'hero.webp' 1x,
        'hero.avif' 1x
    );
}
```

#### 8. Lazy Load Off-Screen Content

```html
<!-- Native lazy loading -->
<img src="image.jpg" loading="lazy" alt="Description">

<!-- Intersection Observer for CSS backgrounds -->
<div class="lazy-bg" data-bg="image.jpg"></div>

<script>
const lazyBackgrounds = document.querySelectorAll('.lazy-bg');

const bgObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            const bg = entry.target.dataset.bg;
            entry.target.style.backgroundImage = `url(${bg})`;
            bgObserver.unobserve(entry.target);
        }
    });
});

lazyBackgrounds.forEach(bg => bgObserver.observe(bg));
</script>
```

### HTML Performance Best Practices

#### 1. Minimize DOM Size

Keep DOM tree shallow and small:

```html
<!-- Bad: Deep nesting -->
<div>
    <div>
        <div>
            <div>
                <div>
                    <p>Content</p>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- Good: Flat structure -->
<div class="container">
    <p>Content</p>
</div>
```

**Guidelines**:
- Keep total nodes under 1500
- Maximum depth of 32 nodes
- Parent nodes with fewer than 60 children

#### 2. Resource Hints

```html
<!-- DNS Prefetch: Resolve DNS early -->
<link rel="dns-prefetch" href="https://api.example.com">

<!-- Preconnect: Establish connection early -->
<link rel="preconnect" href="https://fonts.googleapis.com">

<!-- Prefetch: Fetch resource for next navigation -->
<link rel="prefetch" href="next-page.html">

<!-- Preload: Fetch critical resource -->
<link rel="preload" href="critical.css" as="style">
<link rel="preload" href="font.woff2" as="font" crossorigin>
```

#### 3. Async and Defer Scripts

```html
<!-- Blocks parsing -->
<script src="script.js"></script>

<!-- Downloads in parallel, executes after parsing (order preserved) -->
<script src="script.js" defer></script>

<!-- Downloads and executes asynchronously (no guaranteed order) -->
<script src="analytics.js" async></script>
```

**When to use**:
- `defer`: For scripts that depend on DOM or other scripts
- `async`: For independent scripts (analytics, ads)
- Neither: For critical scripts needed during parsing

#### 4. Optimize Images

```html
<!-- Responsive images with srcset -->
<img 
    src="image-400.jpg"
    srcset="image-400.jpg 400w,
            image-800.jpg 800w,
            image-1200.jpg 1200w"
    sizes="(max-width: 600px) 100vw,
           (max-width: 1200px) 50vw,
           33vw"
    alt="Description">

<!-- Art direction with picture -->
<picture>
    <source media="(min-width: 1200px)" srcset="large.jpg">
    <source media="(min-width: 768px)" srcset="medium.jpg">
    <img src="small.jpg" alt="Description">
</picture>

<!-- Modern formats with fallback -->
<picture>
    <source srcset="image.avif" type="image/avif">
    <source srcset="image.webp" type="image/webp">
    <img src="image.jpg" alt="Description">
</picture>
```

### Performance Monitoring

#### Key Metrics

1. **First Contentful Paint (FCP)**: When first content appears (target: < 1.8s)
2. **Largest Contentful Paint (LCP)**: When main content loads (target: < 2.5s)
3. **First Input Delay (FID)**: Time to interactive (target: < 100ms)
4. **Cumulative Layout Shift (CLS)**: Visual stability (target: < 0.1)
5. **Time to Interactive (TTI)**: Full interactivity (target: < 3.8s)

#### Tools

- **Lighthouse**: Automated audits in Chrome DevTools
- **WebPageTest**: Detailed performance analysis
- **Chrome DevTools Performance**: Real-time performance profiling
- **PageSpeed Insights**: Google's performance scoring

---

## Terminology Tables

### Table 1: Development Phase Terminology

| General Term | HTML/CSS Context | Alternative Names | Description |
|--------------|------------------|-------------------|-------------|
| **Planning** | Information Architecture | Wireframing, Sitemap Creation | Defining structure and content hierarchy |
| **Design** | UI/UX Design | Visual Design, Mockup Creation | Creating visual appearance and user experience |
| **Development** | Markup & Styling | Coding, Implementation | Writing HTML structure and CSS styles |
| **Testing** | Cross-browser Testing | QA, Validation | Ensuring compatibility and correctness |
| **Optimization** | Performance Tuning | Refactoring, Enhancement | Improving speed and efficiency |
| **Deployment** | Publishing | Launch, Going Live | Making website accessible to users |
| **Maintenance** | Updates & Fixes | Support, Iteration | Ongoing improvements and bug fixes |

### Table 2: Layout Terminology Hierarchy

| Level | Term | Scope | Used With | Description |
|-------|------|-------|-----------|-------------|
| **1. Document** | Page Layout | Entire page | Grid, Flexbox | Overall page structure |
| **2. Section** | Section Layout | Major divisions | Grid, Semantic HTML | Header, main, footer, aside |
| **3. Container** | Component Layout | Grouped elements | Flexbox, Grid | Cards, navigation, forms |
| **4. Element** | Box Model | Individual items | CSS Properties | Padding, border, margin |
| **5. Content** | Typography | Text & media | Font properties | Text styling, spacing |

### Table 3: Responsive Design Terminology

| Term | Synonym | Context | Definition |
|------|---------|---------|------------|
| **Breakpoint** | Media Query Point | RWD | Viewport width where layout changes |
| **Viewport** | Screen Size | RWD | Visible area of web page |
| **Fluid Grid** | Flexible Grid | Layout | Grid using relative units (%, fr) |
| **Fixed Layout** | Static Layout | Layout | Layout with pixel-based widths |
| **Adaptive Design** | Progressive Enhancement | Strategy | Distinct layouts for specific breakpoints |
| **Responsive Design** | Fluid Design | Strategy | Continuous adaptation across sizes |
| **Mobile-First** | Progressive Enhancement | Approach | Design for mobile, enhance for desktop |
| **Desktop-First** | Graceful Degradation | Approach | Design for desktop, adapt for mobile |

### Table 4: CSS Layout System Terminology

| System | Primary Axis | Best For | Key Properties |
|--------|--------------|----------|----------------|
| **Block Flow** | Vertical | Documents | `display: block`, `margin`, `padding` |
| **Inline Flow** | Horizontal | Text | `display: inline`, `line-height` |
| **Flexbox** | Single axis | Components | `display: flex`, `justify-content`, `align-items` |
| **Grid** | Two axes | Page layouts | `display: grid`, `grid-template` |
| **Float** | Horizontal | Legacy layouts | `float`, `clear` (avoid for layout) |
| **Position** | Z-axis | Overlays | `position`, `top`, `left`, `z-index` |

### Table 5: Selector Terminology Hierarchy

| Specificity | Selector Type | Example | Weight |
|-------------|---------------|---------|--------|
| **Lowest** | Universal | `*` | 0-0-0 |
| **Low** | Element | `div`, `p` | 0-0-1 |
| **Medium-Low** | Class | `.container` | 0-1-0 |
| **Medium** | Attribute | `[type="text"]` | 0-1-0 |
| **Medium** | Pseudo-class | `:hover`, `:first-child` | 0-1-0 |
| **Medium-High** | ID | `#header` | 1-0-0 |
| **Highest** | Inline | `style="..."` | 1-0-0-0 |
| **Override** | `!important` | `color: red !important` | Overrides all |

### Table 6: Box Model Terminology

| Layer | Direction | Properties | Affects |
|-------|-----------|------------|---------|
| **Content** | Inside → Out | `width`, `height` | Actual content area |
| **Padding** | Inside → Out | `padding`, `padding-*` | Space inside border |
| **Border** | Inside → Out | `border`, `border-*` | Edge around padding |
| **Margin** | Inside → Out | `margin`, `margin-*` | Space outside border |
| **Outline** | Outside | `outline` | Visual indicator (doesn't affect layout) |

### Table 7: Unit Terminology

| Category | Unit | Type | Relative To | Use Case |
|----------|------|------|-------------|----------|
| **Absolute** | `px` | Fixed | Device pixel | Precise control, borders |
| **Absolute** | `pt`, `cm`, `mm` | Fixed | Physical measurement | Print stylesheets |
| **Relative** | `%` | Flexible | Parent element | Widths, responsive sizing |
| **Relative** | `em` | Flexible | Parent font-size | Typography, spacing |
| **Relative** | `rem` | Flexible | Root font-size | Consistent scaling |
| **Relative** | `vw`, `vh` | Flexible | Viewport dimensions | Full-screen sections |
| **Relative** | `vmin`, `vmax` | Flexible | Smaller/larger viewport dimension | Responsive typography |
| **Relative** | `ch` | Flexible | Width of "0" character | Text containers |
| **Relative** | `ex` | Flexible | Height of "x" character | Typography |
| **Grid** | `fr` | Flexible | Available space | Grid track sizing |

### Table 8: Display Property Values

| Value | Behavior | Generates | Use Case |
|-------|----------|-----------|----------|
| `block` | Fills width, stacks vertically | Block-level box | Sections, containers |
| `inline` | Fits content, flows horizontally | Inline box | Text, links |
| `inline-block` | Inline with block properties | Inline-level block | Buttons, badges |
| `flex` | Flexbox container | Block-level flex | Component layouts |
| `inline-flex` | Inline flexbox container | Inline-level flex | Inline component layouts |
| `grid` | Grid container | Block-level grid | Page layouts |
| `inline-grid` | Inline grid container | Inline-level grid | Inline grid layouts |
| `none` | Removes from layout | No box | Hidden elements |
| `contents` | Removes box, keeps children | Children boxes | Semantic wrapper removal |

### Table 9: Position Property Values

| Value | Behavior | Positioned Relative To | Use Case |
|-------|----------|------------------------|----------|
| `static` | Normal flow | N/A | Default positioning |
| `relative` | Offset from normal position | Original position | Minor adjustments |
| `absolute` | Removed from flow | Nearest positioned ancestor | Tooltips, dropdowns |
| `fixed` | Removed from flow | Viewport | Headers, modal overlays |
| `sticky` | Hybrid behavior | Scroll container | Sticky headers |

---

## Advanced Techniques

### CSS Custom Properties (Variables) Deep Dive

```css
/* Global variables */
:root {
    --primary-hue: 210;
    --primary-sat: 80%;
    --primary-light: 50%;
    --primary-color: hsl(var(--primary-hue), var(--primary-sat), var(--primary-light));
    
    --spacing-unit: 8px;
    --spacing-xs: calc(var(--spacing-unit) * 0.5);
    --spacing-sm: var(--spacing-unit);
    --spacing-md: calc(var(--spacing-unit) * 2);
    --spacing-lg: calc(var(--spacing-unit) * 3);
    --spacing-xl: calc(var(--spacing-unit) * 4);
}

/* Component-specific overrides */
.dark-theme {
    --primary-light: 70%;
    --bg-color: #1a1a1a;
    --text-color: #ffffff;
}

/* Using variables with fallback */
.button {
    background-color: var(--primary-color, #3498db);
    padding: var(--spacing-md);
}

/* Dynamic theming with JavaScript */
<script>
document.documentElement.style.setProperty('--primary-hue', '150');
</script>
```

### Modern CSS Features

#### Container Queries
```css
.card-container {
    container-type: inline-size;
    container-name: card;
}

.card {
    padding: 1rem;
}

@container card (min-width: 400px) {
    .card {
        display: grid;
        grid-template-columns: 150px 1fr;
        gap: 1rem;
    }
}
```

#### CSS Nesting (New)
```css
.card {
    padding: 1rem;
    border: 1px solid #ddd;
    
    &:hover {
        border-color: #3498db;
    }
    
    & .card-title {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }
    
    & .card-content {
        color: #666;
    }
}
```

#### Logical Properties
```css
/* Instead of physical properties */
.box {
    margin-top: 20px;
    margin-left: 10px;
    border-left: 2px solid blue;
}

/* Use logical properties (supports RTL) */
.box {
    margin-block-start: 20px;
    margin-inline-start: 10px;
    border-inline-start: 2px solid blue;
}
```

#### `clamp()` for Responsive Values
```css
/* Fluid typography */
h1 {
    font-size: clamp(1.5rem, 4vw + 1rem, 3rem);
    /* min: 1.5rem, preferred: 4vw + 1rem, max: 3rem */
}

/* Fluid spacing */
.container {
    padding: clamp(1rem, 5%, 3rem);
    max-width: clamp(300px, 90%, 1200px);
}
```

#### CSS `aspect-ratio`
```css
/* Maintain aspect ratio */
.video-container {
    aspect-ratio: 16 / 9;
    width: 100%;
}

.square-box {
    aspect-ratio: 1;
    width: 200px;
}
```

### Advanced Flexbox Patterns

#### Equal Height Cards with Footer
```css
.card-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 1.5rem;
}

.card {
    flex: 1 1 300px;
    display: flex;
    flex-direction: column;
    border: 1px solid #ddd;
    border-radius: 8px;
    overflow: hidden;
}

.card-body {
    flex: 1; /* Grows to fill space */
    padding: 1.5rem;
}

.card-footer {
    padding: 1rem 1.5rem;
    background: #f5f5f5;
    border-top: 1px solid #ddd;
}
```

#### Sticky Footer with Flexbox
```css
body {
    display: flex;
    flex-direction: column;
    min-height: 100vh;
    margin: 0;
}

main {
    flex: 1; /* Grows to push footer down */
}

footer {
    flex-shrink: 0;
}
```

### Advanced Grid Patterns

#### Responsive Grid Without Media Queries
```css
.auto-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1.5rem;
}

/* auto-fit: Collapses empty tracks */
/* auto-fill: Maintains empty tracks */
```

#### Named Grid Lines
```css
.layout {
    display: grid;
    grid-template-columns: [full-start] 1fr [content-start] minmax(0, 800px) [content-end] 1fr [full-end];
}

.wide {
    grid-column: full-start / full-end;
}

.content {
    grid-column: content-start / content-end;
}
```

#### Nested Grids
```css
.page-grid {
    display: grid;
    grid-template-columns: 250px 1fr;
    gap: 2rem;
}

.main-content {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1.5rem;
}
```

### Animations and Transitions

```css
/* Smooth transitions */
.button {
    background-color: #3498db;
    transition: background-color 0.3s ease, transform 0.2s ease;
}

.button:hover {
    background-color: #2980b9;
    transform: translateY(-2px);
}

/* Keyframe animations */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.animate-in {
    animation: fadeInUp 0.6s ease-out;
}

/* Reduced motion preference */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}
```

---

## Best Practices Summary

### HTML Best Practices

1. ✅ Use semantic HTML5 elements
2. ✅ Maintain proper heading hierarchy (h1 → h2 → h3)
3. ✅ Include alt text for all images
4. ✅ Use meaningful link text (avoid "click here")
5. ✅ Validate HTML markup
6. ✅ Keep markup clean and well-indented
7. ✅ Use lowercase for elements and attributes
8. ✅ Always close tags properly
9. ✅ Include lang attribute on `<html>`
10. ✅ Use appropriate input types (email, tel, url, etc.)

### CSS Best Practices

1. ✅ Organize styles logically (base → layout → components → utilities)
2. ✅ Use consistent naming conventions (BEM, SMACSS)
3. ✅ Keep specificity low
4. ✅ Avoid `!important` unless absolutely necessary
5. ✅ Use CSS variables for repeated values
6. ✅ Mobile-first responsive design
7. ✅ Optimize for performance (minimize reflows)
8. ✅ Group related properties
9. ✅ Use shorthand properties appropriately
10. ✅ Comment complex or non-obvious code

### Responsive Design Best Practices

1. ✅ Design mobile-first
2. ✅ Use relative units (rem, em, %, vw)
3. ✅ Test on real devices, not just browser tools
4. ✅ Optimize images for different screen sizes
5. ✅ Use flexible grids (Flexbox/Grid)
6. ✅ Implement appropriate breakpoints
7. ✅ Ensure touch targets are minimum 48x48px
8. ✅ Test landscape and portrait orientations
9. ✅ Consider bandwidth constraints
10. ✅ Progressive enhancement over graceful degradation

### Accessibility Best Practices

1. ✅ Use semantic HTML elements
2. ✅ Provide keyboard navigation
3. ✅ Maintain visible focus indicators
4. ✅ Ensure sufficient color contrast
5. ✅ Include ARIA attributes when necessary
6. ✅ Test with screen readers
7. ✅ Avoid relying solely on color for information
8. ✅ Provide text alternatives for non-text content
9. ✅ Create logical heading structure
10. ✅ Make forms accessible with labels and descriptions

### Performance Best Practices

1. ✅ Minimize CSS file size
2. ✅ Eliminate unused CSS
3. ✅ Use CSS containment where appropriate
4. ✅ Optimize images and use modern formats
5. ✅ Lazy load off-screen content
6. ✅ Minimize use of expensive properties (box-shadow, filters)
7. ✅ Use `transform` and `opacity` for animations
8. ✅ Implement critical CSS
9. ✅ Reduce DOM complexity
10. ✅ Monitor Core Web Vitals

---

## Common Patterns and Solutions

### Centering Techniques

#### Horizontal Centering
```css
/* Block element */
.center-block {
    margin-left: auto;
    margin-right: auto;
    width: 80%; /* Must have width */
}

/* Inline/inline-block element */
.center-inline {
    text-align: center;
}
```

#### Vertical Centering
```css
/* Flexbox (modern) */
.center-flex {
    display: flex;
    align-items: center;
    min-height: 100vh;
}

/* Grid (modern) */
.center-grid {
    display: grid;
    place-items: center;
    min-height: 100vh;
}

/* Position (legacy) */
.center-absolute {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
}
```

#### Perfect Centering
```css
/* Flexbox */
.perfect-center-flex {
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 100vh;
}

/* Grid (shortest) */
.perfect-center-grid {
    display: grid;
    place-content: center;
    min-height: 100vh;
}
```

### Truncating Text

```css
/* Single line truncate */
.truncate {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* Multi-line truncate (webkit only) */
.truncate-multiline {
    display: -webkit-box;
    -webkit-line-clamp: 3; /* Number of lines */
    -webkit-box-orient: vertical;
    overflow: hidden;
}

/* Modern multi-line truncate */
.truncate-modern {
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
    text-overflow: ellipsis;
}
```

### Aspect Ratio Boxes

```css
/* Modern way */
.aspect-box {
    aspect-ratio: 16 / 9;
    width: 100%;
}

/* Legacy padding trick */
.aspect-box-legacy {
    position: relative;
    width: 100%;
    padding-bottom: 56.25%; /* 16:9 ratio */
}

.aspect-box-legacy > * {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
}
```

### Clearfix (for Floats - Legacy)

```css
.clearfix::after {
    content: "";
    display: table;
    clear: both;
}
```

### Custom Scrollbar Styling

```css
/* Webkit browsers */
.custom-scroll::-webkit-scrollbar {
    width: 10px;
}

.custom-scroll::-webkit-scrollbar-track {
    background: #f1f1f1;
}

.custom-scroll::-webkit-scrollbar-thumb {
    background: #888;
    border-radius: 5px;
}

.custom-scroll::-webkit-scrollbar-thumb:hover {
    background: #555;
}

/* Firefox */
.custom-scroll {
    scrollbar-width: thin;
    scrollbar-color: #888 #f1f1f1;
}
```

### Smooth Scrolling

```css
/* CSS way */
html {
    scroll-behavior: smooth;
}

/* Respect user preferences */
@media (prefers-reduced-motion: reduce) {
    html {
        scroll-behavior: auto;
    }
}
```

### Glass Morphism Effect

```css
.glass {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 10px;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
}
```

### Gradient Text

```css
.gradient-text {
    background: linear-gradient(45deg, #f06, #3cf);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
```

### CSS Shapes

```css
/* Triangle */
.triangle {
    width: 0;
    height: 0;
    border-left: 50px solid transparent;
    border-right: 50px solid transparent;
    border-bottom: 100px solid #3498db;
}

/* Circle */
.circle {
    width: 100px;
    height: 100px;
    border-radius: 50%;
    background: #3498db;
}

/* Custom shape with clip-path */
.custom-shape {
    clip-path: polygon(50% 0%, 100% 50%, 50% 100%, 0% 50%);
    background: #3498db;
}
```

---

## Debugging and Tools

### CSS Debugging Techniques

```css
/* Visual debugging borders */
* {
    outline: 1px solid red;
}

/* Better debugging with different colors */
* { outline: 1px solid rgba(255, 0, 0, 0.2); }
*:hover { outline: 1px solid rgba(255, 0, 0, 0.6); }

/* Grid debugging */
.grid {
    background-image: 
        repeating-linear-gradient(0deg, rgba(0,0,0,0.1) 0px, transparent 1px, transparent 20px),
        repeating-linear-gradient(90deg, rgba(0,0,0,0.1) 0px, transparent 1px, transparent 20px);
}
```

### Browser DevTools Tips

1. **Inspect Element**: Right-click → Inspect
2. **Force State**: Hover, focus, active, visited states
3. **Computed Styles**: See final calculated values
4. **Layout Panel**: Visualize Flexbox and Grid
5. **Changes Tab**: Track CSS modifications
6. **Coverage Tool**: Find unused CSS
7. **Performance Panel**: Identify bottlenecks
8. **Lighthouse**: Comprehensive audits

### Validation Tools

- **W3C HTML Validator**: validator.w3.org
- **W3C CSS Validator**: jigsaw.w3.org/css-validator
- **WAVE**: wave.webaim.org (accessibility)
- **axe DevTools**: Browser extension for accessibility
- **Lighthouse**: Built into Chrome DevTools
- **Can I Use**: caniuse.com (browser compatibility)

---

## CSS Methodologies

### BEM (Block Element Modifier)

```css
/* Block */
.card { }

/* Element */
.card__title { }
.card__content { }
.card__button { }

/* Modifier */
.card--featured { }
.card--large { }
.card__button--primary { }
```

```html
<div class="card card--featured">
    <h2 class="card__title">Title</h2>
    <p class="card__content">Content</p>
    <button class="card__button card__button--primary">Action</button>
</div>
```

### SMACSS (Scalable and Modular Architecture for CSS)

```css
/* 1. Base */
body, h1, p { margin: 0; }

/* 2. Layout */
.l-header { }
.l-sidebar { }
.l-main { }

/* 3. Module */
.card { }
.button { }

/* 4. State */
.is-active { }
.is-hidden { }

/* 5. Theme */
.theme-dark { }
```

### ITCSS (Inverted Triangle CSS)

```css
/* 1. Settings - Variables */
:root { --primary-color: #3498db; }

/* 2. Tools - Mixins */
/* (requires preprocessor) */

/* 3. Generic - Resets */
* { box-sizing: border-box; }

/* 4. Elements - Base styles */
body { font-family: sans-serif; }

/* 5. Objects - Layout patterns */
.o-container { max-width: 1200px; }

/* 6. Components - UI components */
.c-button { }

/* 7. Utilities - Helper classes */
.u-text-center { text-align: center; }
```

### Utility-First (Tailwind-style)

```html
<div class="flex items-center justify-center h-screen bg-gray-100">
    <div class="p-6 max-w-sm bg-white rounded-lg shadow-lg">
        <h2 class="text-2xl font-bold mb-4">Card Title</h2>
        <p class="text-gray-700 mb-4">Card content goes here.</p>
        <button class="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600">
            Action
        </button>
    </div>
</div>
```

---

## CSS Preprocessors Overview

### Sass/SCSS Features

```scss
// Variables
$primary-color: #3498db;
$spacing: 1rem;

// Nesting
.nav {
    background: $primary-color;
    
    &__item {
        padding: $spacing;
        
        &:hover {
            background: darken($primary-color, 10%);
        }
    }
}

// Mixins
@mixin flex-center {
    display: flex;
    justify-content: center;
    align-items: center;
}

.container {
    @include flex-center;
}

// Functions
@function rem($px) {
    @return #{$px / 16}rem;
}

.text {
    font-size: rem(24); // 1.5rem
}

// Extend/Inheritance
%button-base {
    padding: 0.5rem 1rem;
    border: none;
    cursor: pointer;
}

.button-primary {
    @extend %button-base;
    background: $primary-color;
}
```

### PostCSS Features

```css
/* Autoprefixer */
.box {
    display: flex; /* Auto-adds vendor prefixes */
}

/* CSS Nesting (postcss-nesting) */
.card {
    padding: 1rem;
    
    & .title {
        font-size: 1.5rem;
    }
}

/* Custom properties */
:root {
    --color-primary: #3498db;
}
```

---

## Modern CSS Features (Cutting Edge)

### CSS Cascade Layers

```css
@layer base {
    * { box-sizing: border-box; }
}

@layer components {
    .button { padding: 0.5rem 1rem; }
}

@layer utilities {
    .text-center { text-align: center; }
}

/* Layer order (lowest to highest) */
@layer base, components, utilities;
```

### CSS `@scope`

```css
@scope (.card) to (.card__content) {
    /* Styles only apply within .card but not within .card__content */
    p {
        margin: 1rem 0;
    }
}
```

### CSS `:has()` Selector (Parent Selector)

```css
/* Card with image */
.card:has(img) {
    display: grid;
    grid-template-columns: 200px 1fr;
}

/* Form with errors */
.form:has(.error) {
    border: 2px solid red;
}

/* List items with checkboxes */
li:has(input[type="checkbox"]:checked) {
    text-decoration: line-through;
}
```

### CSS `:is()` and `:where()`

```css
/* :is() - maintains specificity */
:is(h1, h2, h3) {
    line-height: 1.2;
}

/* Equivalent to */
h1, h2, h3 {
    line-height: 1.2;
}

/* :where() - zero specificity */
:where(h1, h2, h3) {
    margin: 0;
}
```

### CSS Color Functions

```css
:root {
    --hue: 210;
    --saturation: 80%;
    --lightness: 50%;
}

.button {
    /* HSL */
    background: hsl(var(--hue), var(--saturation), var(--lightness));
    
    /* Color-mix */
    background: color-mix(in srgb, blue 70%, white);
    
    /* Relative colors (coming) */
    background: hsl(from var(--primary-color) h s calc(l * 1.2));
}
```

### CSS Scroll-Driven Animations

```css
@keyframes fade-in {
    from { opacity: 0; }
    to { opacity: 1; }
}

.animate-on-scroll {
    animation: fade-in linear;
    animation-timeline: view();
    animation-range: entry 0% cover 30%;
}
```

---

## Browser Compatibility

### Feature Detection

```css
/* CSS Feature Queries */
@supports (display: grid) {
    .container {
        display: grid;
    }
}

@supports not (display: grid) {
    .container {
        display: flex;
    }
}

/* Multiple conditions */
@supports (display: grid) and (gap: 1rem) {
    .container {
        display: grid;
        gap: 1rem;
    }
}
```

### Fallbacks

```css
.box {
    background: rgb(52, 152, 219); /* Fallback */
    background: linear-gradient(45deg, #3498db, #2ecc71); /* Modern */
}

.container {
    display: flex; /* Fallback */
    display: grid; /* Modern */
}
```

### Vendor Prefixes

```css
.box {
    -webkit-appearance: none; /* Chrome, Safari */
    -moz-appearance: none; /* Firefox */
    appearance: none; /* Standard */
    
    -webkit-backdrop-filter: blur(10px);
    backdrop-filter: blur(10px);
}
```

### Progressive Enhancement Example

```css
/* Base (works everywhere) */
.card {
    border: 1px solid #ddd;
    padding: 1rem;
}

/* Modern enhancement */
@supports (display: grid) {
    .card {
        display: grid;
        grid-template-columns: 200px 1fr;
    }
}

/* Cutting-edge enhancement */
@supports (aspect-ratio: 1) {
    .card img {
        aspect-ratio: 16 / 9;
        object-fit: cover;
    }
}
```

---

## Real-World Examples

### Complete Responsive Navigation

```html
<nav class="navbar">
    <div class="navbar-brand">
        <a href="/" class="logo">Logo</a>
        <button class="menu-toggle" aria-label="Toggle menu" aria-expanded="false">
            <span></span>
            <span></span>
            <span></span>
        </button>
    </div>
    <ul class="navbar-menu">
        <li><a href="#home">Home</a></li>
        <li><a href="#about">About</a></li>
        <li><a href="#services">Services</a></li>
        <li><a href="#contact">Contact</a></li>
    </ul>
</nav>
```

```css
.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 2rem;
    background: #333;
    color: white;
}

.navbar-brand {
    display: flex;
    align-items: center;
    gap: 1rem;
}

.logo {
    font-size: 1.5rem;
    font-weight: bold;
    color: white;
    text-decoration: none;
}

.menu-toggle {
    display: none;
    flex-direction: column;
    gap: 4px;
    background: none;
    border: none;
    cursor: pointer;
    padding: 0.5rem;
}

.menu-toggle span {
    width: 25px;
    height: 3px;
    background: white;
    transition: transform 0.3s;
}

.navbar-menu {
    display: flex;
    gap: 2rem;
    list-style: none;
    margin: 0;
    padding: 0;
}

.navbar-menu a {
    color: white;
    text-decoration: none;
    transition: color 0.3s;
}

.navbar-menu a:hover {
    color: #3498db;
}

/* Mobile styles */
@media (max-width: 768px) {
    .menu-toggle {
        display: flex;
    }
    
    .navbar-menu {
        position: fixed;
        top: 70px;
        left: 0;
        right: 0;
        flex-direction: column;
        background: #333;
        padding: 1rem;
        gap: 0;
        transform: translateX(-100%);
        transition: transform 0.3s;
    }
    
    .navbar-menu.active {
        transform: translateX(0);
    }
    
    .navbar-menu li {
        padding: 1rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
}
```

### Complete Card Component

```html
<article class="card">
    <img src="image.jpg" alt="Card image" class="card-image">
    <div class="card-body">
        <div class="card-header">
            <h3 class="card-title">Card Title</h3>
            <span class="card-badge">New</span>
        </div>
        <p class="card-description">
            This is a description of the card content that provides context.
        </p>
        <div class="card-footer">
            <button class="btn btn-primary">Read More</button>
            <button class="btn btn-secondary">Share</button>
        </div>
    </div>
</article>
```

```css
.card {
    display: flex;
    flex-direction: column;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    transition: transform 0.3s, box-shadow 0.3s;
    background: white;
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 24px rgba(0, 0, 0, 0.15);
}

.card-image {
    width: 100%;
    aspect-ratio: 16 / 9;
    object-fit: cover;
}

.card-body {
    display: flex;
    flex-direction: column;
    padding: 1.5rem;
    gap: 1rem;
}

.card-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 1rem;
}

.card-title {
    margin: 0;
    font-size: 1.25rem;
    line-height: 1.4;
}

.card-badge {
    padding: 0.25rem 0.75rem;
    background: #3498db;
    color: white;
    border-radius: 20px;
    font-size: 0.875rem;
    white-space: nowrap;
}

.card-description {
    color: #666;
    line-height: 1.6;
    margin: 0;
}

.card-footer {
    display: flex;
    gap: 0.75rem;
    margin-top: auto;
}

.btn {
    padding: 0.5rem 1rem;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.875rem;
    transition: background-color 0.3s;
}

.btn-primary {
    background: #3498db;
    color: white;
}

.btn-primary:hover {
    background: #2980b9;
}

.btn-secondary {
    background: #ecf0f1;
    color: #333;
}

.btn-secondary:hover {
    background: #bdc3c7;
}
```

---

## Cheat Sheet: Quick Reference

### Common Flexbox Patterns

```css
/* Center everything */
.center { display: flex; justify-content: center; align-items: center; }

/* Space between */
.space-between { display: flex; justify-content: space-between; }

/* Column layout */
.column { display: flex; flex-direction: column; }

/* Equal spacing */
.gap { display: flex; gap: 1rem; }

/* Wrap items */
.wrap { display: flex; flex-wrap: wrap; }
```

### Common Grid Patterns

```css
/* Auto-fit cards */
.auto-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; }

/* 12-column grid */
.twelve-col { display: grid; grid-template-columns: repeat(12, 1fr); }

/* Holy grail */
.holy-grail { display: grid; grid-template: "header" auto "main" 1fr "footer" auto / 1fr; }

/* Sidebar layout */
.sidebar { display: grid; grid-template-columns: 250px 1fr; }
```

### Common Media Queries

```css
/* Mobile first breakpoints */
@media (min-width: 576px) { /* Small devices */ }
@media (min-width: 768px) { /* Medium devices */ }
@media (min-width: 992px) { /* Large devices */ }
@media (min-width: 1200px) { /* Extra large devices */ }

/* Common device queries */
@media (max-width: 767px) { /* Mobile */ }
@media (min-width: 768px) and (max-width: 1024px) { /* Tablet */ }
@media (min-width: 1025px) { /* Desktop */ }
```

---

## References

1. <a href="https://developer.mozilla.org/en-US/docs/Web/HTML" target="_blank">MDN Web Docs: HTML</a>
2. <a href="https://developer.mozilla.org/en-US/docs/Web/CSS" target="_blank">MDN Web Docs: CSS</a>
3. <a href="https://www.w3.org/TR/html52/" target="_blank">W3C HTML5 Specification</a>
4. <a href="https://www.w3.org/Style/CSS/" target="_blank">W3C CSS Specifications</a>
5. <a href="https://css-tricks.com/snippets/css/a-guide-to-flexbox/" target="_blank">CSS-Tricks: Complete Guide to Flexbox</a>
6. <a href="https://css-tricks.com/snippets/css/complete-guide-grid/" target="_blank">CSS-Tricks: Complete Guide to Grid</a>
7. <a href="https://www.w3.org/WAI/ARIA/apg/" target="_blank">W3C ARIA Authoring Practices Guide</a>
8. <a href="https://web.dev/learn/css/" target="_blank">Web.dev: Learn CSS</a>
9. <a href="https://web.dev/learn/html/" target="_blank">Web.dev: Learn HTML</a>
10. <a href="https://caniuse.com/" target="_blank">Can I Use: Browser Compatibility Tables</a>
11. <a href="https://www.smashingmagazine.com/guides/css-layout/" target="_blank">Smashing Magazine: CSS Layout Guides</a>
12. <a href="https://responsivedesign.is/" target="_blank">Responsive Design Best Practices</a>
13. <a href="https://www.a11yproject.com/" target="_blank">The A11Y Project</a>
14. <a href="https://web.dev/vitals/" target="_blank">Web.dev: Core Web Vitals</a>
15. <a href="https://html.spec.whatwg.org/" target="_blank">WHATWG HTML Living Standard</a>

---

## Conclusion

HTML5 and CSS3 form the foundation of modern web development. Mastering semantic HTML, responsive design principles, Flexbox, Grid, accessibility standards, and performance optimization techniques enables developers to build fast, accessible, and maintainable websites.

**Key Takeaways:**

1. **Semantic HTML** improves accessibility, SEO, and code maintainability
2. **CSS Flexbox** excels at one-dimensional layouts and component-level design
3. **CSS Grid** provides powerful two-dimensional layout capabilities
4. **Responsive Design** ensures optimal experiences across all devices
5. **Accessibility** makes the web usable for everyone
6. **Performance** directly impacts user experience and business metrics

Continue learning by building projects, experimenting with new features, and staying updated with web standards. The web platform continuously evolves, offering new capabilities while maintaining backward compatibility.

---

*Last updated: November 23, 2025*
