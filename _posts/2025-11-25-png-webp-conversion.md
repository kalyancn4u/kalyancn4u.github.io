---
layout: post
title: "✨ Convert PNG → WebP on Linux"
description: "Concise, clear, and validated revision notes"
author: technical_notes
date: 2025-11-25 07:00:00 +0530
categories: [Tasks, WebP Conversion]
tags: [PNG, WebP, Linux]
image: /assets/img/posts/png-to-webp.webp
toc: true
math: false
mermaid: false
---

# ✨ Convert PNG → WebP on Linux

WebP is a modern image format that gives **much smaller file sizes** while keeping crisp quality. Converting PNGs to WebP on Linux is super easy — just a couple of commands and you're done!

Let’s jump in. 🏃‍♂️💨

---

## 🧰 1. Install the `webp` Tools

The magic tool you need is **`cwebp`**, part of the official **WebP utilities**.

Install it using your distro’s package manager:

### **Debian / Ubuntu / Linux Mint**

```bash
sudo apt install webp
```

### **Fedora / RHEL / CentOS**

```bash
sudo dnf install libwebp-tools
```

### **Arch / Manjaro**

```bash
sudo pacman -S libwebp
```

You're all set! 🚀

---

## 🖼️ 2. Convert a Single PNG → WebP

Navigate to the folder where your PNG lives:

```bash
cd /path/to/images
```

Then convert it:

```bash
cwebp input.png -o output.webp
```

🎯 **Control the quality** using the `-q` flag (0–100):

```bash
cwebp -q 85 input.png -o output.webp
```

A quality value of **80–85** gives a great balance between **crispness** and **small size**.

---

## 📦 3. Batch Convert ALL PNGs in a Folder

Got a whole bunch of PNGs? Convert them all in one go:

```bash
for f in *.png; do
    cwebp -q 80 "$f" -o "${f%.png}.webp"
done
```

Here’s what this cute little loop is doing:

* `for f in *.png` → picks every `.png` file
* `cwebp -q 80 "$f"` → converts it at quality 80
* `"${f%.png}.webp"` → outputs filename with `.webp` extension

Batch conversion… complete! 🎉

---

## ✨ Alternative: Convert PNG → WebP Using ImageMagick

If you already use **ImageMagick**, good news — it works too!

### Install ImageMagick

```bash
sudo apt install imagemagick      # Debian/Ubuntu
sudo dnf install imagemagick      # Fedora
sudo pacman -S imagemagick        # Arch
```

### Convert an Image

```bash
magick input.png output.webp
```

ImageMagick leaves your original file intact. ❤️

---

## 💡 Final Tips

* `cwebp` usually produces **smaller file sizes** than ImageMagick.
* For photos: try **quality 75–85**
* For illustrations: add `-lossless`
* WebP is **browser-friendly** and great for websites.

---

Happy converting! 🌈
