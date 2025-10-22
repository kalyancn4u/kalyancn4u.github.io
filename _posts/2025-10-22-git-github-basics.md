---
title: "Git and GitHub Basics — Comprehensive Revision Notes"
date: 2025-10-22 04:00:00 +0530
categories: [Revision Notes, Git & GitHub]
tags: [git, github, version-control, collaboration, beginners, dsml]
description: "A clear and structured revision guide covering Git and GitHub fundamentals, key concepts, workflows, and collaboration practices."
---

# 🧠 Git and GitHub Basics — Comprehensive Revision Notes

## 📘 Understanding Git and GitHub

---

### 🔹 What is Git?

**Git** is a **Version Control System (VCS)** that allows you to manage changes to your code and collaborate with others on projects efficiently.

#### ✳️ Key Features

- **Version Control:** Enables users to track the history of changes and revert to previous versions if necessary.  
- **Distributed System:** Unlike centralized systems, Git gives each collaborator a local copy of the entire project history — enhancing data reliability and flexibility.

---

### 🧩 Git Terminologies

| **Term** | **Definition** |
|-----------|----------------|
| **Repository (Repo)** | A storage area for project files, acting like a project folder. |
| **Local Repository** | A version of the repository stored on your computer for personal use and updates. |
| **Remote Repository** | An internet-based repository for sharing and collaboration. |
| **Commit** | Saving your changes as snapshots to track changes over time. |
| **Push** | Uploading your local changes to a remote repository to share with collaborators. |
| **Pull** | Downloading the latest changes from the remote repository to your local system. |
| **Clone** | Making a local copy of a repository from a remote source. |

---

## 🌐 Understanding GitHub

**GitHub** is a **web-based platform** that leverages Git for version control.  
It allows hosting repositories online for collaboration, contribution, and backup.

It serves as a **social platform for developers**, encouraging open-source development and community contributions, where users can fork, clone, and collaborate on projects with ease.

---

## ⚙️ Workflows in Git and GitHub

### 1. 🏗️ Creating a Repository
- You can create a repository **locally** using:
```bash
  git init
```

* Or **on GitHub**, by clicking **“New Repository”**, naming it, and optionally initializing it with a `README.md`.

---

### 2. 💾 Committing Changes

Each time you make meaningful code updates:

```bash
git add .
git commit -m "Add initial project files"
```

> 💡 Commits act as **snapshots** of your project, recording progress over time.

---

### 3. 🌿 Branching and Merging

Branches allow independent development of features without affecting the main code base.

```bash
git branch feature-login
git checkout feature-login
# make changes
git commit -m "Add login feature"
git checkout main
git merge feature-login
```

Merging integrates the changes back into the main project.

---

## 🤝 Collaboration Features

| **Feature**             | **Purpose**                                                                                |
| ----------------------- | ------------------------------------------------------------------------------------------ |
| **Pull Requests (PRs)** | Propose changes and request that they be merged into another branch (usually `main`).      |
| **PR Reviews**          | Team members can review, comment, and suggest modifications before approval.               |
| **Merging**             | Once approved, changes are merged into the main branch, integrating new features or fixes. |

---

## ⚠️ Managing Conflicts

When multiple collaborators modify the same section of a file, **merge conflicts** can occur.

To resolve:

1. Open the conflicting file(s).
2. Look for markers:

   ```text
   <<<<<<< HEAD
   your version
   =======
   collaborator’s version
   >>>>>>> branch-name
   ```
3. Manually edit the file to retain the correct lines.
4. Stage and commit the resolution:

   ```bash
   git add <filename>
   git commit -m "Resolve merge conflict"
   ```

---

## 🧪 Practical Example

During hands-on sessions, you can:

1. Create a new repository (`git init` or via GitHub).
2. Add and commit changes.
3. Push them to GitHub:

   ```bash
   git push origin main
   ```
4. Create feature branches and practice merging to reinforce Git fundamentals.

---

## 🧾 Summary

This guide provides a **clear and structured understanding** of Git and GitHub essentials.

* **Git** handles **version control** and local project tracking.
* **GitHub** enables **collaboration**, hosting, and open-source contribution.
* Learn to use essential commands:
  `git init → git add → git commit → git push → git pull → git merge`.

By following these workflows and examples, learners can manage codebases efficiently and collaborate with peers seamlessly — an essential skill set for **software developers, data scientists, and ML engineers** alike.

---

> 💡 **Pro Tip:**
> Use `git status` often — it helps verify what’s staged, unstaged, or untracked before committing changes.
