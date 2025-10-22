---
title: "Git and GitHub Workflows"
date: 2025-10-22 05:00:00 +0530
categories: [Workflows, Git & GitHub]
tags: [git, github, version-control, workflows, TODO]
description: "A clear and structured revision guide covering Git and GitHub fundamentals, key concepts, workflows, and collaboration practices."
published: true
---

# 🧠 Git and GitHub Workflows

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
| **Git** | A **distributed version control system (VCS)** that records changes to files and enables multiple developers to collaborate on a project by maintaining complete version histories locally and remotely. |
| **GitHub** | A cloud-based platform for version control using Git, enabling code collaboration and repository hosting. |
| **Open Source** | Software whose source code is publicly accessible under a license permitting use, modification, and distribution. |
| **CI/CD** | A development practice combining **Continuous Integration** (automated code testing and merging) and **Continuous Deployment** (automated code delivery to production). |
| **VCS (Version Control System)** | A system that manages and records changes to files over time, allowing restoration, comparison, and collaborative editing. |
| | |
| **Repository (Repo)** | A storage area for project files, acting like a project folder. |
| **Local Repository** | A version of the repository stored on your computer for personal use and updates. |
| **Remote Repository** | An internet-based repository for sharing and collaboration. |
| **Commit** | Saving your changes as snapshots to track progress and version history. |
| **Push** | Uploading your local commits to a remote repository to share with collaborators. |
| **Pull** | Downloading the latest updates from a remote repository to your local system. |
| **Clone** | Creating a local copy of a repository from a remote source. |
| **Pull Request (PR)** | A formal request to merge code changes from one branch into another within a version-controlled repository. |
| **Snapshot** | A complete record of the project’s state at a specific point in time, created when a commit is made in Git. |

---

## 🌐 Understanding GitHub

**GitHub** is a **web-based platform** that leverages Git for version control.  
It allows hosting repositories online for collaboration, contribution, and backup.

It serves as a **social platform for developers**, encouraging open-source development and community contributions, where users can fork, clone, and collaborate on projects with ease.

---

## ⚙️ Workflows in Git and GitHub

The **Git SDLC (Software Development Lifecycle)** follows this intuitive sequence:  
**Initialize → Configure → Stage → Commit → Branch → Merge → Sync → Collaborate**

---

### 1. 🏗️ Creating and Initializing a Repository

Start a new project directory and set it up for version control:

```bash
mkdir my-project
cd my-project
git init
````

> Initializes a local repository in the current folder.

* Or **on GitHub**, by clicking **“New Repository”**, naming it, and optionally initializing it with a `README.md`.

If you’re using Git for the first time, configure your identity:

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

View all configurations:

```bash
git config --list
```

Or **clone** an existing remote repository instead of creating a new one:

```bash
git clone https://github.com/username/repository.git
```

*You can also create a new repository directly on GitHub and clone it locally.*

---

### 2. 💾 Tracking and Committing Changes

Check file status before staging:

```bash
git status
```

Stage specific files or all changes:

```bash
git add <filename>
git add .
```

Commit staged files to save the current snapshot:

```bash
git commit -m "Add initial project files"
```

Amend the previous commit if needed:

```bash
git commit --amend
```

View commit history:

```bash
git log --oneline
```

> 💡 Commits act as **snapshots** — milestones of your project over time.

---

### 3. 🌿 Branching and Merging

Create a new branch to work on a feature independently:

```bash
git branch feature-login
git checkout feature-login
```

Or create and switch in one step:

```bash
git checkout -b feature-login
```

Make your changes, then commit them:

```bash
git add .
git commit -m "Add login feature"
```

Switch back to the main branch and merge your work:

```bash
git checkout main
git merge feature-login
```

Delete a merged branch to keep the repository clean:

```bash
git branch -d feature-login
```

List all branches:

```bash
git branch
```

---

### 4. 🔄 Synchronizing with Remote Repository

Connect your local repo to a remote GitHub repository (only once per project):

```bash
git remote add origin https://github.com/username/repository.git
```

Verify remote connection:

```bash
git remote -v
```

Push your local commits to the remote repository:

```bash
git push -u origin main
```

Fetch updates from the remote repository without merging:

```bash
git fetch origin
```

Pull updates and automatically merge them into your local branch:

```bash
git pull origin main
```

---

### 5. 🤝 Collaboration Features

| **Feature**             | **Purpose**                                                                                |
| ----------------------- | ------------------------------------------------------------------------------------------ |
| **Pull Requests (PRs)** | Propose changes and request that they be merged into another branch (usually `main`).      |
| **PR Reviews**          | Team members can review, comment, and suggest modifications before approval.               |
| **Forking**             | Create your own copy of another user’s repository to experiment safely.                    |
| **Merging**             | Once approved, changes are merged into the main branch, integrating new features or fixes. |

---

### 6. ⚠️ Managing Conflicts

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

3. Edit to keep the correct version(s).

4. Stage and commit the fix:

   ```bash
   git add <filename>
   git commit -m "Resolve merge conflict"
   ```

If needed, abort a problematic merge:

```bash
git merge --abort
```

---

### 7. 🧪 Practical Example — Full Workflow Summary

1. Initialize your repo:

   ```bash
   git init
   ```
2. Configure your details:

   ```bash
   git config --global user.name "Alice"
   git config --global user.email "alice@example.com"
   ```
3. Add files and commit:

   ```bash
   git add .
   git commit -m "Initial commit"
   ```
4. Create and switch to a new branch:

   ```bash
   git checkout -b feature
   ```
5. Make edits, then merge back:

   ```bash
   git checkout main
   git merge feature
   ```
6. Add remote and push:

   ```bash
   git remote add origin https://github.com/alice/my-repo.git
   git push -u origin main
   ```
7. Pull updates when others contribute:

   ```bash
   git pull origin main
   ```

---

## 🧾 Summary

**Git SDLC Flow (Simplified):**
`git init → git config → git status → git add → git commit → git branch → git merge → git remote add → git push → git pull`

By following these commands in order, you can smoothly manage local and remote repositories, collaborate confidently, and maintain a clean project history.

---

> 💡 **Pro Tip:**
> Run `git status` and `git log --oneline` frequently to keep track of your current state and progress.

---

## 📚 References (Git → GitHub Workflow)

1. **[Learn Git Basics](https://git-scm.com/learn)** — Understand version control, repositories, commits, and branches.

2. **[Explore Git Commands](https://git-scm.com/doc)** — Review official command references.

3. **[Get Started with GitHub](https://docs.github.com/en/get-started/using-github)** — Learn GitHub workflows.

4. **[Try the Hello World Tutorial](https://guides.github.com/activities/hello-world)** — Practice repository creation, commits, and pull requests.

5. **[W3Schools Git Tutorial](https://www.w3schools.com/git/)** — Beginner-friendly interactive guide.

6. **[Git Cheat Sheet](https://education.github.com/git-cheat-sheet-education.pdf)** — Quick reference of essential Git commands.

7. **[Git Desktop documentation](https://docs.github.com/en/desktop)** — GitHub Desktop official documentation.

8. **[Git Desktop on Linux](https://gist.github.com/berkorbay/6feda478a00b0432d13f1fc0a50467f1)** — Installing GitHub Desktop on Linux (Unofficial, as of publishing date).

9. **[Learn Git Branching](https://learngitbranching.js.org/)** — Mastering Git commands, **Visually**.

---
