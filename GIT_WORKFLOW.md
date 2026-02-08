# Git Workflow Guide - Working with Forks

A simple guide for working with forked repositories (e.g., minimind-v, DiT, or any open-source project).

---

## ⚡ TL;DR - Pull Updates from Original Repo

```bash
git checkout main && git pull upstream main && git checkout - && git merge main
```

That's it! This updates your local main from the original repo, then merges it into your current branch.

---

## 🎯 The Simple Strategy

1. Keep `main` branch clean (always synced with official repo)
2. Do all your work in feature branches
3. Push your branches to your fork
4. Pull updates from official repo anytime without conflicts

---

## 📋 One-Time Setup

### Step 1: Configure Git Identity

```bash
# Set your name and email (only needed once per machine)
git config --global user.name "YourGitHubUsername"
git config --global user.email "your.email@example.com"

# Verify
git config --global --list | grep user
```

### Step 2: Fork the Repository

1. Go to the official repo on GitHub (e.g., `https://github.com/official/minimind-v`)
2. Click the **"Fork"** button (top-right)
3. This creates a copy at `https://github.com/YourUsername/minimind-v`

### Step 3: Clone Your Fork

```bash
# Clone YOUR fork to local machine
git clone https://github.com/YourUsername/minimind-v.git
cd minimind-v
```

### Step 4: Add Upstream Remote

```bash
# Rename 'origin' to 'upstream' (optional but recommended)
git remote rename origin upstream

# Add your fork as 'origin'
git remote add origin https://github.com/YourUsername/minimind-v.git

# Verify remotes
git remote -v
```

**You should see:**
```
origin    https://github.com/YourUsername/minimind-v.git (your fork)
upstream  https://github.com/official/minimind-v.git (official repo)
```

**Alternative (simpler):**
```bash
# Keep origin as your fork, add upstream
git remote add upstream https://github.com/official/minimind-v.git
```

---

## 🚀 Daily Workflow

### Start Working on Something New

```bash
# Always create a new branch for your work
git checkout -b feature/my-feature

# Or for different types of work:
# git checkout -b fix/bug-fix
# git checkout -b experiment/new-idea
# git checkout -b educational/my-learning
```

### Make Changes and Commit

```bash
# Make your changes in files...

# Check what changed
git status

# Stage your changes
git add .                    # Add all changes
# or
git add file1.py file2.py    # Add specific files

# Commit with a descriptive message
git commit -m "Add feature X that does Y"

# Make more changes, commit again...
git add .
git commit -m "Fix issue with feature X"
```

### Push to Your Fork

```bash
# Push your branch to YOUR fork (origin)
git push origin feature/my-feature

# First time pushing this branch, you might need:
git push -u origin feature/my-feature
```

### Continue Working

```bash
# Keep making changes, committing, and pushing
git add .
git commit -m "Improve feature X"
git push origin feature/my-feature
```

---

## 🔄 Getting Updates from Original Repo

### ⚡ Quick Method (When Original Repo Has New Updates)

```bash
# Update your local main branch from original repo
git checkout main
git pull upstream main

# Merge updates into your working branch
git checkout feature/my-feature
git merge main

# Push updated branch to your fork
git push origin feature/my-feature
```

### 📝 Step-by-Step Explanation

**Step 1: Check for updates in original repo**
```bash
# Fetch latest info from original repo (doesn't change your files)
git fetch upstream

# See if there are new commits
git log HEAD..upstream/main --oneline
# If empty = you're up to date
# If shows commits = there are updates
```

**Step 2: Update your local main branch**
```bash
# Switch to main branch
git checkout main

# Pull latest from original repo (upstream)
git pull upstream main

# Output will show what was updated
```

**Step 3: Update your fork on GitHub (optional)**
```bash
# Push updated main to YOUR fork
git push origin main

# Now your fork's main matches the original repo
```

**Step 4: Merge updates into your working branch**
```bash
# Switch to your feature branch
git checkout feature/my-feature

# Merge the updated main into your branch
git merge main

# If no conflicts: Done!
# If conflicts: See below
```

### 🎨 Visual Flow

```
┌─────────────────────────────────────────────────┐
│  Original Repo (upstream/main)                  │
│  - Has new commits                              │
└────────────────┬────────────────────────────────┘
                 │ git pull upstream main
                 ↓
┌─────────────────────────────────────────────────┐
│  Your Local main branch                         │
│  - Now updated with latest changes              │
└────────────────┬────────────────────────────────┘
                 │ git merge main
                 ↓
┌─────────────────────────────────────────────────┐
│  Your Feature Branch (feature/my-work)          │
│  - Has your changes + latest updates            │
└────────────────┬────────────────────────────────┘
                 │ git push origin feature/my-work
                 ↓
┌─────────────────────────────────────────────────┐
│  Your Fork on GitHub (origin/feature/my-work)   │
│  - Everything synced and up to date!            │
└─────────────────────────────────────────────────┘
```

### If There Are Conflicts

```bash
# After git merge main, if conflicts occur:

# 1. Git will tell you which files have conflicts
# 2. Open those files and look for:
#    <<<<<<< HEAD
#    your changes
#    =======
#    upstream changes
#    >>>>>>> main

# 3. Edit files to resolve conflicts (remove markers, keep what you want)

# 4. Stage resolved files
git add <resolved-files>

# 5. Complete the merge
git commit -m "Merge upstream updates"

# 6. Push updated branch
git push origin feature/my-feature
```

---

## 📚 Common Scenarios

### Scenario 1: Save Uncommitted Work

```bash
# You're working but need to switch branches

# Option 1: Stash (temporary save)
git stash save "work in progress on feature X"
git checkout main
# ... do something ...
git checkout feature/my-feature
git stash pop

# Option 2: Commit it
git add .
git commit -m "WIP: work in progress"
```

### Scenario 2: Create Branch from Current Work

```bash
# You forgot to create a branch and worked on main

# Create branch with your changes
git checkout -b feature/forgot-branch

# Now your changes are in the new branch
git add .
git commit -m "Add my changes"

# Reset main to match upstream
git checkout main
git reset --hard upstream/main
```

### Scenario 3: Update Your Fork's Main Branch

```bash
# Your fork's main is behind official repo

git checkout main
git pull upstream main
git push origin main
```

### Scenario 4: Delete a Branch

```bash
# Delete local branch
git branch -d feature/old-feature

# Delete remote branch (on your fork)
git push origin --delete feature/old-feature
```

---

## 🔐 GitHub Authentication

### Option 1: GitHub CLI (Easiest)

```bash
# Install GitHub CLI
sudo apt install gh
# or: brew install gh (macOS)

# Login
gh auth login

# Follow prompts, now git push works!
```

### Option 2: SSH Keys (Most Permanent)

```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your.email@example.com"

# Copy public key
cat ~/.ssh/id_ed25519.pub

# Add to GitHub:
# 1. Go to https://github.com/settings/keys
# 2. Click "New SSH key"
# 3. Paste the public key

# Update remote to use SSH
git remote set-url origin git@github.com:YourUsername/repo.git

# Test connection
ssh -T git@github.com
```

### Option 3: Personal Access Token

```bash
# 1. Go to https://github.com/settings/tokens
# 2. Generate new token (classic)
# 3. Select "repo" scope
# 4. Copy the token

# When pushing, use:
# Username: YourGitHubUsername
# Password: paste-your-token-here
```

---

## 🎓 Best Practices

### ✅ Do's

- **Always work in feature branches**
- **Keep main branch clean** (never commit directly to main)
- **Commit often** with clear messages
- **Pull upstream updates regularly**
- **Use meaningful branch names**: `feature/add-training`, `fix/memory-leak`

### ❌ Don'ts

- **Don't work directly on main branch**
- **Don't force push** unless you know what you're doing
- **Don't commit secrets** (.env files, API keys, passwords)
- **Don't commit large files** (datasets, model weights)

---

## 🛠️ Useful Commands

### Check Status

```bash
git status                    # What's changed
git log                       # Commit history
git log --oneline             # Compact history
git diff                      # See unstaged changes
git diff --staged             # See staged changes
git branch                    # List branches
git branch -a                 # List all branches (including remote)
```

### Undo Things

```bash
git restore file.py           # Discard changes in file
git restore --staged file.py  # Unstage file
git reset HEAD~1              # Undo last commit (keep changes)
git reset --hard HEAD~1       # Undo last commit (discard changes)
git revert <commit-hash>      # Create new commit that undoes a commit
```

### Branch Operations

```bash
git checkout -b new-branch    # Create and switch to branch
git checkout existing-branch  # Switch to branch
git branch -d branch-name     # Delete local branch
git branch -m new-name        # Rename current branch
```

---

## 📊 Workflow Visualization

```
Official Repo (upstream)
    ↓ (fork)
Your Fork (origin) on GitHub
    ↓ (clone)
Local Machine
    ↓ (checkout -b)
Feature Branch (your work)
    ↓ (push)
Your Fork (origin) on GitHub
```

**Flow of changes:**
```
upstream/main → (pull) → local/main → (merge) → local/feature → (push) → origin/feature
```

---

## 🆘 Quick Reference

```bash
# SETUP (once)
git config --global user.name "Username"
git config --global user.email "email@example.com"
git clone https://github.com/YourUsername/repo.git
git remote add upstream https://github.com/official/repo.git

# DAILY WORK
git checkout -b feature/my-work
git add .
git commit -m "description"
git push origin feature/my-work

# GET UPDATES
git checkout main
git pull upstream main
git checkout feature/my-work
git merge main

# CHECK THINGS
git status
git log --oneline
git remote -v
```

---

## 💡 Example: Complete Workflow

Let's say you want to fork and modify `minimind-v`:

```bash
# 1. Fork on GitHub (click Fork button)

# 2. Clone your fork
git clone https://github.com/YourUsername/minimind-v.git
cd minimind-v

# 3. Add upstream
git remote add upstream https://github.com/official/minimind-v.git

# 4. Create feature branch
git checkout -b educational/my-experiments

# 5. Make changes
# ... edit files ...

# 6. Commit
git add .
git commit -m "Add my experimental features"

# 7. Push to your fork
git push origin educational/my-experiments

# 8. (Later) Get official updates
git checkout main
git pull upstream main
git checkout educational/my-experiments
git merge main

# 9. Keep working
# ... more changes ...
git add .
git commit -m "Improve experiments"
git push origin educational/my-experiments
```

---

**Summary:** Keep main clean, work in branches, sync regularly. That's it! 🎉
