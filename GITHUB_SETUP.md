# GitHub Setup Guide

This guide will help you publish the nano_wasm crate to GitHub.

## Prerequisites

- GitHub account
- Git installed
- GitHub CLI (optional, but helpful)

## Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `nano_wasm`
3. Description: `A zero-copy, zero-allocation WebAssembly interpreter for embedded and critical systems`
4. Visibility: Choose Public or Private
5. **Do NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

## Step 2: Update Repository URLs

Update the repository URL in `Cargo.toml`:

```toml
repository = "https://github.com/YOUR_USERNAME/nano_wasm"
```

Also update URLs in:
- `CHANGELOG.md` - Replace `yourusername` with your GitHub username
- `README.md` - Update any repository references if needed

## Step 3: Initialize Git Repository (if standalone)

If you want nano_wasm as a standalone repository:

```bash
cd crates/nano_wasm
git init
git add .
git commit -m "Initial commit: NanoWasm interpreter v0.1.0"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/nano_wasm.git
git push -u origin main
```

## Step 4: Or Add to Existing Repository

If nano_wasm is part of a larger repository (like unikernel-001):

1. The files are already tracked in the parent repository
2. You can create a separate repository and copy the files:

```bash
# Create new directory for standalone repo
cd ..
mkdir nano_wasm-standalone
cd nano_wasm-standalone

# Copy nano_wasm files
cp -r ../unikernel-001/crates/nano_wasm/* .

# Initialize git
git init
git add .
git commit -m "Initial commit: NanoWasm interpreter v0.1.0"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/nano_wasm.git
git push -u origin main
```

## Step 5: Create Initial Release

1. Go to your repository on GitHub
2. Click "Releases" → "Create a new release"
3. Tag: `v0.1.0`
4. Release title: `v0.1.0 - Initial Release`
5. Description: Copy from `CHANGELOG.md`
6. Click "Publish release"

## Step 6: Enable GitHub Actions

The CI workflow will automatically run on:
- Push to main/master
- Pull requests

Check the "Actions" tab to verify it's working.

## Step 7: Add Repository Topics

On GitHub, go to your repository → Settings → Topics, add:
- `wasm`
- `webassembly`
- `interpreter`
- `rust`
- `no-std`
- `embedded`
- `embedded-systems`
- `unikernel`

## Step 8: Optional - Publish to crates.io

When ready to publish to crates.io:

1. Create account at https://crates.io
2. Get API token
3. Run: `cargo publish --dry-run` to test
4. Run: `cargo publish` to publish

**Note**: Make sure to update version in `Cargo.toml` for each release.

## Repository Structure

Your repository should have:

```
nano_wasm/
├── .github/
│   ├── workflows/
│   │   └── ci.yml
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   ├── FUNDING.yml
│   └── PULL_REQUEST_TEMPLATE.md
├── src/
│   ├── lib.rs
│   └── interpreter.rs
├── .gitignore
├── Cargo.toml
├── CHANGELOG.md
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
├── README.md
└── GITHUB_SETUP.md (this file)
```

## Next Steps

- [ ] Create GitHub repository
- [ ] Update repository URLs
- [ ] Push code to GitHub
- [ ] Create initial release (v0.1.0)
- [ ] Verify CI workflow runs
- [ ] Add repository topics
- [ ] Share on social media / Rust communities
- [ ] Consider publishing to crates.io

## Community

Once published, consider:
- Sharing in Rust Discord/Slack channels
- Posting on Reddit (r/rust)
- Sharing on Twitter/X
- Adding to awesome-rust lists
- Writing blog posts about the project

Good luck! 🚀

