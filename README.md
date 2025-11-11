# make.py - a fast, minimalistic, Python-based build tool
**make.py** is a lightweight build system written in Python.

Build rules are defined in `rules.py` scripts: ordinary Python files that give you expressive power, composability,
and simplicity all in one place.

Unlike many build tools that invent their own DSLs, make.py lets you write your build logic directly in Python
(within a controlled subset, see below). The result is a tool that's **fast**, **reliable**, and **pleasant to use**,
but still **small enough to understand**.

## ✨ Highlights

### 🔄 Parallel builds, done right
- Fully parallelized builds by default, using all available CPU cores.
- The scheduler automatically prioritizes deep dependency chains to minimize idle time.
- Works seamlessly across multiple `rules.py` files - no manual coordination needed.

### 🧠 Smart, minimal output
- Interactive builds display a **real-time rolling progress indicator** that shows what's currently building and what's left.
  - When your code is warning-free, this is the *only* output you see - it even erases itself when the build completes.
  - Warnings and errors are captured and printed clearly apart from the progress indicator.
- Regex-based filtering lets you suppress noisy tool output (e.g., `Generating code` messages).
- When stdout is redirected to a file, make.py automatically switches to a clean, non-interactive log format.

### 🧩 Reliable incremental builds
- Automatically rebuilds targets when command lines change.
- Detects and removes stale targets for deleted or renamed rules.
- Deletes outputs of failed rules to avoid leaving corrupted artifacts.
- Canonicalizes paths (including Windows case-insensitivity) to prevent duplicate entries.
- Gracefully exits on `Ctrl-C` without leaving the build in an inconsistent state.

### 🛠 Supported features
- Make-like `.d` dependency files for C/C++ header tracking.
  - Built-in parser for MSVC `/showIncludes` output.
- Order-only dependencies for generated headers.
- Multi-target rules (one command generating multiple outputs).
- Automatic creation of output directories.
- Works on **Windows** (both Win32 and WSL), **Linux**, **macOS**, and other Unix systems (FreeBSD, etc.).
- Built-in host detection (`ctx.host`) exposing normalized OS/architecture information for portable rules.

### ⚙️ Lightweight by design
- Entire tool lives in a **single Python file (~550 lines)**.
- No dependencies beyond Python itself (3.6+).

## 🧱 Python with guardrails (evolving toward a Starlark-like subset)
`rules.py` files are real Python - just not *all* of Python.

Inspired by Starlark, make.py runs AST checks to block
constructs that could hang or make builds non-deterministic or unsafe to parallelize.
For example, `while` and `lambda` are forbidden to ensure all programs terminate, and
`async` is disallowed as unnecessary in this domain.

Today, these restrictions are light - you can use nearly all normal Python, with limited imports (`os`, `platform`
only) - but over time, the checks will tighten to define a well-specified, Starlark-like subset.

The goal is to keep Python's flexibility and readability while gaining the predictability and analyzability of
a structured build DSL. In practice, you write ordinary Python, and make.py keeps your build definitions clean,
reproducible, and future-proof.

## 🚧 Planned Features
- Optional SHA-256–based dependency tracking instead of timestamp comparisons.
- Improved diagnostics for invalid rule definitions and dependency cycles.
- More complete compatibility with other Starlark implementations and .bzl files.

## 🧾 Requirements
Python 3.6 or newer

## 🧭 Philosophy
make.py aims to occupy a unique middle ground:
- Smaller and simpler than heavyweight systems (Bazel, CMake, Ninja).
- More deterministic and parallel than ad-hoc scripts.
- Fully inspectable - you can read the whole thing in one sitting and know what your build system is doing.

It treats builds as hermetic, deterministic processes - a simple, elegant idea that much heavier systems
tend to obscure beneath layers of machinery that might not be necessary after all.

When your build tool fits in a single file, *debugging* and *trust* become much easier.

## 📜 License
MIT License

Copyright © 2012-2025 Matt Craighead
