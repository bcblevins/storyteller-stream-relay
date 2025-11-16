#!/usr/bin/env python3
"""
directory_report.py — Generate a Markdown report embedding every file (recursively) from the
directory containing this script. Each file is wrapped in a fenced code block with a best-effort
language annotation inferred from its extension or well-known filename.

Default behavior:
- Walks ALL subdirectories under the folder where this script resides.
- Fails fast (and writes no report) if any encountered file cannot be mapped to a code-fence
  language (per the requested "cancel the whole action" behavior).
- Skips itself and the output report from inclusion to avoid recursion.

Flexibility features:
- Rich, extensible extension → language map (and special-case filename map).
- CLI flags to override strict behavior or adjust ignores.
- Robust code-fence generation even when file contents contain backticks.
- Sensible default ignores can be toggled off.
- Zero non‑stdlib dependencies.

USAGE
-----
1) Drop 'directory_report.py' into the target directory.
2) Run:    python directory_report.py
3) Result: '<dirname>.report.md' is written next to this script.

CLI OPTIONS (all optional)
--------------------------
  --skip-unknown        : Do NOT cancel when an unmapped extension/name is seen; instead skip it.
  --no-default-ignores  : Do not apply the default ignore list (e.g., .git, node_modules, etc.).
  --ignore PATTERN ...  : Additional glob patterns to ignore (relative to the base dir).
  --include-hidden      : Include dotfiles/directories (hidden) in the report.
  --follow-symlinks     : Follow directory symlinks when walking.
  --base PATH           : Override the base directory (defaults to this script's parent dir).
  --output PATH         : Override the output file (defaults to '<dirname>.report.md').

NOTE
----
This script is intentionally conservative by default (strict mode). If your tree includes images,
binaries, or unusual file types you don't want to fail on, consider --skip-unknown OR add mappings
below.
"""
import argparse
import fnmatch
import os
from pathlib import Path
import sys
from datetime import datetime

# ------------------------------
# Extension/Name → Language Maps
# ------------------------------
# Feel free to extend these as needed. Keys are lowercase.
EXT_LANGUAGE_MAP = {
    # General / text
    ".txt": "text",
    ".log": "text",
    ".md": "markdown",
    ".mdx": "mdx",
    ".rst": "rst",
    ".adoc": "asciidoc",
    ".csv": "csv",
    ".tsv": "tsv",

    # Data / config
    ".json": "json",
    ".jsonc": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".conf": "ini",
    ".properties": "properties",
    ".env": "dotenv",
    ".lock": "text",

    # Web
    ".html": "html",
    ".htm": "html",
    ".xhtml": "html",
    ".xml": "xml",
    ".svg": "xml",
    ".css": "css",
    ".scss": "scss",
    ".sass": "sass",
    ".less": "less",

    # JavaScript / TypeScript
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "jsx",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".vue": "vue",
    ".svelte": "svelte",
    ".astro": "astro",

    # Shell / scripts
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "zsh",
    ".fish": "fish",
    ".ps1": "powershell",
    ".psm1": "powershell",
    ".bat": "batch",
    ".cmd": "batch",

    # Python
    ".py": "python",
    ".pyw": "python",
    ".ipynb": "json",  # Jupyter notebooks are JSON

    # C-family / systems
    ".c": "c",
    ".h": "c",          # ambiguous; default to C
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".hxx": "cpp",
    ".m": "objectivec",
    ".mm": "objectivecpp",
    ".rs": "rust",
    ".go": "go",
    ".swift": "swift",

    # Other languages
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".groovy": "groovy",
    ".gradle": "groovy",  # Gradle Groovy DSL
    ".rb": "ruby",
    ".php": "php",
    ".pl": "perl",
    ".pm": "perl",
    ".r": "r",
    ".jl": "julia",
    ".lua": "lua",
    ".cs": "csharp",
    ".vb": "vbnet",
    ".hs": "haskell",
    ".erl": "erlang",
    ".ex": "elixir",
    ".exs": "elixir",
    ".clj": "clojure",
    ".cljs": "clojure",
    ".edn": "clojure",
    ".scala": "scala",
    ".nim": "nim",
    ".hx": "haxe",
    ".sol": "solidity",

    # Infra / DB / misc
    ".sql": "sql",
    ".psql": "sql",
    ".proto": "protobuf",
    ".graphql": "graphql",
    ".gql": "graphql",
    ".tf": "hcl",
    ".cue": "cue",
    ".plist": "xml",
    ".csproj": "xml",
    ".storyboard": "xml",
}

# Special-case names that lack extensions. Keys are lowercase full filenames.
NAME_LANGUAGE_MAP = {
    "dockerfile": "dockerfile",
    ".dockerignore": "gitignore",
    "makefile": "makefile",
    ".gitignore": "gitignore",
    ".gitattributes": "gitignore",
    ".gitmodules": "ini",
    "license": "text",
    "license.md": "markdown",
    "readme": "markdown",
    "readme.md": "markdown",
    ".editorconfig": "ini",
    "procfile": "procfile",
    "requirements.txt": "text",
    "yarn.lock": "text",
    "pnpm-lock.yaml": "yaml",
    "package.json": "json",
    "package-lock.json": "json",
    "tsconfig.json": "json",
    "compose.yml": "yaml",
    "compose.yaml": "yaml",
}

# Directories commonly excluded. You can disable via --no-default-ignores.
DEFAULT_IGNORES = [
    ".git", ".hg", ".svn",
    "node_modules", "bower_components",
    ".venv", "venv", "env", ".env", ".direnv",
    "__pycache__", ".mypy_cache", ".pytest_cache",
    "dist", "build", "out", "target", "coverage",
    ".idea", ".vscode", ".DS_Store",
]

# Extensions that very likely represent binary assets (not suitable for code fences).
# These are not auto-excluded by default (strict mode may wish to fail on them), but
# are provided for convenience in custom ignore rules.
LIKELY_BINARY_EXTS = {
    ".png",".jpg",".jpeg",".gif",".webp",".bmp",".ico",".tiff",".tif",
    ".mp3",".wav",".flac",".ogg",".m4a",".wma",
    ".mp4",".mkv",".webm",".mov",".avi",".wmv",
    ".eot",".otf",".ttf",".woff",".woff2",
    ".pdf",".zip",".tar",".gz",".tgz",".7z",".rar",".xz",
    ".apk",".bin",".dylib",".dll",".so",".exe",".class",".jar",".war",
    ".pyc",".pyo",".o",".obj",
}

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def best_code_language(path: Path):
    """Infer a code-fence language from filename or extension. Returns None if unknown."""
    name = path.name.lower()
    # Prefer filename-first for well-known extensionless files
    if name in NAME_LANGUAGE_MAP:
        return NAME_LANGUAGE_MAP[name]
    # Otherwise by extension (last suffix)
    ext = path.suffix.lower()
    if ext in EXT_LANGUAGE_MAP:
        return EXT_LANGUAGE_MAP[ext]
    # If there are multiple suffixes (.d.ts), the last one is already considered.
    return None

def longest_backtick_run(s: str) -> int:
    longest = 0
    cur = 0
    for ch in s:
        if ch == "`":
            cur += 1
            if cur > longest:
                longest = cur
        else:
            cur = 0
    return longest

def make_fence(content: str) -> str:
    """Return a fence string using backticks long enough to avoid collisions with content."""
    longest = longest_backtick_run(content)
    # At least 3 backticks; if the content contains a run >= 3, make fence one longer
    needed = max(3, longest + 1)
    return "`" * needed

def should_ignore(rel_path: Path, patterns: list[str]) -> bool:
    """Check if rel_path matches any of the glob patterns (posix-style)."""
    s = rel_path.as_posix()
    for pat in patterns:
        if fnmatch.fnmatch(s, pat):
            return True
        # Also allow simple directory name matches like 'node_modules'
        parts = s.split("/")
        if pat in parts:
            return True
    return False

def iter_files(base: Path, include_hidden: bool, follow_symlinks: bool, ignore_patterns: list[str], output_path: Path) -> list[Path]:
    files = []
    for root, dirs, filenames in os.walk(base, followlinks=follow_symlinks):
        root_path = Path(root)
        # Filter dirs in-place
        kept_dirs = []
        for d in dirs:
            if not include_hidden and d.startswith("."):
                continue
            d_rel = (root_path / d).relative_to(base)
            if should_ignore(d_rel, ignore_patterns):
                continue
            kept_dirs.append(d)
        dirs[:] = kept_dirs

        # Files
        for fname in filenames:
            if not include_hidden and fname.startswith("."):
                continue
            p = root_path / fname
            # Skip output file and this script
            try:
                if p.resolve() == output_path.resolve():
                    continue
            except Exception:
                pass
            if p.name == Path(__file__).name:
                continue
            rel = p.relative_to(base)
            if should_ignore(rel, ignore_patterns):
                continue
            files.append(p)
    files.sort(key=lambda p: str(p.relative_to(base)).lower())
    return files

def read_text_best_effort(path: Path) -> str:
    """
    Read file content as text, prioritizing UTF-8 but falling back to common encodings.
    We use 'errors=replace' to ensure robustness without extra dependencies.
    """
    # Try utf-8 first
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        pass
    # Fallback encodings
    for enc in ("utf-16", "utf-16le", "utf-16be", "latin-1"):
        try:
            return path.read_text(encoding=enc, errors="replace")
        except UnicodeDecodeError:
            continue
        except Exception:
            continue
    # Last resort: binary read then decode with 'utf-8' replace
    try:
        data = path.read_bytes()
        return data.decode("utf-8", errors="replace")
    except Exception:
        # Give up; caller may treat as failure.
        raise

def main():
    parser = argparse.ArgumentParser(description="Generate a Markdown report of an entire directory tree as code blocks.")
    parser.add_argument("--skip-unknown", action="store_true", help="Skip files with unknown language mapping instead of failing.")
    parser.add_argument("--no-default-ignores", action="store_true", help="Do not apply the default ignore list.")
    parser.add_argument("--ignore", nargs="*", default=[], help="Additional ignore glob patterns (relative to base).")
    parser.add_argument("--include-hidden", action="store_true", help="Include dotfiles and dot-directories.")
    parser.add_argument("--follow-symlinks", action="store_true", help="Follow directory symlinks during traversal.")
    parser.add_argument("--base", type=str, default=None, help="Override the base directory (defaults to this script's folder).")
    parser.add_argument("--output", type=str, default=None, help="Override output path (defaults to '<dirname>.report.md' in base).")

    args = parser.parse_args()

    # Establish base directory: the folder WHERE THIS SCRIPT LIVES by default.
    script_path = Path(__file__).resolve()
    base_dir = Path(args.base).resolve() if args.base else script_path.parent

    if not base_dir.is_dir():
        eprint(f"Error: Base directory does not exist or is not a directory: {base_dir}")
        sys.exit(2)

    # Compute default output name: <dirname>.report.md
    default_output = base_dir / f"{base_dir.name}.report.md"
    output_path = Path(args.output).resolve() if args.output else default_output

    # Build ignore patterns
    ignore_patterns = []
    if not args.no_default_ignores:
        # Treat default ignores as simple names and also as '**/name/**' globs
        for name in DEFAULT_IGNORES:
            ignore_patterns.append(name)
            ignore_patterns.append(f"**/{name}/**")
            ignore_patterns.append(f"**/{name}")
    # User-specified patterns
    for pat in (args.ignore or []):
        ignore_patterns.append(pat)

    files = iter_files(
        base=base_dir,
        include_hidden=args.include_hidden,
        follow_symlinks=args.follow_symlinks,
        ignore_patterns=ignore_patterns,
        output_path=output_path,
    )

    # First pass: validate languages (strict by default).
    unknown = []
    lang_by_file = {}
    for p in files:
        lang = best_code_language(p)
        if lang is None:
            unknown.append(p)
        else:
            lang_by_file[p] = lang

    if unknown and not args.skip_unknown:
        eprint("ERROR: Encountered files with unknown/unsupported language mapping.\n"
               "Add appropriate mappings in EXT_LANGUAGE_MAP / NAME_LANGUAGE_MAP, or re-run with --skip-unknown.\n")
        for p in unknown[:25]:  # preview first 25
            eprint(f" - {p.relative_to(base_dir)} (ext='{p.suffix}')")
        if len(unknown) > 25:
            eprint(f" ... and {len(unknown) - 25} more")
        sys.exit(2)

    # Generate the report
    header_lines = [
        f"# Directory Report: {base_dir.name}",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- Base path: `{base_dir}`",
        f"- Strict mode: {'OFF (skip unknown)' if args.skip_unknown else 'ON (fail on unknown)'}",
        "",
        "---",
        "",
    ]

    # Create parent dir for output if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="\n") as out:
        out.write("\n".join(header_lines))

        for p in files:
            lang = lang_by_file.get(p)
            if lang is None and args.skip_unknown:
                # Skip unknown if permitted
                continue
            elif lang is None:
                # Should not happen due to earlier check
                continue

            rel = p.relative_to(base_dir).as_posix()
            try:
                content = read_text_best_effort(p)
            except Exception as ex:
                eprint(f"Warning: failed to read {rel} as text: {ex}")
                if not args.skip_unknown:
                    eprint("Aborting due to strict mode.")
                    sys.exit(2)
                else:
                    continue

            fence = make_fence(content)
            out.write(f"## {rel}\n\n")
            out.write(f"{fence}{lang}\n")
            out.write(content)
            # Ensure trailing newline before closing fence
            if not content.endswith("\n"):
                out.write("\n")
            out.write(f"{fence}\n\n")

    print(f"Wrote report: {output_path}")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        eprint("Interrupted.")
        sys.exit(130)
