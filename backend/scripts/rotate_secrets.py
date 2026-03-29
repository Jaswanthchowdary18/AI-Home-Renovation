#!/usr/bin/env python3
"""
ARKEN — Secret Rotation Checker
=================================
Scans .env (and optionally .env.example) for known-compromised or
placeholder API keys and prints a warning if any are found.

Usage:
    python scripts/rotate_secrets.py              # checks .env
    python scripts/rotate_secrets.py --all-files  # checks all env files
    python scripts/rotate_secrets.py --pre-commit # git pre-commit mode (exits 1 on find)

Add as a pre-commit hook:
    echo 'python scripts/rotate_secrets.py --pre-commit' >> .git/hooks/pre-commit
    chmod +x .git/hooks/pre-commit
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

# ── ANSI colours ──────────────────────────────────────────────────────────────
_RED    = "\033[91m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"

# ── Known-bad / compromised keys ─────────────────────────────────────────────
# Add any key that has been exposed in public commits or CI logs.
KNOWN_BAD_KEYS: list[str] = [
    # ⚠️  EXPOSED in git history — must be rotated immediately
    "AIzaSyBNKExIcDRRsZ0WUgxBVoRBbvegeMs2IHM",
    # Add future compromised keys below:
]

# ── Placeholder / default values that should never reach production ───────────
INSECURE_PATTERNS: list[tuple[str, str]] = [
    (r"SECRET_KEY\s*=\s*arken-dev-secret",      "Default dev SECRET_KEY — generate with: openssl rand -hex 32"),
    (r"SECRET_KEY\s*=\s*your-super-secret",      "Placeholder SECRET_KEY — generate with: openssl rand -hex 32"),
    (r"SECRET_KEY\s*=\s*GENERATE_WITH",          "Unset SECRET_KEY placeholder"),
    (r"GOOGLE_API_KEY\s*=\s*YOUR_GOOGLE",        "Unset GOOGLE_API_KEY placeholder"),
    (r"AWS_SECRET_ACCESS_KEY\s*=\s*your-aws",    "Placeholder AWS secret key"),
    (r"PINECONE_API_KEY\s*=\s*your-pinecone",    "Placeholder Pinecone key"),
]

_BACKEND_DIR = Path(__file__).resolve().parent.parent


def _red(msg: str) -> str:
    return f"{_RED}{_BOLD}{msg}{_RESET}"


def _green(msg: str) -> str:
    return f"{_GREEN}{msg}{_RESET}"


def _yellow(msg: str) -> str:
    return f"{_YELLOW}{msg}{_RESET}"


def scan_file(filepath: Path) -> list[dict]:
    """
    Scan a single file for known-bad keys and insecure patterns.
    Returns list of findings: {line_no, type, value, message, severity}.
    """
    if not filepath.exists():
        return []

    findings: list[dict] = []

    try:
        content = filepath.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"  Could not read {filepath}: {e}")
        return []

    lines = content.splitlines()

    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue  # skip comment lines

        # Check known-bad literal keys
        for bad_key in KNOWN_BAD_KEYS:
            if bad_key in line:
                findings.append({
                    "line_no":  line_no,
                    "type":     "COMPROMISED_KEY",
                    "value":    bad_key[:12] + "...",
                    "message":  (
                        f"COMPROMISED KEY DETECTED: {bad_key[:12]}... "
                        "This key was exposed in public. Rotate immediately at "
                        "https://console.cloud.google.com/apis/credentials"
                    ),
                    "severity": "CRITICAL",
                })

        # Check insecure pattern matches
        for pattern, message in INSECURE_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                findings.append({
                    "line_no":  line_no,
                    "type":     "INSECURE_PLACEHOLDER",
                    "value":    stripped[:50],
                    "message":  message,
                    "severity": "WARNING",
                })

    return findings


def print_finding(filepath: Path, finding: dict) -> None:
    sev = finding["severity"]
    colour = _RED if sev == "CRITICAL" else _YELLOW
    icon   = "🚨" if sev == "CRITICAL" else "⚠️ "
    print(
        f"  {colour}{icon} [{sev}] {filepath.name}:{finding['line_no']}{_RESET}  "
        f"{finding['message']}"
    )


def print_rotation_guide() -> None:
    print(f"""
{_BOLD}═══ HOW TO ROTATE THE COMPROMISED GOOGLE API KEY ══════════════════════════{_RESET}

  1. Open Google Cloud Console:
     https://console.cloud.google.com/apis/credentials

  2. Find the key starting with AIzaSyBNKEx... and click Revoke / Delete.

  3. Create a new API key (restrict it to your server IPs and required APIs).

  4. Update your .env:
       GOOGLE_API_KEY=<your-new-key>

  5. Verify the old key no longer works:
       curl "https://generativelanguage.googleapis.com/v1beta/models?key=AIzaSyBNKEx..."
       # Should return 400 or 403

  6. Update your secret management system (AWS Secrets Manager, GCP Secret Manager,
     Doppler, etc.) with the new key.

  7. Run this script again to confirm clean:
       python scripts/rotate_secrets.py

{_BOLD}═══ HOW TO GENERATE A SECURE SECRET_KEY ═══════════════════════════════════{_RESET}

     openssl rand -hex 32

  Copy the output to your .env:
     SECRET_KEY=<output>

{_BOLD}═══════════════════════════════════════════════════════════════════════════{_RESET}
""")


def main() -> int:
    parser = argparse.ArgumentParser(description="ARKEN secret rotation checker")
    parser.add_argument(
        "--all-files", action="store_true",
        help="Scan all .env* files in the backend directory",
    )
    parser.add_argument(
        "--pre-commit", action="store_true",
        help="Exit with code 1 if any CRITICAL finding is found (for git hooks)",
    )
    parser.add_argument(
        "--env-file", default=None,
        help="Path to a specific .env file to scan",
    )
    args = parser.parse_args()

    # Determine files to scan
    if args.env_file:
        files = [Path(args.env_file)]
    elif args.all_files:
        files = list(_BACKEND_DIR.glob(".env*"))
    else:
        env_path = _BACKEND_DIR / ".env"
        if not env_path.exists():
            # Try current working directory
            env_path = Path.cwd() / ".env"
        files = [env_path]

    print(f"\n{_BOLD}ARKEN Secret Rotation Checker{_RESET}")
    print(f"Scanning: {', '.join(str(f) for f in files)}\n")

    total_critical = 0
    total_warnings = 0
    all_clean      = True

    for filepath in files:
        findings = scan_file(filepath)
        if not findings:
            print(f"  {_green('✓')} {filepath.name} — no known-bad keys found")
            continue

        all_clean = False
        print(f"  {_red('✗')} {filepath.name} — {len(findings)} issue(s) found:")
        for finding in findings:
            print_finding(filepath, finding)
            if finding["severity"] == "CRITICAL":
                total_critical += 1
            else:
                total_warnings += 1

    print()

    if total_critical > 0:
        print(_red(f"  🚨 {total_critical} CRITICAL issue(s) — rotate these keys immediately!"))
        print_rotation_guide()
    elif total_warnings > 0:
        print(_yellow(f"  ⚠️  {total_warnings} warning(s) — update placeholder values before deployment."))
    else:
        print(_green("  ✓ All checks passed — no compromised or placeholder keys detected."))

    print()

    if args.pre_commit and total_critical > 0:
        print(_red("  ✗ Pre-commit check FAILED — commit blocked. Rotate keys before committing."))
        return 1

    return 0 if all_clean else (2 if total_warnings > 0 and total_critical == 0 else 1)


if __name__ == "__main__":
    sys.exit(main())
