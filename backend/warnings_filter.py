"""
ARKEN — Startup Warnings Filter
=================================
Suppresses known harmless dependency version warnings that appear
at import time but have no functional impact on the application.

Import this module as early as possible in main.py (before any other imports)
to ensure filters are applied before the noisy libraries load.

Suppressed:
  1. RequestsDependencyWarning — urllib3/chardet version mismatch in `requests`
     package; harmless in Python 3.11, caused by pip's liberal version pinning.
  2. Any future similar third-party version-mismatch warnings.
"""

import warnings

# ── 1. requests / urllib3 / chardet version mismatch ─────────────────────────
# urllib3 2.x ships chardet 4.x as optional; requests 2.32 checks for older
# chardet/charset_normalizer combinations. The mismatch is cosmetic only.
warnings.filterwarnings(
    "ignore",
    category=Warning,
    message=r".*urllib3.*chardet.*charset_normalizer.*doesn't match.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*doesn't match a supported version.*",
)

# ── 2. Pydantic protected namespace warnings (belt-and-suspenders) ────────────
# Handled at the model level via model_config; this silences any stragglers.
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*conflict with protected namespace.*",
)

# ── 3. DeprecationWarnings from third-party packages we don't control ─────────
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"(pkg_resources|google\.protobuf|grpc|opentelemetry|langchain)",
)
