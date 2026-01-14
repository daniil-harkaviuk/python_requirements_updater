#!/usr/bin/env python3
"""
Find the *first* newer version of each pinned requirement (pkg==X.Y.Z)
that is compatible with Python 3.12 (or 3.12.3), using PyPI metadata.

Example:
  numpy==1.24.4  -> numpy==1.26.0   (first version supporting Python 3.12)

How it works (high level):
- Reads requirements.txt
- For lines like "name==version" (extras allowed: "name[extra]==version"):
  - Queries https://pypi.org/pypi/<name>/json
  - Scans versions newer than the pinned one, in ascending order
  - Picks the first version whose "requires_python" spec allows the target python
    (using file-level requires_python if present; else optionally treat as unknown)

Usage:
  python upgrade_first_py312.py requirements.txt --py 3.12.3 --out requirements.py312.txt
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import requests
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version, InvalidVersion


PYPI_JSON_URL = "https://pypi.org/pypi/{name}/json"
_WHEEL_RE = re.compile(r"^(?P<namever>.+)-(?P<py>[^-]+)-(?P<abi>[^-]+)-(?P<plat>[^-]+)\.whl$")

@dataclass
class LineResult:
    original: str
    updated: str
    changed: bool
    note: str = ""


def strip_inline_comment(line: str) -> str:
    # Keep it simple: split on the first " #" pattern or a leading "#".
    if line.lstrip().startswith("#"):
        return line
    # Attempt to preserve URLs that might contain '#': only treat as comment if preceded by whitespace.
    m = re.search(r"\s+#", line)
    if m:
        return line[: m.start()].rstrip()
    return line.rstrip("\n")


def is_control_line(s: str) -> bool:
    s2 = s.strip()
    return (
        not s2
        or s2.startswith("#")
        or s2.startswith("-r ")
        or s2.startswith("--requirement ")
        or s2.startswith("-c ")
        or s2.startswith("--constraint ")
        or s2.startswith("--find-links ")
        or s2.startswith("-f ")
        or s2.startswith("--index-url ")
        or s2.startswith("--extra-index-url ")
        or s2.startswith("--trusted-host ")
    )


def is_pinned_eq(req: Requirement) -> Optional[Version]:
    """
    Return pinned version if the requirement is exactly "==<version>" (one specifier),
    otherwise None.
    """
    specs = list(req.specifier)
    if len(specs) != 1:
        return None
    sp = specs[0]
    if sp.operator != "==":
        return None
    try:
        return Version(sp.version)
    except InvalidVersion:
        return None


def fetch_pypi_json(name: str, session: requests.Session) -> dict:
    url = PYPI_JSON_URL.format(name=name)
    r = session.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def _parse_cp_tag(tag: str) -> int | None:
    """
    cp312 -> 312, cp38 -> 38
    """
    if not tag.startswith("cp"):
        return None
    rest = tag[2:]
    if not rest.isdigit():
        return None
    return int(rest)

def _wheel_supports_target_py(filename: str, target_py: Version) -> bool:
    """
    Decide wheel compatibility for CPython target (e.g. 3.12.3).

    Accepts if:
      - wheel is universal pure python: py3-none-any (or py2.py3-none-any)
      - wheel has cp312 tag
      - wheel uses abi3 with cpXY where XY <= target (e.g. cp38-abi3 works on 3.12)
      - wheel has multiple python tags including a compatible one (e.g. cp312.cp311)
    """
    m = _WHEEL_RE.match(filename)
    if not m:
        return False

    py_tags = m.group("py").split(".")
    abi = m.group("abi")

    target_cp = int(f"{target_py.major}{target_py.minor:02d}")  # 3.12 -> 312

    # pure python wheels
    if "py3" in py_tags or "py2.py3" in py_tags:
        # This is still a wheel; if it's truly universal it will be none-any,
        # but we don't strictly require that here. You can tighten if you want.
        return True

    # explicit CPython wheel for target
    if f"cp{target_cp}" in py_tags:
        return True

    # abi3 wheels: cp38-abi3, cp39-abi3 ... work on newer CPython too
    if abi == "abi3":
        for t in py_tags:
            cp = _parse_cp_tag(t)
            if cp is not None and cp <= target_cp:
                return True

    return False

def release_supports_python(
    release_files: list,
    target_py: Version,
    allow_unknown: bool,
    allow_sdist: bool = False,
) -> tuple[bool, str]:
    """
    Compatibility rule:
      1) If any wheel in this release is compatible with target python => OK
      2) Else if allow_sdist and there's an sdist and requires_python allows => OK
      3) Else => NOT OK

    This matches real-world "works with Python X" much better for compiled libs.
    """
    wheels = [f for f in (release_files or []) if f.get("packagetype") == "bdist_wheel"]
    sdists = [f for f in (release_files or []) if f.get("packagetype") == "sdist"]

    # 1) Prefer wheel evidence
    for f in wheels:
        fn = f.get("filename") or ""
        if _wheel_supports_target_py(fn, target_py):
            return True, f"wheel ok: {fn}"

    # No compatible wheel found
    if not allow_sdist:
        return False, "no compatible wheel for target python"

    # 2) Optional: sdist fallback (only if requires_python allows)
    # Gather requires_python (file-level) if present
    specs: list[str] = []
    for f in (release_files or []):
        rp = f.get("requires_python")
        if rp:
            specs.append(rp)

    if not specs:
        if allow_unknown and sdists:
            return True, "sdist only, no requires_python metadata (allowed)"
        return False, "sdist only, no requires_python metadata"

    for spec_str in specs:
        try:
            if SpecifierSet(spec_str).contains(str(target_py), prereleases=True) and sdists:
                return True, f"sdist ok with requires_python={spec_str}"
        except Exception:
            continue

    return False, "sdist present but requires_python mismatch"


def first_newer_compatible_version(
    pkg_name: str,
    current: Version,
    target_py: Version,
    include_prereleases: bool,
    allow_unknown: bool,
    session: requests.Session,
) -> Tuple[Optional[Version], str]:
    """
    Return the first Version > current that supports target_py.
    """
    data = fetch_pypi_json(pkg_name, session=session)

    releases = data.get("releases", {})

    candidates: list[Version] = []
    for v_str in releases.keys():
        try:
            v = Version(v_str)
        except InvalidVersion:
            continue
        if v <= current:
            continue
        if v.is_prerelease and not include_prereleases:
            continue
        candidates.append(v)

    candidates.sort()

    for v in candidates:
        files = releases.get(str(v), [])
        # Skip yanked-only releases when possible
        # (PyPI JSON has "yanked" per file, so treat release as yanked if all files yanked.)
        if files and all(bool(f.get("yanked")) for f in files):
            continue

        ok, reason = release_supports_python(files, target_py, allow_unknown=allow_unknown)
        if ok:
            return v, reason

    return None, "no newer compatible release found"


def process_lines(
    lines: Iterable[str],
    target_py: Version,
    include_prereleases: bool,
    allow_unknown: bool,
    session: requests.Session,
) -> list[LineResult]:
    results: list[LineResult] = []

    for raw in lines:
        original_line = raw.rstrip("\n")

        # Keep control/comment/blank lines unchanged
        if is_control_line(original_line):
            results.append(LineResult(original_line, original_line, False))
            continue

        # Separate inline comment (if any) for parsing
        stripped = strip_inline_comment(original_line)
        comment = ""
        if stripped != original_line.rstrip("\n"):
            comment = original_line[len(stripped) :]

        try:
            req = Requirement(stripped)
        except Exception:
            # Unknown format -> keep unchanged
            results.append(LineResult(original_line, original_line, False, note="unparsed line"))
            continue

        pinned = is_pinned_eq(req)
        if pinned is None:
            # Not an exact pin -> keep unchanged (you can extend logic if you want)
            results.append(LineResult(original_line, original_line, False, note="not pinned with =="))
            continue

        name_for_pypi = req.name  # normalized name for PyPI endpoint works fine
        try:
            new_ver, why = first_newer_compatible_version(
                pkg_name=name_for_pypi,
                current=pinned,
                target_py=target_py,
                include_prereleases=include_prereleases,
                allow_unknown=allow_unknown,
                session=session,
            )
        except requests.HTTPError as e:
            results.append(LineResult(original_line, original_line, False, note=f"PyPI error: {e}"))
            continue
        except requests.RequestException as e:
            results.append(LineResult(original_line, original_line, False, note=f"network error: {e}"))
            continue

        if new_ver is None:
            results.append(LineResult(original_line, original_line, False, note=why))
            continue

        # Reconstruct requirement with the same extras, but updated pin
        # Keep markers if present (Requirement preserves them).
        extras = f"[{','.join(sorted(req.extras))}]" if req.extras else ""
        marker = f"; {req.marker}" if req.marker else ""
        updated_req = f"{req.name}{extras}=={new_ver}{marker}"
        updated_line = updated_req + comment
        results.append(LineResult(original_line, updated_line, True, note=why))

    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("requirements", default="requirements.txt", help="Path to requirements.txt")
    ap.add_argument("--py", default="3.12", help="Target Python version (default: 3.12.3)")
    ap.add_argument("--out", default="updated_req.txt", help="Write updated requirements to this file (default: stdout only)")
    ap.add_argument("--include-prereleases", default=False, action="store_true", help="Allow pre-releases when searching")
    ap.add_argument(
        "--allow-unknown",
        default=False,
        action="store_true",
        help="Treat releases with missing requires_python metadata as compatible (less strict)",
    )
    ap.add_argument("--report", default=True, action="store_true", help="Print a small change report to stderr")
    args = ap.parse_args()

    try:
        target_py = Version(args.py)
    except InvalidVersion:
        print(f"Invalid --py version: {args.py}", file=sys.stderr)
        return 2

    with open(args.requirements, "r", encoding="utf-8") as f:
        lines = f.readlines()

    session = requests.Session()
    session.headers.update({"User-Agent": "req-first-py312/1.0 (+https://pypi.org)"})

    results = process_lines(
        lines=lines,
        target_py=target_py,
        include_prereleases=args.include_prereleases,
        allow_unknown=args.allow_unknown,
        session=session,
    )

    updated_text = "\n".join(r.updated for r in results) + "\n"

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(updated_text)
    else:
        sys.stdout.write(updated_text)

    if args.report:
        for r in results:
            if r.changed:
                print(f"CHANGED: {r.original}  ->  {r.updated}   ({r.note})", file=sys.stderr)
            elif r.note:
                print(f"SKIP: {r.original}   ({r.note})", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
