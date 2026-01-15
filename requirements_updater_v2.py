#!/usr/bin/env python3
"""
requirements_updater.py

Updates pinned requirements (pkg==X.Y.Z) to the *first* newer version that has
a CPython 3.12-compatible wheel (cp312 / abi3 / py3 wheel).

NEW: Cross-dependency checker for packages that are ALSO pinned in requirements.txt.
Example:
  eth-abi==3.0.0 requires eth-utils>=2.0.0,<3.0.0
  If requirements.txt pins eth-utils==3.0.0 -> conflict is detected.
The script will try to bump eth-abi to a newer candidate that becomes compatible.
If impossible -> raises with a clear error.

Usage:
  python requirements_updater.py requirements.txt --py 3.12.3 --out requirements.py312.txt --report

Notes:
- Only exact pins "==" are updated. Other lines are preserved as-is.
- Dependency checking uses PyPI "requires_dist" (PEP 508). Markers are evaluated
  only for python_version/python_full_version; extend env if needed.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from packaging.markers import Marker
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version


PYPI_PROJECT_JSON = "https://pypi.org/pypi/{name}/json"
PYPI_VERSION_JSON = "https://pypi.org/pypi/{name}/{version}/json"

_WHEEL_RE = re.compile(r"^(?P<namever>.+)-(?P<py>[^-]+)-(?P<abi>[^-]+)-(?P<plat>[^-]+)\.whl$")


# -----------------------------
# requirements.txt parsing utils
# -----------------------------

def strip_inline_comment(line: str) -> str:
    if line.lstrip().startswith("#"):
        return line
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


def pinned_eq_version(req: Requirement) -> Optional[Version]:
    specs = list(req.specifier)
    if len(specs) != 1 or specs[0].operator != "==":
        return None
    try:
        return Version(specs[0].version)
    except InvalidVersion:
        return None


@dataclass
class ParsedLine:
    raw: str
    req: Optional[Requirement] = None
    pinned: Optional[Version] = None
    comment_suffix: str = ""


def parse_requirement_lines(lines: Iterable[str]) -> List[ParsedLine]:
    parsed: List[ParsedLine] = []
    for raw in lines:
        raw = raw.rstrip("\n")
        if is_control_line(raw):
            parsed.append(ParsedLine(raw=raw))
            continue

        stripped = strip_inline_comment(raw)
        comment = raw[len(stripped):] if stripped != raw else ""

        try:
            req = Requirement(stripped)
        except Exception:
            parsed.append(ParsedLine(raw=raw))
            continue

        pinned = pinned_eq_version(req)
        parsed.append(ParsedLine(raw=raw, req=req, pinned=pinned, comment_suffix=comment))
    return parsed


def render_requirement(req: Requirement, ver: Version, comment_suffix: str) -> str:
    extras = f"[{','.join(sorted(req.extras))}]" if req.extras else ""
    marker = f"; {req.marker}" if req.marker else ""
    return f"{req.name}{extras}=={ver}{marker}{comment_suffix}"


# -----------------------------
# wheel compatibility (CPython)
# -----------------------------

def _target_cp_tag(py: Version) -> str:
    # 3.12 -> cp312
    return f"cp{py.major}{py.minor:02d}"


def _parse_cp_num(tag: str) -> Optional[int]:
    if not tag.startswith("cp"):
        return None
    rest = tag[2:]
    return int(rest) if rest.isdigit() else None


def wheel_supports_target(filename: str, target_py: Version) -> bool:
    """
    Accept if:
      - pure python wheel includes py3 (usually py3-none-any)
      - exact cp312 wheel exists
      - abi3 wheel with cpXY where XY <= target (cp38-abi3 works on 3.12)
    """
    m = _WHEEL_RE.match(filename or "")
    if not m:
        return False

    py_tags = m.group("py").split(".")
    abi = m.group("abi")

    if "py3" in py_tags or "py2.py3" in py_tags:
        return True

    tgt = _target_cp_tag(target_py)
    if tgt in py_tags:
        return True

    if abi == "abi3":
        tgt_num = int(tgt[2:])
        for t in py_tags:
            n = _parse_cp_num(t)
            if n is not None and n <= tgt_num:
                return True

    return False


def release_has_compatible_wheel(files: list, target_py: Version) -> bool:
    for f in files or []:
        if f.get("packagetype") == "bdist_wheel":
            if wheel_supports_target(f.get("filename") or "", target_py):
                return True
    return False


# -----------------------------
# PyPI querying + caching
# -----------------------------

class PypiClient:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "requirements-updater/py312 (+https://pypi.org)"})
        self._project_cache: Dict[str, dict] = {}
        self._version_cache: Dict[Tuple[str, str], dict] = {}

    def project_json(self, name: str) -> dict:
        key = name.lower()
        if key in self._project_cache:
            return self._project_cache[key]
        url = PYPI_PROJECT_JSON.format(name=name)
        r = self.session.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        self._project_cache[key] = data
        return data

    def version_json(self, name: str, version: Version) -> dict:
        key = (name.lower(), str(version))
        if key in self._version_cache:
            return self._version_cache[key]
        url = PYPI_VERSION_JSON.format(name=name, version=str(version))
        r = self.session.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        self._version_cache[key] = data
        return data

    def requires_dist(self, name: str, version: Version) -> List[Requirement]:
        data = self.version_json(name, version)
        reqs = data.get("info", {}).get("requires_dist") or []
        out: List[Requirement] = []
        for s in reqs:
            try:
                out.append(Requirement(s))
            except Exception:
                continue
        return out


def build_candidates(
    client: PypiClient,
    pkg_name: str,
    current: Version,
    target_py: Version,
    include_prereleases: bool,
) -> List[Version]:
    """
    Returns compatible versions >= current (ascending) that have compatible wheels for target_py.
    """
    data = client.project_json(pkg_name)
    releases = data.get("releases", {})

    versions: List[Version] = []
    for v_str in releases.keys():
        try:
            v = Version(v_str)
        except InvalidVersion:
            continue
        if v < current:
            continue
        if v.is_prerelease and not include_prereleases:
            continue
        files = releases.get(v_str, [])
        if files and all(bool(f.get("yanked")) for f in files):
            continue
        if release_has_compatible_wheel(files, target_py):
            versions.append(v)

    versions.sort()
    return versions


# -----------------------------
# Markers evaluation for deps
# -----------------------------

def marker_allows(req: Requirement, target_py: Version) -> bool:
    if req.marker is None:
        return True
    env = {
        "python_version": f"{target_py.major}.{target_py.minor}",
        "python_full_version": str(target_py),
        # Extend if you want stronger correctness:
        # "sys_platform": "linux",
        # "platform_system": "Linux",
    }
    try:
        return req.marker.evaluate(env)
    except Exception:
        # If marker is weird/unparseable, don't block.
        return True


# -----------------------------
# Cross-dependency checking
# -----------------------------

@dataclass
class DepConflict:
    depender: str
    depender_version: Version
    dependency: str
    dependency_version: Version
    required_spec: str


def find_cross_conflicts(
    client: PypiClient,
    selected: Dict[str, Version],
    target_py: Version,
) -> List[DepConflict]:
    """
    Checks dependencies between packages present in `selected` only.

    For each selected package A==v, read requires_dist and if it mentions B
    and B is also selected, ensure selected[B] satisfies the specifier.
    """
    conflicts: List[DepConflict] = []

    for a_name, a_ver in selected.items():
        try:
            deps = client.requires_dist(a_name, a_ver)
        except requests.RequestException:
            continue

        for dep in deps:
            if not marker_allows(dep, target_py):
                continue

            b_name = dep.name.lower()
            if b_name not in selected:
                continue

            spec = dep.specifier
            if not str(spec):
                # no version constraint -> always ok
                continue

            b_ver = selected[b_name]
            if not spec.contains(str(b_ver), prereleases=True):
                conflicts.append(
                    DepConflict(
                        depender=a_name,
                        depender_version=a_ver,
                        dependency=b_name,
                        dependency_version=b_ver,
                        required_spec=str(spec),
                    )
                )

    return conflicts


def try_fix_conflicts_by_bumping_dependers(
    client: PypiClient,
    selected: Dict[str, Version],
    candidates: Dict[str, List[Version]],
    target_py: Version,
    max_rounds: int = 50,
) -> Tuple[Dict[str, Version], List[DepConflict]]:
    """
    Attempts to fix conflicts by bumping ONLY the depender package (A),
    to the next candidate version, until:
      - no conflicts remain, or
      - no more progress is possible.

    This keeps the "upgrade-only" nature intact.
    """
    selected = dict(selected)

    for _ in range(max_rounds):
        conflicts = find_cross_conflicts(client, selected, target_py)
        if not conflicts:
            return selected, []

        progressed = False

        # deterministic order: bump dependers in sorted order
        by_depender = {}
        for c in conflicts:
            by_depender.setdefault(c.depender, []).append(c)

        for depender in sorted(by_depender.keys()):
            curr = selected[depender]
            cand_list = candidates.get(depender, [])
            if not cand_list:
                continue

            try:
                idx = cand_list.index(curr)
            except ValueError:
                # If curr not in list, try to find first >= curr
                idx = -1
                for i, v in enumerate(cand_list):
                    if v >= curr:
                        idx = i
                        break
                if idx == -1:
                    continue

            # Try next candidates until conflicts caused by this depender disappear
            for next_i in range(idx + 1, len(cand_list)):
                trial_ver = cand_list[next_i]
                selected[depender] = trial_ver

                # re-check only; cheap enough
                new_conflicts = find_cross_conflicts(client, selected, target_py)

                # accept the bump if it reduces conflicts for this depender
                still_bad_for_depender = [
                    cc for cc in new_conflicts if cc.depender == depender
                ]
                if not still_bad_for_depender:
                    progressed = True
                    break

            if not progressed:
                # restore if no bump helped
                selected[depender] = curr

        if not progressed:
            # stuck
            return selected, find_cross_conflicts(client, selected, target_py)

    return selected, find_cross_conflicts(client, selected, target_py)


# -----------------------------
# main update flow
# -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("requirements", help="Path to requirements.txt")
    ap.add_argument("--out", default="requirements.py312.txt", help="Output requirements file")
    ap.add_argument("--py", default="3.12.3", help="Target Python version (default 3.12.3)")
    ap.add_argument("--include-prereleases", action="store_true", help="Allow pre-releases")
    ap.add_argument("--report", action="store_true", help="Print changes and dependency conflicts to stderr")
    args = ap.parse_args()

    try:
        target_py = Version(args.py)
    except InvalidVersion:
        print(f"Invalid --py version: {args.py}", file=sys.stderr)
        return 2

    with open(args.requirements, "r", encoding="utf-8") as f:
        lines = f.readlines()

    parsed = parse_requirement_lines(lines)

    # collect pinned top-level packages
    pinned_top: Dict[str, ParsedLine] = {}
    for pl in parsed:
        if pl.req and pl.pinned:
            pinned_top[pl.req.name.lower()] = pl

    if not pinned_top:
        # nothing to do
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("".join(lines))
        return 0

    client = PypiClient()

    # Build candidates and initial selections (first candidate)
    candidates: Dict[str, List[Version]] = {}
    selected: Dict[str, Version] = {}
    for name_l, pl in pinned_top.items():
        try:
            cand = build_candidates(
                client=client,
                pkg_name=pl.req.name,  # type: ignore
                current=pl.pinned,     # type: ignore
                target_py=target_py,
                include_prereleases=args.include_prereleases,
            )
        except requests.RequestException as e:
            cand = [pl.pinned]  # type: ignore
            if args.report:
                print(f"[WARN] PyPI query failed for {pl.req.name}: {e}", file=sys.stderr)  # type: ignore

        if not cand:
            cand = [pl.pinned]  # type: ignore

        candidates[name_l] = cand
        selected[name_l] = cand[0]

    # NEW: cross-dependency check + attempt to fix by bumping dependers
    selected, conflicts = try_fix_conflicts_by_bumping_dependers(
        client=client,
        selected=selected,
        candidates=candidates,
        target_py=target_py,
    )

    # Render updated file
    out_lines: List[str] = []
    changes: List[str] = []
    for pl in parsed:
        if pl.req and pl.pinned:
            name_l = pl.req.name.lower()
            new_ver = selected[name_l]
            out_lines.append(render_requirement(pl.req, new_ver, pl.comment_suffix))
            if new_ver != pl.pinned:
                changes.append(f"{pl.req.name}: {pl.pinned} -> {new_ver}")
        else:
            out_lines.append(pl.raw)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")

    if args.report:
        if changes:
            print("Updated pins:", file=sys.stderr)
            for c in changes:
                print(f"  - {c}", file=sys.stderr)
        else:
            print("No pinned packages were changed.", file=sys.stderr)

    if conflicts:
        # Produce a helpful error and non-zero exit
        msg_lines = ["Dependency conflicts between pinned requirements:"]
        for c in conflicts:
            msg_lines.append(
                f"  - {c.depender}=={c.depender_version} requires {c.dependency}{c.required_spec}, "
                f"but pinned/selected {c.dependency}=={c.dependency_version}"
            )
        msg_lines.append("")
        msg_lines.append("Fix options:")
        msg_lines.append("  1) Bump the depender package to a version that supports the pinned dependency, or")
        msg_lines.append("  2) Change the pinned dependency version to satisfy the depender's constraint.")
        print("\n".join(msg_lines), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
