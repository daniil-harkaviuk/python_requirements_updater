#!/usr/bin/env python3
"""
Simple requirements "package manager":
- Updates direct requirements to newest PyPI versions compatible with target Python
- Resolves dependencies via PyPI metadata (requires_dist)
- Checks cross-dependency constraints and backtracks when conflicts happen
- Outputs:
  - requirements.updated.txt (direct deps pinned)
  - requirements.lock.txt (full lock: direct + transitive pinned)

Limitations (by design, keep it simple):
- Partial support for environment markers (we evaluate common markers for target python)
- No VCS/path/editable requirements
- Does not check wheel/ABI availability (only metadata constraints)
- Prefers stable releases, ignores prereleases unless allowed by constraints
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import requests
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version, InvalidVersion
from packaging.utils import canonicalize_name
from packaging.markers import default_environment


PYPI = "https://pypi.org/pypi"


@dataclass(frozen=True)
class PkgCandidate:
    name: str
    version: Version


def read_requirements_txt(path: str) -> List[str]:
    lines: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # ignore pip options
            if line.startswith(("-", "--")):
                continue
            lines.append(line)
    return lines


def parse_top_level(lines: List[str]) -> List[Requirement]:
    reqs: List[Requirement] = []
    for line in lines:
        # skip unsupported formats early
        if "git+" in line or "://" in line or line.startswith((".", "/")):
            raise ValueError(f"Unsupported requirement line (vcs/url/path): {line}")
        reqs.append(Requirement(line))
    return reqs


def http_get_json(url: str) -> dict:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def get_all_versions(pkg: str) -> List[Version]:
    data = http_get_json(f"{PYPI}/{pkg}/json")
    releases = data.get("releases", {})
    versions: List[Version] = []
    for v in releases.keys():
        try:
            versions.append(Version(v))
        except InvalidVersion:
            continue
    versions.sort(reverse=True)
    return versions


def get_requires_dist(pkg: str, ver: Version) -> List[str]:
    data = http_get_json(f"{PYPI}/{pkg}/{ver}/json")
    info = data.get("info", {}) or {}
    requires = info.get("requires_dist") or []
    # sometimes it can be None
    return list(requires)


def is_python_compatible(pkg: str, ver: Version, target_python: str) -> bool:
    """
    Checks `requires_python` field from PyPI (PEP 345 / PEP 440 specifier)
    """
    data = http_get_json(f"{PYPI}/{pkg}/{ver}/json")
    info = data.get("info", {}) or {}
    rp = info.get("requires_python")
    if not rp:
        return True
    try:
        spec = SpecifierSet(rp)
    except Exception:
        return True

    # packaging compares against a Version
    return Version(target_python) in spec


def evaluate_marker(req: Requirement, target_python: str) -> bool:
    """
    Evaluate env markers for the target python (best-effort).
    """
    if req.marker is None:
        return True
    env = default_environment()
    # Override python_version / python_full_version for target
    pv = ".".join(target_python.split(".")[:2])
    env["python_version"] = pv
    env["python_full_version"] = target_python
    try:
        return bool(req.marker.evaluate(env))
    except Exception:
        # if marker parsing/evaluation fails, be permissive
        return True


def choose_versions_for_requirement(
    name: str,
    spec: SpecifierSet,
    target_python: str,
    allow_prereleases: bool = False,
) -> List[Version]:
    """
    Returns versions in descending order that satisfy the constraint and python compatibility.
    """
    all_vers = get_all_versions(name)
    out: List[Version] = []
    for v in all_vers:
        if not allow_prereleases and v.is_prerelease:
            continue
        if v not in spec:
            continue
        if is_python_compatible(name, v, target_python):
            out.append(v)
    return out


def normalize_req_name(req: Requirement) -> str:
    return canonicalize_name(req.name)


def merge_specifiers(a: SpecifierSet, b: SpecifierSet) -> SpecifierSet:
    # SpecifierSet doesn't have direct "and" merge, but string concatenation works.
    if str(a) and str(b):
        return SpecifierSet(f"{a},{b}")
    return SpecifierSet(str(a) or str(b))


class Resolver:
    def __init__(self, target_python: str, allow_prereleases: bool = False):
        self.target_python = target_python
        self.allow_prereleases = allow_prereleases

        # Cache network calls heavily
        self._versions_cache: Dict[str, List[Version]] = {}
        self._requires_cache: Dict[Tuple[str, str], List[str]] = {}
        self._pyok_cache: Dict[Tuple[str, str, str], bool] = {}

    def get_versions(self, pkg: str) -> List[Version]:
        pkg = canonicalize_name(pkg)
        if pkg not in self._versions_cache:
            self._versions_cache[pkg] = get_all_versions(pkg)
        return self._versions_cache[pkg]

    def py_ok(self, pkg: str, ver: Version) -> bool:
        k = (canonicalize_name(pkg), str(ver), self.target_python)
        if k not in self._pyok_cache:
            self._pyok_cache[k] = is_python_compatible(pkg, ver, self.target_python)
        return self._pyok_cache[k]

    def requires_dist(self, pkg: str, ver: Version) -> List[str]:
        k = (canonicalize_name(pkg), str(ver))
        if k not in self._requires_cache:
            self._requires_cache[k] = get_requires_dist(pkg, ver)
        return self._requires_cache[k]

    def pick_candidates(self, pkg: str, spec: SpecifierSet) -> List[Version]:
        candidates: List[Version] = []
        for v in self.get_versions(pkg):
            if not self.allow_prereleases and v.is_prerelease:
                continue
            if v not in spec:
                continue
            if self.py_ok(pkg, v):
                candidates.append(v)
        return candidates

    def resolve(
        self,
        top_level: List[Requirement],
        max_nodes: int = 500,
    ) -> Dict[str, Version]:
        """
        Backtracking dependency resolver.
        Returns mapping {package_name_normalized: chosen_version}
        """
        # Constraints collected per package
        constraints: Dict[str, SpecifierSet] = {}
        # Track which requirements introduced the constraint (debug)
        # chosen versions
        chosen: Dict[str, Version] = {}

        # Seed constraints from top-level requirements
        queue: List[Requirement] = []
        for r in top_level:
            if not evaluate_marker(r, self.target_python):
                continue
            nm = normalize_req_name(r)
            constraints[nm] = merge_specifiers(constraints.get(nm, SpecifierSet()), r.specifier or SpecifierSet())
            queue.append(r)

        # Expand queue with dependencies as we choose versions.
        # We'll resolve by repeatedly selecting an unresolved package with constraints.
        visited_nodes = 0

        def pick_next_unresolved() -> Optional[str]:
            unresolved = [p for p in constraints.keys() if p not in chosen]
            if not unresolved:
                return None
            # Heuristic: smallest candidate list first (fail fast)
            best = None
            best_len = 10**9
            for p in unresolved:
                cand_len = len(self.pick_candidates(p, constraints[p]))
                if cand_len < best_len:
                    best_len = cand_len
                    best = p
            return best

        # We also need a stack to support backtracking with constraint snapshots
        stack: List[Tuple[str, List[Version], int, Dict[str, SpecifierSet], Dict[str, Version]]] = []

        while True:
            if visited_nodes > max_nodes:
                raise RuntimeError(f"Resolution aborted: exceeded max_nodes={max_nodes}")

            pkg = pick_next_unresolved()
            if pkg is None:
                return chosen  # all resolved

            spec = constraints[pkg]
            candidates = self.pick_candidates(pkg, spec)
            if not candidates:
                # conflict -> backtrack
                if not stack:
                    raise RuntimeError(f"Unresolvable: {pkg} with constraints '{spec}' has no candidates")
                pkg_prev, cand_prev, idx_prev, constraints_prev, chosen_prev = stack.pop()
                # restore snapshots
                constraints = constraints_prev
                chosen = chosen_prev
                # try next candidate
                idx_prev += 1
                if idx_prev >= len(cand_prev):
                    # keep backtracking
                    continue
                # push updated frame back
                stack.append((pkg_prev, cand_prev, idx_prev, constraints.copy(), chosen.copy()))
                # apply next candidate
                chosen[pkg_prev] = cand_prev[idx_prev]
                # add its deps
                visited_nodes += 1
                self._add_deps(pkg_prev, chosen[pkg_prev], constraints)
                continue

            # choose best (latest) candidate first, save frame
            idx = 0
            stack.append((pkg, candidates, idx, constraints.copy(), chosen.copy()))
            chosen[pkg] = candidates[idx]
            visited_nodes += 1
            self._add_deps(pkg, chosen[pkg], constraints)

    def _add_deps(self, pkg: str, ver: Version, constraints: Dict[str, SpecifierSet]) -> None:
        for dep_str in self.requires_dist(pkg, ver):
            try:
                dep = Requirement(dep_str)
            except Exception:
                continue
            if not evaluate_marker(dep, self.target_python):
                continue
            dep_name = normalize_req_name(dep)
            dep_spec = dep.specifier or SpecifierSet()
            constraints[dep_name] = merge_specifiers(constraints.get(dep_name, SpecifierSet()), dep_spec)


def write_updated_files(
    top_level: List[Requirement],
    resolved: Dict[str, Version],
    out_updated: str,
    out_lock: str,
) -> None:
    # requirements.updated.txt: only top-level pinned (keep original names casing)
    updated_lines: List[str] = []
    for r in top_level:
        nm = normalize_req_name(r)
        if nm in resolved:
            updated_lines.append(f"{r.name}=={resolved[nm]}")
        else:
            # marker excluded or unresolved; keep original
            updated_lines.append(str(r))

    with open(out_updated, "w", encoding="utf-8") as f:
        f.write("\n".join(updated_lines) + "\n")

    # requirements.lock.txt: all resolved packages pinned sorted
    lock_lines = [f"{name}=={ver}" for name, ver in sorted(resolved.items(), key=lambda x: x[0])]
    with open(out_lock, "w", encoding="utf-8") as f:
        f.write("\n".join(lock_lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Simple requirements updater/resolver via PyPI metadata.")
    ap.add_argument("-i", "--input", default="requirements.txt", help="Input requirements.txt")
    ap.add_argument("--python", default="3.12.3", help="Target python full version, e.g. 3.12.3")
    ap.add_argument("--allow-prereleases", action="store_true", help="Allow prerelease versions")
    ap.add_argument("--max-nodes", type=int, default=500, help="Limit for resolver steps (avoid huge graphs)")
    ap.add_argument("-o", "--out-updated", default="requirements.updated.txt", help="Output updated top-level pins")
    ap.add_argument("-l", "--out-lock", default="requirements.lock.txt", help="Output full lock file")
    args = ap.parse_args()

    lines = read_requirements_txt(args.input)
    top = parse_top_level(lines)

    resolver = Resolver(target_python=args.python, allow_prereleases=args.allow_prereleases)
    resolved = resolver.resolve(top, max_nodes=args.max_nodes)

    write_updated_files(top, resolved, args.out_updated, args.out_lock)

    print(f"OK: wrote {args.out_updated} (top-level) and {args.out_lock} (full lock)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
