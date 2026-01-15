#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import requests
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version, InvalidVersion
from packaging.markers import Marker


PYPI_PROJECT_JSON = "https://pypi.org/pypi/{name}/json"
PYPI_VERSION_JSON = "https://pypi.org/pypi/{name}/{version}/json"

WHEEL_RE = re.compile(r"^(?P<namever>.+)-(?P<py>[^-]+)-(?P<abi>[^-]+)-(?P<plat>[^-]+)\.whl$")


def target_cp_tag(py: Version) -> str:
    return f"cp{py.major}{py.minor:02d}"  # 3.12 -> cp312


def parse_cp_num(tag: str) -> Optional[int]:
    if not tag.startswith("cp"):
        return None
    rest = tag[2:]
    return int(rest) if rest.isdigit() else None


def wheel_supports_target(filename: str, target_py: Version) -> bool:
    m = WHEEL_RE.match(filename or "")
    if not m:
        return False
    py_tags = m.group("py").split(".")
    abi = m.group("abi")

    if "py3" in py_tags or "py2.py3" in py_tags:
        return True

    tgt = target_cp_tag(target_py)
    if tgt in py_tags:
        return True

    if abi == "abi3":
        tgt_num = int(tgt[2:])
        for t in py_tags:
            n = parse_cp_num(t)
            if n is not None and n <= tgt_num:
                return True

    return False


def release_has_compatible_wheel(files: list, target_py: Version) -> bool:
    for f in files or []:
        if f.get("packagetype") == "bdist_wheel":
            if wheel_supports_target(f.get("filename") or "", target_py):
                return True
    return False


def fetch_json(url: str, session: requests.Session) -> dict:
    r = session.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def build_candidates(name: str, current: Version, target_py: Version, session: requests.Session) -> List[Version]:
    data = fetch_json(PYPI_PROJECT_JSON.format(name=name), session)
    releases = data.get("releases", {})
    vs: List[Version] = []
    for v_str in releases.keys():
        try:
            v = Version(v_str)
        except InvalidVersion:
            continue
        if v < current:
            continue
        files = releases.get(v_str, [])
        if files and all(bool(f.get("yanked")) for f in files):
            continue
        if release_has_compatible_wheel(files, target_py):
            vs.append(v)
    vs.sort()
    return vs


def marker_allows(req: Requirement, target_py: Version) -> bool:
    """
    Evaluate PEP 508 markers for a target env. We only set python_version/python_full_version here.
    You can extend with platform_system, sys_platform, etc.
    """
    if req.marker is None:
        return True
    env = {
        "python_version": f"{target_py.major}.{target_py.minor}",
        "python_full_version": str(target_py),
        # Extend if you want:
        # "sys_platform": "linux",
        # "platform_system": "Linux",
    }
    try:
        return req.marker.evaluate(env)
    except Exception:
        # If marker parsing fails, be conservative and keep it
        return True


def get_requires_dist(name: str, version: Version, session: requests.Session) -> List[Requirement]:
    data = fetch_json(PYPI_VERSION_JSON.format(name=name, version=version), session)
    reqs = data.get("info", {}).get("requires_dist") or []
    out: List[Requirement] = []
    for s in reqs:
        try:
            out.append(Requirement(s))
        except Exception:
            continue
    return out


@dataclass
class Decision:
    name: str
    version: Version


def merge_constraint(constraints: Dict[str, SpecifierSet], pkg: str, spec: SpecifierSet) -> None:
    if pkg not in constraints:
        constraints[pkg] = spec
    else:
        # intersection: SpecifierSet string concat behaves like AND
        constraints[pkg] = SpecifierSet(str(constraints[pkg]) + "," + str(spec))


def satisfies(constraints: Dict[str, SpecifierSet], pkg: str, ver: Version) -> bool:
    spec = constraints.get(pkg)
    if spec is None:
        return True
    return spec.contains(str(ver), prereleases=True)


def resolve_minimal(
    top_level: Dict[str, Version],
    target_py: Version,
    session: requests.Session,
) -> Dict[str, Version]:
    """
    Backtracking resolver selecting the minimal candidate versions satisfying:
    - top-level >= pinned (since candidates built from current upward)
    - all requires_dist constraints (for markers that apply to target_py)
    """
    # Candidate lists for each top-level package
    cand: Dict[str, List[Version]] = {
        name: build_candidates(name, pinned, target_py, session) for name, pinned in top_level.items()
    }

    # If any top-level has no candidates, keep pinned (might still fail later)
    for name, pinned in top_level.items():
        if not cand[name]:
            cand[name] = [pinned]

    chosen: Dict[str, Version] = {}
    constraints: Dict[str, SpecifierSet] = {}

    # seed constraints from top-level pins themselves (exact ==)
    for name, pinned in top_level.items():
        merge_constraint(constraints, name, SpecifierSet(f"=={pinned}"))

    names = sorted(top_level.keys())  # deterministic

    def dfs(i: int) -> bool:
        if i == len(names):
            return True

        name = names[i]

        # Try candidates in ascending order
        for v in cand[name]:
            # Must be >= pinned, and satisfy accumulated constraints
            if not satisfies(constraints, name, v):
                continue

            # Tentatively choose
            chosen[name] = v

            # Save constraints snapshot
            saved = dict(constraints)

            # Apply dependencies constraints
            try:
                deps = get_requires_dist(name, v, session)
            except requests.RequestException:
                deps = []

            for dep in deps:
                if not marker_allows(dep, target_py):
                    continue
                dep_name = dep.name.lower()
                # Add constraint from specifier
                if str(dep.specifier):
                    merge_constraint(constraints, dep_name, dep.specifier)

                # If we already chose dep as top-level, validate immediately
                if dep_name in chosen and not satisfies(constraints, dep_name, chosen[dep_name]):
                    break
            else:
                # no break -> proceed
                if dfs(i + 1):
                    return True

            # backtrack
            constraints.clear()
            constraints.update(saved)
            chosen.pop(name, None)

        return False

    ok = dfs(0)
    if not ok:
        raise RuntimeError("Could not resolve constraints using only PyPI metadata.")

    return chosen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--py", default="3.12.3")
    ap.add_argument("reqs", nargs="+", help="Pinned requirements like eth-abi==3.0.0 numpy==1.24.4")
    args = ap.parse_args()

    target_py = Version(args.py)
    top_level: Dict[str, Version] = {}
    for s in args.reqs:
        r = Requirement(s)
        pinned = None
        for sp in r.specifier:
            if sp.operator == "==":
                pinned = Version(sp.version)
        if pinned is None:
            raise SystemExit(f"Only pinned == supported in this demo: {s}")
        top_level[r.name.lower()] = pinned

    session = requests.Session()
    session.headers.update({"User-Agent": "no-pip-resolver/0.1 (+https://pypi.org)"})

    solved = resolve_minimal(top_level, target_py, session)
    for k, v in solved.items():
        print(f"{k}=={v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
