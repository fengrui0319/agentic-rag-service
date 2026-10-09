from __future__ import annotations

import re
from pathlib import Path


SKILLS_DIR = Path(__file__).parent / "skills"

VALID_SKILL_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """
    Parse a minimal YAML-like frontmatter block:

    ---
    name: sql-formatter
    description: Format messy SQL queries.
    ---

    Returns:
        metadata, body
    """
    if not text.startswith("---"):
        raise ValueError("SKILL.md is missing frontmatter")

    parts = text.split("---", 2)

    if len(parts) != 3:
        raise ValueError("invalid SKILL.md frontmatter")

    raw_meta = parts[1]
    body = parts[2].lstrip()

    metadata: dict[str, str] = {}

    for line in raw_meta.splitlines():
        line = line.strip()

        if not line or ":" not in line:
            continue

        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()

    return metadata, body


def discover_skills() -> list[dict[str, str]]:
    """
    Discover available skills without loading their full bodies.

    This implements progressive disclosure:
    only name + description are placed in the agent context.
    """
    skills: list[dict[str, str]] = []

    if not SKILLS_DIR.exists():
        return skills

    for skill_file in sorted(SKILLS_DIR.glob("*/SKILL.md")):
        text = skill_file.read_text(encoding="utf-8")

        try:
            metadata, _ = _parse_frontmatter(text)
        except ValueError:
            continue

        name = metadata.get("name", "").strip()
        description = metadata.get("description", "").strip()

        if not name or not description:
            continue

        if not VALID_SKILL_NAME.fullmatch(name):
            continue

        skills.append(
            {
                "name": name,
                "description": description,
            }
        )

    return skills


def read_skill(name: str) -> str:
    """
    Load the complete body of one skill on demand.
    """
    if not VALID_SKILL_NAME.fullmatch(name):
        raise ValueError(f"invalid skill name: {name!r}")

    skill_file = SKILLS_DIR / name / "SKILL.md"

    if not skill_file.is_file():
        raise ValueError(f"unknown skill: {name}")

    text = skill_file.read_text(encoding="utf-8")
    metadata, body = _parse_frontmatter(text)

    actual_name = metadata.get("name", "").strip()

    if actual_name != name:
        raise ValueError(
            f"skill name mismatch: requested {name!r}, "
            f"file declares {actual_name!r}"
        )

    return body


def build_skill_catalog() -> str:
    """
    Build the lightweight catalog inserted into the system prompt.
    """
    skills = discover_skills()

    if not skills:
        return "No skills are currently installed."

    lines = ["Available skills:"]

    for skill in skills:
        lines.append(
            f"- {skill['name']}: {skill['description']}"
        )

    lines.append(
        "\nUse the read_skill tool when a skill is relevant. "
        "Do not assume you know the full skill instructions "
        "from the catalog alone."
    )

    return "\n".join(lines)