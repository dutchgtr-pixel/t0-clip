"""Offline PostgreSQL syntax and direct relation-dependency inventory.

No connection, environment credentials or database execution is performed.
The PostgreSQL parser does not resolve columns/types or execute function bodies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from pglast.parser import parse_sql_json


ROOT = Path(__file__).resolve().parents[2]


def canonical_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def prepare_sql(text: str) -> tuple[str, dict]:
    """Normalize psql client directives for syntax inspection only."""
    directives = [line for line in text.splitlines() if line.lstrip().startswith("\\")]
    sql = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("\\"))
    sql, quoted = re.subn(r"(?<!:):'([a-zA-Z_]\w*)'", "'syntax_fixture'", sql)
    sql, numbers = re.subn(r"(?<!:):([a-zA-Z_]\w*)", "1", sql)
    return sql, {"omitted_client_directives": len(directives), "substituted_client_variables": quoted + numbers}


def walk(value):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk(item)


def qualified(relation):
    return f"{relation['schemaname']}.{relation['relname']}" if relation.get("schemaname") and relation.get("relname") else None


def inspect_sql(path: Path, root: Path = ROOT) -> dict:
    text, normalization = prepare_sql(path.read_text(encoding="utf-8-sig"))
    result = {"path": path.relative_to(root).as_posix(), "sha256_lf": canonical_hash(path), **normalization}
    fragment = path.name in {"03_strict_anchor_lateral.sql", "04_advanced_anchor_v1_lateral.sql"}
    if fragment:
        prefix = "SELECT " if path.name.startswith("04_") else "SELECT b.* FROM public.listings b\n"
        text = prefix + text
        result["fragment_wrapper"] = prefix.strip() + "; parser context only, not a source adapter"
    try:
        parsed = json.loads(parse_sql_json(text))
    except Exception as exc:
        result.update(status="parser_rejected", error_type=type(exc).__name__)
        return result
    definitions, references = set(), set()
    dynamic_bodies = 0
    for node in walk(parsed):
        if "RangeVar" in node:
            name = qualified(node["RangeVar"])
            if name:
                references.add(name)
        for kind, field in (("CreateStmt", "relation"), ("ViewStmt", "view")):
            if kind in node:
                name = qualified(node[kind].get(field, {}))
                if name:
                    definitions.add(name)
        if "CreateTableAsStmt" in node:
            name = qualified(node["CreateTableAsStmt"].get("into", {}).get("rel", {}))
            if name:
                definitions.add(name)
        if "CreateFunctionStmt" in node or "DoStmt" in node:
            dynamic_bodies += 1
    result.update(status="fragment_parsed" if fragment else "syntax_parsed", statement_count=len(parsed["stmts"]), defines=sorted(definitions),
                  directly_references=sorted(references - definitions), function_or_do_bodies_not_semantically_checked=dynamic_bodies)
    return result


def build_inventory(root: Path = ROOT) -> dict:
    paths = sorted((root / "feature-stores").rglob("*.sql"))
    paths += sorted((root / "research/feature_store_ddl/archived").glob("*.sql"))
    paths += sorted((root / "research/feature_store_ddl/portable").glob("*.sql"))
    entries = [inspect_sql(path, root) for path in paths]
    return {"schema_version": 1, "validation": "offline PostgreSQL AST parsing; not database execution",
            "limitations": ["Qualified relation edges only; dynamic SQL and function-body dependencies require manual review.",
                            "Definitions spread over historical revisions are not one executable migration sequence.",
                            "Parsing does not check missing relations, column types, extensions, permissions or temporal correctness."],
            "files": entries, "counts": {"sql_files": len(entries), "syntax_parsed": sum(e["status"] == "syntax_parsed" for e in entries),
                                           "fragment_parsed": sum(e["status"] == "fragment_parsed" for e in entries),
                                           "parser_rejected": sum(e["status"] == "parser_rejected" for e in entries)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_inventory()
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["counts"], indent=2))


if __name__ == "__main__":
    main()
