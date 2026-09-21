"""Audit changed public-release files without displaying potentially secret text.

Defaults to tracked changes relative to HEAD plus untracked files, honoring Git
ignore rules. This is a heuristic release gate, not a guarantee of anonymity or
secret absence. Review the diff and data provenance as well.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Iterable
import zipfile


MAX_BYTES = 2 * 1024 * 1024
ORIGINAL_PUBLIC_BASE = "dc41f873b82ab5c1ac07862c7610bcf2d6a40e62"
BLOCKED_SUFFIXES = {
    ".pt", ".pth", ".ckpt", ".parquet", ".arrow", ".feather", ".npy", ".npz",
    ".pkl", ".pickle", ".sqlite", ".sqlite3", ".db", ".dump", ".tar", ".gz",
    ".zip", ".7z", ".log", ".jsonl", ".csv", ".tsv", ".exe", ".dll", ".pyc",
}
RULES = {
    "credential_in_url": re.compile(r"[a-z][a-z0-9+.-]*://[^\s/:@]+:[^\s/@]+@", re.I),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "api_token": re.compile(r"\b(?:sk-(?:proj-)?[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[A-Z0-9]{16})\b"),
    "private_absolute_path": re.compile(r"(?:[A-Z]:[\\/]+(?:Users|surv_artifacts|docker_backup|docker_emergency_recovery)[\\/]|/ho(?:m)e/(?!app/)[^/\s]+/|/mnt/[a-z]/(?:Users|surv_artifacts|docker_backup)/)", re.I),
    "provider_fingerprint": re.compile(r"(?<![a-z])(?:fi(?:n)n(?:\.no|_id|id)?|ti(?:s)e(?:\.com)?|e(?:bay)|ama(?:zon)|face(?:book)(?:[ _-]+marketplace)?|craigs(?:list)|vin(?:ted)|mer(?:cari)|gum(?:tree)|o(?:lx)|tor(?:get)|i(?:phone)\d*)(?![a-z])", re.I),
    "acquisition_terminology": re.compile(r"scra(?:p)(?:e[rsd]?|ers|ing|y)", re.I),
    "private_network": re.compile(r"\b(?:192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b"),
}
SECRET_ASSIGNMENT = re.compile(
    r'''(?ix)["']?(?:password|passwd|api[_-]?key|access[_-]?token|client[_-]?secret|signing[_-]?secret)["']?\s*[:=]\s*["']([^"'\n]+)["']'''
)
PLACEHOLDER = re.compile(r"^(?:<[^>]+>|\$\{[^}]+\}|YOUR_[A-Z_]+|REPLACE_ME|\*+)$")


@dataclass(frozen=True)
class Finding:
    path: str
    rule: str
    line: int | None = None


def git(root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(["git", "-C", str(root), *arguments], capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError("Git could not enumerate release scope")
    return completed.stdout


def changed_paths(root: Path, base: str = "HEAD") -> list[Path]:
    tracked = git(root, "diff", "--name-only", "--diff-filter=ACMRT", "-z", base, "--")
    untracked = git(root, "ls-files", "--others", "--exclude-standard", "-z")
    names = {name.decode("utf-8", errors="surrogateescape") for name in (tracked + untracked).split(b"\0") if name}
    return sorted((root / name for name in names if (root / name).is_file()), key=str)


def all_public_paths(root: Path) -> list[Path]:
    """Versioned and nonignored new files; local environments remain excluded."""
    paths = git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
    names = {name.decode("utf-8", errors="surrogateescape") for name in paths.split(b"\0") if name}
    return sorted((root / name for name in names if (root / name).is_file()), key=str)


def terminology_findings(relative: str, text: str) -> list[Finding]:
    """Report text locations and filenames, without echoing matched content."""
    findings = []
    for name in ("provider_fingerprint", "acquisition_terminology"):
        pattern = RULES[name]
        if pattern.search(relative):
            findings.append(Finding(relative, name + "_in_path"))
        findings.extend(Finding(relative, name, number) for number, line in enumerate(text.splitlines(), 1) if pattern.search(line))
    return findings


def known_public_pdf(root: Path, path: Path, baseline: str = ORIGINAL_PUBLIC_BASE) -> bool:
    """Allow unchanged PDFs from an explicit public baseline, never moving HEAD."""
    relative = path.relative_to(root).as_posix()
    try:
        prior_object = git(root, "rev-parse", f"{baseline}:{relative}").strip()
        current_object = git(root, "hash-object", "--", str(path)).strip()
    except RuntimeError:
        return False
    return prior_object == current_object


def reviewed_artifacts(root: Path, manifest: Path, kind: str, suffix: str) -> dict[str, str]:
    """An explicit path/hash review record is not inferred from Git history."""
    if not manifest.exists():
        return {}
    contents = json.loads(manifest.read_text(encoding="utf-8"))
    if contents.get("schema_version") != 1 or contents.get("kind") != kind:
        raise ValueError("Invalid public artifact review manifest")
    approved = {}
    for item in contents["files"]:
        relative, digest = item["path"], item["sha256"]
        candidate = root / relative
        candidate.resolve().relative_to(root.resolve())
        if Path(relative).is_absolute() or Path(relative).suffix.lower() != suffix or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Invalid artifact path/hash in review manifest")
        if relative in approved:
            raise ValueError("Duplicate artifact in review manifest")
        if not candidate.is_file() or hashlib.sha256(candidate.read_bytes()).hexdigest() != digest:
            raise ValueError("Reviewed artifact hash does not match current bytes")
        approved[relative] = digest
    return approved


def audit_archive(path: Path, relative: str) -> list[Finding]:
    """An approved archive still needs safe, bounded, text-only members."""
    findings = []
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) > 1000 or sum(info.file_size for info in infos) > 8 * MAX_BYTES:
            return [Finding(relative, "archive_size_limit")]
        for info in infos:
            member = Path(info.filename)
            label = relative + "!" + info.filename
            if member.is_absolute() or ".." in member.parts or ":" in info.filename or "\\" in info.filename:
                findings.append(Finding(relative, "unsafe_archive_member_path"))
                continue
            if info.is_dir():
                continue
            raw = archive.read(info)
            if member.suffix.lower() in BLOCKED_SUFFIXES or b"\0" in raw:
                findings.append(Finding(label, "archive_data_or_binary_member"))
                continue
            try:
                contents = raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                findings.append(Finding(label, "archive_non_utf8_member"))
                continue
            findings.extend(screen_text(label, contents))
    return findings


def audit_pdf_text(path: Path, relative: str) -> list[Finding]:
    """Text extraction complements the recorded visual review, not OCR."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF inspection requires the development dependencies") from exc
    reader = PdfReader(path)
    if reader.is_encrypted or len(reader.pages) > 2000:
        return [Finding(relative, "pdf_requires_manual_review")]
    findings = terminology_findings(relative, "")
    for number, page in enumerate(reader.pages, 1):
        contents = page.extract_text() or ""
        findings.extend(terminology_findings(relative + f"#page={number}", contents))
    metadata = reader.metadata
    if metadata:
        findings.extend(terminology_findings(relative + "#metadata", str(metadata)))
    return findings


def screen_text(relative: str, text: str) -> list[Finding]:
    findings = []
    for name in ("provider_fingerprint", "acquisition_terminology"):
        if RULES[name].search(relative):
            findings.append(Finding(relative, name + "_in_path"))
    for line_number, line in enumerate(text.splitlines(), 1):
        for name, pattern in RULES.items():
            if pattern.search(line):
                findings.append(Finding(relative, name, line_number))
        for match in SECRET_ASSIGNMENT.finditer(line):
            if not PLACEHOLDER.fullmatch(match.group(1)):
                findings.append(Finding(relative, "literal_secret_assignment", line_number))
    return findings


def audit_file(root: Path, path: Path, *, max_bytes: int = MAX_BYTES, public_pdf_hashes: frozenset[str] = frozenset(), public_baseline: str = ORIGINAL_PUBLIC_BASE, reviewed_pdfs: dict[str, str] | None = None, reviewed_archives: dict[str, str] | None = None) -> list[Finding]:
    root = root.resolve()
    absolute = path if path.is_absolute() else root / path
    try:
        relative = absolute.relative_to(root).as_posix()
        absolute.resolve().relative_to(root)
    except ValueError:
        return [Finding("<outside-release-root>", "path_outside_root")]
    if absolute.is_symlink():
        return [Finding(relative, "symlink_requires_review")]
    if not absolute.is_file():
        return []
    if absolute.name == ".env" or (absolute.name.startswith(".env.") and not absolute.name.endswith((".example", ".template"))):
        return [Finding(relative, "environment_file")]
    if absolute.suffix.lower() == ".zip" and reviewed_archives and relative in reviewed_archives:
        if hashlib.sha256(absolute.read_bytes()).hexdigest() != reviewed_archives[relative]:
            return [Finding(relative, "reviewed_artifact_hash_mismatch")]
        return audit_archive(absolute, relative)
    if absolute.suffix.lower() in BLOCKED_SUFFIXES:
        return [Finding(relative, "data_or_binary_artifact")]
    if absolute.suffix.lower() == ".pdf":
        approved = known_public_pdf(root, absolute, public_baseline)
        if not approved and reviewed_pdfs and relative in reviewed_pdfs:
            approved = hashlib.sha256(absolute.read_bytes()).hexdigest() == reviewed_pdfs[relative]
        if not approved and public_pdf_hashes:
            approved = hashlib.sha256(absolute.read_bytes()).hexdigest() in public_pdf_hashes
        return [] if approved else [Finding(relative, "unreviewed_pdf")]
    if absolute.stat().st_size > max_bytes:
        return [Finding(relative, "oversized_file")]
    data = absolute.read_bytes()
    if b"\0" in data:
        return [Finding(relative, "binary_content")]
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return [Finding(relative, "non_utf8_content")]
    return screen_text(relative, text)


def audit_paths(root: Path, paths: Iterable[Path], **kwargs) -> list[Finding]:
    return [finding for path in paths for finding in audit_file(root, path, **kwargs)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Optional explicit files within the repository")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--base", default="HEAD", help="Compare tracked files with this commit; default HEAD")
    parser.add_argument("--all-public-text", action="store_true", help="Check source-neutral terminology in all tracked and nonignored new UTF-8 text, including the legacy baseline; binary files are listed as requiring separate review")
    parser.add_argument("--include-reviewed-documents", action="store_true", help="With --all-public-text, require exact PDF/ZIP review manifests and screen extracted PDF text plus all ZIP members; requires pypdf")
    parser.add_argument("--max-bytes", type=int, default=MAX_BYTES)
    parser.add_argument("--public-baseline", default=ORIGINAL_PUBLIC_BASE, help="Explicit reviewed public baseline for PDF allowlisting; defaults to the original public release")
    parser.add_argument("--public-pdf-sha256", action="append", default=[], help="Allow only an independently reviewed public PDF with this SHA-256")
    args = parser.parse_args()
    if args.include_reviewed_documents and not args.all_public_text:
        parser.error("--include-reviewed-documents requires --all-public-text")
    root = args.root.resolve()
    try:
        candidates = args.paths or (all_public_paths(root) if args.all_public_text else changed_paths(root, args.base))
        separate_review = []
        if args.all_public_text:
            findings = []
            pdfs = reviewed_artifacts(root, root / "docs/research/PDF_RELEASE_MANIFEST.json", "reviewed_public_pdf_derivatives", ".pdf") if args.include_reviewed_documents else {}
            archives = reviewed_artifacts(root, root / "docs/research/ARCHIVE_RELEASE_MANIFEST.json", "reviewed_public_archive_derivatives", ".zip") if args.include_reviewed_documents else {}
            for path in candidates:
                if not path.is_file():
                    continue
                relative = path.relative_to(root).as_posix()
                if args.include_reviewed_documents and path.suffix.lower() in {".pdf", ".zip"}:
                    approved = pdfs if path.suffix.lower() == ".pdf" else archives
                    if relative not in approved:
                        findings.append(Finding(relative, "unreviewed_document"))
                    elif path.suffix.lower() == ".pdf":
                        findings.extend(audit_pdf_text(path, relative))
                    else:
                        findings.extend(audit_archive(path, relative))
                    continue
                if path.is_symlink() or path.stat().st_size > args.max_bytes:
                    separate_review.append(relative)
                    continue
                try:
                    raw = path.read_bytes()
                    if b"\0" in raw:
                        separate_review.append(relative)
                        continue
                    contents = raw.decode("utf-8-sig")
                except UnicodeDecodeError:
                    separate_review.append(relative)
                    continue
                findings.extend(terminology_findings(relative, contents))
            if args.include_reviewed_documents:
                findings.extend(Finding(relative, "unreviewed_nontext_file") for relative in separate_review)
        else:
            pdfs = reviewed_artifacts(root, root / "docs/research/PDF_RELEASE_MANIFEST.json", "reviewed_public_pdf_derivatives", ".pdf")
            archives = reviewed_artifacts(root, root / "docs/research/ARCHIVE_RELEASE_MANIFEST.json", "reviewed_public_archive_derivatives", ".zip")
            findings = audit_paths(root, candidates, max_bytes=args.max_bytes, public_pdf_hashes=frozenset(args.public_pdf_sha256), public_baseline=args.public_baseline, reviewed_pdfs=pdfs, reviewed_archives=archives)
    except (OSError, RuntimeError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        print(json.dumps({"status": "error", "error_type": type(exc).__name__}))
        return 2
    print(json.dumps({"status": "fail" if findings else "pass", "files_checked": len(candidates) - len(separate_review), "findings": [asdict(f) for f in findings], "separate_review": separate_review, "scope": ("All public text plus reviewed PDF text and ZIP members" if args.include_reviewed_documents else "All public UTF-8 text; terminology only") if args.all_public_text else ("explicit files" if args.paths else f"Git changes relative to {args.base} and untracked nonignored files"), "limitations": "Heuristic screening; does not prove absence of encoded secrets, personal data, source fingerprints or legal restrictions. PDF extraction is not OCR. Separate-review files are not text-screened. Git history is outside scope."}, indent=2))
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
