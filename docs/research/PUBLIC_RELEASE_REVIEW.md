# Current-tree publication review

The public reference uses generic observation, device and source-record names.
Legacy SQL schemas, columns, Python and Go identifiers, environment examples,
Compose labels, filenames and their documentation references were generalized
consistently. These are reference definitions: adapting an existing deployment
requires mapping its own identifiers and applying a reviewed migration. Renaming
these public examples does not migrate a running database or reproduce a private
deployment configuration.

Database connection examples use explicit caller-supplied DSN placeholders.
The preserved historical model excerpt contains architecture and loss definitions,
source hashes and dimensions. It contains no trained weights, fitted category
values, records, source URLs, credentials or private filesystem paths.

The source archive review covers all twelve UTF-8 members, their member paths and
their actual content. Members are documentation and SQL definitions, including
view creation and certification routines. The INSERT statements generate audit
baselines or registry entries from the caller's database; they do not embed record
rows. No CSV, model weight, database dump, binary member, external URL or COPY data
block is included. The reviewed archive's exact path and hash are recorded in
[the archive manifest](ARCHIVE_RELEASE_MANIFEST.json).

The PDF review is recorded separately in [the PDF manifest](PDF_RELEASE_MANIFEST.json).
Review manifests describe explicit reviewed bytes. Committing a file does not
automatically make it approved. The original public commit used for the release
diff and unchanged-document baseline is pinned in the audit script.

Run both gates after installing the development dependencies:

```sh
python scripts/audit_public_release.py --base dc41f873b82ab5c1ac07862c7610bcf2d6a40e62
python scripts/audit_public_release.py --all-public-text --include-reviewed-documents
```

The first gate screens changed and new deliverables for data/binary artifacts,
credential patterns, private paths and source fingerprints. The second scans
all current public text for source-neutral terminology, validates document hashes,
extracts PDF text and checks every source-archive member. Git-ignored data,
environments and generated result directories stay outside publication scope.

These are heuristic checks. They do not establish legal rights, discover every
encoded credential or personal record, or perform OCR on raster images in PDFs.
They audit the current file tree; earlier public Git history remains unchanged.
Aggregate benchmark outputs also need review before publication: caller-supplied
feature names and configuration fields are retained even when record rows and
input paths are excluded.
