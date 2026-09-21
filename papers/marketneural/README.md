# MarketNeural research monograph

**Author: Ghaffar Masomi**

Read [Learning Time in Moving Markets](marketneural-thesis.pdf), the thesis-style
PDF, or its [assembled Markdown manuscript](manuscript.md). Fifteen chapters and
roughly 33,000 words combine the mathematical method with the implemented system,
real image examples, architecture diagrams, historical results and reproducible
engineering checks.

The chapters cover survival foundations, temporal contracts, feature stores,
generative image/text enrichment, semantic vectors, Stage 0/1/2, the later K8
network, meta-tuning, leakage, recency weighting, operational recovery, implemented
agentic decisions, existing spam controls and a separately labeled fraud extension.

The controlled real-data comparison is **not run**. Historical cohorts and public
synthetic results are reported separately; neither establishes neural superiority.
Raw-plus-whitened vectors are an operator-described design for which a fitted
transform artifact was not recovered; the inspected selected network consumes one
listing-text vector. That distinction remains explicit.

## Rebuild

Install the project's `paper` extra and [Pandoc](https://pandoc.org/installing.html)
and [Tectonic](https://tectonic-typesetting.github.io/en-US/install/), then run from
the repository root:

```sh
python -m pip install -e ".[paper]"
python scripts/build_thesis.py --pandoc pandoc --tectonic tectonic
```

The reviewed build used Pandoc 3.11 and Tectonic 0.17.0. The first typesetting run
may download a public TeX bundle. ReportLab creates the title page; LaTeX typesets
the mathematical body. Source chapters, aggregate JSON, BibTeX and plotting code
are editable. Intermediate files stay in ignored `results/thesis/`.

[Research coverage](RESEARCH_COVERAGE.md) maps nine historical PDFs and private
evidence families. The [figure catalogue](FIGURE_CATALOG.json) identifies selected
historical graphics and their interpretation limits. Public image examples carry
[pairing provenance](../../research/thesis_evidence/image_case_studies.json).
They are selected illustrations, not an image-model accuracy benchmark.

The underlying datasets are proprietary and remain outside the public repository.
Researchers may [request controlled access](../../DATA_ACCESS.md); approval and
terms must be agreed separately.

The [release review](../../docs/research/THESIS_REVIEW.md) records agent reviews and
checks. [PDF](../../docs/research/PDF_RELEASE_MANIFEST.json) and
[image](../../docs/research/IMAGE_RELEASE_MANIFEST.json) manifests bind review to
exact artifact bytes. Rebuilt artifacts require renewed inspection. Current-tree
anonymization does not rewrite earlier Git history.
