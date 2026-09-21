"""Build the evidence monograph from reviewed chapters and aggregate figures.

Requires Pandoc and Tectonic executables, plus the project's `paper` extra.
No private data, network services, trained weights or credentials are inputs.
Tectonic may download its public TeX bundle on first use.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "papers/marketneural"
BUILD = ROOT / "results/thesis"
CHAPTERS = [
    "papers/marketneural/chapters/00_orientation.md",
    "papers/marketneural/chapters/01_foundations.md",
    "papers/marketneural/chapters/02_temporal_platform.md",
    "research/feature_store_ddl/METHODS.md",
    "papers/marketneural/chapters/03_generative_enrichment.md",
    "papers/marketneural/chapters/04_image_case_studies.md",
    "research/cascade/METHODS.md",
    "papers/marketneural/chapters/05_multimodal_comparison.md",
    "research/leakage/LEAKAGE_STUDY.md",
    "papers/marketneural/chapters/06_empirical_evidence.md",
    "papers/marketneural/chapters/07_operations.md",
    "papers/marketneural/chapters/08_discussion.md",
    "papers/marketneural/chapters/10_agentic_decisions.md",
    "papers/marketneural/chapters/11_fraud_and_spam.md",
    "papers/marketneural/chapters/09_reproduction.md",
]
NEW_FIGURES = {
    "00_orientation.md": [("r01_platform_method", "The data platform as part of the research method. This synthesis distinguishes content observation, enrichment, decision-time eligibility, model fitting and operational evidence; arrows are logical dependencies, not measured throughput.")],
    "07_operations.md": [("r02_platform_scale", "Retained database snapshot dated 28 April 2026: 55,260 listing rows and 240,622 image assets within 29.1 GB. The categories have different units and coverage; they must not be added as independent observations. Source: operational_counts.json.")],
    "LEAKAGE_STUDY.md": [("r03_missingness_shortcut", "Lifecycle-dependent missingness in the earlier frozen validation and holdout cohorts. Every observed FAST72 positive belongs to the missing-pattern group. Numerators and denominators are shown; this demonstrates a shortcut opportunity, not a numerical causal estimate of metric inflation."), ("r08_tail_overlap", "Key overlap in a later rebuilt source: both exported tail cohorts are contained in its training table. This reconstruction does not establish that the original locked training matrix was identical.")],
    "06_empirical_evidence.md": [("r04_cohort_shift", "Changing composition and operating characteristics under the locked short-horizon policy. Validation n=584, holdout n=748, recent n=283. The recent cohort is wholly nested within holdout; it is not an independent prospective replication. Prevalence is conditional on eligible labeled records, not market-wide turnover.")],
    "METHODS.md@cascade": [("r05_multimodal_network", "Selected later K8 network: structured features, listing text, eight image vectors and eight image-report vectors form 40 input tokens. The 17,145,736-parameter model mixes dense expert survival distributions and has a separate scalar head. This later model is distinct from the older three-stage cascade."), ("r06_cascade_routes", "Historical ordered policy routing. Stage 0 tests the long tail first, Stage 1 applies the 168-hour gate, and Stage 2 applies the 72-hour gate. The output names denote policy buckets; they are not a partition obtained by subtracting calibrated probabilities from one common survival curve.")],
    "02_temporal_platform.md": [("r09_recency_weighting", "Continuous age weighting implied by the retained 30-day and 23-day half-life configurations, before normalization and other sample weights. This is a visualization of implemented weight functions, not an empirical estimate of market decay or proof that a 60-day window is optimal.")],
    "METHODS.md": [("r11_feature_contracts", "Information dependencies in the governed research design. Outcomes support labels and later audits; they do not authorize future information in a historical feature snapshot. Four experimental roles depict the full proposed protocol; the implemented compact benchmark uses three chronological blocks. Historical SQL and the portable contract have separately documented execution scopes.")],
    "03_generative_enrichment.md": [("r12_visual_measurement", "Implemented visual measurement pipeline summarized across task-specific enrichment, validation, frozen encoders and aligned image/report slots. Structured generation controls output form and field ownership; it does not certify perceptual correctness.")],
    "05_multimodal_comparison.md": [("r13_training_selection", "Historical fit, selection and evaluation roles. SVAL is used for model and policy selection; an untouched evaluation cohort must remain outside that feedback loop. The proposed real-data protocol additionally separates calibration, as described in the text.")],
    "10_agentic_decisions.md": [("r10_agentic_architecture", "The inspected agentic decision architecture. The model proposes one bounded action; deterministic policy and a separate operator-approved execution path determine what can be attempted. Rust services coordinate realtime state and a separate analytical query path. The diagram does not claim autonomous trading profitability.")],
}
HISTORICAL_ASSIGNMENT={"01_foundations.md":["H01","H02"],"02_temporal_platform.md":["H03"],"03_generative_enrichment.md":["H08"],"06_empirical_evidence.md":["H04","H05","H06","H07"]}

def run(args):
    subprocess.run([str(x) for x in args], cwd=ROOT, check=True)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def figure(caption,path):
    return f"\n\n![{caption}]({path.as_posix()}){{width=96%}}\n\n"

def assemble(allow_partial=False, pandoc="pandoc"):
    catalogue=json.loads((PAPER/"FIGURE_CATALOG.json").read_text(encoding="utf-8"))
    historic={f["id"]:f for f in catalogue["figures"]}
    pieces=[]; included=[]
    for rel in CHAPTERS:
        p=ROOT/rel
        if not p.exists():
            if allow_partial: continue
            raise FileNotFoundError(f"Required chapter missing: {rel}")
        content=p.read_text(encoding="utf-8")
        def resolve_image(match):
            label, target, attrs=match.groups()
            target_path=Path(target)
            if not target_path.is_absolute():
                candidates=[p.parent/target,ROOT/target]
                target_path=next((candidate.resolve() for candidate in candidates if candidate.exists()),target_path)
            # Typeset published vector figures through the paired generated PDF.
            paired_pdf=BUILD/"figures"/(target_path.stem+".pdf")
            if target_path.suffix.lower()==".svg" and paired_pdf.exists():
                target_path=paired_pdf
            return f"![{label}]({target_path.as_posix()})"+(attrs or "{width=96%}")
        content=re.sub(r"!\[([^\]]*)\]\(([^)]+)\)(\{[^}]+\})?",resolve_image,content)
        def resolve_link(match):
            label,target=match.groups()
            if re.match(r"^[a-z]+://|^#|^mailto:",target):
                return match.group(0)
            location,sep,anchor=target.partition('#')
            candidates=[p.parent/location,ROOT/location]
            resolved=next((candidate.resolve() for candidate in candidates if candidate.exists()),None)
            if resolved is None:
                raise FileNotFoundError(f"Broken chapter link in {rel}: {target}")
            return f"[{label}]({resolved.as_posix()}{sep}{anchor})"
        content=re.sub(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)",resolve_link,content)
        content=re.sub(r"^#\s+(?:\d+\.\s+)?", "# ", content, count=1)
        key=p.name+('@cascade' if 'cascade' in p.parts else '')
        for name,caption in NEW_FIGURES.get(key,[]):
            content+=figure(caption,BUILD/"figures"/(name+".pdf"))
        for hid in HISTORICAL_ASSIGNMENT.get(p.name,[]):
            f=historic[hid]
            if Path(f['image_path']).name in content:
                continue
            cap=f["caption"]+" "+f["claim_limits"]+f" Source: historical report, p. {f['page']} ({hid})."
            if hid=="H08": cap+=" Post-event reconciliation in this historical diagram does not authorize using future information in decision-time inputs."
            content+=figure(cap,ROOT/f["image_path"])
        pieces.append(content.strip())
        included.append({"path":rel,"sha256":digest(p),"words":len(content.split())})
    # Keep code links useful in the PDF without making local filesystem links.
    text="\n\n".join(pieces)
    text=text.replace("\\(","$").replace("\\)","$").replace("\\[","$$").replace("\\]","$$")
    for dash in ["\u2011","\u2013","\u2014"]: text=text.replace(dash,"-")
    text += "\n\n# References {-}\n\n::: {#refs}\n:::\n"
    pdf_text=re.sub(r"(?<!!)\[([^\]]+)\]\(("+re.escape(ROOT.as_posix())+r"/[^)]+)\)",
        lambda m:f"[{m[1]}](https://github.com/dutchgtr-pixel/t0-clip/blob/main/{m[2][len(ROOT.as_posix())+1:]})",text)
    (BUILD/"body.md").write_text("---\nnocite: |\n  @*\n---\n\n"+pdf_text,encoding="utf-8")
    # Public assembled source uses repository-relative figures; vector PDFs are build intermediates.
    public=text.replace((BUILD/"figures").as_posix()+"/", "figures/")
    public=re.sub(r"(figures/r\d+_[a-z_]+)\.pdf",r"\1.svg",public)
    public=public.replace(ROOT.as_posix()+"/", "../../")
    public=re.sub(r"\{width=\d+%\}","",public)
    if not allow_partial:
        bibliography_input="---\nnocite: |\n  @*\n---\n\n# References\n\n::: {#refs}\n:::\n"
        bibliography=subprocess.run([pandoc,"--from","markdown","--to","gfm","--citeproc","--bibliography",str(PAPER/"references.bib"),"--wrap=none"],
            input=bibliography_input,capture_output=True,text=True,encoding="utf-8",check=True).stdout
        public=public.replace("# References {-}\n\n::: {#refs}\n:::\n",bibliography)
        (PAPER/"manuscript.md").write_text("# MarketNeural: Learning Time in Moving Markets\n\nGhaffar Masomi\n\nTechnical edition 0.3 / 22 September 2026.\n\n"+public,encoding="utf-8")
    (BUILD/"source-manifest.json").write_text(json.dumps(included,indent=2)+"\n",encoding="utf-8")
    return included

def cover(path):
    c=canvas.Canvas(str(path),pagesize=A4,pageCompression=1,invariant=1)
    w,h=A4;navy=HexColor("#17334a");teal=HexColor("#197f83")
    c.setFillColor(navy);c.rect(0,h-20,w,20,fill=1,stroke=0)
    c.setFont("Helvetica",10);c.drawString(65,h-85,"MARKETNEURAL / RESEARCH MONOGRAPH")
    c.setFont("Times-Bold",43);c.drawString(65,h-187,"Learning Time")
    c.drawString(65,h-239,"in Moving Markets")
    c.setFillColor(teal);c.rect(65,h-280,65,3,fill=1,stroke=0)
    c.setFillColor(navy);c.setFont("Times-Roman",19)
    for i,line in enumerate(["A multimodal survival system,", "a three-stage decision cascade,", "and the engineering of temporal evidence"]): c.drawString(65,h-328-i*28,line)
    c.setFont("Helvetica",11);c.drawString(65,229,"Ghaffar Masomi")
    c.setFont("Helvetica",10);c.drawString(65,207,"Technical edition 0.3 / 22 September 2026")
    c.setStrokeColor(HexColor("#ccd7de"));c.line(65,184,w-65,184)
    c.setFont("Helvetica",9)
    for i,line in enumerate(["Multimodal learning, temporal evidence and operational engineering.","Historical observations, released algorithms and reproducible tests."]): c.drawString(65,163-i*15,line)
    c.setFillColor(teal);c.setFont("Helvetica",9);c.drawString(65,69,"PUBLIC RESEARCH / t0-clip")
    c.setTitle("MarketNeural: Learning Time in Moving Markets");c.setAuthor("Ghaffar Masomi")
    c.save()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pandoc",default="pandoc");p.add_argument("--tectonic",default="tectonic")
    p.add_argument("--allow-partial",action="store_true",help="Local layout development only; never a release build")
    args=p.parse_args();BUILD.mkdir(parents=True,exist_ok=True)
    run([sys.executable,ROOT/"scripts/plot_thesis.py","--output",BUILD/"figures"])
    if not args.allow_partial:
        (PAPER/"figures").mkdir(exist_ok=True)
        for svg in (BUILD/"figures").glob("*.svg"):
            shutil.copyfile(svg,PAPER/"figures"/svg.name)
    sources=assemble(args.allow_partial,args.pandoc)
    front=BUILD/"frontmatter.tex"
    front.write_text(r"""\chapter*{Abstract}
\addcontentsline{toc}{chapter}{Abstract}
Predicting market exposure duration requires more than fitting a model to completed outcomes. Listings, photographs, descriptions, enrichment records and labels become available on different clocks. Their asynchronous construction creates both predictive information and opportunities for temporal leakage. This monograph examines a substantial operational platform and its transition from structured survival models to multimodal neural models and a three-stage decision cascade.

The retained evidence includes a 29.1 GB database snapshot containing 55,260 listing rows and 240,622 image assets, scheduler records, restoration reports, neural configurations, saved predictions, historical technical papers and leakage investigations. The platform completed more than 70,000 successful Airflow workflow runs over its operating lifetime. A retained scheduler snapshot contains 15,360 runs from a subset of that history.

The contribution is a detailed, inspectable account of feature construction, constrained generative enrichment, temporal contracts, survival objectives, ensemble and policy selection, operational validation and failure analysis. The public release provides historical numerical algorithms, portable training and inference adapters, SQL contracts, aggregate evidence and a reproducible comparison harness. A central case study shows how outcome-dependent feature availability can bypass otherwise careful temporal controls. Later evidence also exposes dependent diagnostic cohorts and sensitivity to zero-duration observations.

Historical tail-screening results improve from F1 0.8462 for the earlier XGBoost AFT model to 0.9209 for a neural meta-ensemble, with precision increasing from 0.8314 to 0.9802. These results retain their respective 941-row and 964-row evaluation cohorts. The archive also preserves hundreds of neural trial outputs and thousands of ensemble candidates. The manuscript presents this completed experimental development alongside a protocol for reconstructing matched comparisons and testing how the recorded advantage transfers to later market conditions. Public synthetic experiments separately verify the comparison software.

\bigskip
\noindent\textbf{Keywords:} survival analysis; multimodal learning; feature stores; temporal leakage; generative enrichment; marketplace dynamics; reproducibility.
\clearpage
\chapter*{Reader's guide}
\addcontentsline{toc}{chapter}{Reader's guide}
This monograph presents the survival models, data platform and operational methods developed by the MarketNeural research project. The chapters connect mathematical definitions with implementation, historical measurements and reproducible code.

Evidence classes are kept separate: retained artifact measurements, historical report statements, source-code behavior, platform lifetime operating totals, newly executed public tests and proposed experiments. A figure labeled historical retains the limits of its original cohort. Illustrative diagrams are not experimental results. Anonymous case examples show model outputs, not independently certified damage diagnoses.

The three-stage historical cascade and the later independent K8 network are distinct configurations. Their architecture sizes, training populations and routing rules must not be combined into a single fictional model. Likewise, validation used for selection is not an untouched final test.

Readers interested in the scientific method should start with the foundations, temporal platform and leakage chapters. Readers reproducing the software should use the released-methods and reproduction chapters alongside the repository. The engineering chapter explains why data readiness, versioning, scheduling and recoverability were prerequisites for the research.
\clearpage
""",encoding="utf-8")
    tex=BUILD/"body.tex"
    run([args.pandoc,BUILD/"body.md","--from","markdown+tex_math_dollars+raw_tex", "--to","latex","--standalone",
        "--top-level-division=chapter","--number-sections","--toc","--toc-depth=2","--citeproc","--bibliography",PAPER/"references.bib",
        "--metadata","reference-section-title: References",
        "--include-in-header",PAPER/"typesetting/header.tex","--include-before-body",front,
        "--lua-filter",PAPER/"typesetting/wrap_code.lua",
        "-V","documentclass=book","-V","classoption=oneside,openany","-V","papersize=a4","-V","fontsize=11pt",
        "-V","geometry=left=28mm,right=25mm,top=25mm,bottom=26mm,headsep=9mm",
        "-V","linestretch=1.10","-V","lof=true","-V","colorlinks=true","-V","lang=en-GB","-o",tex])
    run([args.tectonic,"--keep-logs","--keep-intermediates","--outdir",BUILD,tex])
    cover(BUILD/"cover.pdf")
    output=(BUILD/"partial-thesis.pdf") if args.allow_partial else (PAPER/"marketneural-thesis.pdf")
    writer=PdfWriter();writer.append(BUILD/"cover.pdf");writer.append(BUILD/"body.pdf")
    writer.add_metadata({"/Title":"MarketNeural: Learning Time in Moving Markets","/Author":"Ghaffar Masomi","/Subject":"Evidence-led research monograph, technical edition 0.3"})
    with output.open("wb") as stream:writer.write(stream)
    receipt={"schema_version":1,"output":output.relative_to(ROOT).as_posix(),"sha256":digest(output),"pages":len(PdfReader(output).pages),
        "partial_development_build":args.allow_partial,"chapters":sources,"word_count":sum(s["words"] for s in sources),
        "build_tools":{tool:subprocess.check_output([getattr(args,tool),"--version"],text=True).splitlines()[0] for tool in ["pandoc","tectonic"]},
        "scope":"Public inputs only; historical images and aggregates are not new model experiments."}
    (BUILD/"build-receipt.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in receipt.items() if k!='chapters'},indent=2))

if __name__=="__main__": main()
