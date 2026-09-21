"""Replot audited aggregate evidence; never access private rows or services."""
from pathlib import Path
import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAVY, TEAL, GOLD, RED, GREY = "#17334a", "#197f83", "#bc8736", "#b35151", "#617181"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": NAVY, "text.color": NAVY, "axes.titleweight": "bold",
    "svg.hashsalt": "marketneural-thesis", "pdf.fonttype": 42})

def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))

def save(fig, name, out):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / (name + ".pdf"), bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(out / (name + ".svg"), bbox_inches="tight", metadata={"Date": None})
    fig.savefig(out / (name + ".png"), bbox_inches="tight", dpi=160)
    plt.close(fig)

def canvas(height=4):
    fig, ax = plt.subplots(figsize=(9, height))
    ax.set(xlim=(0, 10), ylim=(0, height)); ax.axis("off")
    return fig, ax

def box(ax, x, y, w, h, title, text="", color=TEAL):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.07,rounding_size=0.07", fc="#f5f8fa",ec=color,lw=1.2))
    ax.text(x+w/2, y+h*(.79 if text.count('\n')>=2 else .68) if text else y+h/2, title, ha="center", va="center", weight="bold",fontsize=10,color=color)
    if text: ax.text(x+w/2,y+h*.3,text,ha="center",va="center",fontsize=8 if text.count('\n')>=2 else 8.5,linespacing=1.3)

def arrow(ax,a,b,color=GREY):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=12,lw=1.2,color=color))

def make_figures(out):
    ops=read("research/thesis_evidence/operational_counts.json")
    ev=read("research/leakage/aggregate_evidence.json")
    fig,ax=plt.subplots(figsize=(8,4.2))
    labels=["Listing rows","Image assets","Image feature rows","Text enrichment rows","Text vector rows","Image vector rows","Post-event audit rows"]
    values=[x["count"] for x in ops["record_counts"]]
    ax.barh(labels[::-1],values[::-1],color=[GREY,TEAL,TEAL,TEAL,TEAL,NAVY,NAVY],height=.62)
    for i,v in enumerate(values[::-1]): ax.text(v+3600,i,f"{v:,}",va="center",fontsize=9)
    ax.set_xlim(0,290000); ax.set_xlabel("Stored records or assets (different units; do not sum)")
    ax.ticklabel_format(style="plain",axis="x"); ax.grid(axis="x",alpha=.15);ax.set_axisbelow(True)
    fig.tight_layout();save(fig,"r02_platform_scale",out)

    fig,ax=plt.subplots(figsize=(8,4))
    x=np.arange(5); width=.36
    for j,c in enumerate(ev["lifecycle_availability"]):
        b=c["duration_buckets"]; rates=[100*v["missing"]/v["total"] for v in b]
        xpos=x+(j-.5)*width
        ax.bar(xpos,rates,width,label=f"{['Validation','Holdout'][j]} (n={c['rows']})",color=[TEAL,NAVY][j])
        for xx,yy,v in zip(xpos,rates,b): ax.text(xx,yy+2,f"{v['missing']}/{v['total']}",ha="center",fontsize=8)
    ax.set_xticks(x,[v["hours"] for v in b]);ax.set_xlabel("Observed duration band (hours)")
    ax.set_ylabel("Missing pattern (%)");ax.set_ylim(0,125);ax.set_yticks([0,25,50,75,100]);ax.legend(frameon=False,ncol=2,loc="upper right")
    fig.tight_layout();save(fig,"r03_missingness_shortcut",out)

    comparison=read("research/thesis_evidence/historical_neural_comparison.json")
    by_id={row["model_id"]:row for row in comparison["historical_comparison"]["results"]}
    rows=[by_id[key] for key in ["xgboost_aft","neural_top20","neural_seed50","neural_meta"]]
    fig,ax=plt.subplots(figsize=(8.3,4.2))
    vals=[row["f1"] for row in rows]
    labels=["XGBoost AFT\nn=941", "Neural top-20\nn=964", "Neural optimized\nn=964", "Neural meta\nn=964"]
    ax.bar(range(4),vals,color=[GOLD,"#6b969c",TEAL,NAVY],width=.6)
    for x0,value in enumerate(vals):ax.text(x0,value+.018,f"{value:.4f}",ha="center",fontsize=11,weight="bold")
    ax.set_xticks(range(4),labels);ax.tick_params(axis="x",labelsize=9)
    ax.set_ylabel("Recorded F1");ax.set_ylim(0,1.14);ax.set_yticks(np.arange(0,1.01,.2))
    ax.set_title("21-day tail screening: historical model results",fontsize=12,pad=18)
    ax.annotate("",xy=(0,1.035),xytext=(3,1.035),arrowprops={"arrowstyle":"|-|","color":TEAL,"lw":1.2})
    ax.text(1.5,1.065,"Neural meta vs earlier AFT: +7.47 F1 points",ha="center",fontsize=10,color=TEAL)
    ax.grid(axis="y",alpha=.15);ax.set_axisbelow(True)
    fig.tight_layout();save(fig,"r14_historical_neural_advantage",out)

    fig,axes=plt.subplots(1,3,figsize=(9,3.3),sharey=True)
    for ax,key,title in zip(axes,["prevalence","precision","recall"],["FAST72 prevalence","Selected precision","Selected recall"]):
        cohorts=[ev["locked_policy"][k]["overall"] for k in ["sval","holdout","forward"]]
        vals=[v[key]*100 for v in cohorts]
        ax.bar(["Validation","Holdout","Recent*"],vals,color=[TEAL,NAVY,GOLD],width=.65)
        for i,v in enumerate(vals):ax.text(i,v+2,f"{v:.1f}%",ha="center",fontsize=9)
        ax.set_ylim(0,105);ax.set_title(title,fontsize=10);ax.tick_params(axis="x",labelsize=8)
    axes[0].set_ylabel("Percent");fig.tight_layout();save(fig,"r04_cohort_shift",out)

    fig,ax=canvas(4.7)
    box(ax,.2,3.3,2.2,1,"Structured","239 numeric fields\n81 categorical fields\n16 learned tokens")
    box(ax,2.7,3.3,1.9,1,"Text","1 x 768 vector\n8 tokens")
    box(ax,4.9,3.3,2.2,1,"Images","8 x 512 vectors\n8 tokens")
    box(ax,7.4,3.3,2.3,1,"Image reports","8 x 768 vectors\n8 tokens")
    for x0 in [1.3,3.65,6,8.55]:arrow(ax,(x0,3.2),(5,2.8))
    box(ax,1.5,1.8,7,1,"Latent cross-attention and fusion","40 input tokens | width 256 | 32 latents | 6 blocks | 8 heads")
    arrow(ax,(5,1.7),(5,1.32))
    box(ax,.8,.15,4,1.1,"Survival mixture","7 dense experts | 128 bins | 504 hours\nMixture of expert survival distributions")
    box(ax,5.2,.15,4,1.1,"Separate scalar head","Slow-positive historical convention\nFAST72 = 1 - sigmoid(logit)",GOLD)
    arrow(ax,(5,1.4),(2.8,1.3));arrow(ax,(5,1.4),(7.2,1.3))
    save(fig,"r05_multimodal_network",out)

    fig,ax=canvas(5.8)
    stages=[(4.55,"Stage 0: long tail","p(tail) >= threshold?"),(2.85,"Stage 1: 168 h","p(fast168) < threshold?"),(1.15,"Stage 2: 72 h","p(fast72) >= threshold?")]
    outputs=["TAIL_21PLUS","SLOW_168PLUS","FAST_72H"]
    for (y,title,txt),outlabel in zip(stages,outputs):
        box(ax,.4,y,5.2,.9,title,txt);box(ax,7,y,2.6,.9,outlabel,color=GOLD)
        arrow(ax,(5.7,y+.45),(6.9,y+.45));ax.text(6.3,y+.65,"yes",ha="center",fontsize=9)
    for upper,lower in [(4.55,2.85),(2.85,1.15)]:
        arrow(ax,(3,upper-.06),(3,lower+.98));ax.text(3.2,(upper+lower+.9)/2,"no",va="center",fontsize=9)
    arrow(ax,(3,1.08),(3,.68));box(ax,1.5,.12,3,.45,"MID_72_168H",color=GOLD)
    ax.text(3.2,.81,"no",va="center",fontsize=9)
    save(fig,"r06_cascade_routes",out)

    fig,ax=canvas(5.1)
    row1=[(.15,"Observed content","Versioned images and text"),(3.5,"Constrained enrichment","Schema, prompts, nulls"),(6.85,"Validated representations","Attributes and frozen vectors")]
    for x,title,txt in row1:box(ax,x,3.65,3,1,title,txt)
    arrow(ax,(3.22,4.15),(3.42,4.15));arrow(ax,(6.57,4.15),(6.78,4.15))
    row2=[(.15,"Decision-time snapshot","Availability and outcome boundary"),(3.5,"Model and policy","Train / select / lock / evaluate"),(6.85,"Operational decision","Recorded score and version")]
    for x,title,txt in row2:box(ax,x,1.8,3,1,title,txt)
    arrow(ax,(8.35,3.57),(8.35,3.08));arrow(ax,(8.35,3.08),(1.65,3.08));arrow(ax,(1.65,3.08),(1.65,2.87))
    arrow(ax,(3.22,2.3),(3.42,2.3));arrow(ax,(6.57,2.3),(6.78,2.3))
    box(ax,.7,.15,8.6,.95,"Evidence and recovery","Scheduler history | content versions | tests | missingness audits | restoration proofs",NAVY)
    for x in [1.65,5,8.35]:arrow(ax,(x,1.72),(x,1.17))
    save(fig,"r01_platform_method",out)

    fig,ax=plt.subplots(figsize=(8,3.8))
    splits=ev["split_overlap"]["splits"]
    lab=["Validation","Holdout","Tail validation","Tail evaluation"]
    total=[v["rows"] for v in splits]; overlap=[v["also_in_training"] for v in splits]
    ax.barh(lab,total,color="#dee7ed",label="Exported cohort rows")
    ax.barh(lab,overlap,color=RED,label="Also in rebuilt training table")
    for i,(n,k) in enumerate(zip(total,overlap)):ax.text(n+25,i,f"{k:,} / {n:,}",va="center",fontsize=9)
    ax.set_xlim(0,1900);ax.invert_yaxis();ax.set_xlabel("Rows (later rebuilt source; not the original locked fit matrix)")
    ax.legend(frameon=False,loc="lower right",fontsize=8);fig.tight_layout();save(fig,"r08_tail_overlap",out)

    fig,ax=plt.subplots(figsize=(8,3.5))
    age=np.arange(0,121)
    for half,color,label in [(30,TEAL,"Stage 0 saved setting: 30 days"),(23,NAVY,"Stage 1 saved setting: 23 days")]:
        weight=2.**(-age/half);ax.plot(age,weight,color=color,lw=2,label=label)
        ax.scatter([60],[2.**(-60/half)],color=color,zorder=3)
    ax.axvline(60,color=GREY,lw=.8,ls="--");ax.set(xlabel="Observation age (days)",ylabel="Age-only relative weight",xlim=(0,120),ylim=(0,1.05))
    ax.legend(frameon=False,fontsize=9);ax.grid(alpha=.15);fig.tight_layout();save(fig,"r09_recency_weighting",out)

    fig,ax=canvas(5.5)
    box(ax,.25,4.1,2.7,1,"Evidence packet","Listing | prices | thread\nReadiness | policy | source refs")
    box(ax,3.65,4.1,2.7,1,"Proposal model","One typed JSON action\nNo direct browser actuator")
    box(ax,7.05,4.1,2.7,1,"Policy validator","Allowed action | campaign\nOffer ceiling | evidence refs")
    arrow(ax,(3.02,4.6),(3.58,4.6));arrow(ax,(6.42,4.6),(6.98,4.6))
    box(ax,7.05,2.25,2.7,1,"Operator approval","Exact staged draft\nFresh state and duplicate checks",GOLD)
    arrow(ax,(8.4,4.03),(8.4,3.32))
    box(ax,3.65,2.25,2.7,1,"Execution service","Authenticated stateful runtime\nAttempt / uncertain / complete")
    arrow(ax,(6.98,2.75),(6.42,2.75))
    box(ax,.25,2.25,2.7,1,"Durable evidence","Run | event | outcome\nReconcile observed remote state")
    arrow(ax,(3.58,2.75),(3.02,2.75))
    box(ax,.6,.35,8.8,1,"Rust realtime and analytical services","Cache, per-run serialization, refresh queues and SSE | Separate guarded analytical queries",NAVY)
    for x in [1.6,5,8.4]:arrow(ax,(x,2.18),(x,1.42))
    save(fig,"r10_agentic_architecture",out)

    fig,ax=canvas(5.5)
    box(ax,.4,4.1,4.2,1,"Versioned source facts","Event time | observed time | content hash")
    box(ax,5.4,4.1,4.2,1,"Outcome ledger","Event / censor time | label maturity",GOLD)
    box(ax,.4,2.5,4.2,1,"Feature construction","Structured | images | text | prior anchors")
    arrow(ax,(2.5,4.03),(2.5,3.58))
    box(ax,.4,.85,4.2,1,"Certified decision-time snapshot","Availability cutoff | row contract | fingerprint")
    arrow(ax,(2.5,2.43),(2.5,1.93))
    box(ax,5.4,.85,4.2,1,"Temporal experimental export","Fit / select / calibrate / final test\nCensor outcomes at each boundary")
    arrow(ax,(4.67,1.35),(5.33,1.35));arrow(ax,(7.5,4.03),(7.5,1.93))
    ax.text(7.5,2.7,"Labels join after\nfeature eligibility is fixed",ha="center",va="center",fontsize=10,color=GOLD)
    ax.text(5,.15,"Dependency summary of the released contracts; historical layers require their documented prerequisites.",ha="center",fontsize=8.5)
    save(fig,"r11_feature_contracts",out)

    fig,ax=canvas(5.6)
    box(ax,.3,4.15,2.7,1,"Versioned image","Stable identity and slot\nQuality / role / availability")
    box(ax,3.65,4.15,2.7,1,"Task-specific prompt","Taxonomy and bounded fields\nUnknowns and explicit evidence")
    box(ax,7,4.15,2.7,1,"Validated response","Parse | field ownership\nRange and conflict checks")
    arrow(ax,(3.07,4.65),(3.58,4.65));arrow(ax,(6.42,4.65),(6.93,4.65))
    box(ax,.3,2.1,2.7,1,"Frozen visual encoder","512-dimensional image vector")
    box(ax,3.65,2.1,2.7,1,"Structured attributes","Role | quality | damage\nMissingness and version")
    box(ax,7,2.1,2.7,1,"Report encoder","Structured report text\n768-dimensional vector")
    arrow(ax,(1.65,4.08),(1.65,3.17));arrow(ax,(8.35,4.08),(8.35,3.17));arrow(ax,(7.4,4.08),(5,3.17))
    box(ax,1.8,.25,6.4,1,"Aligned multimodal survival inputs","Up to eight role-aware image slots | Reports remain generated evidence",NAVY)
    for x in [1.65,5,8.35]:arrow(ax,(x,2.03),(5,1.32))
    save(fig,"r12_visual_measurement",out)

    fig,ax=canvas(4.5)
    for x,title,txt,col in [(.2,"TRAIN","Fit weights and transforms\nRecency weights / loss objectives",TEAL),(3.6,"SVAL","Checkpoint + meta policy\nThresholds and calibration\nSelection constraints",GOLD),(7,"EVAL","Locked predictions and metrics\nNo feedback into selection",NAVY)]:
        box(ax,x,2.45,2.8,1.3,title,txt,col)
    arrow(ax,(3.07,3.1),(3.53,3.1));arrow(ax,(6.47,3.1),(6.93,3.1))
    box(ax,.9,.45,8.2,1,"Selection boundary is part of the model","Repeated SVAL tuning can overfit selection.\nNeural seeds, ensemble-search seeds and refits differ.",NAVY)
    arrow(ax,(5,2.38),(5,1.52));save(fig,"r13_training_selection",out)

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"results/thesis/figures")
    make_figures(parser.parse_args().output)
