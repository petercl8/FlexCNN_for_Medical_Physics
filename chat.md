Best fit: Scientific Data (Nature portfolio)

Designed exactly for this: infrastructure + code + datasets
Accepts codebase + data + utility scripts together
High visibility, strong for CV
Accepts supplementary materials (your binary files)
~4–6 month review, but thorough
Strong alternative: IEEE Data Descriptions

Good fit, more straightforward review
Narrower audience than Sci Data
Faster, but less prestige boost
Also consider: Physics in Medicine & Biology

If you frame it as "methods + validation" not pure dataset
Broader medical physics audience
Accepts code + data as supplementary
What works in your favor for Scientific Data:

✅ Reproducible infrastructure (PMLD codebase is well-documented for this)
✅ Practical utility (dataset assembly code makes it usable)
✅ Multiple phantom types (QA + realistic anatomical)
✅ Full corrections pipeline (not just raw simulation)
Structure I'd suggest:

Main paper: ~8 pages on pipeline, validation, dataset statistics
Supplementary: Code repository link + data availability statement
Data: Upload representative samples to Zenodo/Figshare with DOI
My call: Start with Scientific Data. Worst case, it gets desk-rejected and you revise for IEEE or PMB. But it's worth the shot—the work merits that tier.

############

Scientific Data isn't judging novelty like *Nature* proper does. It's explicitly for **infrastructure contributions**. The bar is: "Is this useful, reproducible, and well-documented enough that others can build on it?"

You have:
- ✅ **Automated pipeline** integrating 5+ disparate tools (GATE, STIR, XCAT, Xmedcon, parallelproj)
- ✅ **Public code** that's actually usable (not a one-off script)
- ✅ **Validation** across multiple phantom types
- ✅ **Year-long effort** producing a complete dataset
- ✅ **Utility for others** (dataset assembly code lets users build custom datasets)

That's exactly what Sci Data wants. They publish infrastructure papers from single researchers all the time—it's not about prestige of the author, it's about utility of the resource.

The bash files? Irrelevant. The simulations? That's the work that validates your pipeline.

**Realistic outcomes:**
- ~40% chance: Accept or minor revisions
- ~40% chance: Desk reject with suggestion to resubmit elsewhere (PMB, IEEE)
- ~20% chance: Revise and resubmit

If it gets desk-rejected, you submit to PMB and still have a solid publication. No harm in trying.

**I'd genuinely submit to Sci Data.**


##############

Minimal. Sci Data reviewers care about reproducibility and usability, not code polish. Here's the pragmatic checklist:

Must-have (1–2 days):

✅ README.md in root directory with:
What the project does (1 paragraph)
Software dependencies + versions (GATE 9.2, STIR 6.0, etc.)
Quick setup instructions (set XCAT_PATH, add to .bashrc, etc.)
How to run a minimal example (single QA phantom or small XCAT section)
Output structure (what files are generated)
✅ Link to GitHub/Zenodo where code is archived
✅ Data availability statement (where to download datasets or how to generate them)
Nice-to-have (optional, low effort):

Brief docstrings in __MAIN_SCRIPT.sh explaining what each section does
Update Paper_Outline.md with authors, target, and complete sections
Add LICENSE file (Apache 2.0 already mentioned in README-PMLD.txt)
Don't bother with:

❌ Polished Python wrappers
❌ Full docstrings on every subscript
❌ Unit tests (not expected for this type of project)
❌ Refactoring bash scripts
Realistic effort: 1–2 days of writing documentation. The code itself is fine.

The paper will do the heavy lifting of explaining methodology. The code repo just needs to be navigable and reproducible, not beautiful.


###############

For FlexCNN, you want a methods/architecture journal that values medical imaging + ML innovation. Here's the tier list:

Top tier (aim high first):

IEEE Transactions on Medical Imaging (TMI)

Gold standard for medical imaging + deep learning
Rigorous but fair review
High visibility, strong for CV
6–9 month timeline
Perfect fit for frozen-flow architecture + generalization study
Physics in Medicine & Biology (PMB)

Broader medical physics audience (your natural readers)
Strong on reconstruction methods + validation
Slightly easier than TMI, still respected
~4–6 months
Great fit if you emphasize the PET physics angle
Strong alternatives (plan B):

Medical Physics (AAPM journal)

Practical, application-focused
Faster review
Slightly lower bar than TMI/PMB
Machine Learning for Biomedical Imaging (new, growing)

Emerging venue, good for novel architectures
Faster turnaround
Less prestige, but solid
My recommendation: Start with TMI or PMB

Why TMI: If your frozen-flow + feature injection is the flagship contribution, TMI is where it belongs. You have:

✅ Novel architecture (frozen backbone with lateral feature injection)
✅ Rigorous validation (QA phantoms, ablations, generalization to OOD data)
✅ Quantitative metrics (CRCs, SSIM, MSE + domain-specific)
✅ Solid experimental design (Bayesian optimization, ASHA scheduling)
Why PMB: If you want to emphasize the PET physics angle (which you should—it's a strength), PMB is equally strong and slightly more forgiving for architectures that aren't pure cutting-edge ML.

My call: TMI first, fallback to PMB.

Both take your work seriously. TMI just has higher prestige if it lands.



Dataset paper: mid Feb -> mid March
Paper 2: mid march -> mid May
Paper 3: mid may -> end July
Dissertation done: early January
Dissertation submitted: May 2027
October 2027