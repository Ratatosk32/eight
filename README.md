# The figure eight is the unique action minimiser in its symmetry class

Oleksii Stavriianov (ORCID [0009-0002-8504-3626](https://orcid.org/0009-0002-8504-3626))

Archive: [doi:10.5281/zenodo.24248043](https://doi.org/10.5281/zenodo.24248043) (release v1.0: certificate, checker, inputs, logs).

The paper is `paper/eight/main.pdf` (source `paper/eight/main.tex`).

In the planar three-body problem with equal masses and Newtonian attraction, Chenciner and Montgomery proved in 2000
that the action has a collision-free minimiser among loops with the symmetry group of the figure eight. The paper
proves that this minimiser is unique (the figure eight and its images under the isometries of the plane that preserve
the class) and nondegenerate. The proof is a calibration argument in shape space: a piecewise polynomial subsolution
of the Hamilton-Jacobi equation from the Euler configurations, and a field of extremals without focal points from the
isosceles configurations. The finitely many inequalities it needs are verified in interval arithmetic; the
verification is packaged as a certificate that a separate checker reads as untrusted data.

## Repository

- `paper/eight/`: the paper (LaTeX source with the specification appendix `appendix_spec.tex`; figures and the scripts
  that make their data in `fig/`). It builds with any LaTeX distribution with pgfplots (pdflatex, twice).
- `words/out/eight/certificate/eight_cert_v4.npz`: the certificate (leaves for the field, record-only leaves, slab
  boxes, and a header with the hashes of the inputs).
- `words/out/eight/block/cert_pieces_E_goalEfit.json`: the subsolution (cells and nodal values).
- `words/out/eight/tubeb/label_tube_*.npz`: the tube (label ellipsoids at 41 nodes; the tapered tube, the earlier
  tube used by the top check, and the untapered fit).
- `verify/`: the checker and the code it uses.
  - `eight_check_all.py`: one command for the six stages (A hashes and tube facts, B Krawczyk and the action of the
    eight, C the subsolution, D the field boxes, E the covering, F the boundary records, (ii-1), (iii-L), the top
    check and the cover).
  - `eight_check.py` (stages A to E), `eight_cell_verify.py` (stage C), `eight_check_records.py` (stage F records,
    (ii-1), (iii-L), top check), `eight_check_f5v2.py` (stage F cover).
  - `eight_tube_boundary.py`: the end condition (ii') and the bands; `eight_cell_arb.py`: the independent re-check of
    the subsolution inequality in Arb ball arithmetic (python-flint).
- `tools/`: research code from which the checker imports helpers (tube geometry, the small-s lemma
  `words_eight_tubeb.py small_s_rig`).
- `logs/`: the output of the run of the checker on the certificate, of the separate scripts, of the Arb re-check, and
  of the mutation tests (`logs/mutations/`).

Appendix A of the paper specifies the certificate format and what every stage computes and guarantees.

## Running the checker

    pip install numpy==2.4.6 scipy psutil python-flint
    python verify/eight_check_all.py words/out/eight/certificate/eight_cert_v4.npz run1

About 13 hours on 10 processes (most of it in stages D and F); it ends with CHECK ALL PASS or CHECK ALL FAIL. Run it
from the repository root. Each stage prints the md5 of its code and inputs. The separate scripts:

    C=words/out/eight/block/cert_pieces_E_goalEfit.json
    export TUBEB_A8HI=2.5198721686724825                           # A8_hi of the certificate header
    python tools/words_eight_tubeb.py endell $C 8 fit              # (ii') on I outside the two discs, seconds
    python tools/words_eight_tubeb.py band $C 0.0045 0.92 1.03      # band q_D in [0.92, 1.03], seconds
    python tools/words_eight_tubeb.py band $C 0.0001 0.86 0.94      # band q_D in [0.86, 0.94], seconds
    python tools/words_eight_tubeb.py smallrig $C 0.92 1.03         # small-s lemma, about 9 hours on one process

(logged in `logs/end_condition_ii_prime.log`, `logs/band_*.log`, `logs/small_s_lemma.log`). Platform of the logged run: Python 3.11.4,
numpy 2.4.6, Windows 10/11 x64.

The searches, computations and scripts were carried out with the assistance of large language models.

## Licence

Code (`verify/`, `tools/`): MIT, see [`LICENSE`](LICENSE). Paper text and figures (`paper/`): CC BY 4.0, see
[`LICENSE-CC-BY-4.0.md`](LICENSE-CC-BY-4.0.md). Data (`words/`, `logs/`): CC BY 4.0.
