"""Figure-eight certificate checker: one command for all six stages.

    python verify/eight_check_all.py <certificate.npz> <run_dir> [old_tube.npz]

Runs, in order and stopping at the first failure:
  stages A, B, D, E   verify/eight_check.py <cert> <run_dir> ABDE
  stage C             verify/eight_check.py <cert> <run_dir> C
  stage F, F1-F4      verify/eight_check_records.py <run_dir> <run_dir>/saves <run_dir>/F <run_dir>/leaves.npy 8 <old_tube> 1 -
                      (records, (ii-1), (iii-L), top check; its own cover line is not used)
  stage F, F5         verify/eight_check_f5v2.py <run_dir> <run_dir>/leaves.npy 2 <old_tube> 1 -
                      (the cover: requires the F1-F4 lines of the log above, matches every leaf to the box built and
                      proved in stage D, accepts certainly empty cells below EPS_MIN as stage E does, certifies tops
                      with 4000 s-slabs where 1000 do not suffice)
Prints one line "CHECK ALL PASS" or "CHECK ALL FAIL <stage>"; exit 0 / 1. Each stage prints the md5 of its code and
inputs. The separate interval scripts for (ii'), the bands and the small-s lemma are run on their own."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OLD_TUBE = os.path.join(ROOT, "words/out/eight/tubeb/label_tube_ellip0.3_0.13_0.1_fit_taper.npz")


def run(args, verdict_ok):
    print("$ " + " ".join(args), flush=True)
    p = subprocess.run([sys.executable, "-u"] + args, capture_output=True, text=True)
    out = p.stdout + p.stderr
    tail = [ln for ln in out.splitlines() if ln.strip()][-3:]
    print("\n".join(tail), flush=True)
    return p.returncode == 0 and any(verdict_ok in ln for ln in out.splitlines())


def main():
    cert, run_dir = sys.argv[1], sys.argv[2]
    old = sys.argv[3] if len(sys.argv) > 3 else OLD_TUBE
    steps = [
        ("A, B, D, E", [os.path.join(HERE, "eight_check.py"), cert, run_dir, "ABDE"], "STAGES ABDE PASS"),
        ("C", [os.path.join(HERE, "eight_check.py"), cert, run_dir, "C"], "STAGES C PASS"),
    ]
    for name, args, ok in steps:
        if not run(args, ok):
            print(f"CHECK ALL FAIL stage {name}")
            sys.exit(1)
    # stage F, F1-F4: the run writes F/stage_f.npz and F/stage_f.log; its verdict line also covers its own cover
    # step, which is superseded by F5 below, so only completion is required here (F5 re-reads and requires the
    # F1-F4 lines of the log).
    print("$ stage F (F1-F4)", flush=True)
    subprocess.run([sys.executable, "-u", os.path.join(HERE, "eight_check_records.py"), run_dir,
                    os.path.join(run_dir, "saves"), os.path.join(run_dir, "F"), os.path.join(run_dir, "leaves.npy"),
                    "8", old, "1", "-"])
    if not os.path.exists(os.path.join(run_dir, "F", "stage_f.npz")):
        print("CHECK ALL FAIL stage F (no output)")
        sys.exit(1)
    if not run([os.path.join(HERE, "eight_check_f5v2.py"), run_dir, os.path.join(run_dir, "leaves.npy"), "2", old,
                "1", "-"], "STAGE F5v2 PASS"):
        print("CHECK ALL FAIL stage F (F5 cover)")
        sys.exit(1)
    print("CHECK ALL PASS")


if __name__ == "__main__":
    main()
