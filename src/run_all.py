"""
run_all.py - run the four tasks one after another (or just the ones you pick).

    python src/run_all.py             # everything
    python src/run_all.py 1 2         # only tasks 1 and 2
    python src/run_all.py --report    # run everything and build the Word report
"""
import sys
import time
import traceback

import task1_restoration_segmentation as t1
import task2_features as t2
import task3_detection as t3
import task4_motion as t4

TASKS = {"1": t1, "2": t2, "3": t3, "4": t4}


def main():
    args = sys.argv[1:]
    want_report = "--report" in args
    chosen = [a for a in args if a in TASKS] or list(TASKS)

    failed = []
    for key in chosen:
        print(f"\n===== TASK {key} =====")
        t0 = time.time()
        try:
            TASKS[key].main()
        except SystemExit as e:  # friendly error messages from the scripts
            print(e)
            failed.append(key)
        except Exception:
            traceback.print_exc()
            failed.append(key)
        print(f"(task {key} took {time.time() - t0:.1f} s)")

    if want_report:
        import make_report
        make_report.main()

    print("\n==========================================")
    if failed:
        print("Finished, but these tasks had problems:", ", ".join(failed))
    else:
        print("All selected tasks finished. Check the outputs/ folder.")


if __name__ == "__main__":
    main()
