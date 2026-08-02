"""End-to-end verification of the pipeline against the capture-realistic test set.

Checks, against known ground truth:
  1. recovered f1 within 1% for motor excitation, 1.5% for tap (decaying signal = fewer cycles)
  2. damaged twin triggers ALERT, healthy re-take stays HEALTHY
"""
import testdata
import modal
import shs


def main() -> int:
    print("== TremorLens verification (capture-realistic test set) ==")
    testdata.render_set()

    res = {name: modal.extract(f"data/{name}.mp4") for name in testdata.SET}
    base = res["test_healthy_motor"]
    noise = abs(res["test_healthy_take2"].f1 - base.f1) / base.f1 * 100

    ok = True
    for name, r in res.items():
        gt = testdata.GT[name]
        err = abs(r.f1 - gt) / gt * 100
        tol = 1.5 if "tap" in name else 1.0
        flag = "OK " if err < tol else "BAD"
        ok &= err < tol
        print(f"{name:22s} GT {gt:.2f} Hz -> {r.f1:6.3f} Hz  err {err:4.2f}%  [{flag}]")
    print(f"run-to-run scatter (noise floor): {noise:.2f}%")

    v_ok = shs.score(base, res["test_healthy_take2"], noise_pct=max(noise, 0.2))
    v_bad = shs.score(base, res["test_damaged_motor"], noise_pct=max(noise, 0.2))
    print(f"healthy re-take: SHS {v_ok.shs:5.1f}  {v_ok.status}")
    print(f"damaged twin  : SHS {v_bad.shs:5.1f}  {v_bad.status}  ({v_bad.detail})")
    ok &= v_ok.status == "HEALTHY" and v_bad.status == "ALERT"

    print("RESULT:", "PASS ✅" if ok else "FAIL ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
