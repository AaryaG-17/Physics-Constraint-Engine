import time
import lightkurve as lk
from transitleastsquares import transitleastsquares


LIGHT_CURVE = "data/real_lightcurves/TIC_420814525_sector56_spoc120s_flat.fits"


def main():
    print("Loading light curve...")
    lc = lk.read(LIGHT_CURVE)

    time_data = lc.time.value
    flux = lc.flux.value

    print(f"Data points: {len(time_data)}")
    print(f"Time span: {time_data.max() - time_data.min():.3f} days")

    print("\nRunning TLS baseline...")

    model = transitleastsquares(
        time_data,
        flux,
        verbose=True,
    )

    start = time.perf_counter()

    result = model.power(
        R_star=0.72300321,
        M_star=0.6968689,
    )

    elapsed = time.perf_counter() - start

    print("\n========== TLS BASELINE ==========")
    print(f"Runtime       : {elapsed:.3f} s")
    print(f"Best period   : {result.period:.6f} days")
    print(f"Duration      : {result.duration:.6f} days")
    print(f"Depth         : {result.depth:.6f}")
    print(f"SDE           : {result.SDE:.4f}")
    print(f"SNR           : {result.snr:.4f}")
    print(f"FAP           : {result.FAP:.6g}")

    if hasattr(result, "periods"):
        print(f"Trial periods : {len(result.periods)}")
    elif hasattr(result, "test_statistic_periods"):
        print(f"Trial periods : {len(result.test_statistic_periods)}")

    print("==================================")


if __name__ == "__main__":
    main()