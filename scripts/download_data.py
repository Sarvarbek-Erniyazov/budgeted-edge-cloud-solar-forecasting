"""Download the non-image Folsom files (Zenodo record 2826939) and verify MD5.

Standard library only. Re-running skips files that already verify.
MD5 values were read from the Zenodo record page on 2026-10-07. If a file fails,
compare the value below with the record page before assuming a bad download.
"""
import hashlib
import sys
import urllib.parse
import urllib.request
from pathlib import Path

RECORD = "https://zenodo.org/records/2826939/files/{name}?download=1"
OUT = Path(__file__).resolve().parents[1] / "data" / "raw"

FILES = {
    "Folsom_irradiance.csv": "f7deba7ccd089dbd3f52a46405a7dfc2",
    "Folsom_weather.csv": "b04e0dc7edf3513a769ea2c8c59beb27",
    "Folsom_satellite.csv": "f68086048ee5d764d1d992404147c421",
    "Folsom_sky_image_features.csv": "86d58b6b84393399735a93ce1657cfab",
    "Folsom_NAM_lat38.579454_lon-121.260320.csv": "3d917eeecdf967d1f90f803fad5e5467",
    "Folsom_NAM_lat38.599891_lon-121.126680.csv": "30024faae0123990cf29c81c281eaccc",
    "Folsom_NAM_lat38.683880_lon-121.286556.csv": "c0d6db7093b957603cb05c90fff23167",
    "Folsom_NAM_lat38.704328_lon-121.152788.csv": "792f830c261e2c041d35ebeb6eadbeac",
    "Irradiance_features_intra-hour.csv": "9e25e78b816e51b95d4349f304155f56",
    "Irradiance_features_intra-day.csv": "971eee5f86677536b6238e73d923cedc",
    "Irradiance_features_day-ahead.csv": "889efab48e0c0c690c45b11e641ba388",
    "Sky_image_features_intra-hour.csv": "a81c753c308213e2b506b94e0412403a",
    "Sat_image_features_intra-day.csv": "8af401d02a090108b1863cb953ef64cf",
    "NAM_nearest_node_day-ahead.csv": "978905d0c0d1b1488325b33456446d23",
    "Target_intra-hour.csv": "ac6ebc385b6f6112c68ea967fc437c69",
    "Target_intra-day.csv": "9d530ea7cbe0f122bc26041e9da74afd",
    "Target_day-ahead.csv": "ed4959b21d282177cedcefe2e8e27f83",
    "Forecast_intra-hour.py": "7dd387b298e4c75f84a5fe7093bde2dd",
    "Forecast_intra-day.py": "6030752b33ce675859d131833a5e127d",
    "Forecast_day-ahead.py": "763f1666ff1485d631b7417cc8c4a5e8",
    "Postprocess.py": "73601ae78e2e49942673688650abfa3d",
}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    bad = []
    for name, want in FILES.items():
        dest = OUT / name
        if dest.exists() and md5(dest) == want:
            print(f"ok (cached)  {name}")
            continue
        url = RECORD.format(name=urllib.parse.quote(name))
        print(f"downloading  {name}")
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        got = md5(tmp)
        if got != want:
            bad.append((name, want, got))
            print(f"MD5 MISMATCH {name}: expected {want}, got {got}")
            continue
        tmp.replace(dest)
        print(f"ok           {name}")
    total = sum(p.stat().st_size for p in OUT.glob("*") if p.is_file() and p.suffix != ".part")
    print(f"\n{len(FILES) - len(bad)}/{len(FILES)} files verified, {total / 1e6:.0f} MB in {OUT}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
