"""Generate a small synthetic corpus + Q11 query table with the same layout as the real one.

Everything is invented (seeded RNG, fictitious place names, ``SYN-####`` file numbers).  It reproduces the
phenomena the Paper 3 scorers have to cope with: several files per property (historical re-valuations), formatting
variants of survey numbers, empty fields that shift into the next label line ("structural contamination"),
shared common values and near-duplicate localities.

    python scripts/make_synthetic_p3.py --out sample_data
"""

import argparse
import random
from pathlib import Path

import pandas as pd

VILLAGES = ["Alderbank", "Northvale", "Kestrel Hill", "Marrow Cross", "Lowmere", "Ashgrove", "Tinwick", "Oakreach"]
MANDALS = ["Eastmark", "Westmark", "Brightwater"]
DISTRICTS = ["Redfield", "Bluefield"]
TYPES = ["Flat", "Villa", "Land", "Old_Land_and_Building", "Semi_Finished_Flat"]
BUILDINGS = ["Green Court", "Sunrise Heights", "Lakeview Residency", "Maple Towers", "River Park", ""]

LAYOUT = [("Door Number", "door_num"), ("Ward Number", "ward_num"), ("Block Number", "block_num"),
          ("Plot Number", "plot_num"), ("Survey Number", "survey_num"), ("Village", "village"),
          ("Mandal", "mandal"), ("District", "district"), ("State", "state"), ("Pincode", "pincode"),
          ("Building Name", "building_name"), ("Latitude", "latitude"), ("Longitude", "longitude")]


def survey_variants(nums, rng):
    style = rng.choice(["comma", "amp", "nos", "sy", "and"])
    if style == "comma":
        return ",".join(nums)
    if style == "amp":
        return " & ".join(nums)
    if style == "nos":
        return "Nos. " + ", ".join(nums)
    if style == "sy":
        return "Sy.No. " + ",".join(nums)
    return " and ".join(nums)


def make(n_properties=90, seed=7):
    rng = random.Random(seed)
    docs, queries, serial = [], [], 0
    for p in range(n_properties):
        village = rng.choice(VILLAGES)
        base = {
            "door_num": rng.choice(["", f"{rng.randint(1, 40)}-{rng.randint(1, 99)}", f"Flat No.{rng.randint(100, 520)}"]),
            "plot_num": rng.choice(["", str(rng.randint(1, 300)), f"{rng.randint(1, 300)} North Part"]),
            "survey_nums": [str(rng.randint(1, 400)) + rng.choice(["", "/A", "/B"]) for _ in range(rng.randint(0, 3))],
            "village": village, "mandal": MANDALS[VILLAGES.index(village) % 3],
            "district": DISTRICTS[VILLAGES.index(village) % 2],
            "pincode": str(500000 + 11 * (VILLAGES.index(village) + 1)),
            "building_name": rng.choice(BUILDINGS), "property_type": rng.choice(TYPES),
        }
        versions = rng.choice([1, 1, 2, 3, 4])
        first_of_property = None
        for v in range(versions):
            serial += 1
            fn = f"SYN-{serial:04d}"
            raw = {
                "door_num": base["door_num"] if rng.random() > 0.1 else "",
                "plot_num": base["plot_num"] if rng.random() > 0.1 else "",
                "survey_num": survey_variants(base["survey_nums"], rng) if base["survey_nums"] and rng.random() > 0.1 else "",
                "village": base["village"] if rng.random() > 0.05 else base["village"].lower(),
                "mandal": base["mandal"], "district": rng.choice([base["district"], base["district"].lower()]),
                "pincode": base["pincode"] if rng.random() > 0.15 else "",
                "building_name": base["building_name"], "property_type": base["property_type"],
                "ward_num": "", "block_num": "", "state": "Testland", "latitude": "", "longitude": "",
            }
            text = [f"FILE NUMBER: {fn}", f"Property Type: {raw['property_type']}"]
            text += [f"{label}: {raw[col]}" for label, col in LAYOUT]
            docs.append({"file_number": fn, "rag_text": "\n".join(text)})
            if v == 0:
                first_of_property = fn
            elif rng.random() < 0.8:   # later versions have an earlier valuation report attached -> Q11 query
                queries.append({"file_number": fn, **{k: raw[k] for k in
                                ["door_num", "plot_num", "survey_num", "village", "mandal", "district",
                                 "pincode", "building_name", "property_type"]}})
    return pd.DataFrame(docs), pd.DataFrame(queries)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sample_data")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    corpus, queries = make(seed=a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    corpus.to_csv(out / "synthetic_rag_documents.csv", index=False)
    queries.to_csv(out / "synthetic_q11_queries.csv", index=False)
    print(len(corpus), "documents,", len(queries), "queries")
