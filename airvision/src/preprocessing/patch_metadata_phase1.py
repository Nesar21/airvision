# src/preprocessing/patch_metadata_phase1.py

import pandas as pd
from pathlib import Path
from src.utils.env_utils import init_env
from src.utils.manifest import log_run


# --------------------------------------------
# CONFIG
# --------------------------------------------
DATA_ROOT = Path(".")
META_IMAGE = DATA_ROOT / "metadata_image_only.csv"
META_FUSION = DATA_ROOT / "metadata_fusion.csv"


# --------------------------------------------
# COUNTRY MAPPING LOGIC
# --------------------------------------------
def infer_country_group(loc: str) -> str:
    """Map each row to IND_NEP or PM25VISION."""
    loc_low = loc.lower()

    # India or Nepal → IND_NEP
    if "nepal" in loc_low:
        return "IND_NEP"
    if (
        "india" in loc_low
        or "delhi" in loc_low
        or "mumbai" in loc_low
        or "kanpur" in loc_low
        or "kolkata" in loc_low
        or "chennai" in loc_low
        or "bangalore" in loc_low
        or "hyderabad" in loc_low
    ):
        return "IND_NEP"

    # Everything else (PM25Vision global set)
    return "PM25VISION"


# --------------------------------------------
# PATCH PIPELINE
# --------------------------------------------
def patch_metadata():
    init_env(anomaly_for_debug=False)

    # Load
    df_img = pd.read_csv(META_IMAGE)
    df_fus = pd.read_csv(META_FUSION)

    # Validate identical length
    if len(df_img) != len(df_fus):
        raise ValueError(
            f"metadata sizes mismatch: image={len(df_img)}, fusion={len(df_fus)}"
        )

    # -------------------------------------------------
    # 1) Inject image_id
    # -------------------------------------------------
    df_img.insert(0, "image_id", range(len(df_img)))
    df_fus.insert(0, "image_id", range(len(df_fus)))

    # -------------------------------------------------
    # 2) Inject country_group
    # -------------------------------------------------
    df_img["country_group"] = df_img["Location"].apply(infer_country_group)
    df_fus["country_group"] = df_fus["Location"].apply(infer_country_group)

    # -------------------------------------------------
    # 3) Save back
    # -------------------------------------------------
    df_img.to_csv(META_IMAGE, index=False)
    df_fus.to_csv(META_FUSION, index=False)

    # -------------------------------------------------
    # 4) Log manifest
    # -------------------------------------------------
    log_run(
        phase="phase1_patch",
        stage="metadata_augmented",
        description="Patched metadata with image_id + country_group.",
        config={"added_columns": ["image_id", "country_group"]},
        metrics={"rows": len(df_img)},
    )

    print("PATCH COMPLETE.")
    print("metadata_image_only.csv and metadata_fusion.csv updated.")


if __name__ == "__main__":
    patch_metadata()
