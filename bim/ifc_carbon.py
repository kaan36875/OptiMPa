"""
OptiMPa — embodied carbon of the concrete in an IFC (BIM) model

Finds the concrete elements in an IFC file, takes their volumes, assigns each
one a concrete mix and multiplies volume by the mix's kg CO2e/m³ (carbon.py).

Usage:
  python bim/ifc_carbon.py model.ifc --mixes bim/mixes.example.json
  python bim/ifc_carbon.py model.ifc --mixes my_mixes.json --csv elements.csv --api http://localhost:8000

How it works:
  Concrete   An element counts as concrete if its material name, category or
             description contains "concrete", "beton" or a class like C30/37.
             For layered materials (e.g. concrete wall + insulation) only the
             concrete share of the layer thickness is counted.
  Volume     Taken from the element's quantity sets (NetVolume, then
             GrossVolume). If there is none, it is computed from the geometry.
  Mix        Strength class from Pset_ConcreteElementGeneral.StrengthClass, or
             from the material name (C30/37 ...). The class is looked up in the
             mixes file; anything else uses its "default" mix.
  --api      Sends each mix to the OptiMPa API (/predict) and checks whether its
             predicted strength reaches the class it is used for.
"""

import argparse
import csv
import json
import re
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element as uel
import ifcopenshell.util.shape as ushape
import ifcopenshell.util.unit as uunit

from carbon import DEFAULT_FACTORS, mix_carbon

CONCRETE_PATTERN = re.compile(r"concrete|beton|\bC\d{1,3}/\d{1,3}\b", re.IGNORECASE)
CLASS_PATTERN = re.compile(r"\bC(\d{1,3})/(\d{1,3})\b", re.IGNORECASE)
VOLUME_KEYS = ("NetVolume", "GrossVolume")
API_KEYS = ["cement", "slag", "fly_ash", "water", "superplasticizer", "coarse_agg", "fine_agg", "age"]


# ─────────────────────────────────────────────
# 1. MATERIALS
# ─────────────────────────────────────────────

def is_concrete(material) -> bool:
    text = " ".join(
        str(getattr(material, attr, None) or "") for attr in ("Name", "Category", "Description")
    )
    return bool(CONCRETE_PATTERN.search(text))


def concrete_share(element):
    """
    Which part of the element is concrete.

    Returns (share, concrete material names, note). share is 0..1, or None when
    it cannot be determined (the element is then listed for manual review).
    """
    mat = uel.get_material(element, should_skip_usage=True)
    if mat is None:
        return 0.0, [], None

    if mat.is_a("IfcMaterial"):
        return (1.0 if is_concrete(mat) else 0.0), [mat.Name], None

    if mat.is_a("IfcMaterialLayerSet"):
        layers = [l for l in mat.MaterialLayers if l.Material]
        concrete = [l for l in layers if is_concrete(l.Material)]
        total = sum(l.LayerThickness or 0 for l in layers)
        if not concrete:
            return 0.0, [], None
        if total <= 0:
            return None, [l.Material.Name for l in concrete], "layer thicknesses missing"
        share = sum(l.LayerThickness or 0 for l in concrete) / total
        note = None if share == 1 else f"layered, {share:.0%} concrete"
        return share, [l.Material.Name for l in concrete], note

    if mat.is_a("IfcMaterialProfileSet"):
        materials = [p.Material for p in mat.MaterialProfiles if p.Material]
        concrete = [m for m in materials if is_concrete(m)]
        if not concrete:
            return 0.0, [], None
        if len(concrete) < len(materials):
            return None, [m.Name for m in concrete], "profile set mixes concrete and other materials"
        return 1.0, [m.Name for m in concrete], None

    if mat.is_a("IfcMaterialConstituentSet"):
        constituents = [c for c in (mat.MaterialConstituents or []) if c.Material]
        concrete = [c for c in constituents if is_concrete(c.Material)]
        names = [c.Material.Name for c in concrete]
        if not concrete:
            return 0.0, [], None
        if len(concrete) == len(constituents):
            return 1.0, names, None
        if all(c.Fraction is not None for c in concrete):
            share = sum(c.Fraction for c in concrete)
            return share, names, f"constituents, {share:.0%} concrete"
        return None, names, "constituent set without fractions"

    if mat.is_a("IfcMaterialList"):
        concrete = [m for m in mat.Materials if is_concrete(m)]
        if not concrete:
            return 0.0, [], None
        if len(concrete) == len(mat.Materials):
            return 1.0, [m.Name for m in concrete], None
        return None, [m.Name for m in concrete], "material list mixes concrete and other materials"

    return 0.0, [], None


def strength_class(element, material_names) -> str | None:
    """C30/37 style class from Pset_ConcreteElementGeneral or the material names."""
    pset = uel.get_psets(element, psets_only=True).get("Pset_ConcreteElementGeneral", {})
    for text in [pset.get("StrengthClass"), *material_names]:
        match = CLASS_PATTERN.search(str(text or ""))
        if match:
            return f"C{match[1]}/{match[2]}"
    return None


# ─────────────────────────────────────────────
# 2. VOLUMES
# ─────────────────────────────────────────────

def quantity_volume(element, volume_scale):
    """NetVolume, then GrossVolume, from the element's own quantity sets [m³]."""
    qtos = uel.get_psets(element, qtos_only=True, should_inherit=False)
    for key in VOLUME_KEYS:
        for qto_name, props in qtos.items():
            value = props.get(key)
            if isinstance(value, (int, float)) and value > 0:
                return value * volume_scale, f"{qto_name}.{key}"
    return None, None


def geometry_volume(element, settings):
    """Volume of the element's body geometry (openings subtracted) [m³]."""
    if not element.Representation:
        return None, None
    try:
        shape = ifcopenshell.geom.create_shape(settings, element)
        volume = ushape.get_volume(shape.geometry)
    except Exception:
        return None, None
    return (volume, "geometry") if volume > 0 else (None, None)


# ─────────────────────────────────────────────
# 3. OPTIMPA STRENGTH CHECK
# ─────────────────────────────────────────────

def class_fck(grade: str | None) -> int:
    match = CLASS_PATTERN.search(grade or "")
    return int(match[1]) if match else 0


def predict_strength(api_url: str, mix: dict):
    """Ask the OptiMPa API for the mix's predicted strength. Returns (result, error)."""
    missing = [k for k in API_KEYS if k not in mix]
    if missing:
        return None, f"mix is missing {', '.join(missing)}"
    request = urllib.request.Request(
        api_url.rstrip("/") + "/predict",
        data=json.dumps({k: mix[k] for k in API_KEYS}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.load(response), None
    except urllib.error.HTTPError as e:
        detail = json.load(e).get("detail", e.reason)
        if isinstance(detail, list):
            detail = "; ".join(f"{d['loc'][-1]}: {d['msg']}" for d in detail)
        return None, str(detail)
    except (urllib.error.URLError, TimeoutError) as e:
        return None, f"API not reachable ({e})"


# ─────────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────────

def analyse(ifc_path: Path, mixes: dict, factors: dict):
    model = ifcopenshell.open(str(ifc_path))
    volume_scale = uunit.calculate_unit_scale(model, "VOLUMEUNIT")
    settings = ifcopenshell.geom.settings()  # geometry comes out in metres

    carbon_per_m3 = {name: mix_carbon(mix, factors)["total"] for name, mix in mixes.items()}
    rows, review = [], []

    for element in model.by_type("IfcElement"):
        if element.is_a("IfcFeatureElement"):  # openings, projections
            continue
        share, names, note = concrete_share(element)
        if share == 0:
            continue

        base = {
            "global_id": element.GlobalId,
            "ifc_class": element.is_a(),
            "name": element.Name or "",
            "material": ", ".join(n for n in names if n),
        }
        if share is None:
            review.append({**base, "issue": note})
            continue

        volume, source = quantity_volume(element, volume_scale)
        if volume is None:
            volume, source = geometry_volume(element, settings)
        if volume is None:
            review.append({**base, "issue": "no volume in quantities or geometry"})
            continue

        cls = strength_class(element, names)
        mix_name = cls if cls in mixes else ("default" if "default" in mixes else None)
        if mix_name is None:
            review.append({**base, "issue": f"no mix for class {cls or '(none)'} and no default mix"})
            continue

        concrete_volume = volume * share
        rows.append({
            **base,
            "strength_class": cls or "",
            "mix": mix_name,
            "volume_m3": round(concrete_volume, 4),
            "volume_source": source,
            "note": note or "",
            "kgco2e_per_m3": round(carbon_per_m3[mix_name], 1),
            "kgco2e": round(concrete_volume * carbon_per_m3[mix_name], 1),
        })

    return rows, review, carbon_per_m3


def print_report(rows, review, mixes, carbon_per_m3, api_url):
    total_volume = sum(r["volume_m3"] for r in rows)
    total_carbon = sum(r["kgco2e"] for r in rows)

    print(f"\nConcrete elements: {len(rows)} | volume: {total_volume:.2f} m3 | "
          f"embodied carbon (A1-A3): {total_carbon / 1000:.2f} t CO2e\n")

    by_mix = defaultdict(lambda: [0, 0.0, 0.0])
    for r in rows:
        agg = by_mix[r["mix"]]
        agg[0] += 1
        agg[1] += r["volume_m3"]
        agg[2] += r["kgco2e"]

    print(f"  {'Mix':12s}{'Elements':>9s}{'Volume m3':>11s}{'kgCO2e/m3':>11s}{'t CO2e':>9s}{'Share':>7s}")
    for name, (count, volume, carbon) in sorted(by_mix.items(), key=lambda kv: -kv[1][2]):
        share = carbon / total_carbon if total_carbon else 0
        print(f"  {name:12s}{count:9d}{volume:11.2f}{carbon_per_m3[name]:11.1f}{carbon / 1000:9.2f}{share:7.0%}")

    by_class = defaultdict(lambda: [0, 0.0, 0.0])
    for r in rows:
        agg = by_class[r["ifc_class"]]
        agg[0] += 1
        agg[1] += r["volume_m3"]
        agg[2] += r["kgco2e"]
    print(f"\n  {'IFC class':20s}{'Elements':>9s}{'Volume m3':>11s}{'t CO2e':>9s}")
    for name, (count, volume, carbon) in sorted(by_class.items(), key=lambda kv: -kv[1][2]):
        print(f"  {name:20s}{count:9d}{volume:11.2f}{carbon / 1000:9.2f}")

    if api_url:
        print("\n  OptiMPa strength check (28-day prediction for each mix used):")
        for name in sorted(by_mix):
            result, error = predict_strength(api_url, mixes[name])
            if error:
                print(f"  {name:12s} -> {error}")
                continue
            mpa, grade = result["strength_mpa"], result["strength_grade"]
            line = f"  {name:12s} -> {mpa:5.1f} MPa, {grade:8s} {carbon_per_m3[name] / mpa:5.1f} kgCO2e/MPa"
            if CLASS_PATTERN.fullmatch(name):
                ok = class_fck(grade) >= class_fck(name)
                line += "   meets class" if ok else f"   DOES NOT reach {name}"
            print(line)

    if review:
        print(f"\n  Needs review ({len(review)}), not included in the total:")
        for r in review:
            print(f"  - {r['ifc_class']} '{r['name']}' ({r['global_id']}): {r['issue']}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Embodied carbon of the concrete in an IFC model.")
    parser.add_argument("ifc", type=Path, help="IFC file")
    parser.add_argument("--mixes", type=Path, required=True,
                        help='JSON: {"C30/37": {mix}, "default": {mix}, ...} with kg/m3 values')
    parser.add_argument("--factors", type=Path,
                        help=f"JSON with emission factors [kg CO2e/kg] to override: {', '.join(DEFAULT_FACTORS)}")
    parser.add_argument("--csv", type=Path, help="write one row per concrete element")
    parser.add_argument("--api", help="OptiMPa API URL, e.g. http://localhost:8000")
    args = parser.parse_args()

    mixes = json.loads(args.mixes.read_text(encoding="utf-8"))
    mixes = {k: v for k, v in mixes.items() if not k.startswith("_")}  # allow "_comment" keys
    factors = json.loads(args.factors.read_text(encoding="utf-8")) if args.factors else {}
    unknown = set(factors) - set(DEFAULT_FACTORS)
    if unknown:
        sys.exit(f"Unknown emission factor(s): {', '.join(sorted(unknown))}")

    rows, review, carbon_per_m3 = analyse(args.ifc, mixes, factors)
    print_report(rows, review, mixes, carbon_per_m3, args.api)

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["global_id"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Element list written to {args.csv}")


if __name__ == "__main__":
    main()
