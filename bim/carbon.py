"""
OptiMPa — embodied carbon of a concrete mix (A1–A3, cradle to gate)

The mix uses the same 8 inputs as the strength model (kg per m³ of concrete,
plus age, which does not affect carbon). Each ingredient mass is multiplied by
an emission factor to get kg CO2e per m³.

Default factors are approximate generic values, mainly from the ICE database
v3.0 (Circular Ecology, 2019); the superplasticizer value is from the EFCA
admixture EPD. They are good enough to compare mixes, but for reporting you
should replace them with your suppliers' EPD values (see --factors in
ifc_carbon.py). Transport (A4), reinforcement and formwork are not included.
"""

# kg CO2e per kg of material
DEFAULT_FACTORS = {
    "cement": 0.912,            # CEM I Portland cement
    "slag": 0.083,              # GGBS (ground granulated blast furnace slag)
    "fly_ash": 0.004,           # pulverised fuel ash
    "water": 0.000344,          # mains water
    "superplasticizer": 1.88,   # EFCA generic EPD
    "coarse_agg": 0.0075,       # general aggregate
    "fine_agg": 0.0075,         # general aggregate (sand)
}

MIX_KEYS = list(DEFAULT_FACTORS)


def mix_carbon(mix: dict, factors: dict | None = None) -> dict:
    """
    Carbon of 1 m³ of the given mix.

    Returns {"total": kg CO2e/m³, "by_material": {ingredient: kg CO2e/m³}}.
    Missing ingredients count as 0 kg/m³.
    """
    factors = {**DEFAULT_FACTORS, **(factors or {})}
    by_material = {k: float(mix.get(k, 0.0)) * factors[k] for k in MIX_KEYS}
    return {"total": sum(by_material.values()), "by_material": by_material}


if __name__ == "__main__":
    example = {"cement": 350, "slag": 0, "fly_ash": 0, "water": 175, "superplasticizer": 6,
               "coarse_agg": 1040, "fine_agg": 755, "age": 28}
    result = mix_carbon(example)
    print(f"Example mix: {result['total']:.1f} kg CO2e/m³")
    for k, v in sorted(result["by_material"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:17s}{v:8.1f}")
