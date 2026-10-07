# OptiMPa — embodied carbon from a BIM model

`ifc_carbon.py` reads an IFC file, finds the concrete elements and their volumes, and calculates the embodied carbon of the concrete (A1–A3, cradle to gate). It uses [IfcOpenShell](https://ifcopenshell.org/).

```bash
pip install -r requirements.txt
python ifc_carbon.py examples/sample_building.ifc --mixes mixes.example.json
```

Output for the sample model:

```
Concrete elements: 8 | volume: 37.70 m3 | embodied carbon (A1-A3): 11.90 t CO2e

  Mix          Elements  Volume m3  kgCO2e/m3   t CO2e  Share
  C30/37              2      26.00      325.9     8.47    71%
  C25/30              1       7.20      258.7     1.86    16%
  default             1       2.58      325.9     0.84     7%
  C35/45              4       1.92      375.1     0.72     6%
  ...
  Needs review (1), not included in the total:
  - IfcStair 'Stair (placeholder)' (...): no volume in quantities or geometry
```

## How it works

1. **Concrete elements:** an element counts as concrete if its material name, category or description contains "concrete", "beton" or a class like `C30/37`. In layered materials (for example a concrete wall with insulation), only the concrete share of the total thickness is counted.
2. **Volume:** the tool uses `NetVolume`, or else `GrossVolume`, from the element's quantity sets (`Qto_...BaseQuantities`). If neither is there, it computes the volume from the 3D geometry, with openings subtracted. Units are converted to m³ in both cases.
3. **Mix:** each element gets a concrete mix by strength class. The class is read from `Pset_ConcreteElementGeneral.StrengthClass` or from the material name, then looked up in the mixes file. Elements without a class or without a matching mix use the `default` mix.
4. **Carbon:** `carbon.py` multiplies each ingredient (kg/m³) by its emission factor, giving kg CO₂e per m³. This is then multiplied by the volume.

Elements whose concrete share or volume can't be determined are not guessed. They are listed under "Needs review" and left out of the total.

## Options

| Option | |
| :--- | :--- |
| `--mixes file.json` | Mix designs by class, in the same 8 inputs as the strength model (see `mixes.example.json`) |
| `--factors file.json` | Replace any of the default emission factors, e.g. with your suppliers' EPD values: `{"cement": 0.82}` |
| `--csv out.csv` | One row per concrete element: GlobalId, volume, where the volume came from, mix, kg CO₂e |
| `--api http://localhost:8000` | Sends each mix to the OptiMPa API and checks whether its predicted 28-day strength reaches the class it is used for. Also prints kg CO₂e per MPa. |

With `--api`, the sample model gives:

```
  C25/30       ->  35.0 MPa, C25/30     7.4 kgCO2e/MPa   meets class
  C30/37       ->  40.7 MPa, C30/37     8.0 kgCO2e/MPa   meets class
  C35/45       ->  51.1 MPa, C40/50     7.3 kgCO2e/MPa   meets class
```

## Limits

- The default emission factors are approximate generic values, mostly from ICE v3.0. They are fine for comparing mixes. For a real assessment, use EPD values for the actual cement and materials (`--factors`).
- Only concrete is counted. Reinforcement, formwork, transport (A4) and later stages are not included.
- The result depends on the IFC model. Elements with the wrong material, or missing from the model, will be wrong or missing in the total too.

`examples/make_sample_ifc.py` regenerates the sample model. Its docstring lists the expected volumes for each element.
