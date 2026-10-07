# OptiMPa — website

Next.js site for OptiMPa. The project description, model details and results are in the [main README](../README.md).

Everything runs in the browser; there is no backend:

- **`/` Predictor:** strength prediction, EN 206 class, SHAP breakdown, embodied carbon and the comparison with a Portland-cement-only mix. Updates as you move the sliders.
- **`/bim` BIM carbon:** open an IFC file and get the concrete volumes and the embodied carbon. The file is read with web-ifc (WebAssembly) and is not uploaded.
- **`/model` Model:** data, method, results, how carbon is calculated.

## Running

```bash
npm install
npm run dev        # http://localhost:3000
npm run build      # production build (all pages are static)
```

`npm run dev` and `npm run build` first copy `web-ifc.wasm` from `node_modules` into `public/wasm/` (`scripts/copy-wasm.mjs`).

## Tests

```bash
npm run test:model   # browser model vs. Python predictions and SHAP values (400 mixes)
npm run test:ifc     # IFC reader on ../bim/examples/sample_building.ifc
```

## Where things are

| File | |
| :--- | :--- |
| `app/lib/model.json` | Exported XGBoost trees and metrics. Regenerate with `python ml_pipeline/export_web_model.py` after retraining |
| `app/lib/model.ts` | Prediction, TreeSHAP, EN 206 class |
| `app/lib/carbon.ts` | Emission factors and the reference mix (kept in sync with `bim/carbon.py`) |
| `app/lib/ifc.ts` | Concrete elements and volumes from an IFC file (same rules as `bim/ifc_carbon.py`) |
| `app/lib/mixes.ts` | Slider ranges, default and example mixes |

## Deploying to Vercel

Import the repository and set **Root Directory** to `frontend`. No other settings or environment variables are needed.
