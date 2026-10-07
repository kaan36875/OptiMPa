# OptiMPa — Frontend

Next.js interface for OptiMPa. The full project description, model details and results are in the [main README](../README.md).

You set the 8 mix parameters with sliders and press **Predict Strength**. The page sends a single `POST /explain` request to the API and shows:

- the predicted compressive strength (MPa) and the estimated EN 206 class
- a SHAP chart of how much each input raised or lowered the prediction

## Running

The API has to be running first (see the main README). Then:

```bash
npm install
npm run dev
```

and open `http://localhost:3000`.

By default the page calls `http://localhost:8000`. To use a different address, set it in `.env.local`:

```
NEXT_PUBLIC_API_URL=http://your-api-host:8000
```

## Example API call

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"cement": 350, "slag": 0, "fly_ash": 0, "water": 175,
       "superplasticizer": 6, "coarse_agg": 1040, "fine_agg": 780, "age": 28}'
```

```json
{
  "strength_mpa": 38.06,
  "strength_grade": "C30/37",
  "input_summary": { ... }
}
```
