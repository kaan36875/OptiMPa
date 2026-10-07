import type { Metadata } from "next";
import Image from "next/image";
import { Card, Label } from "../components/ui";
import { EMISSION_FACTORS } from "../lib/carbon";
import { FEATURE_LABELS } from "../lib/mixes";
import { MODEL } from "../lib/modelData";

export const metadata: Metadata = {
  title: "Model & method — OptiMPa",
  description: "How the OptiMPa strength model was trained and validated, and how embodied carbon is calculated.",
};

const H2 = ({ children, id }: { children: React.ReactNode; id?: string }) => (
  <h2 id={id} style={{ fontSize: 20, fontWeight: 800, letterSpacing: "-0.02em", color: "var(--text-1)", marginBottom: 12, scrollMarginTop: 80 }}>
    {children}
  </h2>
);
const P = ({ children }: { children: React.ReactNode }) => (
  <p style={{ fontSize: 14, color: "var(--text-2)", lineHeight: 1.7, marginBottom: 12 }}>{children}</p>
);

export default function ModelPage() {
  const { metrics, data_ranges: ranges } = MODEL;
  const x = metrics.xgboost;
  const rf = metrics.rf_baseline;

  return (
    <div style={{ maxWidth: 860, margin: "0 auto", padding: "48px 24px 0", display: "flex", flexDirection: "column", gap: 24 }}>
      <div>
        <h1 style={{ fontSize: "clamp(1.8rem, 4vw, 2.6rem)", fontWeight: 900, letterSpacing: "-0.03em", color: "var(--text-1)", marginBottom: 12 }}>
          Model &amp; method
        </h1>
        <P>
          OptiMPa predicts the compressive strength of a concrete mix from its ingredients and curing age. The model
          is an XGBoost regressor trained on the UCI Concrete Compressive Strength dataset. It runs in your browser:
          the trained trees are shipped with the page, and SHAP values are computed exactly with TreeSHAP.
        </P>
      </div>

      <Card style={{ padding: 28 }}>
        <H2>Data</H2>
        <P>
          UCI Concrete Compressive Strength (I-Cheng Yeh, 1998): 1030 laboratory tests. After removing 25 exact duplicate
          rows, <strong>{metrics.dataset.rows}</strong> observations of <strong>{metrics.dataset.mix_designs}</strong> different
          mixes remain, most of them tested at several ages.
        </P>
        <table className="data">
          <thead>
            <tr><th>Input</th><th className="num">Min</th><th className="num">Max</th><th>Unit</th></tr>
          </thead>
          <tbody>
            {Object.entries(ranges).map(([k, [lo, hi]]) => (
              <tr key={k}>
                <td>{k === "strength" ? "Compressive strength" : FEATURE_LABELS[k]}</td>
                <td className="num">{lo}</td>
                <td className="num">{hi}</td>
                <td>{k === "age" ? "days" : k === "strength" ? "MPa" : "kg/m³"}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p style={{ fontSize: 11, color: "var(--text-3)", marginTop: 10 }}>
          The predictor warns when an input is outside these ranges, because the model is extrapolating there.
        </p>
      </Card>

      <Card style={{ padding: 28 }}>
        <H2>Method</H2>
        <P>
          <strong>Features.</strong> Besides the 8 raw inputs, the model gets the ratios that concrete technology says matter:
          water/cement (Abrams&apos; law), water/binder, total binder, fine/coarse aggregate, and slag/cement and fly ash/cement.
        </P>
        <P>
          <strong>No leakage between train and test.</strong> The same mix is tested at several ages, so a random split would
          put the same mix in both sets and overstate the accuracy. Rows are grouped by mix design: {metrics.split.train_samples} rows
          (80% of the mixes) are used for tuning with {metrics.split.cv_folds}-fold GroupKFold, and {metrics.split.test_samples} rows
          from mixes the model has never seen are held out for the final test.
        </P>
        <P>
          <strong>Tuning.</strong> 50 Optuna trials (seeded), minimising the grouped cross-validation RMSE.
        </P>
      </Card>

      <Card style={{ padding: 28 }}>
        <H2>Results</H2>
        <table className="data" style={{ marginBottom: 20 }}>
          <thead>
            <tr><th>Model</th><th className="num">CV R²</th><th className="num">CV RMSE</th><th className="num">Test R²</th><th className="num">Test RMSE</th></tr>
          </thead>
          <tbody>
            <tr>
              <td>Random Forest baseline (8 raw features)</td>
              <td className="num">{rf.cv_r2.toFixed(3)}</td><td className="num">{rf.cv_rmse.toFixed(2)}</td>
              <td className="num">{rf.test_r2.toFixed(3)}</td><td className="num">{rf.test_rmse.toFixed(2)}</td>
            </tr>
            <tr className="strong">
              <td>XGBoost (14 features, tuned)</td>
              <td className="num">{x.cv_r2.toFixed(3)}</td><td className="num">{x.cv_rmse.toFixed(2)}</td>
              <td className="num">{x.test_r2.toFixed(3)}</td><td className="num">{x.test_rmse.toFixed(2)}</td>
            </tr>
          </tbody>
        </table>
        <P>
          RMSE is in MPa. On mixes it has never seen, the model is typically within about ±{x.test_rmse.toFixed(1)} MPa,
          so treat a prediction as an estimate for comparing mixes, not as a replacement for testing.
        </P>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 16, marginTop: 8 }}>
          <figure>
            <Image src="/reports/model_evaluation.png" alt="Predicted vs. actual strength on the held-out mixes" width={1200} height={1050} style={{ width: "100%", height: "auto", borderRadius: 12 }} />
            <figcaption style={{ fontSize: 11, color: "var(--text-3)", marginTop: 6 }}>Predicted vs. actual strength, held-out mixes</figcaption>
          </figure>
          <figure>
            <Image src="/reports/shap_summary.png" alt="SHAP summary: impact of each feature on the prediction" width={1164} height={1050} style={{ width: "100%", height: "auto", borderRadius: 12, background: "#fff" }} />
            <figcaption style={{ fontSize: 11, color: "var(--text-3)", marginTop: 6 }}>
              SHAP summary: curing age, w/b ratio, total binder and w/c ratio have the largest effect
            </figcaption>
          </figure>
        </div>
      </Card>

      <Card style={{ padding: 28 }}>
        <H2 id="carbon">Embodied carbon</H2>
        <P>
          The carbon of 1 m³ of concrete is each ingredient&apos;s mass (kg/m³) times its emission factor (kg CO₂e/kg),
          summed. This covers production of the materials (stages A1–A3, cradle to gate). Transport, reinforcement,
          formwork and later life stages are not included.
        </P>
        <table className="data" style={{ marginBottom: 16 }}>
          <thead><tr><th>Material</th><th className="num">kg CO₂e per kg</th></tr></thead>
          <tbody>
            {Object.entries(EMISSION_FACTORS).map(([k, v]) => (
              <tr key={k}><td>{FEATURE_LABELS[k]}{k === "cement" ? " (CEM I)" : k === "slag" ? " (GGBS)" : ""}</td><td className="num">{v}</td></tr>
            ))}
          </tbody>
        </table>
        <P>
          These are approximate generic values, mostly from the ICE database v3.0 (Circular Ecology, 2019). The
          superplasticizer value is from the EFCA admixture EPD. They are good for comparing mixes with each
          other. For an actual assessment, use the EPDs of the specific cement and materials.
        </P>
        <P>
          <strong>Reference mix.</strong> The predictor compares your mix with the same mix where all binder is Portland
          cement (slag and fly ash replaced 1:1 by mass), keeping the binder content and w/b ratio. Both are run through
          the strength model, so you see the carbon saved and what it costs in predicted strength.
        </P>
      </Card>

      <Card style={{ padding: 28 }}>
        <H2>Limitations</H2>
        <ul style={{ fontSize: 14, color: "var(--text-2)", lineHeight: 1.7, paddingLeft: 20, listStyle: "disc" }}>
          <li>One laboratory dataset of about 1000 tests. It has only cement, slag and fly ash as binders, so other binders (silica fume, calcined clay, limestone cement ...) can&apos;t be modelled.</li>
          <li>Strength is predicted as a mean cylinder strength. The EN 206 class is an estimate (fck = fcm − 8 MPa), not a conformity check.</li>
          <li>The carbon numbers use generic factors and cover the concrete only.</li>
        </ul>
      </Card>

      <Card style={{ padding: 28 }}>
        <Label style={{ marginBottom: 10 }}>Tuned hyperparameters</Label>
        <table className="data">
          <tbody>
            {Object.entries(metrics.best_params).map(([k, v]) => (
              <tr key={k}><td>{k}</td><td className="num">{Number.isInteger(v) ? v : v.toPrecision(3)}</td></tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  );
}
