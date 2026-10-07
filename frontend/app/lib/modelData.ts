import data from "./model.json";
import type { ModelData } from "./model";

/** The exported XGBoost model (ml_pipeline/export_web_model.py). */
export const MODEL = data as unknown as ModelData;
