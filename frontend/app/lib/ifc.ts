/**
 * Concrete quantities from an IFC model, in the browser (web-ifc).
 *
 * Same rules as bim/ifc_carbon.py:
 *   - concrete = material name/category/description contains "concrete",
 *     "beton" or a class like C30/37; layered materials count only the
 *     concrete share of the thickness
 *   - volume  = NetVolume, else GrossVolume, from the element's own quantity
 *     sets (converted with the project volume unit); else the body geometry
 *   - class   = Pset_ConcreteElementGeneral.StrengthClass, else the material name
 * Elements whose share or volume can't be determined are returned for review.
 */

import {
  IFCELEMENT, IFCELEMENTQUANTITY, IFCFEATUREELEMENT, IFCMATERIAL, IFCMATERIALCONSTITUENTSET,
  IFCMATERIALLAYERSET, IFCMATERIALLAYERSETUSAGE, IFCMATERIALLIST, IFCMATERIALPROFILESET,
  IFCMATERIALPROFILESETUSAGE, IFCPROPERTYSET, IFCQUANTITYVOLUME, IFCRELASSOCIATESMATERIAL,
  IFCRELDEFINESBYPROPERTIES, IFCRELDEFINESBYTYPE, IFCSIUNIT, IFCCONVERSIONBASEDUNIT,
  IFCUNITASSIGNMENT, type IfcAPI,
} from "web-ifc";

const CONCRETE = /concrete|beton|\bC\d{1,3}\/\d{1,3}\b/i;
const CLASS = /\bC(\d{1,3})\/(\d{1,3})\b/i;
const VOLUME_KEYS = ["NetVolume", "GrossVolume"];
const SI_PREFIX: Record<string, number> = {
  EXA: 1e18, PETA: 1e15, TERA: 1e12, GIGA: 1e9, MEGA: 1e6, KILO: 1e3, HECTO: 1e2, DECA: 1e1,
  DECI: 1e-1, CENTI: 1e-2, MILLI: 1e-3, MICRO: 1e-6, NANO: 1e-9, PICO: 1e-12, FEMTO: 1e-15, ATTO: 1e-18,
};

export interface ConcreteElement {
  expressId: number;
  globalId: string;
  ifcClass: string;
  name: string;
  material: string;
  strengthClass: string | null;
  volume: number; // m³ of concrete (share already applied)
  volumeSource: string;
  note: string;
}

export interface ReviewItem {
  globalId: string;
  ifcClass: string;
  name: string;
  issue: string;
}

export interface IfcResult {
  schema: string;
  elements: ConcreteElement[];
  review: ReviewItem[];
}

/* eslint-disable @typescript-eslint/no-explicit-any -- web-ifc lines are untyped */
type Line = any;
const ref = (x: any): number | null => (x && x.type === 5 ? x.value : null);
const val = (x: any) => (x == null ? null : x.value);
const refs = (x: any): number[] => (Array.isArray(x) ? x.map(ref).filter((r) => r != null) : ref(x) != null ? [ref(x) as number] : []);

export function analyseIfc(api: IfcAPI, data: Uint8Array): IfcResult {
  const model = api.OpenModel(data, { COORDINATE_TO_ORIGIN: false });
  try {
    return analyseModel(api, model);
  } finally {
    api.CloseModel(model);
  }
}

function analyseModel(api: IfcAPI, model: number): IfcResult {
  const line = (id: number): Line => api.GetLine(model, id);
  const ids = (type: number, inherited = false): number[] => {
    const v = api.GetLineIDsWithType(model, type, inherited);
    return Array.from({ length: v.size() }, (_, i) => v.get(i));
  };

  // ── Relationships ──
  const materialOf = new Map<number, number>();
  for (const id of ids(IFCRELASSOCIATESMATERIAL)) {
    const rel = line(id);
    for (const obj of refs(rel.RelatedObjects)) materialOf.set(obj, ref(rel.RelatingMaterial)!);
  }
  const typeOf = new Map<number, number>();
  for (const id of ids(IFCRELDEFINESBYTYPE)) {
    const rel = line(id);
    for (const obj of refs(rel.RelatedObjects)) typeOf.set(obj, ref(rel.RelatingType)!);
  }
  const propDefsOf = new Map<number, number[]>();
  for (const id of ids(IFCRELDEFINESBYPROPERTIES)) {
    const rel = line(id);
    for (const obj of refs(rel.RelatedObjects)) {
      propDefsOf.set(obj, [...(propDefsOf.get(obj) ?? []), ...refs(rel.RelatingPropertyDefinition)]);
    }
  }

  const volumeScale = projectVolumeScale(api, model, ids(IFCUNITASSIGNMENT));
  const features = new Set(ids(IFCFEATUREELEMENT, true));

  // ── Materials ──
  const isConcrete = (m: Line) =>
    CONCRETE.test([val(m.Name), val(m.Category), val(m.Description)].filter(Boolean).join(" "));
  const name = (m: Line) => String(val(m.Name) ?? "");

  /** [share 0..1 | null, concrete material names, note] — mirrors concrete_share() in Python */
  const concreteShare = (elementId: number): [number | null, string[], string] => {
    let matId = materialOf.get(elementId);
    if (matId == null && typeOf.has(elementId)) matId = materialOf.get(typeOf.get(elementId)!);
    if (matId == null) return [0, [], ""];
    let mat = line(matId);
    if (mat.type === IFCMATERIALLAYERSETUSAGE) mat = line(ref(mat.ForLayerSet)!);
    if (mat.type === IFCMATERIALPROFILESETUSAGE) mat = line(ref(mat.ForProfileSet)!);

    if (mat.type === IFCMATERIAL) return [isConcrete(mat) ? 1 : 0, [name(mat)], ""];

    if (mat.type === IFCMATERIALLAYERSET) {
      const layers = refs(mat.MaterialLayers).map(line).filter((l) => ref(l.Material) != null);
      const withMat = layers.map((l) => ({ t: Number(val(l.LayerThickness) ?? 0), m: line(ref(l.Material)!) }));
      const concrete = withMat.filter((l) => isConcrete(l.m));
      const total = withMat.reduce((s, l) => s + l.t, 0);
      if (!concrete.length) return [0, [], ""];
      const names = concrete.map((l) => name(l.m));
      if (total <= 0) return [null, names, "layer thicknesses missing"];
      const share = concrete.reduce((s, l) => s + l.t, 0) / total;
      return [share, names, share === 1 ? "" : `layered, ${Math.round(share * 100)}% concrete`];
    }

    if (mat.type === IFCMATERIALPROFILESET) {
      const mats = refs(mat.MaterialProfiles).map(line).map((p) => ref(p.Material)).filter((r) => r != null).map((r) => line(r!));
      const concrete = mats.filter(isConcrete);
      if (!concrete.length) return [0, [], ""];
      if (concrete.length < mats.length) return [null, concrete.map(name), "profile set mixes concrete and other materials"];
      return [1, concrete.map(name), ""];
    }

    if (mat.type === IFCMATERIALCONSTITUENTSET) {
      const cons = refs(mat.MaterialConstituents).map(line).filter((c) => ref(c.Material) != null);
      const concrete = cons.filter((c) => isConcrete(line(ref(c.Material)!)));
      const names = concrete.map((c) => name(line(ref(c.Material)!)));
      if (!concrete.length) return [0, [], ""];
      if (concrete.length === cons.length) return [1, names, ""];
      if (concrete.every((c) => val(c.Fraction) != null)) {
        const share = concrete.reduce((s, c) => s + Number(val(c.Fraction)), 0);
        return [share, names, `constituents, ${Math.round(share * 100)}% concrete`];
      }
      return [null, names, "constituent set without fractions"];
    }

    if (mat.type === IFCMATERIALLIST) {
      const mats = refs(mat.Materials).map(line);
      const concrete = mats.filter(isConcrete);
      if (!concrete.length) return [0, [], ""];
      if (concrete.length === mats.length) return [1, concrete.map(name), ""];
      return [null, concrete.map(name), "material list mixes concrete and other materials"];
    }
    return [0, [], ""];
  };

  // ── Properties & quantities ──
  const propertySets = (elementId: number): Line[] => {
    const own = (propDefsOf.get(elementId) ?? []).map(line).filter((p) => p.type === IFCPROPERTYSET);
    const typeId = typeOf.get(elementId);
    const inherited = typeId != null ? refs(line(typeId).HasPropertySets).map(line).filter((p) => p.type === IFCPROPERTYSET) : [];
    return [...own, ...inherited]; // element values take precedence over the type's
  };

  const strengthClass = (elementId: number, materialNames: string[]): string | null => {
    const pset = propertySets(elementId).find((p) => val(p.Name) === "Pset_ConcreteElementGeneral");
    const prop = pset && refs(pset.HasProperties).map(line).find((p) => val(p.Name) === "StrengthClass");
    for (const text of [prop ? val(prop.NominalValue) : null, ...materialNames]) {
      const m = CLASS.exec(String(text ?? ""));
      if (m) return `C${m[1]}/${m[2]}`;
    }
    return null;
  };

  const quantityVolume = (elementId: number): [number, string] | null => {
    const qtos = (propDefsOf.get(elementId) ?? []).map(line).filter((p) => p.type === IFCELEMENTQUANTITY);
    for (const key of VOLUME_KEYS) {
      for (const qto of qtos) {
        const q = refs(qto.Quantities).map(line).find((x) => x.type === IFCQUANTITYVOLUME && val(x.Name) === key);
        const v = q ? Number(val(q.VolumeValue)) : NaN;
        if (v > 0) return [v * volumeScale, `${val(qto.Name)}.${key}`];
      }
    }
    return null;
  };

  const geometryVolume = (elementId: number): [number, string] | null => {
    const mesh = api.GetFlatMesh(model, elementId); // metres, openings applied
    let volume = 0;
    try {
      for (let g = 0; g < mesh.geometries.size(); g++) {
        const placed = mesh.geometries.get(g);
        const geo = api.GetGeometry(model, placed.geometryExpressID);
        const verts = api.GetVertexArray(geo.GetVertexData(), geo.GetVertexDataSize());
        const index = api.GetIndexArray(geo.GetIndexData(), geo.GetIndexDataSize());
        volume += meshVolume(verts, index, placed.flatTransformation);
        (geo as Line).delete?.(); // typed as required, but not present in every web-ifc build
      }
    } finally {
      (mesh as Line).delete?.();
    }
    volume = Math.abs(volume);
    return volume > 1e-9 ? [volume, "geometry"] : null;
  };

  // ── Elements ──
  const elements: ConcreteElement[] = [];
  const review: ReviewItem[] = [];
  for (const id of ids(IFCELEMENT, true)) {
    if (features.has(id)) continue; // openings, projections
    const [share, names, note] = concreteShare(id);
    if (share === 0) continue;
    const el = line(id);
    const base = {
      globalId: String(val(el.GlobalId) ?? ""),
      ifcClass: api.GetNameFromTypeCode(el.type),
      name: String(val(el.Name) ?? ""),
    };
    if (share === null) {
      review.push({ ...base, issue: note });
      continue;
    }
    const volume = quantityVolume(id) ?? geometryVolume(id);
    if (!volume) {
      review.push({ ...base, issue: "no volume in quantities or geometry" });
      continue;
    }
    elements.push({
      ...base,
      expressId: id,
      material: names.filter(Boolean).join(", "),
      strengthClass: strengthClass(id, names),
      volume: volume[0] * share,
      volumeSource: volume[1],
      note,
    });
  }
  return { schema: api.GetModelSchema(model), elements, review };
}

/** Factor from the project volume unit to m³ (1 if none is defined). */
function projectVolumeScale(api: IfcAPI, model: number, assignments: number[]): number {
  const siScale = (u: Line) => {
    const prefix = SI_PREFIX[val(u.Prefix) ?? ""] ?? 1;
    return val(u.UnitType) === "VOLUMEUNIT" || val(u.Name) === "CUBIC_METRE" ? prefix ** 3 : prefix;
  };
  for (const id of assignments) {
    for (const uid of refs(api.GetLine(model, id).Units)) {
      const u = api.GetLine(model, uid);
      if (val(u.UnitType) !== "VOLUMEUNIT") continue;
      if (u.type === IFCSIUNIT) return siScale(u);
      if (u.type === IFCCONVERSIONBASEDUNIT) {
        const factor = api.GetLine(model, ref(u.ConversionFactor)!);
        const component = api.GetLine(model, ref(factor.UnitComponent)!);
        return Number(val(factor.ValueComponent)) * (component.type === IFCSIUNIT ? siScale(component) : 1);
      }
    }
  }
  return 1;
}

/** Signed volume of a triangle mesh (vertices: x,y,z,nx,ny,nz) under a column-major 4×4 transform. */
function meshVolume(verts: Float32Array, index: Uint32Array, m: number[]): number {
  const p = (i: number): [number, number, number] => {
    const x = verts[i * 6], y = verts[i * 6 + 1], z = verts[i * 6 + 2];
    return [
      m[0] * x + m[4] * y + m[8] * z + m[12],
      m[1] * x + m[5] * y + m[9] * z + m[13],
      m[2] * x + m[6] * y + m[10] * z + m[14],
    ];
  };
  let v = 0;
  for (let t = 0; t < index.length; t += 3) {
    const [a, b, c] = [p(index[t]), p(index[t + 1]), p(index[t + 2])];
    v += a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
  }
  return v / 6;
}
