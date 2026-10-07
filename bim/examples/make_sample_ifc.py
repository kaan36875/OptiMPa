"""
Builds a small test model (sample_building.ifc) for ifc_carbon.py.

Expected concrete volumes:
  slab            10 x 8 x 0.25             = 20.00 m3  (C30/37, NetVolume in Qto)
  4 columns       0.4 x 0.4 x 3.0 each      =  1.92 m3  (C35/45, geometry only)
  layered wall    10 x 3 x (0.2 of 0.3)     =  6.00 m3  (C30/37 layer + insulation)
  strip footing   12 x 1.0 x 0.6            =  7.20 m3  (class only in Pset, GrossVolume in Qto)
  wall with door  5 x 3 x 0.2 - 1 x 2.1 x 0.2 = 2.58 m3  (no class -> default mix)
  -------------------------------------------------------
  total                                       37.70 m3
Not counted: a steel beam, and a concrete element with no geometry or quantities
(should be listed for review).
"""

from pathlib import Path

import numpy as np
import ifcopenshell
import ifcopenshell.api.aggregate
import ifcopenshell.api.context
import ifcopenshell.api.feature
import ifcopenshell.api.geometry
import ifcopenshell.api.material
import ifcopenshell.api.project
import ifcopenshell.api.pset
import ifcopenshell.api.root
import ifcopenshell.api.spatial
import ifcopenshell.api.unit

api = ifcopenshell.api
OUT = Path(__file__).with_name("sample_building.ifc")


def placed(f, product, x=0.0, y=0.0, z=0.0):
    matrix = np.eye(4)
    matrix[:3, 3] = (x, y, z)
    api.geometry.edit_object_placement(f, product=product, matrix=matrix)


def main():
    f = api.project.create_file(version="IFC4")
    project = api.root.create_entity(f, ifc_class="IfcProject", name="OptiMPa sample")
    api.unit.assign_unit(
        f,
        length={"is_metric": True, "raw": "METERS"},
        area={"is_metric": True, "raw": "SQUARE_METERS"},
        volume={"is_metric": True, "raw": "CUBIC_METERS"},
    )
    model = api.context.add_context(f, context_type="Model")
    body = api.context.add_context(f, context_type="Model", context_identifier="Body",
                                   target_view="MODEL_VIEW", parent=model)

    site = api.root.create_entity(f, ifc_class="IfcSite", name="Site")
    building = api.root.create_entity(f, ifc_class="IfcBuilding", name="Building")
    storey = api.root.create_entity(f, ifc_class="IfcBuildingStorey", name="Ground floor")
    api.aggregate.assign_object(f, products=[site], relating_object=project)
    api.aggregate.assign_object(f, products=[building], relating_object=site)
    api.aggregate.assign_object(f, products=[storey], relating_object=building)

    c30 = api.material.add_material(f, name="Concrete C30/37", category="concrete")
    c35 = api.material.add_material(f, name="Beton C35/45", category="concrete")
    plain = api.material.add_material(f, name="Concrete", category="concrete")
    wool = api.material.add_material(f, name="Mineral wool", category="insulation")
    steel = api.material.add_material(f, name="Steel S355", category="steel")
    elements = []

    # Slab — volume given in its quantity set
    slab = api.root.create_entity(f, ifc_class="IfcSlab", name="Ground slab")
    rep = api.geometry.add_slab_representation(f, context=body, depth=0.25,
                                               polyline=[(0, 0), (10, 0), (10, 8), (0, 8)])
    api.geometry.assign_representation(f, product=slab, representation=rep)
    placed(f, slab)
    api.material.assign_material(f, products=[slab], material=c30)
    qto = api.pset.add_qto(f, product=slab, name="Qto_SlabBaseQuantities")
    api.pset.edit_qto(f, qto=qto, properties={"NetVolume": 20.0, "GrossVolume": 20.0})
    elements.append(slab)

    # Columns — geometry only
    profile = f.create_entity("IfcRectangleProfileDef", ProfileType="AREA", XDim=0.4, YDim=0.4)
    for i, (x, y) in enumerate([(0.5, 0.5), (9.5, 0.5), (0.5, 7.5), (9.5, 7.5)], start=1):
        column = api.root.create_entity(f, ifc_class="IfcColumn", name=f"Column {i}")
        rep = api.geometry.add_profile_representation(f, context=body, profile=profile, depth=3.0)
        api.geometry.assign_representation(f, product=column, representation=rep)
        placed(f, column, x, y, 0.25)
        api.material.assign_material(f, products=[column], material=c35)
        elements.append(column)

    # Layered wall — 200 mm concrete + 100 mm insulation
    layer_set = api.material.add_material_set(f, name="Concrete wall + insulation",
                                              set_type="IfcMaterialLayerSet")
    for material, thickness in [(c30, 0.2), (wool, 0.1)]:
        layer = api.material.add_layer(f, layer_set=layer_set, material=material)
        api.material.edit_layer(f, layer=layer, attributes={"LayerThickness": thickness})
    wall = api.root.create_entity(f, ifc_class="IfcWall", name="External wall")
    rep = api.geometry.add_wall_representation(f, context=body, length=10, height=3, thickness=0.3)
    api.geometry.assign_representation(f, product=wall, representation=rep)
    placed(f, wall, 0, -0.3, 0.25)
    api.material.assign_material(f, products=[wall], type="IfcMaterialLayerSetUsage", material=layer_set)
    elements.append(wall)

    # Strip footing — class only in Pset_ConcreteElementGeneral
    footing = api.root.create_entity(f, ifc_class="IfcFooting", name="Strip footing",
                                     predefined_type="STRIP_FOOTING")
    rep = api.geometry.add_slab_representation(f, context=body, depth=0.6,
                                               polyline=[(0, 0), (12, 0), (12, 1), (0, 1)])
    api.geometry.assign_representation(f, product=footing, representation=rep)
    placed(f, footing, -1, -1, -0.6)
    api.material.assign_material(f, products=[footing], material=plain)
    pset = api.pset.add_pset(f, product=footing, name="Pset_ConcreteElementGeneral")
    api.pset.edit_pset(f, pset=pset, properties={"StrengthClass": "C25/30"})
    qto = api.pset.add_qto(f, product=footing, name="Qto_FootingBaseQuantities")
    api.pset.edit_qto(f, qto=qto, properties={"GrossVolume": 7.2})
    elements.append(footing)

    # Wall with a door opening — no class, no quantities
    inner = api.root.create_entity(f, ifc_class="IfcWall", name="Internal wall")
    rep = api.geometry.add_wall_representation(f, context=body, length=5, height=3, thickness=0.2)
    api.geometry.assign_representation(f, product=inner, representation=rep)
    placed(f, inner, 2, 4, 0.25)
    api.material.assign_material(f, products=[inner], material=plain)
    opening = api.root.create_entity(f, ifc_class="IfcOpeningElement", name="Door opening")
    rep = api.geometry.add_wall_representation(f, context=body, length=1, height=2.1, thickness=0.2)
    api.geometry.assign_representation(f, product=opening, representation=rep)
    placed(f, opening, 4, 4, 0.25)
    api.feature.add_feature(f, feature=opening, element=inner)
    elements.append(inner)

    # Steel beam — must be ignored
    beam = api.root.create_entity(f, ifc_class="IfcBeam", name="Steel beam")
    ipe = f.create_entity("IfcIShapeProfileDef", ProfileType="AREA", OverallWidth=0.15,
                          OverallDepth=0.3, WebThickness=0.0071, FlangeThickness=0.0107)
    rep = api.geometry.add_profile_representation(f, context=body, profile=ipe, depth=8.0)
    api.geometry.assign_representation(f, product=beam, representation=rep)
    placed(f, beam, 5, 0, 3.25)
    api.material.assign_material(f, products=[beam], material=steel)
    elements.append(beam)

    # Concrete element with no geometry and no quantities — should go to review
    stair = api.root.create_entity(f, ifc_class="IfcStair", name="Stair (placeholder)")
    api.material.assign_material(f, products=[stair], material=c30)
    elements.append(stair)

    api.spatial.assign_container(f, products=elements, relating_structure=storey)
    f.write(str(OUT))
    print(f"Written {OUT}")


if __name__ == "__main__":
    main()
