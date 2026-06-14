# Data Audit

```json
{
  "ifc": {
    "path": "data/raw/craslabbim.ifc",
    "exists": true,
    "size_bytes": 67553572,
    "ifc_schema": "IFC2X3",
    "entity_count": 1328372,
    "ifcproduct_like_count": 1328369,
    "top_entities": {
      "IFCPOLYLOOP": 368890,
      "IFCFACEOUTERBOUND": 368753,
      "IFCFACE": 368753,
      "IFCCARTESIANPOINT": 209933,
      "IFCAXIS2PLACEMENT3D": 1319,
      "IFCSTYLEDITEM": 1201,
      "IFCAXIS2PLACEMENT2D": 884,
      "IFCEXTRUDEDAREASOLID": 828,
      "IFCPROPERTYSET": 819,
      "IFCRECTANGLEPROFILEDEF": 716,
      "IFCRELDEFINESBYPROPERTIES": 644,
      "IFCPROPERTYSINGLEVALUE": 535,
      "IFCCOMPOSITECURVESEGMENT": 502,
      "IFCPOLYLINE": 481,
      "IFCSHAPEREPRESENTATION": 396,
      "IFCCLOSEDSHELL": 387,
      "IFCFACETEDBREP": 387,
      "IFCRELASSOCIATESMATERIAL": 280,
      "IFCLOCALPLACEMENT": 261,
      "IFCPRODUCTDEFINITIONSHAPE": 256,
      "IFCMATERIALLIST": 197,
      "IFCFACEBOUND": 137,
      "IFCCIRCLE": 132,
      "IFCTRIMMEDCURVE": 132,
      "IFCMAPPEDITEM": 127
    },
    "material_count_text": 35,
    "material_properties_count_text": 0,
    "has_ifcopenshell": false,
    "ifcopenshell_error": "No module named 'ifcopenshell'"
  },
  "pointcloud_zip": {
    "path": "data/raw/craslabannotated.zip",
    "exists": true,
    "size_bytes": 4267281425,
    "file_count": 1,
    "first_files": [
      "CRASLAB_annotated.asc"
    ],
    "total_uncompressed_bytes": 26752658348,
    "suffix_counts": {
      ".asc": 1
    }
  },
  "md5": {
    "ifc": "e20658f0d2d9e13c62363169b7fa3193",
    "pointcloud_zip": "e5ecedab8f2a1d1f91861a3aec028a72"
  },
  "edf": {
    "path": "data/raw/edf",
    "exists": false,
    "note": "EDF Challenge requires user login after download; continuing CRAS primary experiment."
  },
  "ts40k": {
    "path": "external/TS40K",
    "exists": false,
    "note": "TS40K repository is optional and should not download full large data before CRAS smoke completion."
  }
}
```

## ASC Sample Inspection

- ZIP member: `CRASLAB_annotated.asc`
- Full stream line count from `unzip -p ... | wc -l`: `584701979`
- Header line: `//X Y Z R G B Intensity Classification`
- First 5000 non-empty rows after opening the stream:
  - Column-count distribution: `{1: 1, 8: 4999}`
  - Valid numeric point rows in sample: `4999`
  - Sample coordinate min: `[6.5095, 15.6035, 1.4935]`
  - Sample coordinate max: `[6.6135, 16.2405, 1.6615]`
  - Sample RGB min: `[76, 77, 87]`
  - Sample RGB max: `[190, 181, 206]`
  - Sample intensity min/max: `[0.025, 0.441]`
  - Sample classification values: `[0]`

The ZIP manifest contains one fused ASC file, not 21 independent scan-station files. Therefore the current CRAS path must synthesize virtual scan positions from geometry or a fused point cloud if original sensor poses are not available elsewhere.
