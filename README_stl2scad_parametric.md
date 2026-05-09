# STL to OpenSCAD Parametric Converter

This is a reusable Python CLI for converting STL meshes into OpenSCAD. It tries simple parametric reconstruction first, then falls back to an embedded `polyhedron()` mesh when the STL cannot be reliably rebuilt as primitives.

## Install

```bash
python3 -m pip install numpy trimesh scipy
```

## Usage

```bash
python3 stl2scad_parametric.py input.stl -o output.scad
```

Useful modes:

```bash
# Automatic primitive detection, with mesh fallback
python3 stl2scad_parametric.py input.stl -o output.scad --strategy auto

# Force the best detected primitive or extrusion
python3 stl2scad_parametric.py input.stl -o output.scad --strategy primitive

# Always embed the STL triangles as OpenSCAD polyhedron()
python3 stl2scad_parametric.py input.stl -o output.scad --strategy polyhedron

# Increase geometric tolerance for noisy STL exports
python3 stl2scad_parametric.py input.stl -o output.scad --tolerance 0.05

# Try looser reconstruction before falling back to polyhedron()
python3 stl2scad_parametric.py input.stl -o output.scad --aggressive --tolerance 0.05

# Print mesh statistics and troubleshooting hints
python3 stl2scad_parametric.py input.stl -o output.scad --diagnose
```

## What it can rebuild

- Axis-aligned boxes as `cube()`
- Rotated or bevel-heavy blocky parts as `multmatrix(...) cube(...)` approximation
- Z-axis cylinders as `cylinder()`
- Rotated cylinders as `rotate(a, v) cylinder()` using PCA axis detection
- Axis-aligned boxes with cylindrical through-holes as `difference() { cube(); cylinder(); }`
- Simple cylindrical shells or centered cylindrical through-holes as `difference() { cylinder(); cylinder(); }`
- Approximate spheres as `sphere()`
- Simple Z-axis convex extrusions as `linear_extrude() polygon()`
- Aggressive silhouette approximation as `linear_extrude() polygon()` for complex clamp/bracket-like parts
- Aggressive visual-hull approximation as `intersection()` of X/Y/Z silhouettes for complex parts that are not clean primitives
- Any mesh as `polyhedron()` fallback

## Hole and boolean detection

The boolean-style output is heuristic and intentionally conservative. It works best on clean STL exports of simple mechanical parts, especially:

- rectangular blocks with round through-holes parallel to X, Y, or Z
- hollow cylinders, tubes, washers, or cylinders with centered through-holes
- rotated cylinders where the STL mesh still has a clear dominant axis

When a hole is detected, the generated OpenSCAD uses `difference()` so you can edit the base primitive and subtractor dimensions manually.

## Important limitation

STL does not contain CAD history, sketch constraints, boolean operations, dimensions, fillets, chamfers, holes, or feature names. Because of that, a perfect editable OpenSCAD rebuild is not generally possible from STL alone. This tool uses heuristics, so it is most useful for simple mechanical parts or as a starting point for manual cleanup.

For complex objects, use the generated `polyhedron()` or manually remodel the part in OpenSCAD using the mesh as a reference.

## If you keep getting `polyhedron_fallback`

That message means the script could not safely prove that the STL is a simple primitive. Try this workflow:

```bash
python3 stl2scad_parametric.py part.stl -o part.scad --diagnose
python3 stl2scad_parametric.py part.stl -o part.scad --aggressive --tolerance 0.05
python3 stl2scad_parametric.py part.stl -o part.scad --aggressive --tolerance 0.1
```

Common reasons for fallback:

- the STL is organic or sculpted rather than CAD-like
- the part has bevels, fillets, chamfers, text, threads, or embossed details
- the mesh is not watertight
- the model is a boolean combination of many primitives
- the part is rotated and not close to an axis-aligned primitive
- the STL has too many tiny triangles from high-resolution export

For difficult parts, simplify the STL first in MeshLab, Blender, FreeCAD, or `trimesh`, then rerun with `--aggressive`.

If your local Python does not have OpenCV and you want the silhouette fallback, install it:

```bash
python3 -m pip install opencv-python
```
