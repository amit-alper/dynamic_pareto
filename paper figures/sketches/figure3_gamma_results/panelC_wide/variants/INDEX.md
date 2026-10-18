# variants of the wide panel-C sketch

Intermediate versions, kept for comparison, oldest first (the order of the design iterations). Descriptions are from the iteration log; "early example" is the 2-target gamma of the right-hand 2-D box.

| file | what differed |
|---|---|
| `_g35_zoom127` | early 2-target example gamma = 3.5; viridis dark maps; 3-D zoom 1.27 |
| `_g4_viridis` | early example gamma = 4; viridis dark maps |
| `_ygb_white` | gamma = 4; YlGnBu maps on a white background |
| `_g4_cream` | gamma = 4; YlGnBu maps on the cream background |
| `_g3_zoom119` | back to gamma = 3; continuous log-density over the whole box; 3-D zoom 1.19 |
| `_floorlow` | gamma = 3; same, with a very low colour floor (stray late-splitter lines and a very dense look) |
| *(`../figure3_panelC_wide_floor15.png`)* | colour floor at the 15th percentile (a favourite, kept one level up) |
| `_pathfraction_uncentred` | first path-fraction maps (fine bins, faint); 3-D clouds not yet centred in their frames |
| `_pathfraction_centred_faint` | 3-D clouds centred, z* triangle thicker; maps still faint |
| `_2dfraction_3dpale` | 2-D maps use coarse bins and a stronger colour exponent (clear); 3-D colours still pale |
| `_3dscatter_v1` | 3-D colours from a coarse voxel fraction, semi-transparent points |
| `_3dscatter_v2` | 3-D points opaque in the core, fading fringe |
| `_triangle_behind` | 3-D colours from the screen-space fraction; archetype triangle drawn behind the cloud (depth sorting) |
| `_triangle_solid` | triangle and circles in front of the cloud; triangle bold and solid |
| `_triangle_dotted_bold` | triangle bold dotted, z* triangle dotted |
| *(`../figure3_panelC_wide.png`)* | current: thin solid archetype triangle, thin dotted z* triangle, both in front |
