"""stl2bem: sound scattering by rigid objects from STL files (CBIE + CHIEF boundary element method), for plane
waves (far field) or point sources (near field), on the GPU (CuPy) or the CPU (NumPy)."""
from .analytic import rigid_sphere, rigid_sphere_point
from .backend import HAS_CUPY, Backend, gpu_available
from .bem_core import (compute_p_total_panel_major, estimate_cost, solve_table,
                       spherical_doa_grid, write_free_field)
from .chief import chief_points, points_inside
from .panels import (auto_panel_targets, load_mic_positions, map_to_elements, project_to_surface,
                     surface_receivers)
from .stl_io import Mesh, build_mesh, mesh_from_arrays, read_stl, write_stl

__version__ = "0.1.1"

__all__ = [
    "build_mesh", "mesh_from_arrays", "read_stl", "write_stl", "Mesh",
    "chief_points", "points_inside",
    "load_mic_positions", "auto_panel_targets", "map_to_elements", "project_to_surface", "surface_receivers",
    "spherical_doa_grid", "estimate_cost", "solve_table", "compute_p_total_panel_major",
    "write_free_field", "rigid_sphere", "rigid_sphere_point", "Backend", "HAS_CUPY", "gpu_available",
]
