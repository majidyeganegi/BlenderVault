bl_info = {
    "name": "Folding Vaults (Relaxed option added)",
    "author": "Majid Yeganegi",
    "version": (0, 7, 0), # Updated version
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Folding Vaults",
    "description": "Fast TNA Form-finding with Regularized Topology and Advanced Smoothing.",
    "category": "Mesh",
}

import bpy
import bmesh
import mathutils
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree
import numpy as np
import random
import time

# ============================================================
# --- Sparse Matrix & Solver ---------------------------------
# ============================================================

class SparseMatrixCOO:
    def __init__(self, n_rows, n_cols):
        self.n = n_rows
        self.rows = []
        self.cols = []
        self.data = []
        self._row_np = None
        self._col_np = None
        self._data_np = None

    def add(self, r, c, val):
        self.rows.append(r)
        self.cols.append(c)
        self.data.append(val)

    def freeze(self):
        self._row_np = np.array(self.rows, dtype=np.int32)
        self._col_np = np.array(self.cols, dtype=np.int32)
        self._data_np = np.array(self.data, dtype=np.float64)

    def dot(self, x):
        y = np.zeros(self.n, dtype=np.float64)
        np.add.at(y, self._row_np, self._data_np * x[self._col_np])
        return y

def cg_solve(A_sparse, b, x0=None, max_iter=500, tol=1e-5):
    n = len(b)
    x = x0 if x0 is not None else np.zeros(n)
    r = b - A_sparse.dot(x)
    p = r.copy()
    rsold = np.dot(r, r)
    
    for i in range(max_iter):
        Ap = A_sparse.dot(p)
        p_Ap = np.dot(p, Ap)
        if p_Ap == 0: break 
        alpha = rsold / p_Ap
        x += alpha * p
        r -= alpha * Ap
        rsnew = np.dot(r, r)
        if np.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x

# ============================================================
# ----------------------- CORE ENGINE ------------------------
# ============================================================

class VaultEngine:

    @staticmethod
    def get_anchors_via_proximity(new_bm, source_obj, group_name):
        anchors = set()
        grp = source_obj.vertex_groups.get(group_name)
        valid_coords = []
        
        if grp:
            src_mesh = source_obj.data
            gid = grp.index
            for v in src_mesh.vertices:
                for g in v.groups:
                    if g.group == gid and g.weight > 0.1:
                        valid_coords.append(v.co)
                        break
        
        if not valid_coords:
            for v in new_bm.verts:
                if v.is_boundary:
                    anchors.add(v.index)
            return list(anchors)

        kd = KDTree(len(valid_coords))
        for i, co in enumerate(valid_coords):
            kd.insert(co, i)
        kd.balance()
        
        # Increased threshold slightly to catch resampled points
        threshold = 0.15  
        
        for v in new_bm.verts:
            co, index, dist = kd.find(v.co)
            if dist < threshold:
                anchors.add(v.index)
        
        if not anchors:
             for v in new_bm.verts:
                if v.is_boundary:
                    anchors.add(v.index)
                    
        return list(anchors)

    @staticmethod
    def resample_boundary(bm_src, target_len):
        """
        Walks the boundary of the source mesh and returns a list of 
        evenly spaced coordinates (2D xy) to serve as strict constraints.
        """
        boundary_coords = []
        # Find boundary edges
        b_edges = [e for e in bm_src.edges if e.is_boundary]
        
        for edge in b_edges:
            v1 = edge.verts[0].co
            v2 = edge.verts[1].co
            length = (v1 - v2).length
            
            # Determine how many segments we need
            if length <= target_len:
                boundary_coords.append(v1)
                boundary_coords.append(v2)
            else:
                count = int(max(2, length / target_len))
                for i in range(count):
                    t = i / count
                    p = v1.lerp(v2, t)
                    boundary_coords.append(p)
                boundary_coords.append(v2)

        # Remove duplicates (simple proximity check or set tuple)
        unique_map = {}
        unique_list = []
        for co in boundary_coords:
            k = (round(co.x, 4), round(co.y, 4), round(co.z, 4))
            if k not in unique_map:
                unique_map[k] = co
                unique_list.append(co)
                
        return unique_list

    @staticmethod
    def scatter_points_on_mesh(bm_source, num_points):
        """
        Scatter points inside the mesh faces.
        """
        points = []
        bm_tri = bm_source.copy()
        bmesh.ops.triangulate(bm_tri, faces=bm_tri.faces)
        bm_tri.faces.ensure_lookup_table()
        
        faces = bm_tri.faces
        weights = [f.calc_area() for f in faces]
        weight_sum = sum(weights)
        
        if weight_sum > 0:
            probs = np.array(weights) / weight_sum
            selected_indices = np.random.choice(len(faces), size=num_points, p=probs)

            for idx in selected_indices:
                face = faces[idx]
                v1, v2, v3 = face.verts[0].co, face.verts[1].co, face.verts[2].co
                r1, r2 = random.random(), random.random()
                if r1 + r2 > 1: r1, r2 = 1 - r1, 1 - r2
                p = v1 + r1 * (v2 - v1) + r2 * (v3 - v1)
                points.append(p)
            
        bm_tri.free()
        return points

    @staticmethod
    def remesh_delaunay_clean(bm, source_obj, density_factor=1.0, relax_iters=0):
        """
        Delaunay with Boundary Resampling, Lloyd's Relaxation, and Boundary Dissolve.
        """
        bm_src = bmesh.new()
        bm_src.from_mesh(source_obj.data)
        bm_src.verts.ensure_lookup_table()
        
        # 1. Calculate Target Edge Length based on area
        total_area = sum(f.calc_area() for f in bm_src.faces)
        
        base_points_count = int(total_area * density_factor * 100)
        base_points_count = max(10, min(base_points_count, 20000))
        
        # Derive target edge length from desired point count
        if base_points_count > 0:
            avg_area_per_point = total_area / base_points_count
            target_edge_len = (avg_area_per_point ** 0.5) * 1.2
        else:
            target_edge_len = 1.0

        # 2. Resample Boundary (Strategy B)
        boundary_co = VaultEngine.resample_boundary(bm_src, target_edge_len)
        
        # 3. Scatter Internal Points
        scattered_co = VaultEngine.scatter_points_on_mesh(bm_src, base_points_count)
        
        # Setup BVH for Z-mapping later
        bvh = BVHTree.FromBMesh(bm_src)
        bm_src.free()

        # Combine points
        all_points_3d = boundary_co + scattered_co
        points_2d = [p.xy for p in all_points_3d]
        
        # 4. Delaunay
        try:
            result = mathutils.geometry.delaunay_2d_cdt(
                points_2d, 
                [],  
                [], 
                0, 
                0.001
            )
        except Exception as e:
            print(f"Delaunay Error: {e}")
            return

        new_coords, _, new_faces, _, _, _ = result

        # 5. Build FLAT Mesh first
        bm.clear()
        for v_2d in new_coords:
            bm.verts.new((v_2d[0], v_2d[1], 0))
            
        bm.verts.ensure_lookup_table()
        for f_indices in new_faces:
            try:
                verts = [bm.verts[i] for i in f_indices]
                bm.faces.new(verts)
            except ValueError: pass

        bm.faces.ensure_lookup_table()
        
        # 6. Cut Holes (2D check)
        faces_to_delete = []
        down_vec = mathutils.Vector((0, 0, -1))
        
        for face in bm.faces:
            center = face.calc_center_median()
            ray_origin = mathutils.Vector((center.x, center.y, 10000))
            loc, _, _, _ = bvh.ray_cast(ray_origin, down_vec)
            if not loc:
                faces_to_delete.append(face)

        bmesh.ops.delete(bm, geom=faces_to_delete, context='FACES')
        
        # Cleanup loose verts
        bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.is_wire], context='VERTS')
        
        # 7. Lloyd's Relaxation (2D Smooth)
        if relax_iters > 0:
            # Pin boundary verts
            
            for _ in range(relax_iters):
                bmesh.ops.smooth_vert(
                    bm, 
                    verts=[v for v in bm.verts if not v.is_boundary], 
                    factor=0.5, 
                    use_axis_x=True, 
                    use_axis_y=True, 
                    use_axis_z=False
                )

        # 8. Project Z (Lift to 3D)
        bm.verts.ensure_lookup_table()
        for v in bm.verts:
            ray_origin = mathutils.Vector((v.co.x, v.co.y, 10000))
            loc, _, _, _ = bvh.ray_cast(ray_origin, down_vec)
            if loc:
                v.co.z = loc.z
            else:
                v.co.z = 0

        # **NEW: Boundary Dissolve to Clean Jagged Edges (Post Z-Project)**
        # Dissolve edges on the boundary that are near-straight to simplify and smooth
        boundary_edges = [e for e in bm.edges if e.is_boundary]
        # Use a small angle limit (0.017 radians ~= 1 degree)
        bmesh.ops.dissolve_limit(
            bm, 
            edges=boundary_edges, 
            angle_limit=0.017 
        )

        bm.normal_update()


    @staticmethod
    def solve_tna_sparse(bm, props, anchors):
        verts = bm.verts
        n = len(verts)
        if n == 0: return
        
        v_indices = {v: i for i, v in enumerate(verts)}
        pts = np.array([v.co for v in verts], dtype=np.float64)
        
        anchor_set = set(anchors)
        free_map = []
        global_to_free = {}
        free_count = 0
        
        for i in range(n):
            if i not in anchor_set:
                free_map.append(i)
                global_to_free[i] = free_count
                free_count += 1
        
        n_free = len(free_map)
        if n_free == 0: return 

        # --- FORCE DENSITY SCALING IMPROVEMENT ---
        # The internal Q value (q_internal) is inversely proportional to the resulting vault height.
        # We scale the user input (force_density_user) inversely to make smaller inputs generate larger vaults.
        BASE_Q_INTERNAL = 100.0 # Arbitrary reference value
        
        if props.force_density_user > 0:
            # Internal Q is BASE_Q_INTERNAL / user_input
            q_internal = BASE_Q_INTERNAL / props.force_density_user
        else:
            q_internal = BASE_Q_INTERNAL

        edges_idx = []
        for e in bm.edges:
            u, v = v_indices[e.verts[0]], v_indices[e.verts[1]]
            edges_idx.append((u, v))
        
        q = np.full(len(edges_idx), q_internal) # Use the scaled internal value

        # Sparse Laplacian Construction
        L_ff = SparseMatrixCOO(n_free, n_free)
        rhs_fa_x = np.zeros(n_free)
        rhs_fa_y = np.zeros(n_free)
        D = np.zeros(n)

        for idx, (u, v) in enumerate(edges_idx):
            w = q[idx]
            D[u] += w
            D[v] += w
            
            u_free = u not in anchor_set
            v_free = v not in anchor_set
            
            if u_free and v_free:
                uf, vf = global_to_free[u], global_to_free[v]
                L_ff.add(uf, vf, -w)
                L_ff.add(vf, uf, -w)
            elif u_free:
                uf = global_to_free[u]
                rhs_fa_x[uf] += w * pts[v, 0]
                rhs_fa_y[uf] += w * pts[v, 1]
            elif v_free:
                vf = global_to_free[v]
                rhs_fa_x[vf] += w * pts[u, 0]
                rhs_fa_y[vf] += w * pts[u, 1]

        for i_glob in free_map:
            if_idx = global_to_free[i_glob]
            L_ff.add(if_idx, if_idx, D[i_glob])
            
        L_ff.freeze()

        # Horizontal Relax
        if props.do_horizontal_relax:
            pts[free_map, 0] = cg_solve(L_ff, rhs_fa_x, x0=pts[free_map, 0])
            pts[free_map, 1] = cg_solve(L_ff, rhs_fa_y, x0=pts[free_map, 1])

        for i, v in enumerate(verts):
            v.co.x, v.co.y = pts[i, 0], pts[i, 1]

        # Vertical Equilibrium
        z_current = pts[:, 2].copy()
        load_val = -props.load if not props.flip else props.load
        
        for _ in range(props.iterations):
            rhs_z = np.full(n_free, load_val)
            
            for idx, (u, v) in enumerate(edges_idx):
                w = q[idx]
                u_free = u not in anchor_set
                v_free = v not in anchor_set
                
                if u_free and not v_free:
                    rhs_z[global_to_free[u]] += w * z_current[v]
                elif not u_free and v_free:
                    rhs_z[global_to_free[v]] += w * z_current[u]

            z_new = cg_solve(L_ff, rhs_z, x0=z_current[free_map], max_iter=100)
            z_current[free_map] = z_new
        
        for i, v in enumerate(verts):
            v.co.z = z_current[i]

        VaultEngine.visualize_stress(bm, verts, edges_idx, q)

    @staticmethod
    def visualize_stress(bm, verts, edges_idx, q):
        vertex_stress = np.zeros(len(verts))
        vertex_counts = np.zeros(len(verts))
        
        # Use the force density array q, which contains the internal scaled values
        for idx, (u, v) in enumerate(edges_idx):
            l = (verts[u].co - verts[v].co).length
            f = q[idx] * l
            vertex_stress[u] += f
            vertex_stress[v] += f
            vertex_counts[u] += 1
            vertex_counts[v] += 1
            
        mask = vertex_counts > 0
        vertex_stress[mask] /= vertex_counts[mask]
        
        if len(vertex_stress) > 0:
            s_min, s_max = vertex_stress.min(), vertex_stress.max()
            rng = s_max - s_min if s_max > s_min else 1.0
            
            if not bm.loops.layers.color:
                col_layer = bm.loops.layers.color.new("Vault_Stress")
            else:
                col_layer = bm.loops.layers.color.active or bm.loops.layers.color.new("Vault_Stress")
                
            for face in bm.faces:
                for loop in face.loops:
                    val = (vertex_stress[loop.vert.index] - s_min) / rng
                    # Color ramp: Blue (Low) -> Green -> Red (High)
                    if val < 0.5:
                        t = val * 2
                        c = (0.0, t, 1.0 - t, 1.0)
                    else:
                        t = (val - 0.5) * 2
                        c = (t, 1.0 - t, 0.0, 1.0)
                    loop[col_layer] = c

    @staticmethod
    def process(obj, props):
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        
        if props.use_delaunay:
            # remesh_delaunay_clean now handles the 2D relax and boundary dissolve
            VaultEngine.remesh_delaunay_clean(
                bm, 
                obj, 
                density_factor=props.delaunay_density,
                relax_iters=props.topology_relax_iters
            )
        else:
            # Handles non-Delaunay mesh conformity by ensuring triangulation
            if props.subdiv_level > 0:
                bmesh.ops.subdivide_edges(bm, edges=bm.edges, cuts=props.subdiv_level, use_grid_fill=True)
            if props.use_triangles:
                bmesh.ops.triangulate(bm, faces=bm.faces)

        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        
        # Anchors
        if props.use_delaunay:
            anchors = VaultEngine.get_anchors_via_proximity(bm, obj, props.anchor_group)
        else:
            grp = obj.vertex_groups.get(props.anchor_group)
            anchors = []
            if grp:
                deform = bm.verts.layers.deform.active
                if deform:
                    gid = grp.index
                    for v in bm.verts:
                        if gid in v[deform] and v[deform][gid] > 0.1:
                            anchors.append(v.index)
            if not anchors: 
                for v in bm.verts:
                    if v.is_boundary: anchors.append(v.index)

        # --- NEW STEP: 3D Pre-TNA Smoothing for Regularity ---
        if props.pre_tna_smooth_iters > 0:
            anchor_verts = {bm.verts[i] for i in anchors}
            smooth_verts = [v for v in bm.verts if v not in anchor_verts]
            
            for _ in range(props.pre_tna_smooth_iters):
                # Smooth all axes (XYZ) to regularize the 3D surface and reduce turbulences
                bmesh.ops.smooth_vert(
                    bm, 
                    verts=smooth_verts, 
                    factor=0.3, 
                    use_axis_x=True, 
                    use_axis_y=True, 
                    use_axis_z=True
                )

        VaultEngine.solve_tna_sparse(bm, props, anchors)
        return bm

# ============================================================
# ---------------- UI AND UPDATE SYSTEM ----------------------
# ============================================================

def update_vault(self, context):
    if not context.scene.vault_props.auto_update: return
    run_solver(context)

def run_solver(context):
    props = context.scene.vault_props
    obj = props.base_mesh
    if not obj or obj.type != 'MESH': return
    
    start_time = time.time()
    bm = VaultEngine.process(obj, props)
    
    if not bm: return
    
    name = "Vault_Output"
    res_obj = bpy.data.objects.get(name)
    if not res_obj:
        mesh = bpy.data.meshes.new(name + "_Mesh")
        res_obj = bpy.data.objects.new(name, mesh)
        context.collection.objects.link(res_obj)
    
    res_obj.matrix_world = obj.matrix_world
    
    bm.to_mesh(res_obj.data)
    bm.free()
    
    mat_name = "Vault_Stress_Mat"
    mat = bpy.data.materials.get(mat_name)
    if not mat:
        mat = bpy.data.materials.new(mat_name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        out = nodes.new('ShaderNodeOutputMaterial')
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        vcol = nodes.new('ShaderNodeVertexColor')
        vcol.layer_name = "Vault_Stress"
        links.new(vcol.outputs['Color'], bsdf.inputs['Base Color'])
        links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    
    if not res_obj.data.materials:
        res_obj.data.materials.append(mat)
    
    if res_obj.data.attributes:
        res_obj.data.attributes.active_color_index = 0
    
    res_obj.show_in_front = True

class VAULT_OT_Solve(bpy.types.Operator):
    bl_idname = "vault.solve"
    bl_label = "Compute Vault"
    def execute(self, context):
        run_solver(context)
        return {'FINISHED'}

class VaultProperties(bpy.types.PropertyGroup):
    base_mesh: bpy.props.PointerProperty(type=bpy.types.Object, update=update_vault, name="Base Mesh")
    anchor_group: bpy.props.StringProperty(name="Anchor Group", update=update_vault, description="Vertex Group for Supports")
    
    # Topology settings
    use_delaunay: bpy.props.BoolProperty(name="Use Delaunay", default=True, update=update_vault)
    delaunay_density: bpy.props.FloatProperty(name="Mesh Density", default=1.0, min=0.1, max=10.0, update=update_vault)
    # RENAME to clarify this is a 2D pre-process step
    topology_relax_iters: bpy.props.IntProperty(name="2D Relax", default=10, min=0, max=100, update=update_vault, description="Smooths the mesh layout on the XY plane before lifting (Delaunay)")
    # NEW 3D Smoothing property
    pre_tna_smooth_iters: bpy.props.IntProperty(name="3D Smooth", default=5, min=0, max=50, update=update_vault, description="Smooths the full 3D vault topology before TNA solve to reduce noise.")
    
    subdiv_level: bpy.props.IntProperty(name="Subdivision", default=1, min=0, max=4, update=update_vault)
    use_triangles: bpy.props.BoolProperty(name="Triangulate", default=True, update=update_vault)
    
    # Solver settings
    do_horizontal_relax: bpy.props.BoolProperty(name="Horizontal Relax", default=True, update=update_vault)
    # Renamed property for explicit scaling logic
    force_density_user: bpy.props.FloatProperty(
        name="Force Density", 
        default=1.0, 
        min=0.01, 
        update=update_vault, 
        description="Lower values produce higher vaults. Value 1.0 is a good starting point."
    )
    load: bpy.props.FloatProperty(name="Load", default=1.0, update=update_vault)
    flip: bpy.props.BoolProperty(name="Flip (Compression)", default=True, update=update_vault)
    iterations: bpy.props.IntProperty(name="Z Solver Iters", default=15, update=update_vault)
    
    auto_update: bpy.props.BoolProperty(name="Interactive", default=False)

class VAULT_PT_Main(bpy.types.Panel):
    bl_label = "Folding Vaults (TNA Solver)"
    bl_idname = "VAULT_PT_Main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Vaults"

    def draw(self, context):
        layout = self.layout
        props = context.scene.vault_props
        
        layout.prop(props, "base_mesh")
        if props.base_mesh:
            layout.prop_search(props, "anchor_group", props.base_mesh, "vertex_groups")
        
        box = layout.box()
        box.label(text="Topology Generation")
        box.prop(props, "use_delaunay")
        if props.use_delaunay:
            box.prop(props, "delaunay_density")
            box.prop(props, "topology_relax_iters")
            # New 3D smooth property added to Topology section
            box.prop(props, "pre_tna_smooth_iters") 
        else:
            box.prop(props, "subdiv_level")
            box.prop(props, "use_triangles")

        box = layout.box()
        box.label(text="TNA Solver")
        box.prop(props, "do_horizontal_relax", icon="GRID")
        # Updated property name
        box.prop(props, "force_density_user", text="Force Density") 
        box.prop(props, "load")
        box.prop(props, "flip")
        box.prop(props, "iterations")
        
        layout.prop(props, "auto_update", toggle=True, icon="FILE_REFRESH")
        layout.operator("vault.solve", icon="PLAY")

def register():
    bpy.utils.register_class(VaultProperties)
    bpy.utils.register_class(VAULT_OT_Solve)
    bpy.utils.register_class(VAULT_PT_Main)
    bpy.types.Scene.vault_props = bpy.props.PointerProperty(type=VaultProperties)

def unregister():
    del bpy.types.Scene.vault_props
    bpy.utils.unregister_class(VAULT_PT_Main)
    bpy.utils.unregister_class(VAULT_OT_Solve)
    bpy.utils.unregister_class(VaultProperties)

if __name__ == "__main__":
    register()