bl_info = {
    "name": "Folding Vaults (Stable II)",
    "author": "Majid Yeganegi + AI Partner",
    "version": (0, 9, 8), #Version 098b
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Vaults",
    "description": "Stable TNA with Stiffness Propagation and Precise Anchors.",
    "category": "Mesh",
}

import bpy
import bmesh
import mathutils
import numpy as np
import random

# ============================================================
# --- NUMPY LINEAR ALGEBRA (STABLE CORE) ---------------------
# ============================================================

class FastSparseMatrix:
    def __init__(self, size):
        self.n = size
        self.row = np.array([], dtype=np.int32)
        self.col = np.array([], dtype=np.int32)
        self.val = np.array([], dtype=np.float64)

    def set_data(self, rows, cols, vals):
        self.row = rows
        self.col = cols
        self.val = vals

    def dot(self, x):
        y = np.zeros(self.n, dtype=np.float64)
        np.add.at(y, self.row, self.val * x[self.col])
        return y

def cg_solve_fast(A_sparse, b, x0=None, max_iter=500, tol=1e-6):
    n = len(b)
    x = x0 if x0 is not None else np.zeros(n, dtype=np.float64)
    
    Ax = A_sparse.dot(x)
    r = b - Ax
    p = r.copy()
    rsold = np.dot(r, r)
    
    if rsold < 1e-15:
        return x

    for i in range(max_iter):
        Ap = A_sparse.dot(p)
        p_Ap = np.dot(p, Ap)
        
        if abs(p_Ap) < 1e-15: break
            
        alpha = rsold / p_Ap
        x += alpha * p
        r -= alpha * Ap
        
        rsnew = np.dot(r, r)
        if np.sqrt(rsnew) < tol: break
            
        p = r + (rsnew / rsold) * p
        rsold = rsnew
        
    return x

# ============================================================
# ----------------------- MESH PROCESSING --------------------
# ============================================================

class VaultMesher:
    
    @staticmethod
    def get_anchor_indices(obj, bm, group_name):
        """
        Identifies anchors. 
        CRITICAL FIX: Threshold set to 0.9 to prevents point supports 
        from 'bleeding' into neighbors during subdivision.
        """
        anchors = set()
        grp = obj.vertex_groups.get(group_name)
        
        if grp:
            d_layer = bm.verts.layers.deform.active
            if d_layer:
                gid = grp.index
                for v in bm.verts:
                    # STRICT THRESHOLD: 0.9 ensures only true 1.0 weights or fully anchored edges persist
                    if gid in v[d_layer] and v[d_layer][gid] > 0.9:
                        anchors.add(v.index)
        
        # Fallback: Boundary
        if not anchors:
            for v in bm.verts:
                if v.is_boundary:
                    anchors.add(v.index)
        return anchors

    @staticmethod
    def subdivide_quad_fan(bm, anchor_indices, obj, stiff_group_name):
        """
        Subdivides mesh (Fan) while propagating:
        1. Anchor logic
        2. Stiffness Weights (Vertex Group)
        """
        new_bm = bmesh.new()
        old_v_map = {} 
        input_anchors = set(anchor_indices)
        output_anchors = set()

        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        # Prepare Weight Layer for propagation
        src_d_layer = bm.verts.layers.deform.active
        dst_d_layer = new_bm.verts.layers.deform.new() # Create layer in new mesh
        
        stiff_gid = -1
        if stiff_group_name:
            grp = obj.vertex_groups.get(stiff_group_name)
            if grp: stiff_gid = grp.index

        def get_w(v):
            if src_d_layer and stiff_gid != -1 and stiff_gid in v[src_d_layer]:
                return v[src_d_layer][stiff_gid]
            return 0.0

        def set_w(nv, val):
            if val > 0:
                nv[dst_d_layer][stiff_gid] = val

        # 1. Copy Existing Verts
        for v in bm.verts:
            nv = new_bm.verts.new(v.co)
            old_v_map[v.index] = nv
            set_w(nv, get_w(v)) # Copy weight
            
            if v.index in input_anchors:
                output_anchors.add(nv)
        
        # 2. Edge Mids
        edge_map = {} 
        for e in bm.edges:
            v1, v2 = e.verts
            k = tuple(sorted((v1.index, v2.index)))
            
            mid_co = (v1.co + v2.co) * 0.5
            nv = new_bm.verts.new(mid_co)
            
            # Propagate Weight (Average)
            avg_w = (get_w(v1) + get_w(v2)) * 0.5
            set_w(nv, avg_w)
            
            edge_map[k] = nv
            
            # Propagate Anchor only if BOTH ends are anchors (Line support)
            # This prevents point supports from expanding
            if v1.index in input_anchors and v2.index in input_anchors:
                output_anchors.add(nv)
        
        # 3. Faces
        for f in bm.faces:
            # Face Center Weight = Average of all corners
            f_verts = list(f.verts)
            center_w = sum(get_w(v) for v in f_verts) / len(f_verts)
            
            center_v = new_bm.verts.new(f.calc_center_median())
            set_w(center_v, center_w)
            
            n = len(f_verts)
            for i in range(n):
                v_curr = f_verts[i]
                v_prev = f_verts[(i-1)%n]
                v_next = f_verts[(i+1)%n]
                
                nv_curr = old_v_map[v_curr.index]
                mid_prev = edge_map[tuple(sorted((v_prev.index, v_curr.index)))]
                mid_curr = edge_map[tuple(sorted((v_curr.index, v_next.index)))]
                
                new_bm.faces.new((nv_curr, mid_curr, center_v, mid_prev))
                
        new_bm.verts.index_update()
        return new_bm, {v.index for v in output_anchors}

    @staticmethod
    def remesh_delaunay(bm, source_obj, anchor_group_name, density_factor=1.0):
        bm_src = bmesh.new()
        bm_src.from_mesh(source_obj.data)
        bm_src.verts.ensure_lookup_table()
        bm_src.faces.ensure_lookup_table() # Fix for IndexError
        
        anchor_indices = VaultMesher.get_anchor_indices(source_obj, bm_src, anchor_group_name)
        anchor_map = {} 
        
        points_3d = []
        for idx in anchor_indices:
            co = bm_src.verts[idx].co
            k = (round(co.x, 4), round(co.y, 4))
            anchor_map[k] = co.z
            points_3d.append(co.copy())

        total_area = sum(f.calc_area() for f in bm_src.faces)
        count = int(total_area * density_factor * 50)
        count = max(10, min(count, 10000))
        
        areas = np.array([f.calc_area() for f in bm_src.faces])
        cum_areas = np.cumsum(areas)
        if len(cum_areas) > 0:
            total_area = cum_areas[-1]
            r_vals = np.random.rand(count) * total_area
            face_indices = np.searchsorted(cum_areas, r_vals)
            
            for f_idx in face_indices:
                f = bm_src.faces[f_idx]
                v = [v.co for v in f.verts]
                r1, r2 = random.random(), random.random()
                if r1 + r2 > 1: r1, r2 = 1 - r1, 1 - r2
                p = v[0] + r1*(v[1]-v[0]) + r2*(v[2]-v[0])
                points_3d.append(p)

        points_2d = [p.xy for p in points_3d]
        try:
            result = mathutils.geometry.delaunay_2d_cdt(points_2d, [], [], 0, 0.001)
            new_coords_2d, _, new_faces, _, _, _ = result
        except:
            bm_src.free()
            return set()

        bm.clear()
        new_verts = []
        final_anchors = set()
        
        for co_2d in new_coords_2d:
            k = (round(co_2d[0], 4), round(co_2d[1], 4))
            z = anchor_map.get(k, 0.0)
            v = bm.verts.new((co_2d[0], co_2d[1], z))
            new_verts.append(v)
            if k in anchor_map:
                final_anchors.add(v)
        
        bm.verts.ensure_lookup_table()
        for f_idx in new_faces:
            try:
                bm.faces.new([new_verts[i] for i in f_idx])
            except: pass
            
        bm.verts.index_update()
        bm_src.free()
        return {v.index for v in final_anchors}

# ============================================================
# ----------------------- TNA SOLVER -------------------------
# ============================================================

class VaultEngine:
    
    @staticmethod
    def get_force_densities(bm, obj, fd_group_name, global_fd):
        n_edges = len(bm.edges)
        q = np.full(n_edges, global_fd, dtype=np.float64)
        
        if not fd_group_name: return q
        grp = obj.vertex_groups.get(fd_group_name)
        if not grp: return q
            
        d_layer = bm.verts.layers.deform.active
        if not d_layer: return q
            
        gid = grp.index
        v_weights = np.zeros(len(bm.verts))
        for i, v in enumerate(bm.verts):
            if gid in v[d_layer]:
                v_weights[i] = v[d_layer][gid]
        
        edge_indices = np.array([[e.verts[0].index, e.verts[1].index] for e in bm.edges])
        avg_w = (v_weights[edge_indices[:, 0]] + v_weights[edge_indices[:, 1]]) * 0.5
        
        # 1.0 Weight = 10x Stiffness
        q = global_fd * (1.0 + (avg_w * 9.0))
        return q

    @staticmethod
    def solve(bm, props, anchor_indices, obj):
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        
        n_verts = len(bm.verts)
        if n_verts == 0: return

        anchors = set(anchor_indices)
        is_anchor = np.zeros(n_verts, dtype=bool)
        if anchors: is_anchor[list(anchors)] = True
        
        global_to_free = np.full(n_verts, -1, dtype=np.int32)
        free_indices_glob = np.where(~is_anchor)[0]
        n_free = len(free_indices_glob)
        global_to_free[free_indices_glob] = np.arange(n_free)
        
        if n_free == 0: return

        coords = np.array([v.co for v in bm.verts], dtype=np.float64)
        edges = np.array([[e.verts[0].index, e.verts[1].index] for e in bm.edges], dtype=np.int32)
        
        q_base = 100.0 / max(0.001, props.force_density_user)
        q = VaultEngine.get_force_densities(bm, obj, props.stiff_group, q_base)
        
        u, v = edges[:, 0], edges[:, 1]
        
        D = np.zeros(n_verts, dtype=np.float64)
        np.add.at(D, u, q)
        np.add.at(D, v, q)
        
        mask_ff = (~is_anchor[u]) & (~is_anchor[v])
        
        u_ff = global_to_free[u[mask_ff]]
        v_ff = global_to_free[v[mask_ff]]
        vals_ff = -q[mask_ff]
        
        rows = np.concatenate([u_ff, v_ff])
        cols = np.concatenate([v_ff, u_ff])
        data = np.concatenate([vals_ff, vals_ff])
        
        diag_idx = np.arange(n_free)
        diag_val = D[free_indices_glob]
        
        rows = np.concatenate([rows, diag_idx])
        cols = np.concatenate([cols, diag_idx])
        data = np.concatenate([data, diag_val])
        
        L_ff = FastSparseMatrix(n_free)
        L_ff.set_data(rows, cols, data)
        
        load_val = -props.load if not props.flip else props.load
        rhs_z = np.full(n_free, load_val, dtype=np.float64)
        
        mask_fa = (~is_anchor[u]) & (is_anchor[v])
        mask_af = (is_anchor[u]) & (~is_anchor[v])
        
        if np.any(mask_fa):
            idx = global_to_free[u[mask_fa]]
            np.add.at(rhs_z, idx, q[mask_fa] * coords[v[mask_fa], 2])
            
        if np.any(mask_af):
            idx = global_to_free[v[mask_af]]
            np.add.at(rhs_z, idx, q[mask_af] * coords[u[mask_af], 2])
            
        z_free_new = cg_solve_fast(L_ff, rhs_z, x0=coords[free_indices_glob, 2])
        coords[free_indices_glob, 2] = z_free_new
        
        for i, idx in enumerate(free_indices_glob):
            bm.verts[idx].co.z = z_free_new[i]

        VaultEngine.calc_stress_color(bm, edges, q, coords)

    @staticmethod
    def calc_stress_color(bm, edge_indices, q, coords):
        pts_u = coords[edge_indices[:, 0]]
        pts_v = coords[edge_indices[:, 1]]
        forces = q * np.linalg.norm(pts_u - pts_v, axis=1)
        
        n_verts = len(bm.verts)
        v_stress = np.zeros(n_verts)
        v_count = np.zeros(n_verts)
        
        np.add.at(v_stress, edge_indices[:, 0], forces)
        np.add.at(v_stress, edge_indices[:, 1], forces)
        np.add.at(v_count, edge_indices[:, 0], 1)
        np.add.at(v_count, edge_indices[:, 1], 1)
        
        mask = v_count > 0
        v_stress[mask] /= v_count[mask]
        
        if n_verts > 0:
            s_min, s_max = v_stress.min(), v_stress.max()
            rng = s_max - s_min if s_max > s_min else 1.0
            
            if not bm.loops.layers.color:
                col_layer = bm.loops.layers.color.new("Vault_Stress")
            else:
                col_layer = bm.loops.layers.color.active or bm.loops.layers.color.new("Vault_Stress")
            
            for face in bm.faces:
                for loop in face.loops:
                    val = (v_stress[loop.vert.index] - s_min) / rng
                    # Color Scheme: Blue(Low) -> Red(High)
                    loop[col_layer] = (val, 0.0, 1.0 - val, 1.0)

# ============================================================
# ----------------------- MAIN UI ----------------------------
# ============================================================

def run_solver(context):
    props = context.scene.vault_props
    obj = props.base_mesh
    if not obj or obj.type != 'MESH': return
    
    bm = bmesh.new()
    depsgraph = context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    bm.from_mesh(obj_eval.data)
    
    final_anchors = set()
    
    if props.use_delaunay:
        final_anchors = VaultMesher.remesh_delaunay(bm, obj, props.anchor_group, props.delaunay_density)
    else:
        # Initial anchors
        anchors = VaultMesher.get_anchor_indices(obj, bm, props.anchor_group)
        
        if props.subdiv_level > 0:
            if props.subdivide_mode == 'QUAD':
                for _ in range(props.subdiv_level):
                    bm_new, new_anchors = VaultMesher.subdivide_quad_fan(bm, anchors, obj, props.stiff_group)
                    bm.free()
                    bm = bm_new
                    anchors = new_anchors
                final_anchors = anchors
            else:
                # Triangulate/Standard Subdiv
                bmesh.ops.subdivide_edges(bm, edges=bm.edges, cuts=props.subdiv_level, use_grid_fill=True)
                bmesh.ops.triangulate(bm, faces=bm.faces)
                # Re-detect with High Threshold to avoid extra anchors
                final_anchors = VaultMesher.get_anchor_indices(obj, bm, props.anchor_group)
        else:
            if props.use_triangles:
                bmesh.ops.triangulate(bm, faces=bm.faces)
            final_anchors = anchors

    bm.verts.index_update()
    VaultEngine.solve(bm, props, final_anchors, obj)
    
    name = "Vault_Output"
    mesh_data = bpy.data.meshes.new(name + "_Mesh")
    bm.to_mesh(mesh_data)
    bm.free()
    
    res_obj = bpy.data.objects.get(name)
    if not res_obj:
        res_obj = bpy.data.objects.new(name, mesh_data)
        context.collection.objects.link(res_obj)
    else:
        old_mesh = res_obj.data
        res_obj.data = mesh_data
        if old_mesh.users == 0: bpy.data.meshes.remove(old_mesh)
            
    res_obj.matrix_world = obj.matrix_world
    
    mat_name = "Vault_Stress_Mat"
    mat = bpy.data.materials.get(mat_name)
    if not mat:
        mat = bpy.data.materials.new(mat_name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodes.clear()
        out = nodes.new('ShaderNodeOutputMaterial')
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        vcol = nodes.new('ShaderNodeVertexColor')
        vcol.layer_name = "Vault_Stress"
        mat.node_tree.links.new(vcol.outputs['Color'], bsdf.inputs['Base Color'])
        mat.node_tree.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
        
    if not res_obj.data.materials:
        res_obj.data.materials.append(mat)
        
    if res_obj.data.attributes:
        res_obj.data.attributes.active_color_index = 0

def update_vault(self, context):
    if context.scene.vault_props.auto_update:
        run_solver(context)

class VAULT_OT_Solve(bpy.types.Operator):
    bl_idname = "vault.solve"
    bl_label = "Compute Vault"
    def execute(self, context):
        run_solver(context)
        return {'FINISHED'}

class VaultProperties(bpy.types.PropertyGroup):
    base_mesh: bpy.props.PointerProperty(type=bpy.types.Object, update=update_vault, name="Base Mesh")
    anchor_group: bpy.props.StringProperty(name="Anchor Group", update=update_vault)
    stiff_group: bpy.props.StringProperty(name="Stiffness Group", update=update_vault)
    
    use_delaunay: bpy.props.BoolProperty(name="Use Delaunay", default=True, update=update_vault)
    delaunay_density: bpy.props.FloatProperty(name="Density", default=1.0, min=0.1, update=update_vault)
    
    subdivide_mode: bpy.props.EnumProperty(
        items=[('TRI', "Triangulate", ""), ('QUAD', "Quad Fan", "")],
        default='QUAD', update=update_vault, name="Mode"
    )
    subdiv_level: bpy.props.IntProperty(name="Subdivision", default=1, min=0, max=6, update=update_vault)
    use_triangles: bpy.props.BoolProperty(name="Triangulate Base", default=True, update=update_vault)
    
    force_density_user: bpy.props.FloatProperty(name="Global Stiffness", default=1.0, min=0.01, update=update_vault)
    load: bpy.props.FloatProperty(name="Load", default=1.0, update=update_vault)
    flip: bpy.props.BoolProperty(name="Flip Z", default=True, update=update_vault)
    auto_update: bpy.props.BoolProperty(name="Interactive", default=True, update=update_vault)

class VAULT_PT_Main(bpy.types.Panel):
    bl_label = "Folding Vaults II"
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
            layout.prop_search(props, "stiff_group", props.base_mesh, "vertex_groups")
        
        box = layout.box()
        box.label(text="Topology")
        box.prop(props, "use_delaunay")
        if props.use_delaunay:
            box.prop(props, "delaunay_density")
        else:
            box.prop(props, "subdivide_mode")
            box.prop(props, "subdiv_level")
            if props.subdivide_mode == 'TRI':
                box.prop(props, "use_triangles")

        box = layout.box()
        box.label(text="Solver Settings")
        box.prop(props, "force_density_user")
        box.prop(props, "load")
        box.prop(props, "flip")
        
        layout.separator()
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