# Quad subdivision added
# Keeping anchor points fixed
# Stiffness feature added to edges - via vertex group could be added.
# ============================================================
bl_info = {
    "name": "BlenderVaults (08 Core + Stiffness Group)",
    "author": "Majid Yeganegi + AI Partner",
    "version": (0, 8, 7),
    "blender": (5, 0, 0),
    "location": "View3D > Sidebar > Folding Vaults",
    "description": "Fast TNA Form-finding with Regularized Topology, Advanced Smoothing, and Stiffness Weight Propagation.",
    "category": "Mesh",
}

import bpy
import bmesh
import mathutils
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree
import numpy as np
import random

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
        if abs(p_Ap) < 1e-12:  # FIX: Safety check against floating-point zero-division
            break 
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
# ----------------------- SUBDIVISION ------------------------
# ============================================================

class VaultSubdivider:
    
    @staticmethod
    def subdivide_quad_fan(bm, anchor_indices, source_obj, stiff_group_name):
        new_bm = bmesh.new()
        old_v_to_new = {}
        bm.verts.ensure_lookup_table()
        
        input_anchor_set = set(anchor_indices)
        all_anchors_new_bm = set() 
        just_created_anchors = set()

        # Handle weight layer propagation setups
        src_d_layer = bm.verts.layers.deform.active
        dst_d_layer = new_bm.verts.layers.deform.new()
        
        stiff_gid = -1
        if stiff_group_name:
            grp = source_obj.vertex_groups.get(stiff_group_name)
            if grp: stiff_gid = grp.index

        def get_w(v):
            if src_d_layer and stiff_gid != -1 and stiff_gid in v[src_d_layer]:
                return v[src_d_layer][stiff_gid]
            return 0.0

        def set_w(nv, val):
            if val > 0:
                nv[dst_d_layer][stiff_gid] = val

        # Copy existing vertices and their weights
        for v in bm.verts:
            nv = new_bm.verts.new(v.co)
            old_v_to_new[v.index] = nv
            set_w(nv, get_w(v))
            if v.index in input_anchor_set:
                all_anchors_new_bm.add(nv)

        # Edge handling: Create midpoints and interpolate weights
        edge_mid_map = {} 
        bm.edges.ensure_lookup_table()
        for e in bm.edges:
            v1, v2 = e.verts
            v1_i, v2_i = v1.index, v2.index
            key = tuple(sorted((v1_i, v2_i)))
            
            mid_co = (v1.co + v2.co) * 0.5
            mid_vert = new_bm.verts.new(mid_co)
            
            avg_w = (get_w(v1) + get_w(v2)) * 0.5
            set_w(mid_vert, avg_w)
            edge_mid_map[key] = mid_vert
            
            if v1_i in input_anchor_set and v2_i in input_anchor_set:
                all_anchors_new_bm.add(mid_vert)
                just_created_anchors.add(mid_vert)

        # Face handling: Create center vertex and reconstruct quads
        bm.faces.ensure_lookup_table()
        for f in bm.faces:
            f_verts = [v for v in f.verts]
            n_verts = len(f_verts)
            
            center_co = f.calc_center_median()
            center_vert = new_bm.verts.new(center_co)
            
            center_w = sum(get_w(v) for v in f_verts) / n_verts
            set_w(center_vert, center_w)
            
            for i in range(n_verts):
                v_curr = f_verts[i]
                nv_curr = old_v_to_new[v_curr.index]
                
                v_prev = f_verts[(i-1) % n_verts]
                v_next = f_verts[(i+1) % n_verts]
                
                edge_key_prev = tuple(sorted((v_prev.index, v_curr.index)))
                mid_prev = edge_mid_map[edge_key_prev]
                
                edge_key_curr = tuple(sorted((v_curr.index, v_next.index)))
                mid_curr = edge_mid_map[edge_key_curr]
                
                try:
                    new_bm.faces.new((nv_curr, mid_curr, center_vert, mid_prev))
                except ValueError:
                    pass

        new_bm.verts.index_update()
        
        final_all_indices = {v.index for v in all_anchors_new_bm}
        final_new_indices = {v.index for v in just_created_anchors}
        
        return new_bm, final_all_indices, final_new_indices

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
        boundary_coords = []
        b_edges = [e for e in bm_src.edges if e.is_boundary]
        for edge in b_edges:
            v1 = edge.verts[0].co
            v2 = edge.verts[1].co
            length = (v1 - v2).length
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
        bm_src = bmesh.new()
        bm_src.from_mesh(source_obj.data)
        bm_src.verts.ensure_lookup_table()
        
        total_area = sum(f.calc_area() for f in bm_src.faces)
        base_points_count = int(total_area * density_factor * 100)
        base_points_count = max(10, min(base_points_count, 20000))
        
        if base_points_count > 0:
            avg_area_per_point = total_area / base_points_count
            target_edge_len = (avg_area_per_point ** 0.5) * 1.2
        else:
            target_edge_len = 1.0

        boundary_co = VaultEngine.resample_boundary(bm_src, target_edge_len)
        scattered_co = VaultEngine.scatter_points_on_mesh(bm_src, base_points_count)
        
        bvh = BVHTree.FromBMesh(bm_src)
        bm_src.free()

        all_points_3d = boundary_co + scattered_co
        points_2d = [p.xy for p in all_points_3d]
        
        try:
            result = mathutils.geometry.delaunay_2d_cdt(points_2d, [], [], 0, 0.001)
        except Exception as e:
            print(f"Delaunay Error: {e}")
            return

        new_coords, _, new_faces, _, _, _ = result

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
        
        faces_to_delete = []
        down_vec = mathutils.Vector((0, 0, -1))
        
        for face in bm.faces:
            center = face.calc_center_median()
            ray_origin = mathutils.Vector((center.x, center.y, 10000))
            loc, _, _, _ = bvh.ray_cast(ray_origin, down_vec)
            if not loc:
                faces_to_delete.append(face)

        bmesh.ops.delete(bm, geom=faces_to_delete, context='FACES')
        bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.is_wire], context='VERTS')
        
        if relax_iters > 0:
            for _ in range(relax_iters):
                bmesh.ops.smooth_vert(
                    bm, 
                    verts=[v for v in bm.verts if not v.is_boundary], 
                    factor=0.5, 
                    use_axis_x=True, 
                    use_axis_y=True, 
                    use_axis_z=False
                )

        bm.verts.ensure_lookup_table()
        for v in bm.verts:
            ray_origin = mathutils.Vector((v.co.x, v.co.y, 10000))
            loc, _, _, _ = bvh.ray_cast(ray_origin, down_vec)
            if loc: v.co.z = loc.z
            else: v.co.z = 0

        boundary_edges = [e for e in bm.edges if e.is_boundary]
        bmesh.ops.dissolve_limit(bm, edges=boundary_edges, angle_limit=0.017)
        bm.normal_update()

    @staticmethod
    def solve_tna_sparse(bm, props, anchors, source_obj):
        verts = bm.verts
        n = len(verts)
        if n == 0: return
        
        v_indices = {v: i for i, v in enumerate(verts)}
        pts = np.array([v.co for v in verts], dtype=np.float64)
        
        # --- ANCHOR SETUP ---
        anchor_set = set(anchors)
        
        # For RhinoVault behavior: Option to pin boundary positions during horizontal relaxation
        horizontal_anchor_set = anchor_set.copy()
        if props.pin_boundaries:
            for v in verts:
                if v.is_boundary:
                    horizontal_anchor_set.add(v.index)
        
        # --- SETUP FREE MAPS FOR HORIZONTAL SOLVER ---
        free_map_h = []
        global_to_free_h = {}
        free_count_h = 0
        for i in range(n):
            if i not in horizontal_anchor_set:
                free_map_h.append(i)
                global_to_free_h[i] = free_count_h
                free_count_h += 1
                
        # --- SETUP FREE MAPS FOR VERTICAL Z SOLVER ---
        free_map_z = []
        global_to_free_z = {}
        free_count_z = 0
        for i in range(n):
            if i not in anchor_set:
                free_map_z.append(i)
                global_to_free_z[i] = free_count_z
                free_count_z += 1

        BASE_Q_INTERNAL = 100.0
        if props.force_density_user > 0:
            q_internal = BASE_Q_INTERNAL / props.force_density_user
        else:
            q_internal = BASE_Q_INTERNAL

        edges_idx = []
        for e in bm.edges:
            u, v = v_indices[e.verts[0]], v_indices[e.verts[1]]
            edges_idx.append((u, v))
        
        # --- STIFFNESS PROPERTY EVALUATION ---
        q = np.full(len(edges_idx), q_internal)
        if props.stiff_group:
            grp = source_obj.vertex_groups.get(props.stiff_group)
            if grp:
                gid = grp.index
                d_layer = bm.verts.layers.deform.active
                v_weights = np.zeros(n)
                
                for i, v in enumerate(verts):
                    if d_layer and gid in v[d_layer]:
                        v_weights[i] = v[d_layer][gid]
                
                for idx, (u, v) in enumerate(edges_idx):
                    avg_w = (v_weights[u] + v_weights[v]) * 0.5
                    # IMPROVEMENT: Use UI controlled stiffness multiplier property
                    q[idx] = q_internal * (1.0 + (avg_w * props.stiffness_multiplier))

        # ============================================================
        # --- HORIZONTAL RELAXATION MATRIX ASSEMBLY ------------------
        # ============================================================
        if props.do_horizontal_relax and free_count_h > 0:
            L_ff_h = SparseMatrixCOO(free_count_h, free_count_h)
            rhs_fa_x = np.zeros(free_count_h)
            rhs_fa_y = np.zeros(free_count_h)
            D_h = np.zeros(n)

            for idx, (u, v) in enumerate(edges_idx):
                w = q[idx]
                D_h[u] += w
                D_h[v] += w
                
                u_free = u not in horizontal_anchor_set
                v_free = v not in horizontal_anchor_set
                
                if u_free and v_free:
                    uf, vf = global_to_free_h[u], global_to_free_h[v]
                    L_ff_h.add(uf, vf, -w)
                    L_ff_h.add(vf, uf, -w)
                elif u_free:
                    uf = global_to_free_h[u]
                    rhs_fa_x[uf] += w * pts[v, 0]
                    rhs_fa_y[uf] += w * pts[v, 1]
                elif v_free:
                    vf = global_to_free_h[v]
                    rhs_fa_x[vf] += w * pts[u, 0]
                    rhs_fa_y[vf] += w * pts[u, 1]

            for i_glob in free_map_h:
                if_idx = global_to_free_h[i_glob]
                L_ff_h.add(if_idx, if_idx, D_h[i_glob])
                
            L_ff_h.freeze()
            pts[free_map_h, 0] = cg_solve(L_ff_h, rhs_fa_x, x0=pts[free_map_h, 0])
            pts[free_map_h, 1] = cg_solve(L_ff_h, rhs_fa_y, x0=pts[free_map_h, 1])

            for i, v in enumerate(verts):
                v.co.x, v.co.y = pts[i, 0], pts[i, 1]

        # ============================================================
        # --- VERTICAL SOLVER (Z CORE) ASSEMBLY ----------------------
        # ============================================================
        if free_count_z > 0:
            L_ff_z = SparseMatrixCOO(free_count_z, free_count_z)
            D_z = np.zeros(n)

            for idx, (u, v) in enumerate(edges_idx):
                w = q[idx]
                D_z[u] += w
                D_z[v] += w
                
                u_free = u not in anchor_set
                v_free = v not in anchor_set
                
                if u_free and v_free:
                    uf, vf = global_to_free_z[u], global_to_free_z[v]
                    L_ff_z.add(uf, vf, -w)
                    L_ff_z.add(vf, uf, -w)

            for i_glob in free_map_z:
                if_idx = global_to_free_z[i_glob]
                L_ff_z.add(if_idx, if_idx, D_z[i_glob])
                
            L_ff_z.freeze()

            z_current = pts[:, 2].copy()
            load_val = -props.load if not props.flip else props.load
            
            for _ in range(props.iterations):
                rhs_z = np.full(free_count_z, load_val)
                for idx, (u, v) in enumerate(edges_idx):
                    w = q[idx]
                    u_free = u not in anchor_set
                    v_free = v not in anchor_set
                    if u_free and not v_free:
                        rhs_z[global_to_free_z[u]] += w * z_current[v]
                    elif not u_free and v_free:
                        rhs_z[global_to_free_z[v]] += w * z_current[u]
                
                z_new = cg_solve(L_ff_z, rhs_z, x0=z_current[free_map_z], max_iter=100)
                z_current[free_map_z] = z_new
            
            for i, v in enumerate(verts):
                v.co.z = z_current[i]

        VaultEngine.visualize_stress(bm, verts, edges_idx, q)

    @staticmethod
    def visualize_stress(bm, verts, edges_idx, q):
        vertex_stress = np.zeros(len(verts))
        vertex_counts = np.zeros(len(verts))
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
        
        anchor_data = {}

        if props.use_delaunay:
            VaultEngine.remesh_delaunay_clean(
                bm, 
                obj, 
                density_factor=props.delaunay_density,
                relax_iters=props.topology_relax_iters
            )
            base_anchors = VaultEngine.get_anchors_via_proximity(bm, obj, props.anchor_group)
            anchor_data[0] = base_anchors
            final_anchors = base_anchors
        
        else:
            grp = obj.vertex_groups.get(props.anchor_group)
            initial_anchors = []
            if grp:
                deform = bm.verts.layers.deform.active
                if deform:
                    gid = grp.index
                    for v in bm.verts:
                        if gid in v[deform] and v[deform][gid] > 0.1:
                            initial_anchors.append(v.index)
            if not initial_anchors: 
                for v in bm.verts:
                    if v.is_boundary: initial_anchors.append(v.index)
            
            anchor_data[0] = initial_anchors
            current_anchors = set(initial_anchors)

            if props.subdiv_level > 0:
                if props.subdivide_mode == 'QUAD':
                    for lvl in range(1, props.subdiv_level + 1):
                        bm_next, all_anchors_set, new_anchors_set = VaultSubdivider.subdivide_quad_fan(
                            bm, current_anchors, obj, props.stiff_group
                        )
                        anchor_data[lvl] = list(new_anchors_set)
                        bm.free()
                        bm = bm_next
                        current_anchors = all_anchors_set
                    
                    final_anchors = list(current_anchors)

                else:
                    bmesh.ops.subdivide_edges(bm, edges=bm.edges, cuts=props.subdiv_level, use_grid_fill=True)
                    if props.use_triangles:
                        bmesh.ops.triangulate(bm, faces=bm.faces)
                    
                    final_anchors = VaultEngine.get_anchors_via_proximity(bm, obj, props.anchor_group)
                    anchor_data[0] = final_anchors 
            else:
                if props.use_triangles:
                    bmesh.ops.triangulate(bm, faces=bm.faces)
                final_anchors = initial_anchors

        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        # --- PRE-TNA SMOOTHING ---
        if props.pre_tna_smooth_iters > 0:
            anchor_verts = {bm.verts[i] for i in final_anchors}
            # IMMEDIATE FIX: Exclude all boundary vertices so outer edges don't migrate/shrink inwards
            smooth_verts = [v for v in bm.verts if (v not in anchor_verts) and (not v.is_boundary)]
            
            if smooth_verts:
                for _ in range(props.pre_tna_smooth_iters):
                    bmesh.ops.smooth_vert(
                        bm, 
                        verts=smooth_verts, 
                        factor=0.3, 
                        use_axis_x=True, 
                        use_axis_y=True, 
                        use_axis_z=True
                    )

        VaultEngine.solve_tna_sparse(bm, props, final_anchors, obj)
        
        return bm, anchor_data

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
    
    bm, anchor_data = VaultEngine.process(obj, props)
    if not bm: return
    
    name = "Vault_Output"
    res_obj = bpy.data.objects.get(name)
    if not res_obj:
        mesh = bpy.data.meshes.new(name + "_Mesh")
        res_obj = bpy.data.objects.new(name, mesh)
        context.collection.objects.link(res_obj)
    else:
        # FIX: Object database link safeguard. If active collection changes, link if not present.
        if res_obj.name not in context.collection.objects:
            context.collection.objects.link(res_obj)
    
    res_obj.matrix_world = obj.matrix_world
    bm.to_mesh(res_obj.data)
    bm.free()
    
    res_obj.vertex_groups.clear()
    main_vg = res_obj.vertex_groups.new(name="anchors_combined")
    
    all_indices = set()
    for lvl, indices in anchor_data.items():
        if lvl > 0:
            lvl_vg = res_obj.vertex_groups.new(name=f"anchorpoint subd level {lvl}")
            lvl_vg.add(list(indices), 1.0, 'REPLACE')
        for idx in indices:
            all_indices.add(idx)
            
    main_vg.add(list(all_indices), 1.0, 'REPLACE')

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
    stiff_group: bpy.props.StringProperty(name="Stiffness Group", update=update_vault, description="Vertex Group for local stiffness distribution")
    
    # IMPROVEMENT: Exposed weight scale multiplier configuration parameters
    stiffness_multiplier: bpy.props.FloatProperty(name="Stiffness Multiplier", default=9.0, min=0.0, max=100.0, update=update_vault, description="Controls internal force scaling based on vertex weights")
    pin_boundaries: bpy.props.BoolProperty(name="Pin Boundaries (XY)", default=True, update=update_vault, description="Keeps non-anchor perimeter boundary vertices locked in X/Y plane during relaxation")
    
    use_delaunay: bpy.props.BoolProperty(name="Use Delaunay", default=True, update=update_vault)
    delaunay_density: bpy.props.FloatProperty(name="Mesh Density", default=1.0, min=0.1, max=10.0, update=update_vault)
    topology_relax_iters: bpy.props.IntProperty(name="2D Relax", default=10, min=0, max=100, update=update_vault)
    pre_tna_smooth_iters: bpy.props.IntProperty(name="3D Smooth", default=5, min=0, max=50, update=update_vault)
    
    subdivide_mode: bpy.props.EnumProperty(
        name="Mode",
        items=[
            ('TRI', "Triangulate", "Standard subdivision and triangulation"),
            ('QUAD', "Quad (Custom)", "Fan subdivision: Center -> Edge Mids. Preserves anchors.")
        ],
        default='QUAD',
        update=update_vault
    )
    subdiv_level: bpy.props.IntProperty(name="Subdivision", default=1, min=0, max=6, update=update_vault)
    use_triangles: bpy.props.BoolProperty(name="Triangulate", default=True, update=update_vault)
    
    do_horizontal_relax: bpy.props.BoolProperty(name="Horizontal Relax", default=True, update=update_vault)
    force_density_user: bpy.props.FloatProperty(name="Force Density", default=1.0, min=0.01, update=update_vault)
    load: bpy.props.FloatProperty(name="Load", default=1.0, update=update_vault)
    flip: bpy.props.BoolProperty(name="Flip (Compression)", default=True, update=update_vault)
    iterations: bpy.props.IntProperty(name="Z Solver Iters", default=15, update=update_vault)
    
    auto_update: bpy.props.BoolProperty(name="Interactive", default=False)

class VAULT_PT_Main(bpy.types.Panel):
    bl_label = "Folding Vaults (TNA Solver v08-Stiff)"
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
            if props.stiff_group:
                layout.prop(props, "stiffness_multiplier")
        
        box = layout.box()
        box.label(text="Topology Generation")
        box.prop(props, "use_delaunay")
        
        if props.use_delaunay:
            box.prop(props, "delaunay_density")
            box.prop(props, "topology_relax_iters")
            box.prop(props, "pre_tna_smooth_iters") 
        else:
            box.prop(props, "subdiv_level")
            box.prop(props, "subdivide_mode")
            if props.subdivide_mode == 'TRI':
                box.prop(props, "use_triangles")
            box.prop(props, "pre_tna_smooth_iters")

        box = layout.box()
        box.label(text="TNA Solver")
        box.prop(props, "do_horizontal_relax", icon="GRID")
        if props.do_horizontal_relax:
            box.prop(props, "pin_boundaries")
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