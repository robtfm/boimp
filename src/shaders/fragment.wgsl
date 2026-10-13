#import bevy_pbr::{
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
    pbr_types::{STANDARD_MATERIAL_FLAGS_UNLIT_BIT, STANDARD_MATERIAL_FLAGS_FOG_ENABLED_BIT},
    view_transformations::{direction_view_to_world, position_view_to_world, position_world_to_clip}
}

#ifdef PREPASS_PIPELINE
 #import bevy_pbr::prepass_io::FragmentOutput;
#else
 #import bevy_pbr::forward_io::FragmentOutput;
#endif

#import boimp::shared::{ImposterVertexOut, unpack_pbrinput, weighted_props};
#import boimp::bindings::{imposter_data, sample_positions_from_camera_dir, sample_uvs_unbounded, sample_tile, sample_tile_material};

@fragment
fn fragment(in: ImposterVertexOut) -> FragmentOutput {
    var out: FragmentOutput;

    let inv_rot = mat3x3(
        in.inverse_rotation_0c,
        in.inverse_rotation_1c,
        in.inverse_rotation_2c,
    );

    let camera_world_position = position_view_to_world(vec3<f32>(0.0));
#ifdef VIEW_PROJECTION_ORTHOGRAPHIC
    let back_vec = direction_view_to_world(vec3<f32>(0.0, 0.0, 1.0));
#else
    let back_vec = camera_world_position - in.world_position;
#endif

    // For looking-up views past horizontal, snap to the closest in-grid
    // angle (the horizon ring). The grid has no tile coverage above
    // horizontal in hemispherical mode, so without this the closest-3
    // tiles selection swings to whatever's nearest in uv and the
    // imposter jumps; clamping freezes it at its horizontal appearance.
    // Only applies in hemispherical mode — spherical covers all
    // directions and horizontal mode is a 1D ring around the equator.
#ifdef GRID_HEMISPHERICAL
    let back = normalize(vec3<f32>(back_vec.x, max(back_vec.y, 0.0), back_vec.z));
#else
    let back = normalize(back_vec);
#endif

    let samples = sample_positions_from_camera_dir(back * inv_rot);

    let uv_a = sample_uvs_unbounded(in.base_world_position, in.world_position, inv_rot, samples.tile_indices[0], in.content_half_extent);
    let uv_b = sample_uvs_unbounded(in.base_world_position, in.world_position, inv_rot, samples.tile_indices[1], in.content_half_extent);

    let props_a = sample_tile_material(uv_a, samples.tile_indices[0], vec2(0.0));
    let props_b = sample_tile_material(uv_b, samples.tile_indices[1], vec2(0.0));

#ifndef GRID_HORIZONTAL
    let uv_c = sample_uvs_unbounded(in.base_world_position, in.world_position, inv_rot, samples.tile_indices[2], in.content_half_extent);
    let props_c = sample_tile_material(uv_c, samples.tile_indices[2], vec2(0.0));
#endif

    let weights = samples.tile_weights;
    let props_ab = weighted_props(props_a, props_b, weights.x / max(weights.x + weights.y, 0.0001));
#ifndef GRID_HORIZONTAL
    let props_final = weighted_props(props_ab, props_c, (weights.x + weights.y) / (weights.x + weights.y + weights.z));
#else 
    let props_final = props_ab;
#endif

    if props_final.rgba.a < 0.01 {
        discard;
        // out.color = vec4(0.0, 0.2, 0.0, 0.2);
        // return out;
    }

// we can discard based on actual depth if we have the depth prepass data
#ifdef DEPTH_PREPASS
#ifndef PREPASS_PIPELINE
    let existing_depth_ndc = bevy_pbr::prepass_utils::prepass_depth(in.position, 0u);
    // Reconstruct surface from imposter *center* (`base_world_position`),
    // not from the cube-face fragment position. The stored depth encodes
    // `(P - C) · T / scale`, so the center is the correct origin to add
    // the parallax offset to.
    let imposted_ndc = position_world_to_clip(in.base_world_position + back * props_final.depth * imposter_data.center_and_scale.w);
    let imposter_depth_ndc = imposted_ndc.z / imposted_ndc.w;
    
    if imposter_depth_ndc < existing_depth_ndc {
        // out.color = vec4<f32>(0.0, 0.5, 0.0, 0.5);
        // return out;
        discard;
    }
#endif
#endif

    var pbr_input = unpack_pbrinput(props_final, in.position);    
    pbr_input.N = inv_rot * normalize(pbr_input.N);
    pbr_input.world_normal = pbr_input.N;

    pbr_input.material.base_color.a *= imposter_data.alpha;

#ifdef PREPASS_PIPELINE
    #ifdef NORMAL_PREPASS
        out.normal = vec4<f32>(pbr_input.N, 0.0);
    #endif
    // we don't support MOTION_VECTOR or DEFERRED
    #ifdef DEPTH_CLAMP_ORTHO
        out.frag_depth = in.position.z;
    #endif
#else 
    if (pbr_input.material.flags & STANDARD_MATERIAL_FLAGS_UNLIT_BIT) == 0u {
        out.color = apply_pbr_lighting(pbr_input);
    } else {
        out.color = pbr_input.material.base_color;
    }

    pbr_input.material.flags |= STANDARD_MATERIAL_FLAGS_FOG_ENABLED_BIT;

    out.color = main_pass_post_lighting_processing(pbr_input, out.color);

#ifdef TRANSPARENT_FOCUS_OUTPUT
    // depth of field: an imposter stands in for distant geometry, so it is always as far away
    // as the effect allows, covering the pixel by its alpha
    out.focus = vec4<f32>(1.0e5, 1.0, 0.0, out.color.a);
#endif
#endif

    // out.color = clamp(out.color, vec4<f32>(0.2, 0.0, 0.0, 0.2), vec4<f32>(1.0, 1.0, 1.0, 0.7));

    return out;
}
