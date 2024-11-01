#import bevy_pbr::{
    view_transformations::position_view_to_world,
}

#import "shaders/shared.wgsl"::{ImposterData, GRID_HEMISPHERICAL, spherical_normal_from_uv, spherical_uv_from_normal};

@group(2) @binding(200)
var<uniform> imposter_data: ImposterData;

@group(2) @binding(201) 
#ifdef IMPOSTER_MATERIAL
    var imposter_texture: texture_2d<u32>;
#else
    var imposter_texture: texture_2d<f32>;
#endif

@group(2) @binding(202) 
var imposter_sampler: sampler;

fn oct_mode_uv_from_normal(dir: vec3<f32>) -> vec2<f32> {
	if ((imposter_data.flags & GRID_HEMISPHERICAL) != 0) {
        var dir2: vec3<f32> = dir;
        dir2.y = max(dir2.y, 0.0);
        dir2 = normalize(dir2);
        var octant: vec3<f32> = sign(dir2);
        var sum: f32 = dot(dir2, octant);
        var octahedron: vec3<f32> = dir2 / sum;
        return (vec2<f32>(octahedron.x + octahedron.z, octahedron.z - octahedron.x) + 1.0) * 0.5;
    } else {
        return spherical_uv_from_normal(dir);
	}
}

struct Basis {
    normal: vec3<f32>,
    up: vec3<f32>,
}

fn oct_mode_normal_from_uv(uv: vec2<f32>, inv_rot: mat3x3<f32>) -> Basis {
    var n: vec3<f32>;
	if ((imposter_data.flags & GRID_HEMISPHERICAL) != 0) {
        var x = uv.x - uv.y;
        var z = -1.0 + uv.x + uv.y;
        var y = 1.0 - abs(x) - abs(z);
        n = normalize(vec3(x, y, z));
    } else {
        n = spherical_normal_from_uv(uv);
    }

    var up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(n.y) > 0.99);

    var basis: Basis;
    basis.normal = inv_rot * n;
    basis.up = inv_rot * up;
    return basis;
}

fn grid_weights(coords: vec2<f32>) -> vec4<f32> {
    var corner_tr = select(0.0, 1.0, coords.x > coords.y);
    var corner_bl = 1.0 - corner_tr;
    var corner = abs(coords.x - coords.y);
    var res = vec4<f32>(
        1.0 - max(coords.x, coords.y),
        corner_tr * corner,
        corner_bl * corner,
        min(coords.x, coords.y),
    );
    return res / (res.x + res.y + res.z + res.w);
}

fn sample_uvs(base_world_position: vec3<f32>, world_position: vec3<f32>, inv_rot: mat3x3<f32>, grid_index: vec2<f32>) -> vec2<f32> {
    var grid_count = f32(imposter_data.grid_size);
    var tile_origin = grid_index / grid_count;
    var tile_size = 1.0 / grid_count;
    var basis = oct_mode_normal_from_uv(tile_origin * grid_count / (grid_count - 1.0), inv_rot);
    var sample_normal = basis.normal;
    var camera_world_position = position_view_to_world(vec3<f32>(0.0));
    var cam_to_fragment = normalize(world_position - camera_world_position);
    var distance = dot(base_world_position - camera_world_position, sample_normal) / dot(cam_to_fragment, sample_normal);
    var intersect = distance * cam_to_fragment + camera_world_position;
    // calculate uv using basis of the sample plane
    var sample_r = normalize(cross(sample_normal, -basis.up));
    var sample_u = normalize(cross(sample_r, sample_normal));
    var v = intersect - base_world_position;
    var x = dot(v, sample_r / (imposter_data.center_and_scale.w * 2.0));
    var y = dot(v, sample_u / (imposter_data.center_and_scale.w * 2.0));
    var uv = vec2<f32>(x, y) + 0.5;
    
    return select(
        vec2(tile_origin + tile_size * uv),
        vec2(-1.0),
        any(clamp(uv, vec2(0.0), vec2(1.0)) != uv)
    );
}

fn sample_tile(base_world_position: vec3<f32>, world_position: vec3<f32>, inv_rot: mat3x3<f32>, grid_index: vec2<f32>) -> vec4<f32> {
    var uv = sample_uvs(base_world_position, world_position, inv_rot, grid_index);
    var sample_tl = textureSample(imposter_texture, imposter_sampler, uv);
    return select(sample_tl, vec4(0.0), uv.x < 0.0);
}

fn sample_tile_material(base_world_position: vec3<f32>, world_position: vec3<f32>, inv_rot: mat3x3<f32>, grid_index: vec2<f32>) -> vec2<u32> {
    var uv = sample_uvs(base_world_position, world_position, inv_rot, grid_index);
    var coords = vec2<u32>(uv * vec2<f32>(textureDimensions(imposter_texture)));
    var pixel = textureLoad(imposter_texture, coords, 0);
    return select(pixel.xy, vec2(0u), uv.x < 0.0);
}