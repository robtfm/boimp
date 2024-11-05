# boimp
boimp is the sound a mesh makes when its lods pop. it's also a library for octahedral imposters in bevy.

# versions
version 0.1.0 requires a slightly modified bevy 0.14.2 (see [cargo.toml](Cargo.toml)).

# bake
generate an imposter with an `ImposterBakeBundle`, specifying the image size, grid count, multisampling and grid mode (spherical / hemispherical / horizontal).

```rs
commands.spawn(ImposterBakeBundle {
    camera: ImposterBakeCamera {
        radius: 10.0, // how large an area to snapshot
        grid_size: 6, // 6x6 separate snapshots
        image_size: 512, // 512x512 texture
        grid_mode: GridMode::Spherical, // how the snapshots are arranged
        multisample: 8, // how many samples to average over (^2, 8 -> 64 samples)
        ..Default::default()
    };
    transform: Transform::from_translation(Vec3::ZERO),
})
```

for anything to be produced, the materials used in the area must implement `ImposterBakeMaterial`. This is automatically implemented for `StandardMaterial`s, other implementations can be registered by adding an `ImposterMaterialPlugin::<M>`. the frag shader is quite simple, see [the standard material version](src/shaders/standard_material_imposter_baker.wgsl).

# render
render the imposter with a `MaterialMeshBundle`:

```rs
commands.spawn(MaterialMeshBundle::<Imposter> {
    mesh: meshes.add(Plane3d::new(Vec3::Z, Vec2::splat(0.5)).mesh()),
    material: asset_server.load_with_settings::<_, ImposterLoaderSettings>(source, move |s| {
        s.multisample = multisample;
    }),
    ..default()
});
```

for billboarding, use a `Rectangle` or `Plane3d::new(Vec3::Z, Vec2::splat(0.5))` mesh. for non-billboarding, any mesh will do.

# examples:
- `dynamic` - runs baking every frame (once 'I' is pressed, and until 'O' is pressed), and spawns a large number of imposters based on the bake results.
- `save_asset` - loads a gltf, bakes and saves an imposter, with baking params from the command line.
- `load_asset` - loads a previously baked imposter, with rendering params from the command line.

# todo
- integrate with visibility ranges
- improve asset format
- maybe store/adjust for depths
- maybe make the storage more configurable maybe - currently 5bit/channel color and alpha, 6bit metallic and roughness, 24bit normal, 8bit flags (only unlit flag currently passed)
- maybe add "image" mode that records the actual view rather than the material properties
- update to 0.15 and upstream