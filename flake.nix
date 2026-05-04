{
  description = "Rust + MuJoCo Environment pinned to nixpkgs commit 09061f748ee21f68a089cd5d91ec1859cd93d0be";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/09061f748ee21f68a089cd5d91ec1859cd93d0be";
    # 1. Add rust-overlay to get target-specific toolchains
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = { self, nixpkgs, rust-overlay }:
    let
      system = "x86_64-linux"; 
      
      # Apply the overlay
      overlays = [ (import rust-overlay) ];
      pkgs = import nixpkgs { inherit system overlays; };

      # 2. Define a custom Rust toolchain that includes the aarch64 target
      rustToolchain = pkgs.rust-bin.stable.latest.default.override {
        targets = [ "aarch64-unknown-linux-gnu" ];
      };

      runtimeLibs = with pkgs; [
        mujoco udev libGL glfw wayland
        libxkbcommon libdecor fontconfig
        libx11 libxrandr libxi libxcursor
        libxext libxinerama stdenv.cc.cc.lib
      ];
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        nativeBuildInputs = with pkgs; [
          rustToolchain     # Replaces 'rustc' and 'cargo'
          cargo-cross
          rust-analyzer
          pkg-config gcc binutils mujoco
        ];

        buildInputs = runtimeLibs;

        MUJOCO_DYNAMIC_LINK_DIR = "${pkgs.mujoco}/lib";
        CPATH = "${pkgs.mujoco}/include";
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibs;
        MUJOCO_GL = "glfw"; 

        shellHook = ''
          echo "========================================="
          echo "🦀 Rust + MuJoCo Environment (Fixed Linker)"
          echo "========================================="
          export RUSTFLAGS="-C linker=cc"
          
          # 3. Inform cross-rs that the toolchain is manually managed
          export CROSS_CUSTOM_TOOLCHAIN=1
        '';
      };
    };
}
