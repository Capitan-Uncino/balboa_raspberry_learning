{
  description = "Rust + MuJoCo Environment (Native Cross-Compile)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/09061f748ee21f68a089cd5d91ec1859cd93d0be";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = { self, nixpkgs, rust-overlay }:
    let
      system = "x86_64-linux";
      
      overlays = [ (import rust-overlay) ];
      pkgs = import nixpkgs { inherit system overlays; };

      # ✨ NEW: The entire ARM64 cross-compiled package set
      crossPkgs = pkgs.pkgsCross.aarch64-multiplatform;

      rustToolchain = pkgs.rust-bin.stable.latest.default.override {
        targets = [ "aarch64-unknown-linux-gnu" ];
      };

      # These are your x86_64 host libraries for running the simulation locally
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
          rustToolchain
          rust-analyzer
          pkg-config mujoco binutils
          crossPkgs.stdenv.cc # Injects the aarch64 gcc into the shell
        ];

        buildInputs = runtimeLibs;

        MUJOCO_DYNAMIC_LINK_DIR = "${pkgs.mujoco}/lib";
        CPATH = "${pkgs.mujoco}/include";
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibs;
        MUJOCO_GL = "glfw"; 

        shellHook = ''
          echo "========================================="
          echo "🦀 Rust Native & Cross-Compilation Shell Active"
          echo "========================================="
          
          # Remove the old global RUSTFLAGS so we don't force 'cc' everywhere
          unset RUSTFLAGS
          
          # 1. Point cargo to the cross-linker
          export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER="aarch64-unknown-linux-gnu-gcc"

          # 2. Tell the pkg-config rust crate that cross-compiling is allowed
          export PKG_CONFIG_ALLOW_CROSS=1

          # 3. ✨ THE FIX: Explicitly feed the ARM64 version of libudev to the cross-compiler
          export PKG_CONFIG_PATH_aarch64_unknown_linux_gnu="${crossPkgs.udev.dev}/lib/pkgconfig"
        '';
      };
    };
}
