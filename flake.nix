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

      rustToolchain = pkgs.rust-bin.stable.latest.default.override {
        targets = [ "aarch64-unknown-linux-gnu" ];
      };

      # ✨ THE MAGIC: Nix's native ARM64 GCC cross-compiler
      crossLinker = pkgs.pkgsCross.aarch64-multiplatform.stdenv.cc;

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
          crossLinker # Injects the aarch64 gcc into the shell
        ];

        buildInputs = runtimeLibs;

        MUJOCO_DYNAMIC_LINK_DIR = "${pkgs.mujoco}/lib";
        CPATH = "${pkgs.mujoco}/include";
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibs;
        MUJOCO_GL = "glfw"; 

        shellHook = ''
          echo "========================================="
          echo "🦀 Rust Native Cross-Compilation Active"
          echo "========================================="
          
          # Remove the old global RUSTFLAGS so we don't force 'cc' everywhere
          unset RUSTFLAGS
          
          # Tell Cargo exactly which linker to use when targeting aarch64
          export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER="aarch64-unknown-linux-gnu-gcc"
        '';
      };
    };
}
