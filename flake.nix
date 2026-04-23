{
  description = "Rust + MuJoCo Environment pinned to nixpkgs commit 09061f748ee21f68a089cd5d91ec1859cd93d0be";

  inputs = {
    # Pinned to the specific commit providing your target MuJoCo version
    nixpkgs.url = "github:NixOS/nixpkgs/09061f748ee21f68a089cd5d91ec1859cd93d0be";
  };

  outputs = { self, nixpkgs }:
    let
      # Define the system you are building for
      system = "x86_64-linux"; 
      pkgs = nixpkgs.legacyPackages.${system};

      # Grouped runtime libraries needed for both buildInputs and LD_LIBRARY_PATH
      runtimeLibs = with pkgs; [
        mujoco
        udev
        libGL
        glfw
        wayland
        libxkbcommon
        libdecor 

        fontconfig
        
        # X11 dependencies
        libx11
        libxrandr
        libxi
        libxcursor
        libxext
        libxinerama
        
        # Very important for libstdc++
        stdenv.cc.cc.lib
      ];
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        # nativeBuildInputs: Tools needed at compile-time
        nativeBuildInputs = with pkgs; [
          mujoco
          pkg-config
          rustc
          cargo 
          rust-analyzer
          gcc             # Provides the 'cc' linker
          binutils        # Provides 'ld' and other binary tools
        ];

        # buildInputs: Libraries needed at compile-time and run-time
        buildInputs = runtimeLibs;

        # ==========================================
        # RUST BUILD-TIME FIXES
        # ==========================================
        MUJOCO_DYNAMIC_LINK_DIR = "${pkgs.mujoco}/lib";
        CPATH = "${pkgs.mujoco}/include";

        # ==========================================
        # RUNTIME FIXES
        # ==========================================
        # This ensures egui and mujoco can find the .so files at runtime
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibs;
        
        MUJOCO_GL = "glfw"; 

        shellHook = ''
          echo "========================================="
          echo "🦀 Rust + MuJoCo Environment (Fixed Linker)"
          echo "========================================="
          # This points Cargo to the correct linker provided by Nix
          export RUSTFLAGS="-C linker=cc"
        '';
      };
    };
}
