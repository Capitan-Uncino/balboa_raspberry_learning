{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  # nativeBuildInputs: Tools needed at compile-time
  nativeBuildInputs = with pkgs; [
    pkg-config
    # Add these to override your broken global rustup
    rustc
    cargo 
    rust-analyzer
    gcc             # Provides the 'cc' linker
    binutils        # Provides 'ld' and other binary tools
  ];

  # buildInputs: Libraries needed at compile-time and run-time
  buildInputs = with pkgs; [
    mujoco
    udev
    
    # Graphical libraries (Required by egui 0.33)
    libGL
    glfw
    wayland
    libxkbcommon
    libdecor
    
    # X11 dependencies (using your preferred flat names)
    libx11
    libxrandr
    libxi
    libxcursor
    libxext
    libxinerama
  ];

  # ==========================================
  # RUST BUILD-TIME FIXES
  # ==========================================
  MUJOCO_DYNAMIC_LINK_DIR = "${pkgs.mujoco}/lib";
  CPATH = "${pkgs.mujoco}/include";

  # ==========================================
  # RUNTIME FIXES
  # ==========================================
  # This ensures egui and mujoco can find the .so files at runtime
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [
    mujoco
    udev
    libGL
    glfw
    wayland
    libxkbcommon
    libdecor
    libx11
    libxrandr
    libxi
    libxcursor
    libxext
    libxinerama
    stdenv.cc.cc.lib # Very important for libstdc++
  ]);
  
  MUJOCO_GL = "glfw"; 

  shellHook = ''
    echo "========================================="
    echo "🦀 Rust + MuJoCo Environment (Fixed Linker)"
    echo "========================================="
    # This points Cargo to the correct linker provided by Nix
    export RUSTFLAGS="-C linker=cc"
  '';
}
