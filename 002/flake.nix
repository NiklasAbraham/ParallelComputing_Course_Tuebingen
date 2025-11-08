{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs = inputs@{ self, nixpkgs, flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];

      perSystem = { system, ... }:
        let
          pkgs = import nixpkgs {
            inherit system;
            config.allowUnfree = true;
          };
          pythonEnv = pkgs.python312.withPackages (ps: [
            ps.numpy
            ps.pyopencl
            ps.pycuda
            ps.torchWithCuda
          ]);
        in {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            futhark
            pocl
            ocl-icd
            cudaPackages.cudatoolkit
            pythonEnv
          ];
          shellHook = ''
            if [ -z "''${OCL_ICD_VENDORS+x}" ]; then
              export OCL_ICD_VENDORS=${pkgs.pocl}/etc/OpenCL/vendors
            fi
            if [ -z "''${PYOPENCL_CTX+x}" ]; then
              export PYOPENCL_CTX="portable"
            fi
            if [ -z "''${CUDA_PATH+x}" ]; then
              export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
            fi
            export CUDA_INCLUDE_DIRS=${pkgs.cudaPackages.cudatoolkit}/include
            if [ -z "''${FUTHARK_DEVICE_PREF+x}" ]; then
              export FUTHARK_DEVICE_PREF="NVIDIA"
            fi
            if [ -z "''${PYOPENCL_COMPILER_OUTPUT+x}" ]; then
              export PYOPENCL_COMPILER_OUTPUT=1
            fi
            gpu_driver_link_dir=''${TMPDIR:-/tmp}/nix-gpu-driver-links
            mkdir -p "$gpu_driver_link_dir"
            for drv in /lib/x86_64-linux-gnu/libcuda.so* /lib/x86_64-linux-gnu/libnvidia-opencl.so* /lib/x86_64-linux-gnu/libOpenCL.so* /usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so* /usr/lib/x86_64-linux-gnu/libnvrtc.so* /usr/lib/x86_64-linux-gnu/libcudart.so*; do
              if [ -e "$drv" ]; then
                ln -sf "$drv" "$gpu_driver_link_dir/$(basename "$drv")"
              fi
            done
            export LD_LIBRARY_PATH="$gpu_driver_link_dir":${pkgs.ocl-icd}/lib:${pkgs.pocl}/lib:''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
            export LIBRARY_PATH="$gpu_driver_link_dir":''${LIBRARY_PATH:+:$LIBRARY_PATH}
            export NIX_LDFLAGS="-L$gpu_driver_link_dir ''${NIX_LDFLAGS:+$NIX_LDFLAGS}"
          '';
        };
      };
    };
    
}
