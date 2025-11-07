{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs = inputs@{ self, nixpkgs, flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];

      perSystem = { pkgs, ... }: {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            futhark
            pocl
            ocl-icd
            (python312.withPackages (ps: [
              ps.numpy
              ps.pyopencl
              ps.torch
            ]))
          ];
          shellHook = ''
            export OCL_ICD_VENDORS=${pkgs.pocl}/etc/OpenCL/vendors
            export PYOPENCL_CTX="portable"
          '';
        };
      };
    };
    
}
