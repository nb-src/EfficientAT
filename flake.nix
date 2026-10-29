{
  description = "EfficientAT dev shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" ] (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python312;

        runtimeLibs = with pkgs; [
          stdenv.cc.cc.lib # libstdc++ for torch
          zlib
          portaudio # sounddevice (live mic input)
          libsndfile # soundfile (librosa audio loading)
          alsa-lib
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            python
            pkgs.alsa-utils
          ];

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibs;

          # tkinter is a separate package in nixpkgs; expose it to the venv
          PYTHONPATH = "${python.pkgs.tkinter}/${python.sitePackages}";
          MPLBACKEND = "TkAgg";

          shellHook = ''
            if [ ! -d .venv ]; then
              echo "Creating .venv (first run)..."
              ${python}/bin/python -m venv .venv
              source .venv/bin/activate
              pip install --upgrade pip
              # CPU-only 
              pip install torch torchvision torchaudio \
                --index-url https://download.pytorch.org/whl/cpu
              pip install librosa sounddevice numpy matplotlib
            else
              source .venv/bin/activate
            fi
          '';
        };
      }
    );
}
