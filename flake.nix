{
  description = "Wayland speech-to-text daemon";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    fenix.url = "github:nix-community/fenix";
    fenix.inputs.nixpkgs.follows = "nixpkgs";
    naersk.url = "github:nix-community/naersk";
    naersk.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, fenix, naersk }:
    let
      forAllSystems = nixpkgs.lib.genAttrs nixpkgs.lib.platforms.linux;

      mkPkgs = system: import nixpkgs {
        inherit system;
        overlays = [ fenix.overlays.default ];
      };
    in
    {
      packages = forAllSystems (system:
        let
          pkgs = mkPkgs system;
          rust = fenix.packages.${system}.stable.toolchain;
          naersk' = naersk.lib.${system}.override {
            cargo = rust;
            rustc = rust;
          };
        in
        rec {
          default = stt-wayland;

          stt-wayland = naersk'.buildPackage {
            src = ./.;
            nativeBuildInputs = with pkgs; [
              pkg-config cmake clang
              autoPatchelfHook
            ];
            buildInputs = with pkgs; [
              pipewire
              gtk4
              gtk4-layer-shell
              wayland
              wayland-protocols
              libclang.lib
            ];
            LIBCLANG_PATH = "${pkgs.libclang.lib}/lib";
          };
        });

      devShells = forAllSystems (system:
        let
          pkgs = mkPkgs system;
          rust = fenix.packages.${system}.stable.toolchain;
        in
        {
          default = pkgs.mkShell {
            name = "stt-wayland-dev";

            packages = with pkgs; [
              rust
              pkg-config cmake clang
              pipewire gtk4 gtk4-layer-shell wayland wayland-protocols
              libclang.lib
              stdenv.cc.cc.lib
            ];

            LIBCLANG_PATH = "${pkgs.libclang.lib}/lib";
            LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
          };
        });

      apps = forAllSystems (system: {
        default = {
          type = "app";
          program = "${self.packages.${system}.default}/bin/sttd";
        };
      });
    };
}