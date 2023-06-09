{
  description = "Symbolator is a component diagramming tool for VHDL and Verilog.";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-22.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        composeExtensions =
          f: g: self: super:
          let
            fApplied = f self super;
            super' = super // fApplied;
          in
          fApplied // g self super';
        pkgs = nixpkgs.legacyPackages.${system};
        packageOverrides = pkgs.callPackage ./python-packages.nix { };
        pythonPackagesLocalOverrides = self: super: with pkgs; {
          symbolator = super.symbolator.override (attrs: {
            nativeBuildInputs = [ wrapGAppsHook gobject-introspection ] ++ attrs.nativeBuildInputs;

            propagatedBuildInputs = [
              pango
              fontconfig
              python3Packages.pygobject3
            ] ++ attrs.propagatedBuildInputs;

            meta = with lib; {
              description = "A component diagramming tool for VHDL and Verilog";
              longDescription = ''
                Symbolator is a component diagramming tool for VHDL and Verilog. It will parse HDL source files, extract components or modules and render them as an image.
              '';
              homepage = "https://hdl.github.io/symbolator/";
              maintainer = "zebreus";
              license = licenses.mit;
              platforms = lib.platforms.linux;
              mainProgram = "symbolator";
            };
          });
        };
        pythonPackages = (pkgs.python3.override { packageOverrides = composeExtensions packageOverrides pythonPackagesLocalOverrides; }).pkgs;
      in
      rec
      {
        name = "Symbolator";
        packages.symbolator = pythonPackages.symbolator;
        packages.default = packages.symbolator;
      }
    );
}
