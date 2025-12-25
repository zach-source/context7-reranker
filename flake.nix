{
  description = "TF-IDF reranker for Context7 library documentation with pluggable backends";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = nixpkgs.legacyPackages.${system};

        # Python package definition
        context7-reranker = pkgs.python312Packages.buildPythonPackage {
          pname = "context7-reranker";
          version = "0.2.0";
          format = "pyproject";

          src = ./.;

          build-system = with pkgs.python312Packages; [
            hatchling
          ];

          dependencies = with pkgs.python312Packages; [
            # Core has no required dependencies
          ];

          optional-dependencies = {
            tiktoken = with pkgs.python312Packages; [tiktoken];
            http = with pkgs.python312Packages; [httpx];
            semantic = with pkgs.python312Packages; [
              sentence-transformers
              nltk
              scikit-learn
            ];
            all = with pkgs.python312Packages; [
              tiktoken
              httpx
              sentence-transformers
              nltk
              scikit-learn
            ];
          };

          nativeCheckInputs = with pkgs.python312Packages; [
            pytest
            pytest-cov
            pytest-asyncio
            respx
            httpx
          ];

          checkPhase = ''
            runHook preCheck
            pytest tests/ -v --ignore=tests/test_semantic_chunker.py
            runHook postCheck
          '';

          pythonImportsCheck = ["context7_reranker"];

          meta = with pkgs.lib; {
            description = "TF-IDF reranker for Context7 library documentation";
            homepage = "https://github.com/zach-source/context7-reranker";
            license = licenses.mit;
            maintainers = [];
          };
        };

        # Full package with all optional dependencies
        context7-reranker-full = context7-reranker.overridePythonAttrs (old: {
          dependencies =
            (old.dependencies or [])
            ++ (with pkgs.python312Packages; [
              tiktoken
              httpx
            ]);
        });

        # Python environment with the package
        pythonEnv = pkgs.python312.withPackages (ps: [
          context7-reranker
          ps.httpx
          ps.tiktoken
        ]);
      in {
        packages = {
          default = context7-reranker;
          context7-reranker = context7-reranker;
          context7-reranker-full = context7-reranker-full;
        };

        apps.default = {
          type = "app";
          program = "${pythonEnv}/bin/context7-reranker";
        };

        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            python312
            python312Packages.pip
            python312Packages.pytest
            python312Packages.pytest-cov
            python312Packages.pytest-asyncio
            python312Packages.black
            python312Packages.ruff
            python312Packages.httpx
            python312Packages.respx
            python312Packages.tiktoken
            uv
            jj
          ];

          shellHook = ''
            echo "context7-reranker development shell"
            echo ""
            echo "Commands:"
            echo "  uv pip install -e '.[dev]'  - Install in development mode"
            echo "  pytest -v                   - Run tests"
            echo "  black src tests             - Format code"
            echo "  ruff check src tests        - Lint code"
            echo ""
          '';
        };
      }
    )
    // {
      # NixOS/nix-darwin module
      nixosModules.default = ./module.nix;
      nixosModules.context7-reranker = ./module.nix;

      # Overlay for adding to nixpkgs
      overlays.default = final: prev: {
        context7-reranker = self.packages.${prev.system}.context7-reranker;
        context7-reranker-full = self.packages.${prev.system}.context7-reranker-full;
      };
    };
}
