{
  pkgs,
  lib,
  ...
}:
{
  languages.python = {
    enable = true;
    package = pkgs.python312;
    venv = {
      enable = true;
      requirements = ''
        pytest>=7.0
        pytest-cov>=4.0
        black>=23.0
        ruff>=0.1.0
      '';
    };
  };

  packages = with pkgs; [
    python312Packages.pip
  ];

  enterShell = ''
    pip install -e . -q 2>/dev/null || true
    echo "context7-reranker dev environment ready"
    echo "Run: pytest -v"
  '';
}
