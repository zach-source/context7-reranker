{
  config,
  lib,
  pkgs,
  ...
}: let
  cfg = config.services.context7-reranker;
  format = pkgs.formats.json {};
in {
  options.services.context7-reranker = {
    enable = lib.mkEnableOption "context7-reranker service";

    package = lib.mkOption {
      type = lib.types.package;
      default = pkgs.context7-reranker or (throw "context7-reranker package not found. Add the overlay first.");
      description = "The context7-reranker package to use.";
    };

    # Tokenizer configuration
    tokenizer = {
      endpoint = lib.mkOption {
        type = lib.types.nullOr lib.types.str;
        default = null;
        description = "HTTP endpoint for external tokenizer service.";
        example = "http://localhost:8080/v1/embeddings";
      };

      apiKey = lib.mkOption {
        type = lib.types.nullOr lib.types.str;
        default = null;
        description = "API key for tokenizer service.";
      };

      apiKeyFile = lib.mkOption {
        type = lib.types.nullOr lib.types.path;
        default = null;
        description = "File containing the API key for tokenizer service.";
      };

      model = lib.mkOption {
        type = lib.types.str;
        default = "default";
        description = "Model name for tokenizer.";
      };

      timeout = lib.mkOption {
        type = lib.types.float;
        default = 30.0;
        description = "Request timeout in seconds.";
      };
    };

    # Reranker configuration
    reranker = {
      endpoint = lib.mkOption {
        type = lib.types.nullOr lib.types.str;
        default = null;
        description = "HTTP endpoint for external reranker service.";
        example = "http://localhost:8080/v1/rerank";
      };

      apiKey = lib.mkOption {
        type = lib.types.nullOr lib.types.str;
        default = null;
        description = "API key for reranker service.";
      };

      apiKeyFile = lib.mkOption {
        type = lib.types.nullOr lib.types.path;
        default = null;
        description = "File containing the API key for reranker service.";
      };

      format = lib.mkOption {
        type = lib.types.enum ["cohere" "openai" "custom"];
        default = "cohere";
        description = "API format for reranker (cohere, openai, or custom).";
      };

      model = lib.mkOption {
        type = lib.types.str;
        default = "default";
        description = "Model name for reranker.";
      };

      timeout = lib.mkOption {
        type = lib.types.float;
        default = 30.0;
        description = "Request timeout in seconds.";
      };
    };

    # Chunker configuration
    chunker = {
      mode = lib.mkOption {
        type = lib.types.enum ["regex" "semantic" "http"];
        default = "regex";
        description = "Chunking mode (regex, semantic, or http).";
      };

      endpoint = lib.mkOption {
        type = lib.types.nullOr lib.types.str;
        default = null;
        description = "HTTP endpoint for embeddings (used in http mode).";
      };

      threshold = lib.mkOption {
        type = lib.types.float;
        default = 0.5;
        description = "Semantic similarity threshold (0-1).";
      };

      model = lib.mkOption {
        type = lib.types.str;
        default = "all-mpnet-base-v1";
        description = "Model name for semantic chunking.";
      };
    };

    # LLM configuration for query parsing
    llm = {
      endpoint = lib.mkOption {
        type = lib.types.nullOr lib.types.str;
        default = null;
        description = "OpenAI-compatible chat API endpoint.";
        example = "https://api.openai.com/v1";
      };

      apiKey = lib.mkOption {
        type = lib.types.nullOr lib.types.str;
        default = null;
        description = "API key for LLM service.";
      };

      apiKeyFile = lib.mkOption {
        type = lib.types.nullOr lib.types.path;
        default = null;
        description = "File containing the API key for LLM service.";
      };

      model = lib.mkOption {
        type = lib.types.str;
        default = "gpt-4o-mini";
        description = "Model name for query parsing.";
      };

      temperature = lib.mkOption {
        type = lib.types.float;
        default = 0.0;
        description = "Temperature for LLM responses (0-2).";
      };

      timeout = lib.mkOption {
        type = lib.types.float;
        default = 30.0;
        description = "Request timeout in seconds.";
      };
    };

    # Server configuration (for future HTTP server mode)
    server = {
      enable = lib.mkEnableOption "Run as HTTP server";

      host = lib.mkOption {
        type = lib.types.str;
        default = "127.0.0.1";
        description = "Host to bind the server to.";
      };

      port = lib.mkOption {
        type = lib.types.port;
        default = 8000;
        description = "Port to run the server on.";
      };
    };

    user = lib.mkOption {
      type = lib.types.str;
      default = "context7";
      description = "User to run the service as.";
    };

    group = lib.mkOption {
      type = lib.types.str;
      default = "context7";
      description = "Group to run the service as.";
    };

    extraEnvironment = lib.mkOption {
      type = lib.types.attrsOf lib.types.str;
      default = {};
      description = "Additional environment variables.";
      example = {
        PYTHONPATH = "/extra/path";
      };
    };
  };

  config = lib.mkIf cfg.enable {
    users.users.${cfg.user} = lib.mkIf (cfg.user == "context7") {
      isSystemUser = true;
      group = cfg.group;
      description = "context7-reranker service user";
    };

    users.groups.${cfg.group} = lib.mkIf (cfg.group == "context7") {};

    # Environment file for the service
    systemd.services.context7-reranker = lib.mkIf cfg.server.enable {
      description = "Context7 Reranker Service";
      wantedBy = ["multi-user.target"];
      after = ["network.target"];

      environment =
        {
          # Tokenizer config
          TOKENIZER_ENDPOINT = lib.mkIf (cfg.tokenizer.endpoint != null) cfg.tokenizer.endpoint;
          TOKENIZER_MODEL = cfg.tokenizer.model;
          TOKENIZER_TIMEOUT = toString cfg.tokenizer.timeout;

          # Reranker config
          RERANKER_ENDPOINT = lib.mkIf (cfg.reranker.endpoint != null) cfg.reranker.endpoint;
          RERANKER_FORMAT = cfg.reranker.format;
          RERANKER_MODEL = cfg.reranker.model;
          RERANKER_TIMEOUT = toString cfg.reranker.timeout;

          # Chunker config
          CHUNKER_MODE = cfg.chunker.mode;
          CHUNKER_THRESHOLD = toString cfg.chunker.threshold;
          CHUNKER_MODEL = cfg.chunker.model;
          CHUNKER_ENDPOINT = lib.mkIf (cfg.chunker.endpoint != null) cfg.chunker.endpoint;

          # LLM config
          LLM_ENDPOINT = lib.mkIf (cfg.llm.endpoint != null) cfg.llm.endpoint;
          LLM_MODEL = cfg.llm.model;
          LLM_TEMPERATURE = toString cfg.llm.temperature;
          LLM_TIMEOUT = toString cfg.llm.timeout;

          # Server config
          HOST = cfg.server.host;
          PORT = toString cfg.server.port;
        }
        // cfg.extraEnvironment;

      serviceConfig = {
        Type = "simple";
        User = cfg.user;
        Group = cfg.group;
        ExecStart = "${cfg.package}/bin/context7-reranker server --host ${cfg.server.host} --port ${toString cfg.server.port}";
        Restart = "on-failure";
        RestartSec = 5;

        # Security hardening
        NoNewPrivileges = true;
        PrivateTmp = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectControlGroups = true;
        RestrictAddressFamilies = ["AF_INET" "AF_INET6" "AF_UNIX"];
        RestrictNamespaces = true;
        RestrictRealtime = true;
        RestrictSUIDSGID = true;
        MemoryDenyWriteExecute = true;
        LockPersonality = true;

        # Load API keys from files if specified
        LoadCredential =
          lib.optional (cfg.tokenizer.apiKeyFile != null) "tokenizer-api-key:${cfg.tokenizer.apiKeyFile}"
          ++ lib.optional (cfg.reranker.apiKeyFile != null) "reranker-api-key:${cfg.reranker.apiKeyFile}"
          ++ lib.optional (cfg.llm.apiKeyFile != null) "llm-api-key:${cfg.llm.apiKeyFile}";
      };

      # Set API keys from credential files
      preStart = ''
        ${lib.optionalString (cfg.tokenizer.apiKeyFile != null) ''
          export TOKENIZER_API_KEY="$(cat $CREDENTIALS_DIRECTORY/tokenizer-api-key)"
        ''}
        ${lib.optionalString (cfg.reranker.apiKeyFile != null) ''
          export RERANKER_API_KEY="$(cat $CREDENTIALS_DIRECTORY/reranker-api-key)"
        ''}
        ${lib.optionalString (cfg.llm.apiKeyFile != null) ''
          export LLM_API_KEY="$(cat $CREDENTIALS_DIRECTORY/llm-api-key)"
        ''}
      '';
    };

    # Also provide the package in system path for CLI usage
    environment.systemPackages = [cfg.package];
  };
}
