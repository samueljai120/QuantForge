"""QuantForge — Shared Configuration.

The production runtime host is determined by the QF_PRODUCTION_HOST env var or
by hostname detection. Set QF_BASE_DIR to override the data directory.
"""

import os
import socket

# Auto-detect environment
HOSTNAME = socket.gethostname()
PROD_HOST = os.environ.get("QF_PRODUCTION_HOST", "")
IS_PRODUCTION = bool(PROD_HOST and PROD_HOST.lower() in HOSTNAME.lower())

# Base directory — configurable via env var, defaults to ~/quantforge
PRODUCTION_BASE = os.path.expanduser(os.environ.get("QF_BASE_DIR", "~/quantforge"))
WORKSPACE_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.environ.get("QF_BASE_DIR"):
    BASE_DIR = os.environ["QF_BASE_DIR"]
elif IS_PRODUCTION:
    BASE_DIR = PRODUCTION_BASE
else:
    # Local development only. Runtime execution should still be blocked unless explicitly overridden.
    BASE_DIR = os.path.join(WORKSPACE_BASE, "quantforge-local")


def _env_flag(*names):
    for name in names:
        value = os.environ.get(name, "")
        if value.strip().lower() in {"1", "true", "yes", "on"}:
            return True
    return False


ALLOW_LOCAL_RUNTIME = _env_flag("QF_ALLOW_LOCAL_RUNTIME")


class Config:
    """Centralized path configuration for all QuantForge scripts."""

    def __init__(self):
        self.base = BASE_DIR
        self.brain = os.path.join(BASE_DIR, "brain")
        self.directives = os.path.join(BASE_DIR, "directives")
        self.departments = os.path.join(BASE_DIR, "departments")
        self.projects = os.path.join(BASE_DIR, "projects")
        self.data = os.path.join(BASE_DIR, "data")
        self.portfolio = os.path.join(self.data, "portfolio")
        self.portfolio_reports = os.path.join(self.portfolio, "reports")
        self.manual_project_updates = os.path.join(self.portfolio, "manual-project-updates")
        self.portfolio_blockers = os.path.join(self.portfolio, "blockers")
        self.project_sources = os.path.join(self.portfolio, "project-sources.json")
        self.project_change_intel = os.path.join(self.portfolio, "project-change-intel.json")
        self.blocker_registry = os.path.join(self.portfolio_blockers, "blocker-registry.jsonl")
        self.blocker_snapshot = os.path.join(self.portfolio_blockers, "blocker-registry-latest.json")
        self.project_registry = os.path.join(self.portfolio, "project-registry.json")
        self.project_report_schema = os.path.join(self.portfolio, "project-report-schema.json")
        self.docs = os.path.join(BASE_DIR, "docs")
        self.ml = os.path.join(BASE_DIR, "ml")
        self.logs = os.path.join(BASE_DIR, "logs")

        # Architecture paths
        self.mission_config = os.path.join(BASE_DIR, "config", "mission")
        self.mission_data = os.path.join(BASE_DIR, "data", "mission")
        self.portfolios_data = os.path.join(BASE_DIR, "data", "portfolios")
        self.changelogs = os.path.join(BASE_DIR, "changelogs")
        self.migration_reports = os.path.join(BASE_DIR, "reports", "migration")

        # Supabase config (for sync scripts) — all from env, no defaults
        self.supabase_url = os.environ.get("SUPABASE_URL", "")
        self.supabase_service_key = os.environ.get("SUPABASE_SERVICE_KEY", "")
        self.supabase_anon_key = os.environ.get("SUPABASE_ANON_KEY", "")

        # API keys
        self.anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self.openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
        self.groq_key = os.environ.get("GROQ_API_KEY", "")

        # Remote host SSH config (for zero-budget status script)
        self.prod_host = os.environ.get("QF_PROD_SSH_HOST", "")
        self.prod_user = os.environ.get("QF_PROD_SSH_USER", "youruser")
        self.prod_port = int(os.environ.get("QF_PROD_SSH_PORT", "22"))
        self.prod_base = os.environ.get("QF_BASE_DIR", "~/quantforge")

        # Environment info
        self.hostname = HOSTNAME
        self.is_production = IS_PRODUCTION
        self.environment = "production" if IS_PRODUCTION else "local"
        self.allow_local_runtime = ALLOW_LOCAL_RUNTIME

    def assert_production_runtime(self, job_name, *, allow_env_var="QF_ALLOW_LOCAL_RUNTIME"):
        """Fail closed for production runtime jobs on non-production hosts."""
        if self.is_production or ALLOW_LOCAL_RUNTIME:
            return
        raise SystemExit(
            f"ABORT: {job_name} is production-only. "
            f"Develop locally, then sync and run on your production host via SSH. "
            f"Override only for intentional local testing with {allow_env_var}=1."
        )

    def ensure_dirs(self):
        """Create all required directories."""
        dirs = [
            self.brain,
            os.path.join(self.brain, "reports"),
            self.directives,
            self.departments,
            self.projects,
            self.data,
            self.portfolio,
            self.portfolio_reports,
            self.manual_project_updates,
            self.portfolio_blockers,
            os.path.join(self.data, "revenue"),
            os.path.join(self.data, "accounting"),
            os.path.join(self.data, "rnd"),
            os.path.join(self.data, "marketing-content"),
            self.docs,
            self.ml,
            self.logs,
            self.mission_config,
            self.mission_data,
            self.portfolios_data,
            self.changelogs,
            self.migration_reports,
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    def dept_dir(self, dept_name):
        """Get department directory path."""
        return os.path.join(self.departments, dept_name)

    def dept_reports_dir(self, dept_name):
        """Get department reports directory."""
        d = os.path.join(self.departments, dept_name, "reports")
        os.makedirs(d, exist_ok=True)
        return d

    def require_production_runtime(self, script_name, extra_hint=None):
        """Fail fast when a production runtime script is launched outside the production host.

        Local development is still allowed for editing, previewing, and sync-only tools.
        Production loops, cron jobs, and autonomous runtime scripts must execute on the
        configured production host.
        """
        if self.is_production or self.allow_local_runtime:
            return

        lines = [
            f"ABORT: {script_name} is production-only runtime.",
            "QuantForge production execution must run on the configured production host.",
            f"Sync changes with: rsync -av scripts/{script_name} $QF_PROD_SSH_HOST:~/quantforge/scripts/",
            f"Run remotely with: ssh $QF_PROD_SSH_HOST 'cd ~/quantforge && python3 scripts/{script_name}'",
            "To override intentionally for local testing only, set QF_ALLOW_LOCAL_RUNTIME=1.",
        ]
        if extra_hint:
            lines.append(extra_hint)
        raise SystemExit("\n".join(lines))


# Singleton instance — import this
cfg = Config()
