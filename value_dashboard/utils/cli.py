import argparse
import os
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cdhdashboard",
        description="Launch the Value Dashboard Streamlit application.",
        epilog=(
            "Examples:\n"
            "  cdhdashboard run\n"
            "  cdhdashboard run --server.port 8502\n"
            "  cdhdashboard run --server.headless true --browser.gatherUsageStats false\n"
            "  cdhdashboard run -- --config=config/config.toml --logging_config=config/logging_config.yaml\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", metavar="command")

    run_parser = subparsers.add_parser(
        "run",
        help="Start the dashboard.",
        description=(
            "Start the Value Dashboard application.\n\n"
            "Arguments before `--` are forwarded to `streamlit run`.\n"
            "Arguments after `--` are forwarded to `vd_app.py`.\n\n"
            "Application options:\n"
            "  --config PATH                   Load a specific dashboard TOML config.\n"
            "  --logging_config PATH           Load a specific logging configuration file.\n\n"
            "Common Streamlit options:\n"
            "  --server.port PORT              Run on a custom port.\n"
            "  --server.address ADDRESS        Bind to a specific host.\n"
            "  --server.headless true|false    Disable or enable browser launch.\n"
            "  --browser.gatherUsageStats true|false\n"
            "                                  Control Streamlit usage reporting.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    run_parser.add_argument(
        "--config",
        action="store",
        default="",
        help="Config file to forward to vd_app.py as `--config`.",
    )
    run_parser.add_argument(
        "--logging_config",
        action="store",
        default="",
        help="Logging config file to forward to vd_app.py as `--logging_config`.",
    )
    run_parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help=(
            "Additional arguments to forward. "
            "Use `--` to separate Streamlit options from app arguments."
        ),
    )
    return parser


def _build_run_app_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--config",
        action="store",
        default="",
        help="Config file to forward to vd_app.py as `--config`.",
    )
    parser.add_argument(
        "--logging_config",
        action="store",
        default="",
        help="Logging config file to forward to vd_app.py as `--logging_config`.",
    )
    return parser


def _split_forwarded_args(args: list[str]) -> tuple[list[str], list[str]]:
    if "--" not in args:
        return list(args), []
    split_index = args.index("--")
    return list(args[:split_index]), list(args[split_index + 1:])


def main():
    parser = _build_parser()
    raw_args = sys.argv[1:]

    if not raw_args or raw_args[0] in {"-h", "--help"}:
        parser.parse_args(raw_args)
        return

    if raw_args[0] != "run" or any(arg in {"-h", "--help"} for arg in raw_args[1:]):
        parser.parse_args(raw_args)
        return

    run_args = raw_args[1:]
    streamlit_raw_args, script_raw_args = _split_forwarded_args(run_args)
    run_app_parser = _build_run_app_parser()
    streamlit_app_opts, streamlit_args = run_app_parser.parse_known_args(streamlit_raw_args)
    script_app_opts, script_args = run_app_parser.parse_known_args(script_raw_args)

    config_path = script_app_opts.config or streamlit_app_opts.config
    logging_config_path = script_app_opts.logging_config or streamlit_app_opts.logging_config

    forwarded_script_args = list(script_args)
    if config_path:
        forwarded_script_args = ["--config", config_path, *forwarded_script_args]
    if logging_config_path:
        forwarded_script_args = ["--logging_config", logging_config_path, *forwarded_script_args]

    run(streamlit_args=streamlit_args, script_args=forwarded_script_args)


def run(streamlit_args=None, script_args=None):
    from streamlit.web import cli as stcli

    streamlit_args = list(streamlit_args or [])
    script_args = list(script_args or [])

    filename = os.path.join(os.path.dirname(__file__), "../../vd_app.py")
    sys.argv = ["streamlit", "run", *streamlit_args, filename]
    if script_args:
        sys.argv.extend(["--", *script_args])

    sys.exit(stcli.main())


if __name__ == '__main__':
    main()
