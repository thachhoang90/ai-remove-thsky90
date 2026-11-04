"""Entry point of the desktop app: AI Remove - thsky90."""
import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream is not None:
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        from airemove.cli import main as cli_main
        sys.exit(cli_main(sys.argv[2:]))
    from airemove.gui import main
    sys.exit(main())
