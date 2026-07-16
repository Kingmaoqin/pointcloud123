#!/usr/bin/env python3
from __future__ import annotations

import argparse

from patent_gap.webapp import launch


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the local BIM gap analysis UI")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
