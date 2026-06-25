#!/usr/bin/env python3
"""Measure tier1 parity commands and update a Markdown speed/RSS table.

Input manifest format:

[
  {
    "function": "add_images_weighted",
    "implementation": "rust",
    "command": "cargo bench --bench gpu -- add_images_weighted/256x256"
  },
  {
    "function": "add_images_weighted",
    "implementation": "clic",
    "command": "ctest -R TestArithmeticOperations.add_image_weighted"
  }
]

The command is executed through `sh -c` and measured with `/usr/bin/time -v`.
The table between the marker comments in `--output` is replaced on each run.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


START = "<!-- tier1-parity-measurements:start -->"
END = "<!-- tier1-parity-measurements:end -->"


@dataclass
class Measurement:
    function: str
    implementation: str
    command: str
    status: str
    elapsed_seconds: float | None
    max_rss_kb: int | None


def parse_elapsed(value: str) -> float | None:
    parts = value.strip().split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
        return float(parts[0])
    except ValueError:
        return None


def measure(entry: dict[str, str]) -> Measurement:
    command = entry["command"]
    proc = subprocess.run(
        ["/usr/bin/time", "-v", "sh", "-c", command],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    elapsed_seconds = None
    max_rss_kb = None
    for line in proc.stderr.splitlines():
        if "Elapsed (wall clock) time" in line:
            elapsed_seconds = parse_elapsed(line.rsplit(":", 1)[-1])
        elif "Maximum resident set size" in line:
            match = re.search(r"(\d+)$", line.strip())
            if match:
                max_rss_kb = int(match.group(1))

    return Measurement(
        function=entry["function"],
        implementation=entry["implementation"],
        command=command,
        status="ok" if proc.returncode == 0 else f"exit {proc.returncode}",
        elapsed_seconds=elapsed_seconds,
        max_rss_kb=max_rss_kb,
    )


def fmt_number(value: float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def markdown_table(measurements: list[Measurement]) -> str:
    rows = [
        START,
        "| function | implementation | status | elapsed_s | max_rss_kb | command |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for item in sorted(measurements, key=lambda m: (m.function, m.implementation)):
        command = item.command.replace("|", "\\|")
        rows.append(
            "| "
            + " | ".join(
                [
                    item.function,
                    item.implementation,
                    item.status,
                    fmt_number(item.elapsed_seconds),
                    fmt_number(item.max_rss_kb),
                    f"`{command}`",
                ]
            )
            + " |"
        )
    rows.append(END)
    return "\n".join(rows) + "\n"


def update_output(path: Path, table: str) -> None:
    if not path.exists():
        path.write_text(table)
        return

    text = path.read_text()
    if START in text and END in text:
        before = text.split(START, 1)[0]
        after = text.split(END, 1)[1]
        path.write_text(before + table + after)
    else:
        suffix = "" if text.endswith("\n") else "\n"
        path.write_text(text + suffix + "\n" + table)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    entries = json.loads(args.manifest.read_text())
    measurements = [measure(entry) for entry in entries]
    update_output(args.output, markdown_table(measurements))

    failed = [m for m in measurements if m.status != "ok"]
    for item in measurements:
        print(
            f"{item.function} {item.implementation}: {item.status}, "
            f"{fmt_number(item.elapsed_seconds)}s, {fmt_number(item.max_rss_kb)} KB RSS"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
