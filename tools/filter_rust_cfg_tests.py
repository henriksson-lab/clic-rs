#!/usr/bin/env python3
import shutil
import sys
from pathlib import Path


def skip_test_module(lines, start):
    i = start + 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i >= len(lines) or not lines[i].lstrip().startswith("mod tests"):
        return start + 1

    depth = 0
    saw_open = False
    while i < len(lines):
        for char in lines[i]:
            if char == "{":
                depth += 1
                saw_open = True
            elif char == "}":
                depth -= 1
                if saw_open and depth == 0:
                    return i + 1
        i += 1
    return i


def filter_rust_file(src, dst):
    lines = src.read_text().splitlines(keepends=True)
    out = []
    i = 0
    trait_depth = 0
    while i < len(lines):
        if lines[i].strip() == "#[cfg(test)]":
            next_i = skip_test_module(lines, i)
            if next_i != i + 1:
                i = next_i
                continue

        stripped = lines[i].lstrip()
        if stripped.startswith('extern "C"') or stripped.startswith('unsafe extern "C"'):
            depth = 0
            saw_open = False
            while i < len(lines):
                for char in lines[i]:
                    if char == "{":
                        depth += 1
                        saw_open = True
                    elif char == "}":
                        depth -= 1
                        if saw_open and depth == 0:
                            i += 1
                            break
                else:
                    i += 1
                    continue
                break
            continue

        if trait_depth == 1 and stripped.startswith("fn "):
            signature = []
            depth_delta = 0
            while i < len(lines):
                signature.append(lines[i])
                depth_delta += lines[i].count("{") - lines[i].count("}")
                if ";" in lines[i] or "{" in lines[i]:
                    break
                i += 1
            if any(";" in line for line in signature) and not any("{" in line for line in signature):
                i += 1
                continue
            out.extend(signature)
            trait_depth += depth_delta
            i += 1
            continue

        out.append(lines[i])
        if "trait " in lines[i] and "{" in lines[i]:
            trait_depth += lines[i].count("{") - lines[i].count("}")
        elif trait_depth:
            trait_depth += lines[i].count("{") - lines[i].count("}")
        i += 1
    dst.write_text("".join(out))


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: filter_rust_cfg_tests.py SRC_DIR DST_DIR")
    src_root = Path(sys.argv[1])
    dst_root = Path(sys.argv[2])
    if dst_root.exists():
        shutil.rmtree(dst_root)
    for src in src_root.rglob("*"):
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
        elif src.suffix == ".rs":
            dst.parent.mkdir(parents=True, exist_ok=True)
            filter_rust_file(src, dst)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


if __name__ == "__main__":
    main()
