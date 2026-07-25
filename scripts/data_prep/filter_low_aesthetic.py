#!/usr/bin/env python3
"""
Filter and move images with low aesthetic scores to excluded datasets directory.

This script identifies images with aesthetic scores below a threshold and moves them
to excluded_datasets/laion-high-resolution-all/low_aesthetic/ maintaining the chunk structure.
It can also create excluded JSON files and update original JSON files.

Usage:
    # Dry run to see what would be done (JSON operations included by default)
    python test/scripts/laion_aesthetic/filter_low_aesthetic.py --threshold 4.0 --dry-run

    # Execute filtering (JSON operations included by default)
    python test/scripts/laion_aesthetic/filter_low_aesthetic.py --threshold 4.0 --execute

    # Execute without JSON operations
    python test/scripts/laion_aesthetic/filter_low_aesthetic.py --threshold 4.0 --execute --no-update-scores --no-create-excluded-json
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor


def load_chunk_scores(aes_score_root: Path) -> Dict[str, Dict[str, float]]:
    """Load all chunk aesthetic scores from JSON files."""
    chunk_scores = {}

    if not aes_score_root.exists():
        raise FileNotFoundError(f"AES score directory not found: {aes_score_root}")

    json_files = sorted(aes_score_root.glob("chunk_*_aesthetic_scores.json"))
    if not json_files:
        raise FileNotFoundError(f"No aesthetic score JSON files found in {aes_score_root}")

    print(f"Loading aesthetic scores from {len(json_files)} chunk files...")

    for json_file in json_files:
        chunk_name = json_file.stem.replace("_aesthetic_scores", "")

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                scores = json.load(f)
                chunk_scores[chunk_name] = scores
                print(f"  {chunk_name}: {len(scores)} images loaded")
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")
            continue

    return chunk_scores


def identify_low_aesthetic_files(chunk_scores: Dict[str, Dict[str, float]],
                               threshold: float,
                               images_root: Path) -> Dict[str, List[Tuple[Path, float]]]:
    """Identify image files with aesthetic scores below threshold."""
    low_aesthetic_files = {}

    for chunk_name, scores in chunk_scores.items():
        chunk_low_files = []
        chunk_dir = images_root / chunk_name

        if not chunk_dir.exists():
            print(f"Warning: Chunk directory not found: {chunk_dir}")
            continue

        for filename, score in scores.items():
            if score < threshold:
                image_path = chunk_dir / filename
                if image_path.exists():
                    chunk_low_files.append((image_path, score))
                else:
                    print(f"Warning: Image file not found: {image_path}")

        if chunk_low_files:
            low_aesthetic_files[chunk_name] = chunk_low_files
            print(f"  {chunk_name}: {len(chunk_low_files)} files below threshold {threshold}")

    return low_aesthetic_files


def create_exclusion_summary(low_aesthetic_files: Dict[str, List[Tuple[Path, float]]],
                           output_dir: Path) -> None:
    """Create summary files of excluded images."""
    summary_data = {}
    total_excluded = 0

    for chunk_name, files_scores in low_aesthetic_files.items():
        chunk_summary = []
        for file_path, score in files_scores:
            chunk_summary.append({
                "filename": file_path.name,
                "aesthetic_score": score,
                "original_path": str(file_path)
            })
            total_excluded += 1

        summary_data[chunk_name] = {
            "excluded_count": len(chunk_summary),
            "files": chunk_summary
        }

    # Write overall summary
    summary_file = output_dir / "exclusion_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            "total_excluded": total_excluded,
            "total_chunks": len(low_aesthetic_files),
            "chunks": summary_data
        }, f, indent=2, ensure_ascii=False)

    print(f"Exclusion summary saved to: {summary_file}")


def move_file_safe(src_path: Path, dst_path: Path) -> bool:
    """Safely move a file with error handling."""
    try:
        # Create destination directory if it doesn't exist
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if destination file already exists
        if dst_path.exists():
            print(f"Warning: Destination exists, skipping: {dst_path}")
            return False

        # Move the file
        shutil.move(str(src_path), str(dst_path))
        return True

    except Exception as e:
        print(f"Error moving {src_path} to {dst_path}: {e}")
        return False


def move_low_aesthetic_files(low_aesthetic_files: Dict[str, List[Tuple[Path, float]]],
                           output_root: Path,
                           dry_run: bool = True,
                           max_workers: int = 4) -> Tuple[int, int]:
    """Move low aesthetic files to exclusion directory."""
    total_files = sum(len(files) for files in low_aesthetic_files.values())
    moved_count = 0
    failed_count = 0

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Moving {total_files} files to {output_root}")

    if not dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    def process_chunk(chunk_data):
        chunk_name, files_scores = chunk_data
        chunk_moved = 0
        chunk_failed = 0

        # Create chunk output directory
        chunk_output_dir = output_root / chunk_name

        if not dry_run:
            chunk_output_dir.mkdir(parents=True, exist_ok=True)

        for src_path, score in files_scores:
            dst_path = chunk_output_dir / src_path.name

            if dry_run:
                print(f"  [DRY RUN] Would move: {src_path} -> {dst_path} (score: {score:.3f})")
                chunk_moved += 1
            else:
                if move_file_safe(src_path, dst_path):
                    print(f"  Moved: {src_path.name} (score: {score:.3f})")
                    chunk_moved += 1
                else:
                    chunk_failed += 1

        print(f"  {chunk_name}: {'[DRY RUN] Would move' if dry_run else 'Moved'} {chunk_moved} files"
              + (f", failed {chunk_failed}" if chunk_failed > 0 else ""))

        return chunk_moved, chunk_failed

    # Process chunks in parallel for better performance
    if max_workers > 1 and len(low_aesthetic_files) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_chunk, low_aesthetic_files.items()))
    else:
        results = [process_chunk(item) for item in low_aesthetic_files.items()]

    for chunk_moved, chunk_failed in results:
        moved_count += chunk_moved
        failed_count += chunk_failed

    return moved_count, failed_count


def create_excluded_score_files(chunk_scores: Dict[str, Dict[str, float]],
                               low_aesthetic_files: Dict[str, List[Tuple[Path, float]]],
                               excluded_json_root: Path,
                               dry_run: bool = True) -> None:
    """Create JSON files for excluded images with low aesthetic scores."""
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Creating excluded aesthetic score JSON files...")

    if not dry_run:
        excluded_json_root.mkdir(parents=True, exist_ok=True)

    for chunk_name, files_scores in low_aesthetic_files.items():
        excluded_filenames = {file_path.name for file_path, _ in files_scores}

        if chunk_name in chunk_scores:
            # Create excluded scores with original structure
            excluded_scores = {
                filename: score for filename, score in chunk_scores[chunk_name].items()
                if filename in excluded_filenames
            }

            excluded_count = len(excluded_scores)
            print(f"  {chunk_name}: {excluded_count} excluded files to save")

            if not dry_run and excluded_scores:
                # Save excluded scores
                excluded_file = excluded_json_root / f"{chunk_name}_aesthetic_scores.json"
                try:
                    with open(excluded_file, 'w', encoding='utf-8') as f:
                        json.dump(excluded_scores, f, indent=2, ensure_ascii=False)
                    print(f"    Created: {excluded_file}")
                except Exception as e:
                    print(f"    Error creating {excluded_file}: {e}")
            elif dry_run and excluded_scores:
                excluded_file = excluded_json_root / f"{chunk_name}_aesthetic_scores.json"
                print(f"    [DRY RUN] Would create: {excluded_file}")


def update_remaining_scores(chunk_scores: Dict[str, Dict[str, float]],
                          low_aesthetic_files: Dict[str, List[Tuple[Path, float]]],
                          aes_score_root: Path,
                          dry_run: bool = True) -> None:
    """Update aesthetic score JSON files to remove excluded images."""
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Updating aesthetic score files...")

    for chunk_name, files_scores in low_aesthetic_files.items():
        excluded_filenames = {file_path.name for file_path, _ in files_scores}

        if chunk_name in chunk_scores:
            # Create updated scores without excluded files
            updated_scores = {
                filename: score for filename, score in chunk_scores[chunk_name].items()
                if filename not in excluded_filenames
            }

            original_count = len(chunk_scores[chunk_name])
            updated_count = len(updated_scores)
            excluded_count = original_count - updated_count

            print(f"  {chunk_name}: {original_count} -> {updated_count} "
                  f"(excluded {excluded_count} files)")

            if not dry_run:
                # Save updated scores
                score_file = aes_score_root / f"{chunk_name}_aesthetic_scores.json"
                try:
                    with open(score_file, 'w', encoding='utf-8') as f:
                        json.dump(updated_scores, f, indent=2, ensure_ascii=False)
                    print(f"    Updated: {score_file}")
                except Exception as e:
                    print(f"    Error updating {score_file}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Filter images with low aesthetic scores")
    parser.add_argument('--threshold', type=float, required=True,
                       help='Aesthetic score threshold (images below this will be moved)')
    parser.add_argument('--images-root', type=str,
                       default='datasets/images/laion-high-resolution',
                       help='Root directory containing image chunks')
    parser.add_argument('--aes-score-root', type=str,
                       default='datasets/AES_score/laion-high-resolution',
                       help='Root directory containing aesthetic score JSON files')
    parser.add_argument('--output-root', type=str,
                       default='excluded_datasets/laion-high-resolution-all/low_aesthetic',
                       help='Output directory for excluded images')
    parser.add_argument('--excluded-json-root', type=str,
                       default='excluded_datasets/laion-high-resolution-all/low_aesthetic_json',
                       help='Output directory for excluded JSON files')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without actually moving files')
    parser.add_argument('--execute', action='store_true',
                       help='Actually execute the filtering (moves files)')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Maximum number of parallel workers for file operations')
    parser.add_argument('--update-scores', action='store_true', default=True,
                       help='Update aesthetic score JSON files to remove excluded entries (default: True)')
    parser.add_argument('--no-update-scores', action='store_false', dest='update_scores',
                       help='Skip updating aesthetic score JSON files')
    parser.add_argument('--create-excluded-json', action='store_true', default=True,
                       help='Create JSON files for excluded images in excluded-json-root directory (default: True)')
    parser.add_argument('--no-create-excluded-json', action='store_false', dest='create_excluded_json',
                       help='Skip creating excluded JSON files')

    args = parser.parse_args()

    # Validate arguments
    if not args.dry_run and not args.execute:
        print("Error: Must specify either --dry-run or --execute")
        sys.exit(1)

    if args.dry_run and args.execute:
        print("Error: Cannot specify both --dry-run and --execute")
        sys.exit(1)

    # Setup paths
    images_root = Path(args.images_root)
    aes_score_root = Path(args.aes_score_root)
    output_root = Path(args.output_root)
    excluded_json_root = Path(args.excluded_json_root)

    if not images_root.exists():
        print(f"Error: Images root directory not found: {images_root}")
        sys.exit(1)

    # Load aesthetic scores
    try:
        chunk_scores = load_chunk_scores(aes_score_root)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not chunk_scores:
        print("No aesthetic scores found.")
        sys.exit(1)

    # Identify files below threshold
    print(f"\nIdentifying images with aesthetic scores < {args.threshold}...")
    low_aesthetic_files = identify_low_aesthetic_files(chunk_scores, args.threshold, images_root)

    if not low_aesthetic_files:
        print(f"No images found below threshold {args.threshold}")
        return

    total_low_files = sum(len(files) for files in low_aesthetic_files.values())
    total_all_files = sum(len(scores) for scores in chunk_scores.values())
    percentage = (total_low_files / total_all_files * 100) if total_all_files > 0 else 0

    print(f"\nFound {total_low_files:,} images below threshold {args.threshold}")
    print(f"This represents {percentage:.2f}% of all {total_all_files:,} images")

    # Move files
    moved_count, failed_count = move_low_aesthetic_files(
        low_aesthetic_files, output_root, dry_run=args.dry_run, max_workers=args.max_workers
    )

    # Create summary
    if not args.dry_run:
        create_exclusion_summary(low_aesthetic_files, output_root)

    # Create excluded JSON files if requested
    if args.create_excluded_json:
        create_excluded_score_files(chunk_scores, low_aesthetic_files, excluded_json_root, dry_run=args.dry_run)

    # Update score files if requested
    if args.update_scores:
        update_remaining_scores(chunk_scores, low_aesthetic_files, aes_score_root, dry_run=args.dry_run)

    # Final summary
    print(f"\n=== Summary ===")
    print(f"Threshold: {args.threshold}")
    print(f"Files below threshold: {total_low_files:,}")
    print(f"{'[DRY RUN] Would move' if args.dry_run else 'Successfully moved'}: {moved_count:,}")
    if failed_count > 0:
        print(f"Failed to move: {failed_count:,}")
    print(f"Output directory: {output_root}")

    if args.dry_run:
        print(f"\nTo actually execute the filtering, run with --execute instead of --dry-run")


if __name__ == '__main__':
    main()