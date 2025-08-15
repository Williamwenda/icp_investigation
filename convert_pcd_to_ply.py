#!/usr/bin/env python3
"""
PCD to PLY Converter

This script converts point cloud files from PCD format to PLY format using Open3D.
It can process single files or entire directories.

Usage:
    python convert_pcd_to_ply.py input.pcd                    # Convert single file
    python convert_pcd_to_ply.py input.pcd output.ply         # Convert with custom output name
    python convert_pcd_to_ply.py /path/to/pcd/files/          # Convert all PCD files in directory
    python convert_pcd_to_ply.py --help                       # Show help
"""

import open3d as o3d
import os
import sys
import argparse
import glob
from pathlib import Path


def convert_pcd_to_ply(input_path, output_path=None, verbose=True):
    """
    Convert a single PCD file to PLY format.
    
    Args:
        input_path (str): Path to input PCD file
        output_path (str): Path to output PLY file (optional)
        verbose (bool): Print conversion details
        
    Returns:
        bool: True if conversion successful, False otherwise
    """
    try:
        # Load PCD file
        if verbose:
            print(f"Loading PCD file: {input_path}")
        
        pcd = o3d.io.read_point_cloud(input_path)
        
        if len(pcd.points) == 0:
            print(f"Error: No points found in {input_path}")
            return False
        
        # Generate output path if not provided
        if output_path is None:
            input_file = Path(input_path)
            output_path = str(input_file.with_suffix('.ply'))
        
        # Save as PLY file
        if verbose:
            print(f"Converting {len(pcd.points)} points...")
            if pcd.has_normals():
                print("  - Normals: ✓")
            if pcd.has_colors():
                print("  - Colors: ✓")
        
        success = o3d.io.write_point_cloud(output_path, pcd)
        
        if success:
            if verbose:
                print(f"Successfully saved: {output_path}")
            return True
        else:
            print(f"Error: Failed to save {output_path}")
            return False
            
    except Exception as e:
        print(f"Error converting {input_path}: {e}")
        return False


def convert_directory(input_dir, output_dir=None, verbose=True):
    """
    Convert all PCD files in a directory to PLY format.
    
    Args:
        input_dir (str): Path to directory containing PCD files
        output_dir (str): Path to output directory (optional, defaults to same directory)
        verbose (bool): Print conversion details
        
    Returns:
        tuple: (successful_conversions, total_files)
    """
    input_path = Path(input_dir)
    
    if not input_path.is_dir():
        print(f"Error: {input_dir} is not a directory")
        return 0, 0
    
    # Find all PCD files
    pcd_files = list(input_path.glob("*.pcd"))
    if not pcd_files:
        print(f"No PCD files found in {input_dir}")
        return 0, 0
    
    # Set output directory
    if output_dir is None:
        output_path = input_path
    else:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    successful = 0
    total = len(pcd_files)
    
    print(f"Found {total} PCD files to convert...")
    print("-" * 50)
    
    for i, pcd_file in enumerate(pcd_files, 1):
        print(f"[{i}/{total}] Processing: {pcd_file.name}")
        
        # Generate output filename
        output_file = output_path / pcd_file.with_suffix('.ply').name
        
        # Convert file
        if convert_pcd_to_ply(str(pcd_file), str(output_file), verbose=False):
            successful += 1
            print(f"  ✓ Converted to: {output_file.name}")
        else:
            print(f"  ✗ Failed to convert: {pcd_file.name}")
    
    print("-" * 50)
    print(f"Conversion complete: {successful}/{total} files converted successfully")
    
    return successful, total


def main():
    parser = argparse.ArgumentParser(
        description="Convert PCD files to PLY format",
        epilog="""
        Examples:
          %(prog)s input.pcd                          # Convert single file
          %(prog)s input.pcd output.ply               # Convert with custom output name
          %(prog)s /path/to/pcd/files/                # Convert all PCD files in directory
          %(prog)s /path/to/pcd/ /path/to/output/     # Convert directory with custom output location
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('input', 
                       help='Input PCD file or directory containing PCD files')
    parser.add_argument('output', nargs='?', default=None,
                       help='Output PLY file or directory (optional)')
    parser.add_argument('-v', '--verbose', action='store_true', default=True,
                       help='Verbose output (default: True)')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Quiet mode (minimal output)')
    
    args = parser.parse_args()
    
    # Handle quiet mode
    if args.quiet:
        args.verbose = False
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: Input path '{args.input}' does not exist")
        sys.exit(1)
    
    # Convert single file
    if input_path.is_file():
        if not input_path.suffix.lower() == '.pcd':
            print(f"Error: Input file must have .pcd extension")
            sys.exit(1)
        
        success = convert_pcd_to_ply(str(input_path), args.output, args.verbose)
        sys.exit(0 if success else 1)
    
    # Convert directory
    elif input_path.is_dir():
        successful, total = convert_directory(str(input_path), args.output, args.verbose)
        sys.exit(0 if successful == total else 1)
    
    else:
        print(f"Error: Invalid input path '{args.input}'")
        sys.exit(1)


def quick_convert_workspace():
    """
    Quick function to convert all PCD files in the current workspace.
    """
    print("Converting all PCD files in current workspace...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Convert files in main directory
    convert_directory(script_dir, verbose=True)
    
    # Convert files in aeva_data subdirectory if it exists
    aeva_dir = os.path.join(script_dir, "aeva_data")
    if os.path.exists(aeva_dir):
        print(f"\nConverting files in {aeva_dir}...")
        convert_directory(aeva_dir, verbose=True)


if __name__ == "__main__":
    # If no command line arguments, run quick convert for workspace
    if len(sys.argv) == 1:
        quick_convert_workspace()
    else:
        main()
