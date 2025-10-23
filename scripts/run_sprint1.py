"""
Sprint 1 Setup - Simple Runner

Runs Sprint 1 tasks using direct Python module execution
"""

import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent

def run_module(module_path, args="", description=""):
    """Run a Python module with arguments"""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}\n")
    
    cmd = f"{sys.executable} {module_path} {args}"
    result = subprocess.run(cmd, shell=True, cwd=project_root)
    
    return result.returncode == 0

def main():
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "SPRINT 1: DATA FOUNDATION" + " "*33 + "║")
    print("╚" + "="*78 + "╝\n")
    
    # Step 1: Initialize Database
    success = run_module(
        project_root / "src" / "data" / "database.py",
        "",
        "STEP 1: INITIALIZING DATABASE"
    )
    
    if not success:
        print("\n⚠ Database initialization had issues, but continuing...")
    
    # Step 2: Collect Historical Data
    success = run_module(
        project_root / "src" / "data" / "historical_data.py",
        "--region=default --months=6",
        "STEP 2: COLLECTING 6 MONTHS OF HISTORICAL DATA"
    )
    
    if not success:
        print("\n⚠ Historical data collection had issues, but continuing...")
    
    # Step 3: Label Events
    success = run_module(
        project_root / "src" / "data" / "event_labeling.py",
        "--create-samples --detect --auto-label --report",
        "STEP 3: LABELING CLOUD BURST EVENTS"
    )
    
    if not success:
        print("\n⚠ Event labeling had issues, but continuing...")
    
    print("\n" + "="*80)
    print("🎉 SPRINT 1 SETUP COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Review database: data/cloud_burst.db")
    print("  2. Check logs for data quality")
    print("  3. Proceed to Sprint 2: Feature Engineering")

if __name__ == "__main__":
    main()
