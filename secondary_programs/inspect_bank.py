"""
EDIT CURRENT MODE IN LINE 104
"""

from __future__ import annotations # MUST BE LINE 1

import pickle
import os
import random
from dataclasses import dataclass
from collections import Counter
from typing import Dict, List, Optional, Tuple, Any

@dataclass
class ScheduleSample:
    seed: int
    schedule: Dict[str, List[Optional[str]]]
    energy: float
    offered: Counter
    seat_p: Dict[str, float]
    course_option_masks: Dict[str, List[int]]
    art_bundle_option_masks: List[int]

# IMPORT LOGIC
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main as core

# --- CONFIGURATION ---
BANK_PATH = ROOT / "data" / "schedule_bank.pkl"
PERIODS = ["F1", "F2", "F3", "F4", "S1", "S2", "S3", "S4"]

# FUNCTIONS
def get_detailed_metrics(sample: ScheduleSample, courses: Dict[str, Any], sw: float, ww: float):
    """
    Calculates the full breakdown including Objective and Seat Shortages.
    """
    # Calculate energy using core logic
    energy_ctx = core.build_energy_context(courses, core.FIXED_PERIOD_MAP)
    energy = core.calculate_energy(sample.schedule, energy_ctx)
    
    # Calculate seat penalties
    total_short = 0
    total_waste = 0
    shortage_breakdown = {}
    
    for cname, c in courses.items():
        if c.demand <= 0:
            continue
        off = sample.offered.get(cname, 0)
        seats = off * c.capacity
        short = max(0, c.demand - seats)
        waste = max(0, seats - c.demand)
        
        if short > 0:
            shortage_breakdown[cname] = short
        total_short += short
        total_waste += waste
        
    objective = float(energy) + sw * (float(total_short) + ww * float(total_waste))
    
    return {
        "energy": energy,
        "objective": objective,
        "shortage": total_short,
        "waste": total_waste,
        "shortage_list": shortage_breakdown
    }

def print_table(schedule: Dict[str, List[Optional[str]]]):
    header = ["Teacher"] + PERIODS
    col_width = 15
    print("\n" + " | ".join(f"{h:<{col_width}}" for h in header))
    print("-" * (len(header) * (col_width + 3)))
    
    for tname in sorted(schedule.keys()):
        row = schedule[tname]
        cells = [tname] + [(c[:col_width] if c else "---") for c in row]
        print(" | ".join(f"{cell:<{col_width}}" for cell in cells))

def main():
    if not os.path.exists(BANK_PATH):
        print(f"Error: {BANK_PATH} not found.")
        return

    print(f"Loading {BANK_PATH}...")
    with open(BANK_PATH, "rb") as f:
        # We pass the local ScheduleSample to the unpickler
        data = pickle.load(f)

    samples: List[ScheduleSample] = data["samples"]
    build_params = data.get("build_params", {})
    courses = core.parse_request_matrix(core.CO_REQUEST_TEXT)
    
    sw = build_params.get("BANK_SEAT_WEIGHT", 6.0)
    ww = build_params.get("BANK_WASTE_WEIGHT", 0.0)

    # Mode State: 0 = Objective, 1 = Conflict Energy
    current_mode = 1

    while True:
        # Sort based on current mode
        if current_mode == 0:
            # Sort by the calculated objective (Energy + Seats)
            samples.sort(key=lambda s: get_detailed_metrics(s, courses, sw, ww)["objective"])
            mode_str = "OVERALL OBJECTIVE (Energy + Seats)"
        else:
            # Sort by raw conflict energy
            samples.sort(key=lambda s: s.energy)
            mode_str = "CONFLICT ENERGY ONLY"

        total = len(samples)
        print(f"\n" + "="*40)
        print(f"BANK INSPECTOR - Mode: {mode_str}")
        print(f"Loaded {total} samples. Weights: Seat={sw}, Waste={ww}")
        print("="*40)
        print("Commands: [1-N] View Rank | [m] Toggle Mode | [q] Quit")
        
        user_input = input("Selection >> ").strip().lower()

        if user_input == 'q':
            break
        elif user_input == 'm':
            current_mode = 1 - current_mode
            print("Switched sorting mode...")
            continue

        try:
            rank = int(user_input)
            if 1 <= rank <= total:
                s = samples[rank - 1]
                metrics = get_detailed_metrics(s, courses, sw, ww)
                
                print("\n" + "#"*80)
                print(f" RANK #{rank} (Seed: {s.seed})")
                print(f" Mode: {mode_str}")
                print(f" Conflict Energy: {metrics['energy']:.2f}")
                print(f" Seat Shortage:   {metrics['shortage']}")
                print(f" Final Objective: {metrics['objective']:.2f}")
                print("#"*80)
                
                print_table(s.schedule)
                
                if metrics['shortage_list']:
                    print("\n--- SEAT SHORTAGE BREAKDOWN ---")
                    sorted_shortages = sorted(metrics['shortage_list'].items(), key=lambda x: x[1], reverse=True)
                    for cname, count in sorted_shortages[:10]:
                        print(f"  - {cname:<30}: {count} seats missing")
                    if len(sorted_shortages) > 10:
                        print(f"  ... and {len(sorted_shortages)-10} more.")
                else:
                    print("\n[Perfect Seating: No shortages found]")

                # Validation
                floors, _t = core.compute_hard_floors(courses)
                is_valid = core.validate_hard(s.schedule, core.init_teachers(), courses, floors, {})
                print(f"\nHard Constraint Validation: {'PASS' if is_valid else 'FAIL (Planning/Coach conflict)'}")
                
            else:
                print(f"Out of range (1-{total}).")
        except ValueError:
            print("Invalid input. Enter a number, 'm', or 'q'.")

if __name__ == "__main__":
    main()