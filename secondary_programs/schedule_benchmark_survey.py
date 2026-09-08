"""
IMPORTANT: 3 Things
#1 Full year classes will only show up as 1 top-request or backup, not 2, which is a bug. But the failed = True and failed = False will still work
#2 English IV H GT is not in any of the generated schedules because no one requested English IV H GT on the scheduling request form
#3 MAKE SURE YOU HAVE A SCHEDULE PASTED IN FOR LINES 97-99 TO USE (INSTRUCTIONS ON LINES 95-96)
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Any

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main as core

PeriodRow = List[Optional[str]]
MasterSchedule = Dict[str, PeriodRow]

def _normalize_master_schedule(raw: Dict[str, List[Any]]) -> MasterSchedule:
    sched: MasterSchedule = {}
    for tname, row in raw.items():
        if len(row) != 8:
            raise ValueError(f"Teacher {tname} row must have 8 periods; got {len(row)}")
        out: PeriodRow = []
        for x in row:
            if x is None:
                out.append(None)
            else:
                s = str(x).strip()
                s_upper = s.upper()
                if not s or s_upper == "OFF" or s_upper == "NONE":
                    out.append(None)
                else:
                    out.append(core.normalize_name(s))
        sched[tname] = out
    return sched

def evaluate_master_schedule(
    schedule: MasterSchedule,
    seat_weight: float = 6.0,
    waste_weight: float = 0.0,
) -> Dict[str, Any]:
    courses = core.parse_request_matrix(core.CO_REQUEST_TEXT)
    energy_ctx = core.build_energy_context(courses, core.FIXED_PERIOD_MAP)
    energy = core.calculate_energy(schedule, energy_ctx)
    offered = core.compute_offered_sections(schedule, courses)
    _pen, seat_short, waste_seats = core.seat_penalty(offered, courses, waste_weight=waste_weight)
    objective = float(energy) + float(seat_weight) * float(seat_short + waste_weight * waste_seats)
    return {"energy": float(energy), "seatShort": int(seat_short), "wasteSeats": int(waste_seats), "obj": float(objective)}

def best_student_plan_for_one_schedule(
    schedule: MasterSchedule,
    top8: List[str],
    backups: List[str],
    seed: int = 1,
    top_weight: float = 100_000.0,
    backup_weights: Optional[List[float]] = None,
    fill_bonus: float = 1_000.0,
) -> core.StudentSolveResult:
    courses = core.parse_request_matrix(core.CO_REQUEST_TEXT)
    energy_ctx = core.build_energy_context(courses, core.FIXED_PERIOD_MAP)
    energy = core.calculate_energy(schedule, energy_ctx)
    offered = core.compute_offered_sections(schedule, courses)
    seat_p = core.compute_course_seat_probabilities(offered, courses)
    course_opts, art_bundle_opts = core.precompute_course_option_masks(schedule, courses)

    sample = core.ScheduleSample(
        seed=0, schedule=schedule, energy=float(energy), offered=offered,
        seat_p=seat_p, course_option_masks=course_opts, art_bundle_option_masks=art_bundle_opts,
    )

    if backup_weights is None:
        backup_weights = [9000.0, 8000.0, 7000.0, 6000.0, 5000.0, 4000.0, 3000.0, 2000.0, 1000.0]

    spec = core.prepare_student_spec(
        top8=top8, backups=backups, courses=courses,
        top_weight=top_weight, backup_weights=tuple(backup_weights),
    )

    rng = random.Random(seed)
    return core.solve_student_for_sample(
        sample=sample, courses=courses, spec=spec, rng=rng,
        seat_model="none", fill_bonus=fill_bonus, return_assignment=True,
    )

if __name__ == "__main__":
    # The Master Schedule to test against
    MASTER_SCHEDULE_RAW = {
    # PASTE IN ONE OF THE MASTER SCHEDULES AT THE BOTTOM OF THIS FILE HERE
    # Make sure to REMOVE THE TITLE so it's just "Teacher": ["Class1", "Class2"...]



    }

    # List of all 21 students
    STUDENTS = [
        {"id": 1, "top": ["AP Calculus AB", "AP Calculus AB", "AP Psychology", "AP Statistics", "Dual Enrollment 3", "Dual Enrollment 4", "Dual Enrollment 7", "Dual Enrollment 8"], "backups": ["AP European Hist"]},
        {"id": 2, "top": ["DE Eng IV: 2303", "DE Algebra III", "AP Env Sci 1C", "Amer Hist H", "AP Statistics", "Dual Enrollment 4", "Dual Enrollment 7", "Dual Enrollment 8"], "backups": ["AP Psychology", "Act Prep", "Dual Enrollment 3", "Dual Enrollment 6"]},
        {"id": 3, "top": ["DE Eng IV: 2303", "DE Algebra III", "AP Physics I Alg", "Amer Hist H", "Tal Music", "Indiv Proj H 1C", "AP Physics II Alg", "DE Adv Math PCal"], "backups": ["Fine Art (H)", "World Hist H", "AP English III (11th/12th Grade)", "AP Psychology"]},
        {"id": 4, "top": ["AP Calculus AB", "AP Calculus AB", "AP Psychology", "AP Statistics", "AP Std Art&Draw", "DE Intro to Theatre", "Early Release 4th", "Early Release 8th"], "backups": ["DE Art Hist I", "DE Eng IV: 2303", "AP European Hist", "DE Intro to Engnr"]},
        {"id": 5, "top": ["AP Psychology", "AP European Hist", "World Hist H", "DE Adv Math PCal", "Physics H", "Fine Art (H)", "Afr American H", "Dual Enrollment 4"], "backups": ["DE Art Hist I", "AP Physics I Alg", "Tal Theatre", "Dual Enrollment 8"]},
        {"id": 6, "top": ["DE Eng IV: 2303", "DE Algebra III", "AP Env Sci 1C", "Amer Hist H", "DE Intro to Engnr", "Robotics (LSU)", "AP Psychology", "Int Comp Think H"], "backups": ["AP Statistics", "World Hist H", "AP English III (11th/12th Grade)", "AP Physics I Alg"]},
        {"id": 7, "top": ["AP Physics II Alg", "AP Calculus AB", "AP Calculus AB", "Int Comp Think H", "DE Adv Math PCal", "AP Env Sci 1C", "Afr American H", "AP Psychology"], "backups": ["World Hist H", "AP Biology (full year)", "AP Biology (full year)"]},
        {"id": 8, "top": ["DE Intro to Engnr", "AP English III (11th/12th Grade)", "Physics H", "AP European Hist", "Dual Enrollment 3", "Dual Enrollment 4", "Dual Enrollment 7", "Dual Enrollment 8"], "backups": ["DE Eng IV: 2303", "Tal Theatre", "World Hist H", "AP Psychology"]},
        {"id": 9, "top": ["AP Psychology", "AP Env Sci 1C", "DE Intro to Engnr", "Early Release 4th", "Robotics (LSU)", "AP Physics I Alg", "AP English III (11th/12th Grade)", "Early Release 8th"], "backups": ["Dual Enrollment 3", "Dual Enrollment 4", "Dual Enrollment 7", "Dual Enrollment 8", "DE Intro to Theatre", "AP Research"]},
        {"id": 10, "top": ["English IV H", "DE Algebra III", "AP Env Sci 1C", "Amer Hist H", "AP Psychology", "AP Statistics", "Physics H", "Fine Art (H)"], "backups": ["AP European Hist", "Afr American H", "AP English III (11th/12th Grade)", "Tal Art"]},
        {"id": 11, "top": ["English IV H", "DE Algebra III", "AP Physics I Alg", "Amer Hist H", "AP Physics II Alg", "AP European Hist", "AP Seminar", "AP Research"], "backups": ["Afr American H", "Act Prep", "AP Psychology", "AP Statistics"]},
        {"id": 12, "top": ["AP Calculus AB", "AP Calculus AB", "AP Psychology", "AP Env Sci 1C", "DE Intro to Engnr", "Robotics (LSU)", "Dual Enrollment 4", "Dual Enrollment 8"], "backups": ["Int Comp Think H", "DE Intro to Theatre", "AP European Hist", "World Hist H"]},
        {"id": 13, "top": ["DE Eng IV: 2303", "DE Algebra III", "AP Env Sci 1C", "AP Psychology", "Indiv Proj H 1C", "Physics H", "DE Intro to Engnr", "DE Intro to Theatre"], "backups": ["DE Art Hist I", "Act Prep", "AP Seminar", "Int Comp Think H"]},
        {"id": 14, "top": ["DE Adv Math PCal", "AP European Hist", "Dual Enrollment 3", "Dual Enrollment 4", "AP Psychology", "DE Intro to Engnr", "Dual Enrollment 7", "Dual Enrollment 8"], "backups": ["AP Physics II Alg", "DE Intro to Theatre", "Tal Theatre", "World Hist H"]},
        {"id": 15, "top": ["AP Calculus AB", "AP Calculus AB", "AP Physics I Alg", "AP Psychology", "DE Intro to Engnr", "AP Statistics", "Dual Enrollment 4", "Dual Enrollment 8"], "backups": ["Dual Enrollment 2", "Dual Enrollment 3", "Dual Enrollment 6", "Dual Enrollment 7"]},
        {"id": 16, "top": ["English IV H", "DE Algebra III", "Physics H", "Amer Hist H", "AP Env Sci 1C", "Indiv Proj H 1C", "World Hist H", "Tal Art"], "backups": ["AP Psychology", "Afr American H", "DE Intro to Engnr", "AP US History", "AP US History"]},
        {"id": 17, "top": ["AP Calculus AB", "AP Calculus AB", "DE Adv Math PCal", "AP Psychology", "Int Comp Think H", "Robotics (LSU)", "Afr American H", "AP English III (11th/12th Grade)"], "backups": ["Physics H", "DE Art Hist I", "AP Biology (full year)", "AP Biology (full year)"]},
        {"id": 18, "top": ["AP Biology (full year)", "AP Biology (full year)", "AP Env Sci 1C", "AP English III (11th/12th Grade)", "AP US History", "AP US History", "English IV H", "DE Adv Math PCal"], "backups": ["AP European Hist", "DE Art Hist I", "Physics H", "AP Psychology"]},
        {"id": 19, "top": ["Int Comp Think H", "Physics H", "AP Calculus AB", "AP Calculus AB", "DE Intro to Engnr", "Robotics (LSU)", "Enginr Dev+ LSU", "AP Psychology"], "backups": ["AP Calculus AB", "AP Calculus AB", "DE Intro to Engnr", "Int Comp Think H", "Robotics (LSU)"]},
        {"id": 20, "top": ["AP Psychology", "AP Calculus AB", "AP Calculus AB", "AP Statistics", "DE Intro to Engnr", "Robotics (LSU)", "Int Comp Think H", "AP Env Sci 1C"], "backups": ["DE Art Hist I", "AP English III (11th/12th Grade)", "AP Comp Sci Prin", "DE Intro to Theatre"]},
        {"id": 21, "top": ["AP Psychology", "AP European Hist", "AP English III (11th/12th Grade)", "AP Statistics", "DE Intro to Engnr", "AP Biology (full year)", "AP Biology (full year)", "AP Env Sci 1C"], "backups": ["Robotics (LSU)", "Int Comp Think H", "Afr American H", "AP Calculus AB", "AP Calculus AB"]},
    ]

    master = _normalize_master_schedule(MASTER_SCHEDULE_RAW)

    print(f"{'Student':<12} | {'Status'}")
    print("-" * 40)

    for s_data in STUDENTS:
        res = best_student_plan_for_one_schedule(
            schedule=master,
            top8=s_data["top"],
            backups=s_data["backups"]
        )
        
        # Calculate counts
        num_top = 8 - len(res.missing_top)
        num_backups = len(res.used_backups)
        
        backup_str = f" ({num_backups} backup{'s' if num_backups != 1 else ''})" if num_backups > 0 else ""
        
        print(f"Student {s_data['id']:<3} : Failed = {res.failed}")
        print(f"             {num_top} top classes{backup_str}")
        print("-" * 40)



"""
ACTUAL 2025-2026 MASTER SCHEDULE

  "AliH": ["AP Human Geog", "AP Human Geog", "None", "World Hist H", "AP Human Geog", "Civics H", "None", "AP Human Geog"],
  "AveL": ["AP Physics I Alg", "Physics H", "AP Physics I Alg", "None", "Physics H", "AP Physics II Alg", "None", "Physics H"],
  "Brad": ["PE I", "Health Edu", "None", "None", "Health Edu", "PE I", "None", "None"],
  "CheD": ["Indiv Proj H 1C", "Act Prep", "AP Research", "None", "Act Prep", "None", "AP Seminar", "AP English III (10th Grade)"],
  "ChrB": ["AP Std Art&Draw", "None", "Fine Art (H)", "Tal Art", "Tal Art", "Fine Art (H)", "None", "None"],
  "DanL": ["Algebra II H", "Algebra II H", "Algebra II H GT", "None", "Int Comp Think H", "Int Comp Think H", "Algebra II H", "None"],
  "DerB": ["Health Edu", "None", "None", "None", "Health Edu", "None", "None", "None"],
  "DomN": ["Spanish I H", "Spanish I H", "None", "Spanish I H", "Spanish II H", "None", "Spanish II H", "Spanish II H"],
  "EmH": ["Enginr Dev+ LSU", "None", "DE Intro to Engnr", "Enginr Dev+ LSU", "None", "Enginr Dev+ LSU", "Enginr Dev+ LSU", "DE Intro to Engnr"],
  "EsK": ["Geometry H", "None", "Geometry H", "None", "Geometry H", "None", "Geometry H", "None"],
  "GigB": ["None", "None", "English III H", "English III H", "None", "AP English III (10th Grade)", "None", "None"],
  "GusP": ["AP Calculus AB", "AP Statistics", "AP Statistics", "None", "AP Calculus AB", "AP Pre-Calculus", "None", "AP Pre-Calculus"],
  "KenB": ["None", "None", "PE I", "None", "PE I", "None", "None", "None"],
  "LauM": ["AP European Hist", "None", "AP US History", "AP European Hist", "AP Psychology", "None", "AP US History", "AP Psychology"],
  "MarA": ["DE Algebra III", "None", "DE Algebra III", "DE Algebra III", "DE Adv Math PCal", "None", "DE Adv Math PCal", "Algebra II H"],
  "MarE": ["AP Biology (full year)", "Biology H", "Biology H", "None", "AP Biology (full year)", "Biology H", "Biology H", "None"],
  "MelN": ["None", "None", "Amer Hist H", "Amer Hist H", "None", "None", "Amer Hist H", "Amer Hist H"],
  "MelS": ["None", "Fin Literacy", "None", "Fin Literacy", "None", "Fin Literacy", "None", "Fin Literacy"],
  "MicM": ["Tal Art", "DE Art Hist I", "Tal Art", "None", "Tal Art", "DE Art Hist I", "Tal Art", "None"],
  "MicW": ["None", "Chemistry H", "Chemistry H", "Spanish I H", "None", "Chemistry H", "Chemistry H", "Spanish II H"],
  "NicL": ["None", "None", "None", "None", "None", "None", "Theatre Tech H", "None"],
  "OdaM": ["Civics H", "Civics H", "None", "Afr American H", "Civics H", "Afr American H", "Afr American H", "None"],
  "RebK": ["AP Env Sci 1C", "AP Env Sci 1C", "Robotics (LSU)", "None", "AP Env Sci 1C", "Robotics (LSU)", "AP Physics II Alg", "None"],
  "RobC": ["DE Intro to Theatre", "Tal Theatre", "Tal Theatre", "None", "Tal Theatre", "Tal Theatre", "DE Intro to Theatre", "None"],
  "RobP": ["None", "AP Comp Sci Prin", "None", "None", "None", "None", "None", "AP Comp Sci Prin"],
  "TayL": ["None", "None", "None", "Band", "None", "None", "None", "Band"],
  "ThoC": ["English II H GT", "None", "English II H", "English II H", "AP English III (11th/12th Grade)", "None", "English II H", "English II H"],
  "ValW": ["None", "English IV H", "DE Eng IV: 2303", "English IV H", "None", "DE Eng IV: 2303", "English IV H", "English IV H"],
  "WilL": ["None", "Music Apprec H", "Tal Music", "None", "Tal Music", "None", "Tal Music", "None"]


BEST PROGRAM GENERATED MASTER SCHEDULE

  "AliH": ["AP Human Geog", "None", "AP Human Geog", "AP Human Geog", "Civics H", "AP Human Geog", "None", "Civics H"],
  "AveL": ["AP Physics II Alg", "Physics H", "Physics H", "None", "Physics H", "AP Physics II Alg", "AP Physics I Alg", "None"],
  "Brad": ["PE I", "Health Edu", "Health Edu", "None", "Health Edu", "Health Edu", "Health Edu", "None"],
  "CheD": ["AP Research", "Indiv Proj H 1C", "None", "Act Prep", "Act Prep", "AP Seminar", "None", "AP English III (10th Grade)"],
  "ChrB": ["Tal Art", "Fine Art (H)", "AP Std Art&Draw", "None", "Tal Art", "Fine Art (H)", "Tal Art", "None"],
  "DanL": ["Algebra II H", "None", "Algebra II H GT", "Int Comp Think H", "Int Comp Think H", "Algebra II H", "Algebra II H", "None"],
  "DerB": ["None", "Health Edu", "None", "None", "Health Edu", "None", "None", "None"],
  "DomN": ["Spanish I H", "Spanish I H", "None", "Spanish I H", "None", "Spanish II H", "Spanish I H", "Spanish II H"],
  "EmH": ["Enginr Dev+ LSU", "None", "DE Intro to Engnr", "Enginr Dev+ LSU", "Enginr Dev+ LSU", "None", "Enginr Dev+ LSU", "DE Intro to Engnr"],
  "EsK": ["None", "None", "Geometry H", "Geometry H", "Geometry H", "None", "None", "Geometry H"],
  "GigB": ["AP English III (10th Grade)", "AP English III (10th Grade)", "None", "None", "English III H", "None", "None", "English III H"],
  "GusP": ["AP Calculus AB", "None", "AP Pre-Calculus", "AP Statistics", "AP Calculus AB", "Algebra II H", "AP Statistics", "None"],
  "KenB": ["None", "None", "PE I", "None", "None", "None", "PE I", "None"],
  "LauM": ["AP US History", "AP European Hist", "None", "AP European Hist", "AP US History", "None", "Amer Hist H", "AP Psychology"],
  "MarA": ["DE Algebra III", "None", "DE Algebra III", "DE Algebra III", "DE Adv Math PCal", "DE Adv Math PCal", "None", "AP Pre-Calculus"],
  "MarE": ["Biology H", "None", "AP Biology (full year)", "Biology H", "None", "Biology H", "AP Biology (full year)", "Biology H"],
  "MelN": ["Amer Hist H", "World Hist H", "None", "None", "None", "Amer Hist H", "Amer Hist H", "None"],
  "MelS": ["Fin Literacy", "Fin Literacy", "Fin Literacy", "None", "None", "Fin Literacy", "Fin Literacy", "Fin Literacy"],
  "MicM": ["None", "DE Art Hist I", "Tal Art", "DE Art Hist I", "Fine Art (H)", "Fine Art (H)", "None", "Tal Art"],
  "MicW": ["None", "Chemistry H", "Chemistry H", "Chemistry H", "Spanish II H", "Spanish II H", "None", "Chemistry H"],
  "NicL": ["None", "None", "None", "None", "None", "Theatre Tech H", "None", "None"],
  "OdaM": ["None", "Afr American H", "Civics H", "Civics H", "None", "Afr American H", "Afr American H", "Afr American H"],
  "RebK": ["Robotics (LSU)", "AP Env Sci 1C", "DE Intro to Engnr", "None", "AP Env Sci 1C", "AP Env Sci 1C", "AP Physics I Alg", "None"],
  "RobC": ["Tal Theatre", "DE Intro to Theatre", "DE Intro to Theatre", "None", "DE Intro to Theatre", "DE Intro to Theatre", "Tal Theatre", "None"],
  "RobP": ["AP Comp Sci Prin", "None", "None", "None", "None", "None", "AP Comp Sci Prin", "None"],
  "TayL": ["None", "None", "Band", "None", "None", "None", "None", "Band"],
  "ThoC": ["None", "English II H", "English II H", "AP English III (11th/12th Grade)", "English II H", "English II H GT", "English II H", "None"],
  "ValW": ["English IV H", "DE Eng IV: 2303", "English IV H", "None", "English IV", "None", "English IV H", "English IV H"],
  "WilL": ["Tal Music", "None", "Music Apprec H", "Music Apprec H", "Music Apprec H", "Music Apprec H", "None", "Music Apprec H"]


WORST PROGRAM GENERATED MASTER SCHEDULE

  "AliH": ["AP Human Geog", "AP Human Geog", "None", "AP Human Geog", "AP Human Geog", "Civics H", "None", "World Hist H"],
  "AveL": ["AP Physics II Alg", "AP Physics I Alg", "Physics H", "None", "AP Physics II Alg", "AP Physics I Alg", "None", "Physics H"],
  "Brad": ["Health Edu", "Health Edu", "Health Edu", "None", "PE I", "PE I", "PE I", "None"],
  "CheD": ["AP Research", "Indiv Proj H 1C", "Act Prep", "None", "Act Prep", "None", "Act Prep", "AP Seminar"],
  "ChrB": ["AP Std Art&Draw", "None", "Fine Art (H)", "Fine Art (H)", "Tal Art", "None", "Tal Art", "Tal Art"],
  "DanL": ["Algebra II H", "Algebra II H", "None", "Algebra II H GT", "Int Comp Think H", "None", "Algebra II H", "Int Comp Think H"],
  "DerB": ["None", "None", "Health Edu", "None", "Health Edu", "None", "None", "None"],
  "DomN": ["Spanish I H", "None", "Spanish I H", "Spanish I H", "Spanish II H", "Spanish II H", "None", "Spanish I H"],
  "EmH": ["DE Intro to Engnr", "Enginr Dev+ LSU", "Enginr Dev+ LSU", "None", "DE Intro to Engnr", "Enginr Dev+ LSU", "None", "Enginr Dev+ LSU"],
  "EsK": ["None", "Geometry H", "None", "Geometry H", "Geometry H", "None", "None", "Geometry H"],
  "GigB": ["AP English III (10th Grade)", "None", "AP English III (10th Grade)", "None", "English III H", "None", "None", "English III H"],
  "GusP": ["Algebra II H", "AP Calculus AB", "AP Statistics", "None", "AP Pre-Calculus", "AP Calculus AB", "AP Pre-Calculus", "None"],
  "KenB": ["None", "PE I", "None", "None", "None", "None", "PE I", "None"],
  "LauM": ["AP Psychology", "None", "AP Psychology", "AP US History", "None", "AP European Hist", "AP European Hist", "AP US History"],
  "MarA": ["DE Algebra III", "DE Algebra III", "DE Algebra III", "None", "None", "AP Statistics", "DE Adv Math PCal", "DE Adv Math PCal"],
  "MarE": ["None", "Biology H", "Biology H", "AP Biology (full year)", "Biology H", "None", "Biology H", "AP Biology (full year)"],
  "MelN": ["None", "Amer Hist H", "None", "Amer Hist H", "None", "Amer Hist H", "Amer Hist H", "None"],
  "MelS": ["Fin Literacy", "None", "Fin Literacy", "Fin Literacy", "Fin Literacy", "Fin Literacy", "Fin Literacy", "None"],
  "MicM": ["None", "DE Art Hist I", "Fine Art (H)", "DE Art Hist I", "DE Art Hist I", "Tal Art", "Fine Art (H)", "None"],
  "MicW": ["None", "Chemistry H", "Chemistry H", "Chemistry H", "Spanish II H", "None", "Spanish II H", "Chemistry H"],
  "NicL": ["None", "None", "None", "None", "None", "None", "Theatre Tech H", "None"],
  "OdaM": ["Civics H", "Civics H", "None", "Civics H", "None", "Afr American H", "Afr American H", "Afr American H"],
  "RebK": ["Physics H", "DE Intro to Engnr", "Robotics (LSU)", "None", "AP Env Sci 1C", "AP Env Sci 1C", "AP Env Sci 1C", "None"],
  "RobC": ["DE Intro to Theatre", "None", "Tal Theatre", "DE Intro to Theatre", "DE Intro to Theatre", "None", "DE Intro to Theatre", "DE Intro to Theatre"],
  "RobP": ["None", "None", "None", "AP Comp Sci Prin", "None", "AP Comp Sci Prin", "None", "None"],
  "TayL": ["Band", "None", "None", "None", "None", "None", "Band", "None"],
  "ThoC": ["English II H", "None", "English II H", "AP English III (11th/12th Grade)", "None", "English II H", "English II H", "English II H GT"],
  "ValW": ["English IV H", "None", "DE Eng IV: 2303", "English IV H", "English IV H", "None", "English IV H", "English IV H"],
  "WilL": ["None", "Music Apprec H", "Music Apprec H", "Tal Music", "Music Apprec H", "None", "Music Apprec H", "Tal Music"]


FROM MASSIVE RUN (completely seperate, not in any data used in the experiment)

  "AliH": ["AP Human Geog", "None", "AP Human Geog", "AP Human Geog", "Civics H", "None", "Civics H", "AP Human Geog"],
  "AveL": ["Physics H", "AP Physics II Alg", "None", "Physics H", "AP Physics II Alg", "Physics H", "None", "AP Physics I Alg"],
  "Brad": ["PE I", "Health Edu", "Health Edu", "None", "Health Edu", "Health Edu", "PE I", "None"],
  "CheD": ["Act Prep", "None", "AP Research", "Indiv Proj H 1C", "Act Prep", "Act Prep", "AP Seminar", "None"],
  "ChrB": ["Fine Art (H)", "AP Std Art&Draw", "Tal Art", "None", "None", "Fine Art (H)", "Fine Art (H)", "Tal Art"],
  "DanL": ["None", "Algebra II H", "Int Comp Think H", "Algebra II H", "Algebra II H GT", "None", "Int Comp Think H", "Algebra II H"],
  "DerB": ["None", "None", "Health Edu", "None", "None", "Health Edu", "None", "None"],
  "DomN": ["None", "Spanish I H", "Spanish I H", "Spanish I H", "Spanish II H", "Spanish II H", "None", "Spanish I H"],
  "EmH": ["None", "Enginr Dev+ LSU", "DE Intro to Engnr", "Enginr Dev+ LSU", "DE Intro to Engnr", "None", "Enginr Dev+ LSU", "Enginr Dev+ LSU"],
  "EsK": ["Geometry H", "Geometry H", "None", "None", "Geometry H", "None", "None", "Geometry H"],
  "GigB": ["AP English III (10th Grade)", "AP English III (10th Grade)", "None", "None", "English III H", "English III H", "None", "None"],
  "GusP": ["AP Pre-Calculus", "AP Calculus AB", "None", "AP Pre-Calculus", "AP Statistics", "AP Calculus AB", "AP Statistics", "None"],
  "KenB": ["None", "None", "PE I", "None", "None", "None", "PE I", "None"],
  "LauM": ["None", "Amer Hist H", "AP US History", "AP Psychology", "None", "AP Psychology", "AP US History", "AP European Hist"],
  "MarA": ["DE Algebra III", "DE Algebra III", "None", "DE Algebra III", "Algebra II H", "None", "DE Adv Math PCal", "DE Adv Math PCal"],
  "MarE": ["AP Biology (full year)", "None", "Biology H", "Biology H", "AP Biology (full year)", "Biology H", "None", "Biology H"],
  "MelN": ["None", "Amer Hist H", "World Hist H", "None", "Amer Hist H", "None", "Amer Hist H", "None"],
  "MelS": ["None", "Fin Literacy", "Fin Literacy", "Fin Literacy", "Fin Literacy", "Fin Literacy", "Fin Literacy", "None"],
  "MicM": ["DE Art Hist I", "None", "DE Art Hist I", "DE Art Hist I", "DE Art Hist I", "DE Art Hist I", "Fine Art (H)", "None"],
  "MicW": ["Chemistry H", "Chemistry H", "None", "Chemistry H", "None", "Spanish II H", "Spanish II H", "Chemistry H"],
  "NicL": ["None", "None", "None", "None", "None", "None", "Theatre Tech H", "None"],
  "OdaM": ["None", "Civics H", "Afr American H", "Civics H", "Afr American H", "Afr American H", "Afr American H", "None"],
  "RebK": ["Robotics (LSU)", "Robotics (LSU)", "DE Intro to Engnr", "None", "AP Env Sci 1C", "AP Env Sci 1C", "AP Env Sci 1C", "None"],
  "RobC": ["None", "DE Intro to Theatre", "Tal Theatre", "DE Intro to Theatre", "Tal Theatre", "DE Intro to Theatre", "Tal Theatre", "None"],
  "RobP": ["AP Comp Sci Prin", "None", "None", "None", "AP Comp Sci Prin", "None", "None", "None"],
  "TayL": ["Band", "None", "None", "None", "None", "None", "Band", "None"],
  "ThoC": ["English II H", "AP English III (11th/12th Grade)", "None", "English II H", "English II H", "English II H", "English II H GT", "None"],
  "ValW": ["None", "DE Eng IV: 2303", "English IV H", "English IV H", "English IV H", "English IV H", "None", "English IV H"],
  "WilL": ["Music Apprec H", "Tal Music", "None", "Tal Music", "Tal Music", "Tal Music", "Tal Music", "None"]

"""