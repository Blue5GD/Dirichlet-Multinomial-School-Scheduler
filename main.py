from __future__ import annotations

import os
import re
import math
import csv
import time
import random
import pickle
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple, Set, Iterable, Any

import numpy as np

# ============================================================
# USER-TUNABLE PARAMETERS (edit these first)
# ============================================================

# --- Mode switch ---
BUILD_SCHEDULE_BANK_ONCE = False          # True => build schedule_bank.pkl; False => Monte Carlo only
BANK_PATH = "data/schedule_bank.pkl"

# --- Pickle step (master schedule bank generation) ---
# Quality/runtime scales roughly with: BANK_NUM_SAMPLES * BANK_MCMC_ITERS / BANK_N_WORKERS
BANK_NUM_SAMPLES = 1_000
BANK_SEED0 = 7
BANK_MCMC_ITERS = 50_000
BANK_MAX_ATTEMPTS = 10_000

# I have a Ryzen 9 7900X: 12 physical cores. So I should start with 12 workers.
# (Using more can still work, but may increase overhead or reduce per-process CPU boost clocks.)
BANK_N_WORKERS = None       # set to None for auto-detect
BANK_INFLIGHT = None      # None => auto (≈ 2× workers)
BANK_VERBOSE_EVERY = 10

# Optimization objective used during MCMC:
# objective = conflict_energy + BANK_SEAT_WEIGHT * (seat_shortage + BANK_WASTE_WEIGHT * waste_seats)
BANK_SEAT_WEIGHT = 6.0
BANK_WASTE_WEIGHT = 0.0

# Print the best/worst schedules at the end of the pickle step
BANK_PRINT_TOP_K = 5
BANK_PRINT_BOTTOM_K = 5
BANK_PRINT_SCHEDULE_STYLE = "table"       # "long" = full names; "table" = compact
BANK_EXPORT_EXTREME_SCHEDULES = False    # True => also writes .txt files

# --- Monte Carlo student simulation (demo mode) ---
MC_ITERATIONS = 20_000
MC_SEED = 1
MC_TOP_SCENARIOS = 12
MC_ENABLE_OPTION_CACHE = False            # speed: reuse placement options instead of rebuilding each trial (IDK if this works though)

# IMPORTANT!! MUST READ NOTE TODO FIXME:
# Make sure to edit lines 2728-2732 if you're using Monte Carlo mode

# ============================================================
# 0) PASTE REQUEST MATRIX TEXT HERE (Below is my school's 25-26 request matrix)
# ============================================================

CO_REQUEST_TEXT = r"""
Act Prep: 56
Potential conflicts: Afr American H (21), Algebra II H (2), Amer Hist H (36), AP Calculus AB (1), AP Comp Sci Prin (1), AP English III (11th/12th Grade) (5), AP Env Sci 1C (25), AP European Hist (11), AP Human Geog (2), AP Physics I Alg (19), AP Physics II Alg (12), AP Pre-Calculus (18), AP Psychology (7), AP Research (4), AP Statistics (13), AP Std Art&Draw (2), AP US History (3), Band (2), Chemistry H (3), DE Adv Math PCal (5), DE Algebra III (24), DE Art Hist I (11), DE Eng IV: 2303 (5), DE Intro to Engnr (20), DE Intro to Theatre (13), Dual Enrollment 3 (2), Dual Enrollment 4 (8), Dual Enrollment 5 (1), Dual Enrollment 7 (5), Dual Enrollment 8 (14), Enginr Dev+ LSU (2), English III H (2), English IV H (36), English IV H GT (2), Fine Art (H) (5), Health Edu (1), Int Comp Think H (6), PE I (1), Physics H (17), Robotics (LSU) (5), Tal Art (5), Tal Music (1), Tal Theatre (3), Theatre Tech H (2), World Hist H (6)

Afr American H: 129
Potential conflicts: Act Prep (21), Algebra II H (43), Algebra II H GT (6), Amer Hist H (38), AP Biology (5), AP Calculus AB (16), AP Comp Sci Prin (31), AP English III (10th Grade) (15), AP English III (11th/12th Grade) (13), AP Env Sci 1C (32), AP European Hist (14), AP Human Geog (49), AP Physics I Alg (21), AP Physics II Alg (15), AP Pre-Calculus (16), AP Psychology (17), AP Research (5), AP Seminar (6), AP Statistics (19), AP US History (4), Band (11), Chemistry H (50), DE Adv Math PCal (12), DE Algebra III (33), DE Art Hist I (21), DE Eng IV: 2303 (6), DE Intro to Engnr (28), DE Intro to Theatre (10), Dual Enrollment 3 (5), Dual Enrollment 4 (16), Dual Enrollment 5 (1), Dual Enrollment 7 (7), Dual Enrollment 8 (17), Early Release 3rd (3), Early Release 4th (3), Early Release 7th (2), Early Release 8th (3), Enginr Dev+ LSU (47), English III H (30), English IV H (36), English IV H GT (3), Fin Literacy (2), Fine Art (H) (27), Geometry H (1), Health Edu (45), Indiv Proj H 1C (6), Int Comp Think H (13), Music Apprec H (12), PE I (3), Physics H (20), Robotics (LSU) (11), Tal Art (9), Tal Music (3), Tal Theatre (5), Theatre Tech H (10), World Hist H (9)

Algebra II H: 106
Potential conflicts: Act Prep (2), Afr American H (43), Amer Hist H (2), AP Comp Sci Prin (52), AP English III (10th Grade) (41), AP Env Sci 1C (1), AP European Hist (1), AP Human Geog (103), AP Pre-Calculus (9), AP Psychology (1), AP Seminar (25), Band (10), Chemistry H (105), DE Intro to Engnr (1), Enginr Dev+ LSU (93), English III H (64), Fin Literacy (13), Fine Art (H) (44), Geometry H (3), Health Edu (99), Music Apprec H (8), PE I (7), Tal Art (13), Tal Art I 2020 (2), Tal Music (1), Tal Theatre (3), Theatre Tech H (10), World Hist H (15)

Algebra II H GT: 20
Potential conflicts: Afr American H (6), Amer Hist H (1), AP Comp Sci Prin (12), AP English III (10th Grade) (19), AP Human Geog (19), AP Pre-Calculus (2), AP Seminar (6), Band (2), Chemistry H (20), Enginr Dev+ LSU (19), English III H (1), Fin Literacy (4), Fine Art (H) (1), Health Edu (18), PE I (3), Tal Art (4), Tal Music (2), Tal Theatre (1), Theatre Tech H (1), World Hist H (4)

Amer Hist H: 121
Potential conflicts: Act Prep (39), Afr American H (42), Algebra II H (2), Algebra II H GT (1), AP Biology (4), AP Comp Sci Prin (6), AP English III (11th/12th Grade) (13), AP Env Sci 1C (60), AP European Hist (15), AP Physics I Alg (49), AP Physics II Alg (30), AP Pre-Calculus (32), AP Psychology (3), AP Research (16), AP Seminar (1), AP Statistics (25), Band (5), Chemistry H (3), DE Adv Math PCal (31), DE Algebra III (83), DE Art Hist I (26), DE Eng IV: 2303 (23), DE Intro to Engnr (47), DE Intro to Theatre (14), Dual Enrollment 4 (6), Dual Enrollment 5 (1), Dual Enrollment 7 (3), Dual Enrollment 8 (16), Early Release 8th (1), Enginr Dev+ LSU (3), English III H (3), English IV H (80), English IV H GT (17), Fine Art (H) (11), Health Edu (2), Indiv Proj H 1C (26), Int Comp Think H (14), Music Apprec H (10), PE I (2), Physics H (41), Robotics (LSU) (5), Tal Art (10), Tal Music (3), Tal Theatre (4), Theatre Tech H (4), World Hist H (6)

AP Biology (full year): 13
Potential conflicts: Afr American H (5), Amer Hist H (2), AP English III (10th Grade) (1), AP Env Sci 1C (2), AP European Hist (2), AP Physics I Alg (1), AP Physics II Alg (1), AP Pre-Calculus (1), AP Psychology (4), AP Research (1), AP Statistics (2), AP US History (2), DE Adv Math PCal (4), DE Algebra III (3), DE Art Hist I (2), DE Eng IV: 2303 (2), DE Intro to Engnr (2), DE Intro to Theatre (2), Dual Enrollment 4 (2), Dual Enrollment 7 (1), Dual Enrollment 8 (1), English IV H (1), English IV H GT (1), Indiv Proj H 1C (1), Int Comp Think H (2), Music Apprec H (2), Physics H (2), Tal Art (1), Tal Music (1), Tal Theatre (1)

AP Calculus AB: 31
Potential conflicts: Act Prep (1), Afr American H (16), AP Comp Sci Prin (3), AP English III (11th/12th Grade) (2), AP Env Sci 1C (8), AP European Hist (11), AP Physics I Alg (6), AP Physics II Alg (8), AP Psychology (23), AP Statistics (12), Civics H (1), DE Adv Math PCal (3), DE Art Hist I (9), DE Eng IV: 2303 (2), DE Intro to Engnr (14), DE Intro to Theatre (6), Dual Enrollment 3 (3), Dual Enrollment 4 (15), Dual Enrollment 7 (6), Dual Enrollment 8 (13), Early Release 3rd (2), Early Release 4th (2), Early Release 7th (2), Early Release 8th (2), Enginr Dev+ LSU (1), Int Comp Think H (9), Music Apprec H (1), Physics H (3), Robotics (LSU) (14), Tal Art (1), Tal Theatre (2), World Hist H (3)

AP Comp Sci Prin: 78
Potential conflicts: Act Prep (1), Afr American H (31), Algebra II H (52), Algebra II H GT (12), Amer Hist H (5), AP Calculus AB (3), AP English III (10th Grade) (20), AP English III (11th/12th Grade) (7), AP Env Sci 1C (4), AP European Hist (3), AP Human Geog (63), AP Physics I Alg (3), AP Physics II Alg (4), AP Pre-Calculus (9), AP Psychology (2), AP Research (1), AP Seminar (12), AP Statistics (1), AP US History (1), Band (5), Biology H (1), Chemistry H (65), Civics H (1), DE Adv Math PCal (2), DE Algebra III (1), DE Art Hist I (3), DE Eng IV: 2303 (2), DE Intro to Engnr (5), DE Intro to Theatre (4), Dual Enrollment 3 (2), Dual Enrollment 4 (6), Dual Enrollment 7 (2), Dual Enrollment 8 (5), Early Release 3rd (2), Early Release 4th (1), Early Release 7th (2), Early Release 8th (1), Enginr Dev+ LSU (60), English III H (38), English IV H (2), English IV H GT (1), Fin Literacy (3), Fine Art (H) (23), Geometry H (1), Health Edu (65), Int Comp Think H (2), Music Apprec H (5), PE I (3), Physics H (3), Robotics (LSU) (1), Tal Art (11), Tal Theatre (1), Theatre Tech H (5), World Hist H (11)

AP English III (10th Grade): 62
Potential conflicts: Afr American H (15), Algebra II H (41), Algebra II H GT (19), AP Biology (1), AP Comp Sci Prin (20), AP Human Geog (59), AP Pre-Calculus (5), AP Seminar (15), Band (3), Chemistry H (62), Enginr Dev+ LSU (53), Fin Literacy (13), Fine Art (H) (13), Geometry H (1), Health Edu (56), Music Apprec H (1), PE I (6), Tal Art (10), Tal Music (1), Tal Theatre (2), Theatre Tech H (3), World Hist H (2)

AP English III (11th/12th Grade): 25
Potential conflicts: Act Prep (5), Afr American H (13), Amer Hist H (9), AP Calculus AB (2), AP Comp Sci Prin (7), AP Env Sci 1C (11), AP European Hist (9), AP Physics I Alg (5), AP Physics II Alg (4), AP Pre-Calculus (8), AP Psychology (12), AP Research (2), AP Seminar (5), AP Statistics (9), AP Std Art&Draw (1), AP US History (4), Band (2), DE Adv Math PCal (10), DE Algebra III (6), DE Art Hist I (6), DE Eng IV: 2303 (4), DE Intro to Engnr (10), DE Intro to Theatre (9), Dual Enrollment 3 (4), Dual Enrollment 4 (6), Dual Enrollment 7 (5), Dual Enrollment 8 (7), Early Release 3rd (1), English IV H (9), English IV H GT (1), Fine Art (H) (8), Indiv Proj H 1C (2), Int Comp Think H (3), Music Apprec H (2), Physics H (10), Robotics (LSU) (7), Tal Art (8), Tal Music (1), Tal Theatre (2), Theatre Tech H (2), World Hist H (12)

AP Env Sci 1C: 94
Potential conflicts: Act Prep (25), Afr American H (32), Algebra II H (1), Amer Hist H (60), AP Biology (2), AP Calculus AB (8), AP Comp Sci Prin (4), AP English III (11th/12th Grade) (11), AP European Hist (27), AP Physics I Alg (21), AP Physics II Alg (5), AP Pre-Calculus (24), AP Psychology (21), AP Research (11), AP Statistics (22), AP Std Art&Draw (1), AP US History (7), Band (4), DE Adv Math PCal (19), DE Algebra III (46), DE Art Hist I (22), DE Eng IV: 2303 (27), DE Intro to Engnr (36), DE Intro to Theatre (16), Dual Enrollment 3 (1), Dual Enrollment 4 (11), Dual Enrollment 6 (1), Dual Enrollment 7 (7), Dual Enrollment 8 (20), Early Release 3rd (1), Early Release 4th (1), Early Release 8th (4), English III H (1), English IV H (48), English IV H GT (7), Fine Art (H) (8), Health Edu (1), Indiv Proj H 1C (15), Int Comp Think H (14), Music Apprec H (5), Physics H (25), Robotics (LSU) (11), Tal Art (9), Tal Music (1), Tal Theatre (6), Theatre Tech H (5), World Hist H (5)

AP European Hist: 72
Potential conflicts: Act Prep (11), Afr American H (14), Algebra II H (1), Amer Hist H (12), AP Biology (3), AP Calculus AB (11), AP Comp Sci Prin (3), AP English III (11th/12th Grade) (9), AP Env Sci 1C (27), AP Physics I Alg (7), AP Physics II Alg (10), AP Pre-Calculus (6), AP Psychology (37), AP Research (5), AP Statistics (19), AP Std Art&Draw (4), AP US History (3), Band (3), Civics H (1), DE Adv Math PCal (19), DE Algebra III (11), DE Art Hist I (20), DE Eng IV: 2303 (10), DE Intro to Engnr (28), DE Intro to Theatre (13), Dual Enrollment 3 (14), Dual Enrollment 4 (22), Dual Enrollment 6 (2), Dual Enrollment 7 (17), Dual Enrollment 8 (22), Early Release 3rd (1), Early Release 4th (1), Early Release 6th (1), Early Release 7th (2), Early Release 8th (4), Enginr Dev+ LSU (1), English IV H (12), English IV H GT (2), Fine Art (H) (6), Indiv Proj H 1C (4), Int Comp Think H (16), Music Apprec H (8), PE I (2), Physics H (32), Robotics (LSU) (20), Tal Art (4), Tal Music (3), Tal Theatre (8), Theatre Tech H (6), World Hist H (4)

AP Human Geog: 125
Potential conflicts: Act Prep (2), Afr American H (49), Algebra II H (103), Algebra II H GT (19), AP Comp Sci Prin (63), AP English III (10th Grade) (59), AP Pre-Calculus (10), AP Seminar (30), Band (12), Chemistry H (122), Civics H (3), Enginr Dev+ LSU (110), English II H (3), English III H (63), Fin Literacy (20), Fine Art (H) (44), Geometry H (6), Health Edu (118), Music Apprec H (8), PE I (11), Tal Art (17), Tal Music (3), Tal Theatre (5), Theatre Tech H (11), World Hist H (19)

AP Physics I Alg: 66
Potential conflicts: Act Prep (19), Afr American H (21), Amer Hist H (45), AP Biology (1), AP Calculus AB (6), AP Comp Sci Prin (3), AP English III (11th/12th Grade) (5), AP Env Sci 1C (21), AP European Hist (7), AP Physics II Alg (38), AP Pre-Calculus (7), AP Psychology (14), AP Research (8), AP Statistics (14), AP Std Art&Draw (1), AP US History (4), Band (2), DE Adv Math PCal (27), DE Algebra III (41), DE Art Hist I (18), DE Eng IV: 2303 (10), DE Intro to Engnr (26), DE Intro to Theatre (5), Dual Enrollment 3 (1), Dual Enrollment 4 (10), Dual Enrollment 7 (5), Dual Enrollment 8 (12), Early Release 3rd (2), Early Release 4th (2), Early Release 7th (1), Early Release 8th (2), English III H (1), English IV H (34), English IV H GT (8), Fine Art (H) (5), Indiv Proj H 1C (14), Int Comp Think H (11), Music Apprec H (8), Physics H (1), Robotics (LSU) (3), Tal Art (6), Tal Theatre (2), Theatre Tech H (1), World Hist H (4)

AP Physics II Alg: 49
Potential conflicts: Act Prep (12), Afr American H (15), Amer Hist H (28), AP Biology (1), AP Calculus AB (8), AP Comp Sci Prin (4), AP English III (11th/12th Grade) (4), AP Env Sci 1C (5), AP European Hist (10), AP Physics I Alg (38), AP Pre-Calculus (3), AP Psychology (16), AP Research (4), AP Statistics (11), AP Std Art&Draw (1), AP US History (2), Band (1), DE Adv Math PCal (23), DE Algebra III (24), DE Art Hist I (13), DE Eng IV: 2303 (5), DE Intro to Engnr (18), DE Intro to Theatre (3), Dual Enrollment 3 (3), Dual Enrollment 4 (10), Dual Enrollment 7 (4), Dual Enrollment 8 (9), Early Release 3rd (2), Early Release 4th (1), Early Release 7th (1), Early Release 8th (1), English IV H (18), English IV H GT (7), Fine Art (H) (1), Indiv Proj H 1C (9), Int Comp Think H (11), Music Apprec H (7), Robotics (LSU) (7), Tal Art (9), World Hist H (6)

AP Pre-Calculus: 45
Potential conflicts: Act Prep (18), Afr American H (16), Algebra II H (20), Amer Hist H (26), AP Biology (1), AP Comp Sci Prin (9), AP English III (10th Grade) (5), AP English III (11th/12th Grade) (8), AP Env Sci 1C (24), AP European Hist (6), AP Human Geog (10), AP Physics I Alg (7), AP Physics II Alg (3), AP Research (3), AP Seminar (3), AP Statistics (5), AP US History (6), Chemistry H (12), DE Algebra III (3), DE Art Hist I (3), DE Eng IV: 2303 (2), DE Intro to Engnr (14), DE Intro to Theatre (6), Dual Enrollment 4 (3), Dual Enrollment 7 (1), Dual Enrollment 8 (9), Enginr Dev+ LSU (10), English III H (4), English IV H (29), English IV H GT (2), Fine Art (H) (7), Health Edu (11), Indiv Proj H 1C (1), Int Comp Think H (3), Music Apprec H (1), Physics H (15), Robotics (LSU) (1), Tal Art (1), Tal Music (3), Theatre Tech H (3), World Hist H (3)

AP Psychology: 77
Potential conflicts: Act Prep (7), Afr American H (17), Algebra II H (1), Amer Hist H (2), AP Biology (4), AP Calculus AB (23), AP Comp Sci Prin (2), AP English III (11th/12th Grade) (12), AP Env Sci 1C (21), AP European Hist (37), AP Physics I Alg (14), AP Physics II Alg (16), AP Research (3), AP Statistics (21), AP Std Art&Draw (5), AP US History (1), Band (2), Civics H (1), DE Adv Math PCal (24), DE Algebra III (1), DE Art Hist I (26), DE Eng IV: 2303 (9), DE Intro to Engnr (26), DE Intro to Theatre (19), Dual Enrollment 3 (12), Dual Enrollment 4 (27), Dual Enrollment 6 (2), Dual Enrollment 7 (17), Dual Enrollment 8 (25), Early Release 3rd (5), Early Release 4th (7), Early Release 6th (1), Early Release 7th (5), Early Release 8th (7), Enginr Dev+ LSU (1), English IV H (1), English IV H GT (2), Fine Art (H) (7), First Responder (1), Indiv Proj H 1C (2), Int Comp Think H (17), Music Apprec H (8), PE I (19), Physics H (19), Robotics (LSU) (30), Tal Art (10), Tal Music (1), Tal Theatre (5), Theatre Tech H (6), World Hist H (6)

AP Research: 19
Potential conflicts: Act Prep (4), Afr American H (5), Amer Hist H (15), AP Biology (1), AP Comp Sci Prin (1), AP English III (11th/12th Grade) (2), AP Env Sci 1C (11), AP European Hist (5), AP Physics I Alg (8), AP Physics II Alg (4), AP Pre-Calculus (3), AP Psychology (3), AP Statistics (9), AP US History (1), DE Adv Math PCal (5), DE Algebra III (14), DE Art Hist I (4), DE Eng IV: 2303 (9), DE Intro to Engnr (4), DE Intro to Theatre (3), Dual Enrollment 8 (2), English IV H (7), English IV H GT (2), Indiv Proj H 1C (6), Int Comp Think H (3), Physics H (5)

AP Seminar: 31
Potential conflicts: Afr American H (6), Algebra II H (25), Algebra II H GT (6), Amer Hist H (1), AP Comp Sci Prin (12), AP English III (10th Grade) (15), AP English III (11th/12th Grade) (5), AP Human Geog (30), AP Pre-Calculus (3), Band (1), Chemistry H (31), Enginr Dev+ LSU (27), English III H (11), Fin Literacy (3), Fine Art (H) (6), Health Edu (30), Music Apprec H (2), PE I (1), Tal Art (6), Tal Theatre (1), Theatre Tech H (1), World Hist H (4)

AP Statistics: 60
Potential conflicts: Act Prep (13), Afr American H (19), Amer Hist H (23), AP Biology (2), AP Calculus AB (12), AP Comp Sci Prin (1), AP English III (11th/12th Grade) (9), AP Env Sci 1C (22), AP European Hist (19), AP Physics I Alg (14), AP Physics II Alg (11), AP Pre-Calculus (5), AP Psychology (21), AP Research (9), AP Std Art&Draw (4), AP US History (2), Band (2), DE Adv Math PCal (11), DE Algebra III (16), DE Art Hist I (14), DE Eng IV: 2303 (9), DE Intro to Engnr (20), DE Intro to Theatre (16), Dual Enrollment 4 (10), Dual Enrollment 7 (6), Dual Enrollment 8 (12), Early Release 3rd (2), Early Release 4th (3), Early Release 6th (1), Early Release 7th (1), Early Release 8th (3), English IV H (16), English IV H GT (2), Fine Art (H) (1), Health Edu (1), Indiv Proj H 1C (5), Int Comp Think H (15), Music Apprec H (2), PE I (2), Physics H (19), Robotics (LSU) (7), Tal Art (4), Tal Theatre (1), Theatre Tech H (3), World Hist H (10)

AP Std Art&Draw: 7
Potential conflicts: Act Prep (2), AP English III (11th/12th Grade) (1), AP Env Sci 1C (1), AP European Hist (4), AP Physics I Alg (1), AP Physics II Alg (1), AP Psychology (5), AP Statistics (4), DE Adv Math PCal (2), DE Art Hist I (4), DE Intro to Engnr (2), Dual Enrollment 4 (2), Dual Enrollment 7 (1), Dual Enrollment 8 (1), Early Release 3rd (1), Early Release 7th (1), Early Release 8th (1), Fine Art (H) (1), PE I (1), Physics H (4), Robotics (LSU) (1), Tal Art (4), Theatre Tech H (2)

AP US History: 15
Potential conflicts: Act Prep (3), Afr American H (4), AP Biology (2), AP Comp Sci Prin (1), AP English III (11th/12th Grade) (4), AP Env Sci 1C (7), AP European Hist (3), AP Physics I Alg (4), AP Physics II Alg (2), AP Pre-Calculus (6), AP Psychology (1), AP Research (1), AP Statistics (2), Chemistry H (1), DE Adv Math PCal (5), DE Algebra III (9), DE Art Hist I (2), DE Eng IV: 2303 (2), DE Intro to Engnr (2), Dual Enrollment 4 (1), Dual Enrollment 8 (3), Enginr Dev+ LSU (1), English IV H (11), English IV H GT (2), Fine Art (H) (2), Health Edu (1), Indiv Proj H 1C (3), Music Apprec H (1), Physics H (1), Tal Art (2), Tal Music (1), World Hist H (1)

Band: 20
Potential conflicts: Act Prep (2), Afr American H (11), Algebra II H (10), Algebra II H GT (2), Amer Hist H (5), AP Comp Sci Prin (5), AP English III (10th Grade) (3), AP English III (11th/12th Grade) (2), AP Env Sci 1C (4), AP European Hist (3), AP Human Geog (12), AP Physics I Alg (2), AP Physics II Alg (1), AP Psychology (2), AP Seminar (1), AP Statistics (2), Chemistry H (12), DE Algebra III (5), DE Art Hist I (2), DE Eng IV: 2303 (2), DE Intro to Engnr (2), DE Intro to Theatre (3), Dual Enrollment 3 (1), Dual Enrollment 4 (1), Dual Enrollment 8 (1), Enginr Dev+ LSU (12), English III H (7), English IV H (1), English IV H GT (2), Fin Literacy (1), Fine Art (H) (10), Health Edu (12), Indiv Proj H 1C (1), Int Comp Think H (3), Music Apprec H (2), Physics H (3), Robotics (LSU) (1), Tal Art (2), Theatre Tech H (2), World Hist H (1)

Biology H: 124
Potential conflicts: AP Comp Sci Prin (1), AP Human Geog (3), Civics H (124), Dual Enrollment 4 (1), English II H (104), English II H GT (19), Fin Literacy (117), Fine Art (H) (1), Geometry H (122), Health Edu (7), PE I (90), Spanish I H (118), Spanish II H (119), Tal Art (19), Tal Music (5), Tal Theatre (14)

Chemistry H: 127
Potential conflicts: Act Prep (3), Afr American H (50), Algebra II H (105), Algebra II H GT (20), Amer Hist H (3), AP Comp Sci Prin (65), AP English III (10th Grade) (62), AP Human Geog (122), AP Pre-Calculus (12), AP Seminar (31), AP US History (1), Band (12), DE Algebra III (1), DE Intro to Engnr (1), DE Intro to Theatre (1), Enginr Dev+ LSU (113), English III H (65), English IV H (1), Fin Literacy (17), Fine Art (H) (45), Geometry H (3), Health Edu (118), Music Apprec H (8), PE I (10), Physics H (1), Tal Art (18), Tal Music (3), Tal Theatre (4), Theatre Tech H (11), World Hist H (20)

Civics H: 125
Potential conflicts: AP Calculus AB (1), AP Comp Sci Prin (1), AP European Hist (1), AP Human Geog (3), AP Psychology (1), Biology H (124), DE Intro to Engnr (1), Dual Enrollment 4 (1), Enginr Dev+ LSU (1), English II H (104), English II H GT (19), Fin Literacy (117), Fine Art (H) (1), Geometry H (122), Health Edu (7), PE I (90), Physics H (1), Robotics (LSU) (1), Spanish I H (118), Spanish II H (119), Tal Art (19), Tal Music (5), Tal Theatre (14)

DE Adv Math PCal: 66
Potential conflicts: Act Prep (5), Afr American H (12), Amer Hist H (26), AP Biology (4), AP Calculus AB (3), AP Comp Sci Prin (2), AP English III (11th/12th Grade) (10), AP Env Sci 1C (19), AP European Hist (19), AP Physics I Alg (27), AP Physics II Alg (23), AP Psychology (24), AP Research (5), AP Statistics (11), AP Std Art&Draw (2), AP US History (5), DE Algebra III (27), DE Art Hist I (22), DE Eng IV: 2303 (10), DE Intro to Engnr (24), DE Intro to Theatre (8), Dual Enrollment 3 (10), Dual Enrollment 4 (18), Dual Enrollment 7 (12), Dual Enrollment 8 (19), Early Release 3rd (3), Early Release 4th (3), Early Release 6th (1), Early Release 7th (1), Early Release 8th (3), English IV H (19), English IV H GT (8), Fine Art (H) (7), Indiv Proj H 1C (16), Int Comp Think H (12), Music Apprec H (8), PE I (1), Physics H (17), Robotics (LSU) (11), Tal Art (7), Tal Music (2), Tal Theatre (4), World Hist H (4)

DE Algebra III: 86
Potential conflicts: Act Prep (24), Afr American H (33), Amer Hist H (74), AP Biology (3), AP Comp Sci Prin (1), AP English III (11th/12th Grade) (6), AP Env Sci 1C (46), AP European Hist (11), AP Physics I Alg (41), AP Physics II Alg (24), AP Pre-Calculus (3), AP Psychology (1), AP Research (14), AP Statistics (16), AP US History (9), Band (5), Chemistry H (1), DE Adv Math PCal (27), DE Art Hist I (22), DE Eng IV: 2303 (21), DE Intro to Engnr (33), DE Intro to Theatre (10), Dual Enrollment 1 (1), Dual Enrollment 2 (1), Dual Enrollment 3 (1), Dual Enrollment 4 (3), Dual Enrollment 5 (2), Dual Enrollment 6 (1), Dual Enrollment 7 (3), Dual Enrollment 8 (9), English III H (1), English IV H (55), English IV H GT (14), Fine Art (H) (7), Indiv Proj H 1C (25), Int Comp Think H (11), Music Apprec H (10), Physics H (25), Robotics (LSU) (4), Tal Art (8), Tal Music (2), Tal Theatre (3), Theatre Tech H (2), World Hist H (5)

DE Art Hist I: 66
Potential conflicts: Act Prep (11), Afr American H (21), Amer Hist H (24), AP Biology (2), AP Calculus AB (9), AP Comp Sci Prin (3), AP English III (11th/12th Grade) (6), AP Env Sci 1C (22), AP European Hist (20), AP Physics I Alg (18), AP Physics II Alg (13), AP Pre-Calculus (3), AP Psychology (26), AP Research (4), AP Statistics (14), AP Std Art&Draw (4), AP US History (2), Band (2), DE Adv Math PCal (22), DE Algebra III (22), DE Eng IV: 2303 (4), DE Intro to Engnr (18), DE Intro to Theatre (17), Dual Enrollment 3 (9), Dual Enrollment 4 (21), Dual Enrollment 6 (1), Dual Enrollment 7 (14), Dual Enrollment 8 (23), Early Release 3rd (4), Early Release 4th (5), Early Release 7th (4), Early Release 8th (7), English IV H (20), English IV H GT (5), Fine Art (H) (4), Indiv Proj H 1C (3), Int Comp Think H (10), Music Apprec H (2), Physics H (23), Robotics (LSU) (10), Tal Art (7), Tal Theatre (2), Theatre Tech H (3), World Hist H (2)

DE Eng IV: 2303: 33
Potential conflicts: Act Prep (5), Afr American H (6), Amer Hist H (21), AP Biology (2), AP Calculus AB (2), AP Comp Sci Prin (2), AP English III (11th/12th Grade) (4), AP Env Sci 1C (27), AP European Hist (10), AP Physics I Alg (10), AP Physics II Alg (5), AP Pre-Calculus (2), AP Psychology (9), AP Research (9), AP Statistics (9), AP US History (2), Band (2), DE Adv Math PCal (10), DE Algebra III (21), DE Art Hist I (4), DE Intro to Engnr (7), DE Intro to Theatre (9), Dual Enrollment 3 (1), Dual Enrollment 4 (5), Dual Enrollment 7 (4), Dual Enrollment 8 (7), Early Release 8th (1), English IV H (3), English IV H GT (1), Health Edu (1), Indiv Proj H 1C (12), Int Comp Think H (4), Physics H (9), Tal Art (3), Tal Music (1), World Hist H (3)

DE Intro to Theatre: 51
Potential conflicts: Act Prep (13), Afr American H (10), Amer Hist H (14), AP Biology (2), AP Calculus AB (6), AP Comp Sci Prin (4), AP English III (11th/12th Grade) (9), AP Env Sci 1C (16), AP European Hist (13), AP Physics I Alg (5), AP Physics II Alg (3), AP Pre-Calculus (6), AP Psychology (19), AP Research (3), AP Statistics (16), Band (3), Chemistry H (1), DE Adv Math PCal (8), DE Algebra III (10), DE Art Hist I (14), DE Eng IV: 2303 (9), DE Intro to Engnr (20), Dual Enrollment 2 (1), Dual Enrollment 3 (10), Dual Enrollment 4 (14), Dual Enrollment 6 (1), Dual Enrollment 7 (12), Dual Enrollment 8 (13), Early Release 3rd (5), Early Release 4th (5), Early Release 5th (1), Early Release 6th (3), Early Release 7th (3), Early Release 8th (6), English IV H (11), English IV H GT (2), Fine Art (H) (4), Indiv Proj H 1C (1), Int Comp Think H (9), Music Apprec H (4), PE I (1), Physics H (20), Robotics (LSU) (7), Tal Art (2), Tal Music (3), Tal Theatre (1), Theatre Tech H (1), World Hist H (8)

DE Intro to Engnr: 85
Potential conflicts: Act Prep (20), Afr American H (28), Algebra II H (1), Amer Hist H (45), AP Biology (2), AP Calculus AB (14), AP Comp Sci Prin (5), AP English III (11th/12th Grade) (10), AP Env Sci 1C (36), AP European Hist (28), AP Physics I Alg (26), AP Physics II Alg (18), AP Pre-Calculus (14), AP Psychology (26), AP Research (4), AP Statistics (20), AP Std Art&Draw (2), AP US History (2), Band (2), Chemistry H (1), Civics H (1), DE Adv Math PCal (24), DE Algebra III (33), DE Art Hist I (18), DE Eng IV: 2303 (7), DE Intro to Theatre (20), Dual Enrollment 3 (9), Dual Enrollment 4 (14), Dual Enrollment 7 (10), Dual Enrollment 8 (15), Early Release 3rd (1), Early Release 4th (2), Early Release 8th (2), Enginr Dev+ LSU (1), English III H (1), English IV H (33), English IV H GT (10), Fine Art (H) (7), Indiv Proj H 1C (7), Int Comp Think H (19), Music Apprec H (9), PE I (2), Physics H (22), Robotics (LSU) (22), Tal Art (4), Tal Music (1), Tal Theatre (2), Theatre Tech H (4), World Hist H (7)

Dual Enrollment 1: 8
Potential conflicts: DE Algebra III (1), Dual Enrollment 2 (8), Dual Enrollment 3 (8), Dual Enrollment 4 (8), Dual Enrollment 5 (8), Dual Enrollment 6 (8), Dual Enrollment 7 (8), Dual Enrollment 8 (8)

Dual Enrollment 2: 10
Potential conflicts: DE Algebra III (1), DE Intro to Theatre (1), Dual Enrollment 1 (8), Dual Enrollment 3 (10), Dual Enrollment 4 (10), Dual Enrollment 5 (8), Dual Enrollment 6 (9), Dual Enrollment 7 (9), Dual Enrollment 8 (9), Early Release 1st (1), Physics H (1)

Dual Enrollment 3: 37
Potential conflicts: Act Prep (2), Afr American H (5), AP Calculus AB (3), AP Comp Sci Prin (2), AP English III (11th/12th Grade) (4), AP Env Sci 1C (1), AP European Hist (14), AP Physics I Alg (1), AP Physics II Alg (3), AP Psychology (12), AP Statistics (4), Band (1), DE Adv Math PCal (10), DE Algebra III (1), DE Art Hist I (9), DE Eng IV: 2303 (10), DE Intro to Engnr (9), DE Intro to Theatre (10), Dual Enrollment 1 (8), Dual Enrollment 2 (10), Dual Enrollment 4 (37), Dual Enrollment 5 (8), Dual Enrollment 6 (9), Dual Enrollment 7 (33), Dual Enrollment 8 (35), Early Release 1st (1), Early Release 7th (1), Fine Art (H) (5), Int Comp Think H (2), Music Apprec H (2), Part Tpc Calc H (3), Physics H (16), Robotics (LSU) (3), Tal Art (2), Tal Theatre (2), World Hist H (1)

Dual Enrollment 4: 71
Potential conflicts: Act Prep (8), Afr American H (16), Amer Hist H (6), AP Biology (2), AP Calculus AB (15), AP Comp Sci Prin (6), AP English III (11th/12th Grade) (6), AP Env Sci 1C (11), AP European Hist (22), AP Physics I Alg (10), AP Physics II Alg (10), AP Pre-Calculus (3), AP Psychology (27), AP Statistics (10), AP Std Art&Draw (2), AP US History (1), Band (1), Biology H (1), Civics H (1), DE Adv Math PCal (18), DE Algebra III (3), DE Art Hist I (21), DE Eng IV: 2303 (5), DE Intro to Engnr (14), DE Intro to Theatre (14), Dual Enrollment 1 (8), Dual Enrollment 2 (10), Dual Enrollment 3 (37), Dual Enrollment 5 (9), Dual Enrollment 6 (10), Dual Enrollment 7 (44), Dual Enrollment 8 (60), Early Release 1st (1), Early Release 3rd (6), Early Release 5th (1), Early Release 6th (2), Early Release 7th (5), Early Release 8th (5), English II H GT (1), English IV H (5), English IV H GT (1), Fin Literacy (1), Fine Art (H) (9), Int Comp Think H (8), Music Apprec H (2), PE I (1), Physics H (26), Robotics (LSU) (9), Spanish I H (1), Spanish II H (1), Tal Art (10), Tal Theatre (7), Theatre Tech H (1), World Hist H (4)

Dual Enrollment 5: 9
Potential conflicts: Act Prep (1), Afr American H (1), Amer Hist H (1), DE Algebra III (2), Dual Enrollment 1 (8), Dual Enrollment 2 (8), Dual Enrollment 3 (8), Dual Enrollment 4 (8), Dual Enrollment 6 (8), Dual Enrollment 7 (8), Dual Enrollment 8 (8), English IV H (1), Physics H (1)

Dual Enrollment 6: 11
Potential conflicts: AP Env Sci 1C (1), AP European Hist (2), AP Psychology (2), DE Algebra III (1), DE Art Hist I (1), DE Intro to Theatre (1), Dual Enrollment 1 (8), Dual Enrollment 2 (9), Dual Enrollment 3 (9), Dual Enrollment 4 (10), Dual Enrollment 5 (8), Dual Enrollment 7 (11), Dual Enrollment 8 (11), Physics H (2), Robotics (LSU) (2)

Dual Enrollment 7: 48
Potential conflicts: Act Prep (5), Afr American H (7), Amer Hist H (3), AP Biology (1), AP Calculus AB (6), AP Comp Sci Prin (2), AP English III (11th/12th Grade) (5), AP Env Sci 1C (7), AP European Hist (17), AP Physics I Alg (5), AP Physics II Alg (4), AP Pre-Calculus (1), AP Psychology (17), AP Statistics (6), AP Std Art&Draw (1), Band (1), DE Adv Math PCal (12), DE Algebra III (3), DE Art Hist I (14), DE Intro to Theatre (12), DE Intro to Engnr (10), Dual Enrollment 1 (8), Dual Enrollment 2 (9), Dual Enrollment 3 (33), Dual Enrollment 4 (44), Dual Enrollment 5 (8), Dual Enrollment 6 (11), Dual Enrollment 8 (47), Early Release 3rd (3), Early Release 4th (1), Early Release 6th (2), Early Release 8th (1), English IV H (3), Fine Art (H) (7), Indiv Proj H 1C (1), Int Comp Think H (4), Music Apprec H (2), Physics H (22), Robotics (LSU) (4), Tal Art (4), Tal Theatre (3), World Hist H (1)

Dual Enrollment 8: 74
Potential conflicts: Act Prep (14), Afr American H (17), Amer Hist H (13), AP Biology (1), AP Calculus AB (13), AP Comp Sci Prin (5), AP English III (11th/12th Grade) (7), AP Env Sci 1C (20), AP European Hist (22), AP Physics I Alg (12), AP Physics II Alg (9), AP Pre-Calculus (9), AP Psychology (25), AP Research (2), AP Statistics (12), AP Std Art&Draw (1), AP US History (3), Band (1), DE Adv Math PCal (19), DE Algebra III (9), DE Art Hist I (23), DE Eng IV: 2303 (7), DE Intro to Engnr (15), DE Intro to Theatre (13), Dual Enrollment 1 (8), Dual Enrollment 2 (9), Dual Enrollment 3 (35), Dual Enrollment 4 (60), Dual Enrollment 5 (8), Dual Enrollment 6 (11), Dual Enrollment 7 (47), Early Release 3rd (4), Early Release 4th (1), Early Release 6th (2), Early Release 7th (2), English IV H (14), English IV H GT (1), Fine Art (H) (8), Indiv Proj H 1C (3), Int Comp Think H (8), Music Apprec H (2), Physics H (27), Robotics (LSU) (10), Tal Art (7), Tal Theatre (5), Theatre Tech H (1), World Hist H (4)

Early Release 1st: 2
Potential conflicts: Dual Enrollment 2 (1), Dual Enrollment 3 (1), Dual Enrollment 4 (1), Early Release 2nd (1), Early Release 7th (1), Early Release 8th (1), Fine Art (H) (1), Physics H (1)

Early Release 2nd: 1
Potential conflicts: Early Release 1st (1), Early Release 7th (1), Early Release 8th (1), Fine Art (H) (1), Physics H (1)

Early Release 3rd: 12
Potential conflicts: Afr American H (3), AP Calculus AB (2), AP Comp Sci Prin (2), AP English III (11th/12th Grade) (1), AP Env Sci 1C (1), AP European Hist (1), AP Physics I Alg (2), AP Physics II Alg (2), AP Psychology (5), AP Statistics (2), AP Std Art&Draw (1), DE Adv Math PCal (3), DE Art Hist I (4), DE Intro to Engnr (1), DE Intro to Theatre (5), Dual Enrollment 4 (6), Dual Enrollment 7 (3), Dual Enrollment 8 (4), Early Release 4th (6), Early Release 5th (1), Early Release 6th (4), Early Release 7th (7), Early Release 8th (7), English IV H (1), Fine Art (H) (2), Int Comp Think H (1), Music Apprec H (1), Physics H (5), Tal Art (2), Tal Theatre (2), World Hist H (2)

Early Release 4th: 10
Potential conflicts: Afr American H (3), AP Calculus AB (2), AP Comp Sci Prin (1), AP Env Sci 1C (1), AP European Hist (1), AP Physics I Alg (2), AP Physics II Alg (1), AP Psychology (7), AP Statistics (3), DE Adv Math PCal (3), DE Art Hist I (5), DE Intro to Engnr (2), DE Intro to Theatre (5), Dual Enrollment 7 (1), Dual Enrollment 8 (1), Early Release 3rd (6), Early Release 6th (2), Early Release 7th (4), Early Release 8th (7), Fine Art (H) (1), Int Comp Think H (2), Music Apprec H (2), Physics H (2), Robotics (LSU) (2), Tal Art (1), Tal Theatre (1)

Early Release 5th: 1
Potential conflicts: DE Intro to Theatre (1), Dual Enrollment 4 (1), Early Release 3rd (1), Early Release 6th (1), Early Release 7th (1), Early Release 8th (1), English IV H (1), World Hist H (1)

Early Release 6th: 4
Potential conflicts: AP European Hist (1), AP Psychology (1), AP Statistics (1), DE Adv Math PCal (1), DE Intro to Theatre (3), Dual Enrollment 4 (2), Dual Enrollment 7 (2), Dual Enrollment 8 (2), Early Release 3rd (4), Early Release 4th (2), Early Release 5th (1), Early Release 7th (2), Early Release 8th (2), English IV H (1), Physics H (2), Tal Theatre (1), World Hist H (2)

Early Release 7th: 10
Potential conflicts: Afr American H (2), AP Calculus AB (2), AP Comp Sci Prin (2), AP European Hist (2), AP Physics I Alg (1), AP Physics II Alg (1), AP Psychology (5), AP Statistics (1), AP Std Art&Draw (1), DE Adv Math PCal (1), DE Art Hist I (4), DE Intro to Theatre (3), DE Intro to Engnr (10), Dual Enrollment 3 (1), Dual Enrollment 4 (5), Dual Enrollment 8 (2), Early Release 1st (1), Early Release 2nd (1), Early Release 3rd (7), Early Release 4th (4), Early Release 5th (1), Early Release 6th (2), Early Release 8th (8), English IV H (1), Fine Art (H) (2), Music Apprec H (1), Physics H (5), Robotics (LSU) (2), Tal Theatre (1), World Hist H (2)

Early Release 8th: 15
Potential conflicts: Afr American H (3), Amer Hist H (1), AP Calculus AB (2), AP Comp Sci Prin (1), AP Env Sci 1C (4), AP European Hist (4), AP Physics I Alg (2), AP Physics II Alg (1), AP Psychology (7), AP Statistics (3), AP Std Art&Draw (1), DE Adv Math PCal (1), DE Art Hist I (7), DE Eng IV: 2303 (1), DE Intro to Engnr (2), DE Intro to Theatre (6), Dual Enrollment 3 (1), Dual Enrollment 4 (5), Dual Enrollment 7 (1), Early Release 1st (1), Early Release 2nd (1), Early Release 3rd (7), Early Release 4th (7), Early Release 5th (1), Early Release 6th (2), Early Release 7th (8), English IV H (1), Fine Art (H) (3), Int Comp Think H (2), Music Apprec H (1), Physics H (5), Robotics (LSU) (4), Tal Art (1), Tal Theatre (2), World Hist H (1)

Enginr Dev+ LSU: 114
Potential conflicts: Act Prep (2), Afr American H (47), Algebra II H (93), Algebra II H GT (19), Amer Hist H (3), AP Calculus AB (1), AP Comp Sci Prin (60), AP English III (10th Grade) (53), AP European Hist (11), AP Human Geog (110), AP Pre-Calculus (10), AP Psychology (1), AP Seminar (27), AP US History (1), Band (12), Chemistry H (113), Civics H (1), DE Intro to Engnr (1), English III H (60), Fin Literacy (11), Fine Art (H) (38), Geometry H (2), Health Edu (106), Music Apprec H (8), PE I (8), Physics H (2), Robotics (LSU) (1), Tal Art (14), Tal Music (3), Tal Theatre (1), Theatre Tech H (9), World Hist H (17)

English II H: 104
Potential conflicts: AP Comp Sci Prin (1), AP Human Geog (3), Biology H (104), Civics H (104), Fin Literacy (99), Fine Art (H) (1), Geometry H (104), Health Edu (6), PE I (78), Spanish I H (99), Spanish II H (100), Tal Art (14), Tal Music (5), Tal Theatre (10)

English II H GT: 19
Potential conflicts: Biology H (19), Civics H (19), Dual Enrollment 4 (1), Fin Literacy (18), Geometry H (18), Health Edu (1), PE I (12), Spanish I H (18), Spanish II H (18), Tal Art (5), Tal Theatre (4)

English III H: 66
Potential conflicts: Act Prep (2), Afr American H (30), Algebra II H (64), AP Comp Sci Prin (38), AP Env Sci 1C (1), AP Human Geog (63), AP Physics I Alg (1), AP Pre-Calculus (4), AP Seminar (11), Band (7), Chemistry H (65), DE Algebra III (1), DE Intro to Engnr (1), Enginr Dev+ LSU (60), English IV H (1), Fin Literacy (4), Fine Art (H) (27), Geometry H (2), Health Edu (1), Music Apprec H (8), PE I (4), Tal Art (3), Tal Music (1), Theatre Tech H (9), World Hist H (7)

English IV H: 87
Potential conflicts: Act Prep (36), Afr American H (36), Amer Hist H (69), AP Biology (1), AP Comp Sci Prin (2), AP English III (11th/12th Grade) (9), AP Env Sci 1C (48), AP European Hist (12), AP Physics I Alg (34), AP Physics II Alg (18), AP Pre-Calculus (29), AP Psychology (1), AP Research (7), AP Statistics (16), AP US History (11), Band (1), Chemistry H (1), DE Adv Math PCal (19), DE Algebra III (55), DE Art Hist I (20), DE Eng IV: 2303 (3), DE Intro to Engnr (33), DE Intro to Theatre (11), Dual Enrollment 4 (5), Dual Enrollment 5 (1), Dual Enrollment 7 (3), Dual Enrollment 8 (14), Early Release 3rd (1), Early Release 5th (1), Early Release 6th (1), Early Release 7th (1), Early Release 8th (1), English III H (1), English IV H GT (2), Fine Art (H) (10), Indiv Proj H 1C (11), Int Comp Think H (5), Music Apprec H (4), Physics H (34), Robotics (LSU) (3), Tal Art (6), Tal Music (3), Tal Theatre (3), Theatre Tech H (3), World Hist H (7)

English IV H GT: 18
Potential conflicts: Act Prep (2), Afr American H (3), Amer Hist H (15), AP Biology (1), AP Comp Sci Prin (1), AP English III (11th/12th Grade) (1), AP Env Sci 1C (7), AP European Hist (2), AP Physics I Alg (8), AP Physics II Alg (7), AP Pre-Calculus (2), AP Psychology (2), AP Research (2), AP Statistics (2), AP US History (2), Band (2), DE Adv Math PCal (8), DE Algebra III (14), DE Art Hist I (5), DE Eng IV: 2303 (1), DE Intro to Engnr (10), DE Intro to Theatre (2), Dual Enrollment 4 (1), Dual Enrollment 8 (1), English IV H (2), Indiv Proj H 1C (8), Int Comp Think H (7), Music Apprec H (4), Physics H (6), Robotics (LSU) (3), Tal Art (2), Tal Music (1), Tal Theatre (1), Theatre Tech H (1), World Hist H (1)

Fin Literacy: 134
Potential conflicts: Afr American H (2), Algebra II H (13), Algebra II H GT (4), AP Comp Sci Prin (3), AP English III (10th Grade) (13), AP Human Geog (20), AP Seminar (3), Band (1), Biology H (117), Chemistry H (17), Civics H (117), Dual Enrollment 4 (1), Enginr Dev+ LSU (11), English II H (99), English II H GT (18), English III H (4), Fine Art (H) (5), Geometry H (119), Health Edu (19), PE I (84), Spanish I H (111), Spanish II H (112), Tal Art (25), Tal Music (5), Tal Theatre (16), Theatre Tech H (1), World Hist H (1)

Fine Art (H): 69
Potential conflicts: Act Prep (5), Afr American H (27), Algebra II H (44), Algebra II H GT (1), Amer Hist H (9), AP Comp Sci Prin (23), AP English III (10th Grade) (13), AP English III (11th/12th Grade) (8), AP Env Sci 1C (8), AP European Hist (6), AP Human Geog (44), AP Physics I Alg (5), AP Physics II Alg (1), AP Pre-Calculus (7), AP Psychology (7), AP Seminar (6), AP Statistics (1), AP Std Art&Draw (1), AP US History (2), Band (10), Biology H (1), Chemistry H (45), Civics H (1), DE Adv Math PCal (7), DE Algebra III (7), DE Art Hist I (4), DE Intro to Engnr (7), DE Intro to Theatre (4), Dual Enrollment 3 (5), Dual Enrollment 4 (9), Dual Enrollment 7 (7), Dual Enrollment 8 (8), Early Release 1st (1), Early Release 2nd (1), Early Release 3rd (2), Early Release 4th (1), Early Release 7th (2), Early Release 8th (3), Enginr Dev+ LSU (38), English II H (1), English III H (27), English IV H (10), Fin Literacy (5), Geometry H (2), Health Edu (42), Indiv Proj H 1C (1), Int Comp Think H (1), Music Apprec H (2), PE I (2), Physics H (11), Robotics (LSU) (1), Spanish II H (1), Tal Art (7), Tal Theatre (4), Theatre Tech H (2), World Hist H (1)

Geometry H: 125
Potential conflicts: Afr American H (1), Algebra II H (3), AP Comp Sci Prin (1), AP English III (10th Grade) (1), AP Human Geog (1), Biology H (122), Chemistry H (3), Civics H (122), Enginr Dev+ LSU (2), English II H (104), English II H GT (18), English III H (2), Fin Literacy (119), Fine Art (H) (2), Health Edu (9), PE I (89), Spanish I H (116), Spanish II H (117), Tal Art (20), Tal Music (5), Tal Theatre (14)

Health Edu: 126
Potential conflicts: Act Prep (1), Afr American H (45), Algebra II H (99), Algebra II H GT (18), Amer Hist H (1), AP Comp Sci Prin (65), AP English III (10th Grade) (56), AP Env Sci 1C (3), AP Human Geog (118), AP Pre-Calculus (11), AP Seminar (30), AP Statistics (1), AP US History (1), Band (12), Biology H (7), Chemistry H (118), Civics H (7), DE Eng IV: 2303 (1), Enginr Dev+ LSU (106), English II H (6), English II H GT (1), English III H (62), Fin Literacy (19), Fine Art (H) (42), Geometry H (9), Music Apprec H (7), PE I (14), Physics H (1), Spanish I H (3), Spanish II H (3), Tal Art (16), Tal Music (2), Tal Theatre (14), Theatre Tech H (10), World Hist H (19)

Indiv Proj H 1C: 27
Potential conflicts: Afr American H (6), Amer Hist H (23), AP Biology (1), AP English III (11th/12th Grade) (2), AP English III (10th Grade) (3), AP Env Sci 1C (15), AP European Hist (4), AP Physics I Alg (14), AP Physics II Alg (9), AP Pre-Calculus (1), AP Psychology (2), AP Research (6), AP Statistics (5), AP US History (3), Band (1), DE Adv Math PCal (16), DE Algebra III (25), DE Art Hist I (3), DE Eng IV: 2303 (12), DE Intro to Engnr (7), DE Intro to Theatre (1), Dual Enrollment 7 (1), Dual Enrollment 8 (3), English IV H (11), English IV H GT (8), Fine Art (H) (1), Int Comp Think H (5), Music Apprec H (5), Physics H (8), Robotics (LSU) (1), Tal Art (3), Theatre Tech H (1)

Int Comp Think H: 39
Potential conflicts: Act Prep (6), Afr American H (13), Amer Hist H (14), AP Biology (2), AP Calculus AB (9), AP English III (11th/12th Grade) (3), AP Env Sci 1C (14), AP European Hist (16), AP Physics I Alg (11), AP Physics II Alg (11), AP Pre-Calculus (3), AP Psychology (17), AP Research (3), AP Statistics (15), Band (3), DE Adv Math PCal (12), DE Algebra III (11), DE Art Hist I (10), DE Eng IV: 2303 (4), DE Intro to Engnr (19), DE Intro to Theatre (9), Dual Enrollment 3 (2), Dual Enrollment 4 (8), Dual Enrollment 7 (4), Dual Enrollment 8 (8), Early Release 3rd (1), Early Release 4th (2), Early Release 8th (2), English IV H (5), English IV H GT (7), Fine Art (H) (1), Indiv Proj H 1C (5), Music Apprec H (6), Physics H (9), Robotics (LSU) (8), Tal Art (2), Tal Music (1), Tal Theatre (1), Theatre Tech H (1), World Hist H (3)

Music Apprec H: 29
Potential conflicts: Afr American H (12), Algebra II H (8), Amer Hist H (9), AP Biology (2), AP Calculus AB (1), AP Comp Sci Prin (5), AP English III (10th Grade) (1), AP English III (11th/12th Grade) (2), AP Env Sci 1C (5), AP European Hist (8), AP Human Geog (8), AP Physics I Alg (8), AP Physics II Alg (7), AP Pre-Calculus (1), AP Psychology (8), AP Seminar (2), AP Statistics (2), AP US History (1), Band (2), Chemistry H (8), DE Adv Math PCal (8), DE Algebra III (10), DE Art Hist I (2), DE Eng IV: 2303 (5), DE Intro to Engnr (9), DE Intro to Theatre (4), Dual Enrollment 3 (2), Dual Enrollment 4 (2), Dual Enrollment 7 (2), Dual Enrollment 8 (2), Early Release 3rd (1), Early Release 4th (2), Early Release 7th (1), Early Release 8th (1), Enginr Dev+ LSU (8), English III H (8), English IV H (4), English IV H GT (4), Fine Art (H) (2), Health Edu (7), Indiv Proj H 1C (5), Int Comp Think H (6), PE I (1), Physics H (6), Robotics (LSU) (4), Tal Art (3), Tal Music (4), Theatre Tech H (1), World Hist H (1)

PE I: 102
Potential conflicts: Act Prep (1), Afr American H (3), Algebra II H (7), Algebra II H GT (3), Amer Hist H (2), AP Comp Sci Prin (3), AP English III (10th Grade) (6), AP European Hist (2), AP Human Geog (11), AP Psychology (1), AP Seminar (1), AP Statistics (2), AP Std Art&Draw (1), Biology H (90), Chemistry H (10), Civics H (90), DE Adv Math PCal (1), DE Intro to Engnr (2), DE Intro to Theatre (1), Dual Enrollment 4 (1), Enginr Dev+ LSU (8), English II H (78), English II H GT (12), English III H (4), Fin Literacy (84), Fine Art (H) (2), Geometry H (89), Health Edu (14), Music Apprec H (1), Physics H (2), Spanish I H (84), Spanish II H (85), Tal Art (6), Tal Music (2), Tal Theatre (3), Theatre Tech H (1), World Hist H (1)

Physics H: 85
Potential conflicts: Act Prep (17), Afr American H (20), Amer Hist H (34), AP Biology (2), AP Calculus AB (3), AP Comp Sci Prin (3), AP English III (11th/12th Grade) (10), AP Env Sci 1C (25), AP European Hist (32), AP Physics I Alg (1), AP Pre-Calculus (15), AP Psychology (19), AP Research (5), AP Statistics (19), AP Std Art&Draw (4), AP US History (7), Band (3), Chemistry H (1), Civics H (1), DE Adv Math PCal (17), DE Algebra III (25), DE Art Hist I (23), DE Eng IV: 2303 (9), DE Intro to Engnr (22), DE Intro to Theatre (20), Dual Enrollment 2 (1), Dual Enrollment 3 (16), Dual Enrollment 4 (26), Dual Enrollment 5 (1), Dual Enrollment 6 (2), Dual Enrollment 7 (22), Dual Enrollment 8 (27), Early Release 1st (1), Early Release 2nd (1), Early Release 3rd (5), Early Release 4th (2), Early Release 6th (2), Early Release 7th (5), Early Release 8th (5), Enginr Dev+ LSU (2), English IV H (34), English IV H GT (6), Fine Art (H) (11), Health Edu (1), Indiv Proj H 1C (8), Int Comp Think H (9), Music Apprec H (6), PE I (2), Robotics (LSU) (12), Tal Art (7), Tal Music (5), Tal Theatre (7), Theatre Tech H (6), World Hist H (6)

Robotics (LSU): 39
Potential conflicts: Act Prep (5), Afr American H (11), Amer Hist H (5), AP Calculus AB (14), AP Comp Sci Prin (1), AP English III (11th/12th Grade) (7), AP Env Sci 1C (11), AP European Hist (20), AP Physics I Alg (3), AP Physics II Alg (7), AP Pre-Calculus (1), AP Psychology (30), AP Statistics (7), AP Std Art&Draw (1), Band (1), Civics H (1), DE Adv Math PCal (11), DE Algebra III (4), DE Art Hist I (10), DE Intro to Engnr (22), DE Intro to Theatre (7), Dual Enrollment 3 (3), Dual Enrollment 4 (9), Dual Enrollment 6 (2), Dual Enrollment 7 (4), Dual Enrollment 8 (10), Early Release 4th (2), Early Release 7th (2), Early Release 8th (4), Enginr Dev+ LSU (1), Fine Art (H) (1), Indiv Proj H 1C (1), Int Comp Think H (8), Music Apprec H (4), Physics H (12), Tal Art (4), Tal Music (2), Tal Theatre (1), Theatre Tech H (5), World Hist H (6)

Spanish I H: 118
Potential conflicts: Biology H (118), Civics H (118), Dual Enrollment 4 (1), English II H (99), English II H GT (18), Fin Literacy (111), Geometry H (116), Health Edu (3), PE I (84), Spanish II H (118), Tal Art (18), Tal Music (5), Tal Theatre (13)

Spanish II H: 119
Potential conflicts: Biology H (119), Civics H (119), Dual Enrollment 4 (1), English II H (100), English II H GT (18), Fin Literacy (112), Fine Art (H) (1), Geometry H (117), Health Edu (3), PE I (85), Spanish I H (118), Tal Art (18), Tal Music (5), Tal Theatre (13)

Tal Art: 59
Potential conflicts: Act Prep (5), Afr American H (9), Algebra II H (14), Amer Hist H (10), AP Biology (1), AP Calculus AB (1), AP Comp Sci Prin (7), AP English III (10th Grade) (10), AP English III (11th/12th Grade) (8), AP Env Sci 1C (12), AP European Hist (4), AP Human Geog (17), AP Physics I Alg (6), AP Physics II Alg (9), AP Pre-Calculus (1), AP Psychology (10), AP Seminar (6), AP Statistics (1), AP Std Art&Draw (4), AP US History (2), Band (6), Biology H (19), Chemistry H (17), Civics H (19), DE Adv Math PCal (7), DE Algebra III (2), DE Eng IV: 2303 (3), DE Intro to Engnr (4), DE Intro to Theatre (1), Dual Enrollment 3 (2), Dual Enrollment 4 (10), Dual Enrollment 7 (4), Dual Enrollment 8 (7), Early Release 3rd (2), Early Release 4th (1), Early Release 8th (1), Enginr Dev+ LSU (16), English II H (14), English II H GT (5), English III H (15), English IV H (8), English IV H GT (2), Fin Literacy (25), Fine Art (H) (6), Geometry H (21), Health Edu (15), Int Comp Think H (2), Music Apprec H (3), PE I (5), Physics H (11), Robotics (LSU) (2), Spanish I H (18), Spanish II H (18), Theatre Tech H (1)

Tal Music: 14
Potential conflicts: Act Prep (1), Afr American H (2), Algebra II H (2), Algebra II H GT (1), Amer Hist H (3), AP Biology (1), AP English III (10th Grade) (1), AP English III (11th/12th Grade) (1), AP Env Sci 1C (1), AP European Hist (2), AP Human Geog (3), AP Pre-Calculus (3), AP Psychology (1), AP US History (1), Biology H (5), Chemistry H (3), Civics H (5), DE Adv Math PCal (1), DE Algebra III (2), DE Eng IV: 2303 (1), DE Intro to Engnr (1), DE Intro to Theatre (3), Enginr Dev+ LSU (3), English II H (5), English III H (1), English IV H (3), English IV H GT (1), Fin Literacy (5), Geometry H (5), Health Edu (2), Int Comp Think H (1), Music Apprec H (1), PE I (2), Physics H (3), Robotics (LSU) (1), Spanish I H (2), Spanish II H (5)

Tal Theatre: 31
Potential conflicts: Act Prep (13), Afr American H (5), Algebra II H (3), Algebra II H GT (1), Amer Hist H (4), AP Biology (2), AP Calculus AB (2), AP Comp Sci Prin (1), AP English III (10th Grade) (2), AP English III (11th/12th Grade) (2), AP Env Sci 1C (6), AP European Hist (8), AP Human Geog (18), AP Physics I Alg (3), AP Psychology (5), AP Seminar (1), AP Statistics (1), Biology H (15), Chemistry H (4), Civics H (19), DE Adv Math PCal (2), DE Algebra III (5), DE Art Hist I (2), DE Eng IV: 2303 (2), DE Intro to Engnr (2), DE Intro to Theatre (1), Dual Enrollment 3 (2), Dual Enrollment 4 (3), Dual Enrollment 7 (2), Dual Enrollment 8 (5), Early Release 3rd (2), Early Release 4th (1), Early Release 7th (1), Early Release 8th (2), Enginr Dev+ LSU (1), English II H (10), English II H GT (4), English IV H (4), English IV H GT (1), Fin Literacy (15), Fine Art (H) (2), Geometry H (14), Health Edu (4), Int Comp Think H (1), PE I (3), Physics H (7), Robotics (LSU) (1), Spanish I H (13), Spanish II H (13), Theatre Tech H (1), World Hist H (3)

Theatre Tech H: 22
Potential conflicts: Act Prep (2), Afr American H (10), Algebra II H (10), Algebra II H GT (1), Amer Hist H (4), AP Calculus AB (3), AP Comp Sci Prin (5), AP English III (10th Grade) (3), AP English III (11th/12th Grade) (2), AP Env Sci 1C (5), AP European Hist (6), AP Human Geog (11), AP Physics I Alg (1), AP Pre-Calculus (3), AP Psychology (6), AP Seminar (1), AP Statistics (3), AP Std Art&Draw (2), Band (2), Chemistry H (11), DE Algebra III (2), DE Art Hist I (3), DE Intro to Engnr (4), DE Intro to Theatre (1), Dual Enrollment 4 (1), Dual Enrollment 8 (1), Enginr Dev+ LSU (9), English III H (9), English IV H (3), English IV H GT (1), Fin Literacy (1), Fine Art (H) (2), Health Edu (10), Indiv Proj H 1C (1), Int Comp Think H (1), Music Apprec H (1), PE I (1), Physics H (6), Robotics (LSU) (5), Tal Art (1), Tal Theatre (1), World Hist H (1)

World Hist H: 38
Potential conflicts: Act Prep (6), Afr American H (9), Algebra II H (15), Algebra II H GT (4), Amer Hist H (5), AP Calculus AB (3), AP Comp Sci Prin (11), AP English III (10th Grade) (2), AP English III (11th/12th Grade) (12), AP Env Sci 1C (5), AP European Hist (4), AP Human Geog (19), AP Physics I Alg (4), AP Physics II Alg (6), AP Pre-Calculus (3), AP Psychology (6), AP Seminar (4), AP Statistics (10), AP US History (1), Band (1), Chemistry H (20), DE Adv Math PCal (4), DE Algebra III (5), DE Art Hist I (2), DE Eng IV: 2303 (3), DE Intro to Engnr (7), DE Intro to Theatre (8), Dual Enrollment 3 (1), Dual Enrollment 4 (4), Dual Enrollment 7 (1), Dual Enrollment 8 (4), Early Release 3rd (2), Early Release 5th (1), Early Release 6th (2), Early Release 7th (2), Early Release 8th (1), Enginr Dev+ LSU (17), English III H (7), English IV H (7), English IV H GT (1), Fine Art (H) (1), Health Edu (19), Int Comp Think H (3), Music Apprec H (1), PE I (1), Physics H (6), Robotics (LSU) (6), Tal Theatre (3), Theatre Tech H (1)
"""

# ============================================================
# 1) GLOBAL CONFIG (HARD RULES)
# ============================================================

PERIODS = ["F1", "F2", "F3", "F4", "S1", "S2", "S3", "S4"]

# Pseudo-courses: not teachable; only affect conflict energy
FIXED_PERIOD_MAP = {
    "Dual Enrollment 1": 0, "Early Release 1st": 0,
    "Dual Enrollment 2": 1, "Early Release 2nd": 1,
    "Dual Enrollment 3": 2, "Early Release 3rd": 2,
    "Dual Enrollment 4": 3, "Early Release 4th": 3,
    "Dual Enrollment 5": 4, "Early Release 5th": 4,
    "Dual Enrollment 6": 5, "Early Release 6th": 5,
    "Dual Enrollment 7": 6, "Early Release 7th": 6,
    "Dual Enrollment 8": 7, "Early Release 8th": 7,
}

ALIASES = {"AP Biology": "AP Biology (full year)"}

FULL_YEAR_COURSES = {
    "AP Calculus AB",
    "AP Biology (full year)",
    "AP US History",
}

# Term-only hard rules
FALL_ONLY_COURSES = {
    "AP Research",
    "Indiv Proj H 1C",
    "AP Std Art&Draw",  # paired fall→spring (same teacher)
}
SPRING_ONLY_COURSES = {
    "DE Adv Math PCal",
    "AP Seminar",
}

# Hard: "AP Studio Art in fall implies Tal Art in spring by same teacher"
ART_FALL = "AP Std Art&Draw"
ART_SPRING = "Tal Art"
PAIRED_NEXT_SEMESTER = {ART_FALL: ART_SPRING}

# Hard: coaches off 4th block (index 3) both semesters
COACH_TEACHERS = {"KenB", "Brad", "DerB", "RebK"}

# Hard: fellows distinct planning each semester
FELLOWS = {"AliH", "RebK", "LauM"}

# Hard: mentorship: GusP off overlaps RebK in ONE sem and MarA in the OTHER
MENTORSHIP = ("GusP", "RebK", "MarA")

# Hard: required courses list (explicit)
REQUIRED_COURSES = {
    # 9th
    "Geometry H", "Biology H", "Civics H",
    "English II H", "English II H GT",
    "Spanish I H", "Spanish II H",
    "Fin Literacy",
    # 10th
    "Algebra II H", "Algebra II H GT",
    "Chemistry H",
    "AP Human Geog",
    "Enginr Dev+ LSU",
    "Health Edu",
    "English III H", "AP English III (10th Grade)",
    # 11th
    "English IV H", "English IV H GT", "DE Eng IV: 2303",
    "Amer Hist H", "AP US History",
}

# Hard: fixed full-year placements
FULL_YEAR_FIXED = {
    "AP Calculus AB": "GusP",
    "AP Biology (full year)": "MarE",
    "AP US History": "LauM",
}

# ============================================================
# 2) DATA MODELS
# ============================================================

def normalize_name(name: str) -> str:
    name = re.sub(r"\s+", " ", name.strip())
    return ALIASES.get(name, name)


@dataclass
class Course:
    name: str
    demand: int
    co_requests: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    centrality: float = 0.0

    capacity: int = 33
    is_full_year: bool = False
    term: str = "EITHER"  # "F", "S", "EITHER", "FY"
    is_required: bool = False
    paired_next: Optional[str] = None

    def __post_init__(self) -> None:
        # capacities
        if self.name in {"PE I", "Health Edu"}:
            self.capacity = 40
        if self.name == "Int Comp Think H":
            self.capacity = 20

        # required (exact list)
        self.is_required = (self.name in REQUIRED_COURSES)

        # full-year
        if self.name in FULL_YEAR_COURSES:
            self.is_full_year = True
            self.term = "FY"
            return

        # term-only rules
        if self.name in SPRING_ONLY_COURSES:
            self.term = "S"
        elif self.name in FALL_ONLY_COURSES:
            self.term = "F"
        elif self.name.startswith("DE Algebra"):
            self.term = "F"
        else:
            self.term = "EITHER"

        # pairing rule
        if self.name in PAIRED_NEXT_SEMESTER:
            self.paired_next = PAIRED_NEXT_SEMESTER[self.name]
            self.term = "F"

    def allowed_in_semester(self, sem: str) -> bool:
        if self.term == "FY":
            return sem in ("F", "S")
        if self.term == "F":
            return sem == "F"
        if self.term == "S":
            return sem == "S"
        return sem in ("F", "S")


@dataclass
class Teacher:
    name: str
    allowed_courses: Set[str]
    load_fall: int
    load_spring: int
    coach: bool = False
    ms_paired_off_min: int = 0  # min blocks off in BOTH semesters

    def can_teach(self, course_name: str) -> bool:
        return course_name in self.allowed_courses


# ============================================================
# 3) PARSING REQUEST MATRIX
# ============================================================

def parse_request_matrix(text_data: str) -> Dict[str, Course]:
    courses: Dict[str, Course] = {}

    header_re = re.compile(r"^(.*?):\s*(\d+)\s*$")
    conflict_re = re.compile(r"\s*([^,]+?)\s*\((\d+)\)\s*(?:,|$)")

    current: Optional[Course] = None
    lines = text_data.strip().splitlines()

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        if line.startswith("Potential conflicts:"):
            if current is None:
                continue
            rest = line.replace("Potential conflicts:", "").strip()
            for m in conflict_re.finditer(rest):
                other = normalize_name(m.group(1))
                cnt = int(m.group(2))
                current.co_requests[other] += cnt
            continue

        m = header_re.match(line)
        if not m:
            continue

        name = normalize_name(m.group(1))
        demand = int(m.group(2))

        # skip pseudo-courses as teachable courses
        if name in FIXED_PERIOD_MAP:
            current = None
            continue

        if name not in courses:
            courses[name] = Course(name=name, demand=demand)
        else:
            courses[name].demand = demand

        current = courses[name]

    for c in courses.values():
        c.centrality = float(sum(c.co_requests.values()))

    return courses


# ============================================================
# 4) TEACHERS + ELIGIBILITY
# ============================================================

def init_teachers() -> Dict[str, Teacher]:
    def S(*xs: str) -> Set[str]:
        return {normalize_name(x) for x in xs}

    teachers = [
        Teacher("MarA", S("DE Adv Math PCal", "DE Algebra III", "Algebra II H", "AP Pre-Calculus", "AP Statistics"), 3, 3),
        Teacher("DerB", S("Health Edu"), 1, 1),
        Teacher("KenB", S("PE I"), 1, 1),
        Teacher("GigB", S("English III H", "AP English III (10th Grade)"), 2, 2),
        Teacher("ChrB", S("AP Std Art&Draw", "Fine Art (H)", "Tal Art"), 3, 3),
        Teacher("RobC", S("DE Intro to Theatre", "Tal Theatre"), 3, 3),
        Teacher("ThoC", S("English II H", "English II H GT", "AP English III (11th/12th Grade)"), 3, 3),
        Teacher("CheD", S("Act Prep", "AP Seminar", "AP Research", "Indiv Proj H 1C", "English III H", "AP English III (10th Grade)"), 3, 3),
        Teacher("MarE", S("AP Biology (full year)", "Biology H"), 3, 3),
        Teacher("AliH", S("AP Human Geog", "Civics H", "World Hist H"), 3, 3),
        Teacher("EmH", S("Enginr Dev+ LSU", "DE Intro to Engnr"), 3, 3),
        Teacher("EsK", S("Geometry H"), 2, 2),
        Teacher("RebK", S("AP Env Sci 1C", "Robotics (LSU)", "DE Intro to Engnr", "AP Physics II Alg", "AP Physics I Alg", "Physics H"), 3, 3),
        Teacher("AveL", S("AP Physics II Alg", "AP Physics I Alg", "Physics H"), 3, 3),
        Teacher("DanL", S("Int Comp Think H", "Algebra II H", "Algebra II H GT"), 3, 3),
        Teacher("LauM", S("AP European Hist", "AP Psychology", "AP US History", "Amer Hist H"), 3, 3),
        Teacher("MicM", S("DE Art Hist I", "Fine Art (H)", "Tal Art"), 3, 3),
        Teacher("OdaM", S("Afr American H", "Civics H"), 3, 3),
        Teacher("DomN", S("Spanish I H", "Spanish II H"), 3, 3),
        Teacher("MelN", S("Amer Hist H", "World Hist H"), 2, 2),
        Teacher("RobP", S("AP Comp Sci Prin"), 1, 1),
        Teacher("GusP", S("AP Calculus AB", "AP Pre-Calculus", "Algebra II H", "Algebra II H GT", "AP Statistics"), 3, 3),
        Teacher("ValW", S("English IV H", "English IV H GT", "DE Eng IV: 2303"), 3, 3),
        Teacher("Brad", S("PE I", "Health Edu"), 3, 3),
        Teacher("MicW", S("Chemistry H", "Spanish I H", "Spanish II H"), 3, 3),
        Teacher("MelS", S("Fin Literacy"), 3, 3),
        Teacher("TayL", S("Band"), 1, 1),
        Teacher("WilL", S("Tal Music", "Music Apprec H"), 3, 3),
        Teacher("NicL", S("Theatre Tech H"), 0, 1),
    ]

    for t in teachers:
        t.coach = (t.name in COACH_TEACHERS)

        # MS paired-off constraints (hard)
        if t.name in {"GigB", "MelN", "EsK"}:
            t.ms_paired_off_min = 1
        if t.name in {"DerB", "KenB", "RobP"}:
            t.ms_paired_off_min = 2
        if t.name == "TayL":
            t.ms_paired_off_min = 0

    return {t.name: t for t in teachers}


def build_eligibility(courses: Dict[str, Course], teachers: Dict[str, Teacher]) -> Dict[str, List[str]]:
    eligible = {c: [] for c in courses}
    for t in teachers.values():
        for cname in t.allowed_courses:
            if cname in courses:
                eligible[cname].append(t.name)
    return eligible


# ============================================================
# 5) HARD FLOORS (minimum sections) + TARGETS (ceil(demand/cap))
# ============================================================

def compute_hard_floors(courses: Dict[str, Course]) -> Tuple[Dict[str, int], Dict[str, int]]:
    floors: Dict[str, int] = {}
    targets: Dict[str, int] = {}

    for cname, c in courses.items():
        if c.demand <= 0:
            floors[cname] = 0
            targets[cname] = 0
            continue

        targets[cname] = max(1, math.ceil(c.demand / c.capacity))

        if c.is_required:
            floors[cname] = targets[cname]
        else:
            floors[cname] = 1

        if c.is_full_year:
            floors[cname] = max(floors[cname], 1)

    return floors, targets


# ============================================================
# 6) DINIC MAX-FLOW (floor placement feasibility)
# ============================================================

class Edge:
    __slots__ = ("to", "rev", "cap", "original")
    def __init__(self, to: int, rev: int, cap: int):
        self.to = to
        self.rev = rev
        self.cap = cap
        self.original = cap

class Dinic:
    def __init__(self, n: int):
        self.n = n
        self.g: List[List[Edge]] = [[] for _ in range(n)]
        self.level = [0] * n
        self.it = [0] * n

    def add_edge(self, fr: int, to: int, cap: int):
        fwd = Edge(to, len(self.g[to]), cap)
        rev = Edge(fr, len(self.g[fr]), 0)
        self.g[fr].append(fwd)
        self.g[to].append(rev)

    def bfs(self, s: int, t: int) -> bool:
        self.level = [-1] * self.n
        q = [s]
        self.level[s] = 0
        for v in q:
            for e in self.g[v]:
                if e.cap > 0 and self.level[e.to] < 0:
                    self.level[e.to] = self.level[v] + 1
                    q.append(e.to)
        return self.level[t] >= 0

    def dfs(self, v: int, t: int, f: int) -> int:
        if v == t:
            return f
        for i in range(self.it[v], len(self.g[v])):
            self.it[v] = i
            e = self.g[v][i]
            if e.cap <= 0:
                continue
            if self.level[e.to] != self.level[v] + 1:
                continue
            pushed = self.dfs(e.to, t, min(f, e.cap))
            if pushed > 0:
                e.cap -= pushed
                self.g[e.to][e.rev].cap += pushed
                return pushed
        return 0

    def max_flow(self, s: int, t: int) -> int:
        flow = 0
        INF = 10**9
        while self.bfs(s, t):
            self.it = [0] * self.n
            while True:
                pushed = self.dfs(s, t, INF)
                if pushed == 0:
                    break
                flow += pushed
        return flow


# ============================================================
# 7) STEP 1a — SECTION COUNTS + TEACHER ASSIGNMENT
# ============================================================

def allocate_counts_and_teachers(
    courses: Dict[str, Course],
    teachers: Dict[str, Teacher],
    floors: Dict[str, int],
    targets: Dict[str, int],
    seed: int,
) -> Tuple[Optional[Dict[str, Dict[str, List[str]]]], Optional[Counter], str]:
    """
    Hard-valid construction:
      (A) Place fixed full-year sections
      (B) Place the Studio Art -> Tal Art bundle
      (C) Satisfy remaining floor sections using max-flow (teacher-sem capacities)
      (D) Fill remaining teacher loads using a Dirichlet-multinomial sampler (soft)
    """

    rng = np.random.default_rng(seed)
    eligible = build_eligibility(courses, teachers)

    # Feasibility: every demanded course must have an eligible teacher
    for cname, c in courses.items():
        if c.demand > 0 and len(eligible.get(cname, [])) == 0:
            return None, None, f"INFEASIBLE: No eligible teacher for demanded course '{cname}'."

    # Remaining teacher-sem slots
    rem = {t.name: {"F": t.load_fall, "S": t.load_spring} for t in teachers.values()}
    teach = {tname: {"F": [], "S": []} for tname in teachers}
    section_counts: Counter = Counter()

    # (A) Fixed full-year placements
    for cname, tname in FULL_YEAR_FIXED.items():
        if cname not in courses or courses[cname].demand <= 0:
            continue
        if not teachers[tname].can_teach(cname):
            return None, None, f"INFEASIBLE: fixed FY '{cname}' but {tname} not eligible."
        if rem[tname]["F"] <= 0 or rem[tname]["S"] <= 0:
            return None, None, f"INFEASIBLE: {tname} has no capacity for fixed FY '{cname}'."

        teach[tname]["F"].append(cname)
        teach[tname]["S"].append(cname)
        rem[tname]["F"] -= 1
        rem[tname]["S"] -= 1
        section_counts[cname] += 1

    # (B) Paired AP Studio Art -> Tal Art (same teacher)
    if ART_FALL in courses and courses[ART_FALL].demand > 0:
        spring_c = courses[ART_FALL].paired_next
        if spring_c not in courses:
            return None, None, f"INFEASIBLE: paired partner '{spring_c}' not found in courses."

        candidates = [
            t for t in eligible.get(ART_FALL, [])
            if teachers[t].can_teach(spring_c) and rem[t]["F"] > 0 and rem[t]["S"] > 0
        ]
        if not candidates:
            return None, None, "INFEASIBLE: no teacher can host the AP Std Art (fall) + Tal Art (spring) pair."

        t_pick = max(candidates, key=lambda t: rem[t]["F"] + rem[t]["S"])
        teach[t_pick]["F"].append(ART_FALL)
        teach[t_pick]["S"].append(spring_c)
        rem[t_pick]["F"] -= 1
        rem[t_pick]["S"] -= 1
        section_counts[ART_FALL] += 1
        section_counts[spring_c] += 1

    # (C) Remaining floor needs
    need = {c: max(0, floors[c] - section_counts[c]) for c in courses}

    # Model limitation: we only force exactly 1 Studio Art bundle section
    for cname, c in courses.items():
        if c.paired_next is not None and need[cname] > 0:
            return None, None, (
                f"INFEASIBLE (model limitation): '{cname}' needs >1 section under floors, "
                f"but pairing logic currently only forces 1. Extend pairing if needed."
            )

    # (C1) If additional full-year sections are needed, allocate greedily (rare)
    for cname, k in list(need.items()):
        if k <= 0 or not courses[cname].is_full_year:
            continue
        for _ in range(k):
            candidates = [t for t in eligible.get(cname, []) if rem[t]["F"] > 0 and rem[t]["S"] > 0]
            if not candidates:
                return None, None, f"INFEASIBLE: need more full-year sections for '{cname}', but no teacher has 2-sem capacity."
            t_pick = max(candidates, key=lambda t: min(rem[t]["F"], rem[t]["S"]))
            teach[t_pick]["F"].append(cname)
            teach[t_pick]["S"].append(cname)
            rem[t_pick]["F"] -= 1
            rem[t_pick]["S"] -= 1
            section_counts[cname] += 1
        need[cname] = 0

    # (C2) One-term floors via max-flow
    course_nodes = [c for c in courses if need[c] > 0 and not courses[c].is_full_year]
    total_need = sum(need[c] for c in course_nodes)

    teacher_sem_nodes = [(tname, sem) for tname in teachers for sem in ("F", "S")]

    node_id: Dict[Tuple[Any, ...], int] = {}
    SRC = 0
    idx = 1

    for cname in course_nodes:
        node_id[("course", cname)] = idx
        idx += 1

    for tname, sem in teacher_sem_nodes:
        node_id[("ts", tname, sem)] = idx
        idx += 1

    SNK = idx
    dinic = Dinic(SNK + 1)

    for cname in course_nodes:
        dinic.add_edge(SRC, node_id[("course", cname)], need[cname])

    for cname in course_nodes:
        c = courses[cname]
        cnode = node_id[("course", cname)]
        for tname in eligible.get(cname, []):
            for sem in ("F", "S"):
                if not c.allowed_in_semester(sem):
                    continue
                if rem[tname][sem] <= 0:
                    continue
                tsnode = node_id[("ts", tname, sem)]
                dinic.add_edge(cnode, tsnode, need[cname])  # big enough

    for tname, sem in teacher_sem_nodes:
        cap = rem[tname][sem]
        if cap > 0:
            dinic.add_edge(node_id[("ts", tname, sem)], SNK, cap)

    flow = dinic.max_flow(SRC, SNK)
    if flow != total_need:
        return None, None, f"INFEASIBLE: could not place {flow}/{total_need} hard-floor sections (eligibility/capacity mismatch)."

    rev_node = {nid: key for key, nid in node_id.items()}

    for cname in course_nodes:
        cnode = node_id[("course", cname)]
        for e in dinic.g[cnode]:
            used = e.original - e.cap
            if used <= 0:
                continue
            key = rev_node.get(e.to)
            if not key or key[0] != "ts":
                continue
            tname, sem = key[1], key[2]
            teach[tname][sem].extend([cname] * used)
            rem[tname][sem] -= used
            section_counts[cname] += used

    # (D) Fill remaining teacher load with Dirichlet-multinomial
    concentration = 18.0
    kappa = 0.35
    req_boost = 1.15
    exclusive_boost = 1.25

    for tname, t in teachers.items():
        for sem in ("F", "S"):
            K = rem[tname][sem]
            if K <= 0:
                continue

            candidates = []
            weights = []

            for cname in t.allowed_courses:
                if cname not in courses:
                    continue
                c = courses[cname]
                if c.demand <= 0:
                    continue
                if c.is_full_year:
                    continue
                if cname in PAIRED_NEXT_SEMESTER:
                    continue
                if not c.allowed_in_semester(sem):
                    continue

                target = targets[cname]
                shortfall_sections = max(0, target - section_counts[cname])

                w = (shortfall_sections + 0.25) * ((c.centrality + 1.0) ** kappa)
                if c.is_required:
                    w *= req_boost
                if len(eligible.get(cname, [])) == 1:
                    w *= exclusive_boost

                candidates.append(cname)
                weights.append(w)

            if not candidates:
                return None, None, f"INFEASIBLE: {tname} has remaining load in {sem} but no eligible course candidates."

            w = np.array(weights, dtype=float)
            w = np.maximum(w, 1e-9)
            alpha = w / w.sum() * concentration
            alpha = np.maximum(alpha, 1e-6)

            p = rng.dirichlet(alpha)
            counts = rng.multinomial(K, p)

            for cname, cnt in zip(candidates, counts):
                if cnt <= 0:
                    continue
                teach[tname][sem].extend([cname] * int(cnt))
                section_counts[cname] += int(cnt)
                rem[tname][sem] -= int(cnt)

    for tname in teachers:
        if rem[tname]["F"] != 0 or rem[tname]["S"] != 0:
            return None, None, f"BUG: remaining slots not zero for {tname}: {rem[tname]}"

    for cname in courses:
        if courses[cname].demand > 0 and section_counts[cname] < floors[cname]:
            return None, None, f"BUG: floor violated for {cname}."

    return teach, section_counts, "OK"


# ============================================================
# 8) STEP 1b — PERIOD PLACEMENT (HARD)
# ============================================================

def count_off_pairs(row8: List[Optional[str]]) -> int:
    return sum(1 for b in range(4) if row8[b] is None and row8[b + 4] is None)

def planning_block_for_three_load(row8: List[Optional[str]], sem: str) -> int:
    base = 0 if sem == "F" else 4
    offs = [b for b in range(4) if row8[base + b] is None]
    if len(offs) != 1:
        raise ValueError("Expected exactly 1 off block for a 3-load teacher.")
    return offs[0]

def validate_teacher_row(
    t: Teacher,
    row8: List[Optional[str]],
    courses: Dict[str, Course],
    forced_off: Dict[str, Dict[str, Set[int]]],
) -> bool:
    if len(row8) != 8:
        return False

    f_off = forced_off.get(t.name, {}).get("F", set())
    s_off = forced_off.get(t.name, {}).get("S", set())
    for b in f_off:
        if row8[b] is not None:
            return False
    for b in s_off:
        if row8[b + 4] is not None:
            return False

    for b in range(4):
        f = row8[b]
        s = row8[b + 4]

        if f is not None:
            if not t.can_teach(f):
                return False
            cf = courses[f]
            if cf.term == "S":
                return False
            if cf.is_full_year and s != f:
                return False

        if s is not None:
            if not t.can_teach(s):
                return False
            cs = courses[s]
            if cs.term == "F":
                return False
            if cs.is_full_year and f != s:
                return False

    fall_load = sum(1 for i in range(4) if row8[i] is not None)
    spring_load = sum(1 for i in range(4, 8) if row8[i] is not None)
    if fall_load != t.load_fall:
        return False
    if spring_load != t.load_spring:
        return False

    if t.ms_paired_off_min > 0 and count_off_pairs(row8) < t.ms_paired_off_min:
        return False

    return True

def build_teacher_row(
    t: Teacher,
    fall_courses: List[str],
    spring_courses: List[str],
    courses: Dict[str, Course],
    forced_off: Dict[str, Dict[str, Set[int]]],
    seed: int,
) -> Optional[List[Optional[str]]]:
    rng = random.Random(seed)

    fall = [None] * 4
    spring = [None] * 4

    f_off = set(forced_off.get(t.name, {}).get("F", set()))
    s_off = set(forced_off.get(t.name, {}).get("S", set()))

    if t.coach:
        f_off |= {3}
        s_off |= {3}

    fy_f = Counter([c for c in fall_courses if courses[c].is_full_year])
    fy_s = Counter([c for c in spring_courses if courses[c].is_full_year])
    if fy_f != fy_s:
        return None

    linked = []
    for cname, k in fy_f.items():
        linked.extend([("B", cname, cname)] * k)

    fall_only = [c for c in fall_courses if not courses[c].is_full_year]
    spring_only = [c for c in spring_courses if not courses[c].is_full_year]

    items = linked + [("F", c, None) for c in fall_only] + [("S", None, c) for c in spring_only]

    blocks = [0, 1, 2, 3]

    def ok_off_pairs() -> bool:
        return count_off_pairs(fall + spring) >= t.ms_paired_off_min

    def can_place(kind: str, b: int) -> bool:
        if kind == "B":
            return (b not in f_off) and (b not in s_off) and (fall[b] is None) and (spring[b] is None)
        if kind == "F":
            return (b not in f_off) and (fall[b] is None)
        return (b not in s_off) and (spring[b] is None)

    def backtrack(i: int) -> bool:
        if i == len(items):
            return ok_off_pairs()

        kind, fc, sc = items[i]
        rng.shuffle(blocks)

        for b in blocks:
            if not can_place(kind, b):
                continue
            if kind == "B":
                fall[b] = fc
                spring[b] = sc
                if backtrack(i + 1):
                    return True
                fall[b] = None
                spring[b] = None
            elif kind == "F":
                fall[b] = fc
                if backtrack(i + 1):
                    return True
                fall[b] = None
            else:
                spring[b] = sc
                if backtrack(i + 1):
                    return True
                spring[b] = None
        return False

    if not backtrack(0):
        return None

    row8 = fall + spring
    if not validate_teacher_row(t, row8, courses, forced_off):
        return None
    return row8

def enumerate_forced_off_plans(seed: int) -> Iterable[Dict[str, Dict[str, Set[int]]]]:
    rng = random.Random(seed)
    gus, reb, mar = MENTORSHIP

    reb_plan_f = 3
    reb_plan_s = 3

    fall_pairs = [(a, b) for a in [0, 1, 2] for b in [0, 1, 2] if a != b]
    spring_pairs = [(a, b) for a in [0, 1, 2] for b in [0, 1, 2] if a != b]
    rng.shuffle(fall_pairs)
    rng.shuffle(spring_pairs)

    patterns = [0, 1]  # 0: Gus matches Reb in Fall; 1: Gus matches Reb in Spring
    rng.shuffle(patterns)

    for pattern in patterns:
        for (ali_f, lau_f) in fall_pairs:
            for (ali_s, lau_s) in spring_pairs:
                mar_f_choices = [0, 1, 2, 3]
                mar_s_choices = [0, 1, 2, 3]
                rng.shuffle(mar_f_choices)
                rng.shuffle(mar_s_choices)

                for mar_f in mar_f_choices:
                    for mar_s in mar_s_choices:
                        forced = defaultdict(lambda: {"F": set(), "S": set()})

                        # Coaches forced off block 3
                        for t in ("RebK", "KenB", "Brad"):
                            forced[t]["F"].add(3)
                            forced[t]["S"].add(3)

                        # Fellows planning (AliH + LauM distinct each semester, not block 3)
                        forced["AliH"]["F"].add(ali_f)
                        forced["LauM"]["F"].add(lau_f)
                        forced["AliH"]["S"].add(ali_s)
                        forced["LauM"]["S"].add(lau_s)

                        # MarA planning
                        forced["MarA"]["F"].add(mar_f)
                        forced["MarA"]["S"].add(mar_s)

                        # Mentorship constraint
                        if pattern == 0:
                            forced["GusP"]["F"].add(reb_plan_f)   # 3
                            forced["GusP"]["S"].add(mar_s)
                        else:
                            forced["GusP"]["F"].add(mar_f)
                            forced["GusP"]["S"].add(reb_plan_s)  # 3

                        yield forced

def compute_offered_sections(schedule: Dict[str, List[Optional[str]]], courses: Dict[str, Course]) -> Counter:
    offered = Counter()
    for row in schedule.values():
        for p, cname in enumerate(row):
            if cname is None:
                continue
            c = courses[cname]
            if c.is_full_year:
                if p < 4:
                    offered[cname] += 1
            else:
                offered[cname] += 1
    return offered

def validate_hard(
    schedule: Dict[str, List[Optional[str]]],
    teachers: Dict[str, Teacher],
    courses: Dict[str, Course],
    floors: Dict[str, int],
    forced_off: Dict[str, Dict[str, Set[int]]],
) -> bool:
    for tname, t in teachers.items():
        row = schedule.get(tname)
        if row is None:
            return False
        if not validate_teacher_row(t, row, courses, forced_off):
            return False

    # Fellows: distinct planning each semester (3-load => exactly 1 off)
    try:
        fall_plans = [planning_block_for_three_load(schedule[n], "F") for n in FELLOWS]
        spring_plans = [planning_block_for_three_load(schedule[n], "S") for n in FELLOWS]
    except ValueError:
        return False
    if len(set(fall_plans)) != len(FELLOWS):
        return False
    if len(set(spring_plans)) != len(FELLOWS):
        return False

    # Mentorship
    gus, reb, mar = MENTORSHIP
    try:
        gf, gs = planning_block_for_three_load(schedule[gus], "F"), planning_block_for_three_load(schedule[gus], "S")
        rf, rs = planning_block_for_three_load(schedule[reb], "F"), planning_block_for_three_load(schedule[reb], "S")
        mf, ms = planning_block_for_three_load(schedule[mar], "F"), planning_block_for_three_load(schedule[mar], "S")
    except ValueError:
        return False
    ok = (gf == rf and gs == ms) or (gf == mf and gs == rs)
    if not ok:
        return False

    # Floors satisfied
    offered = compute_offered_sections(schedule, courses)
    for cname, c in courses.items():
        if c.demand <= 0:
            continue
        if offered.get(cname, 0) < floors[cname]:
            return False

    return True

def build_initial_schedule(
    teach: Dict[str, Dict[str, List[str]]],
    teachers: Dict[str, Teacher],
    courses: Dict[str, Course],
    floors: Dict[str, int],
    seed: int,
) -> Tuple[Optional[Dict[str, List[Optional[str]]]], Optional[Dict[str, Dict[str, Set[int]]]], str]:
    rnd = random.Random(seed)
    for tname in teach:
        rnd.shuffle(teach[tname]["F"])
        rnd.shuffle(teach[tname]["S"])

    for forced_off in enumerate_forced_off_plans(seed):
        schedule: Dict[str, List[Optional[str]]] = {}
        ok = True

        order = list(dict.fromkeys(["RebK", "AliH", "LauM", "GusP", "MarA"] + list(teachers.keys())))

        for i, tname in enumerate(order):
            t = teachers[tname]
            row = build_teacher_row(
                t=t,
                fall_courses=teach[tname]["F"],
                spring_courses=teach[tname]["S"],
                courses=courses,
                forced_off=forced_off,
                seed=seed + i * 10007,
            )
            if row is None:
                ok = False
                break
            schedule[tname] = row

        if not ok:
            continue

        if validate_hard(schedule, teachers, courses, floors, forced_off):
            return schedule, forced_off, "OK"

    return None, None, "INFEASIBLE: could not place periods under planning/coach/MS constraints."


# ============================================================
# 9) ENERGY MODEL (conflict graph)
# ============================================================

def build_pair_weights(courses: Dict[str, Course], fixed_conflicts: Dict[str, int]) -> Dict[Tuple[str, str], float]:
    valid = set(courses.keys()) | set(fixed_conflicts.keys())
    w = defaultdict(float)
    for a, ca in courses.items():
        for b, cnt in ca.co_requests.items():
            b = normalize_name(b)
            if b not in valid or b == a:
                continue
            key = tuple(sorted((a, b)))
            w[key] += float(cnt)
    return w

def build_energy_context(courses: Dict[str, Course], fixed_conflicts: Dict[str, int]) -> dict:
    pair_w = build_pair_weights(courses, fixed_conflicts)

    fixed_used = set()
    for (a, b) in pair_w.keys():
        if a in fixed_conflicts:
            fixed_used.add(a)
        if b in fixed_conflicts:
            fixed_used.add(b)

    course_names = sorted(courses.keys())
    fixed_names = sorted(fixed_used)
    all_names = course_names + fixed_names
    idx = {name: i for i, name in enumerate(all_names)}

    is_full_year = np.array([(courses[n].is_full_year if n in courses else False) for n in all_names], dtype=bool)

    edges_by_mode = {0: [], 1: [], 2: [], 3: []}
    for (a, b), wt in pair_w.items():
        i = idx[a]
        j = idx[b]
        ai = bool(is_full_year[i])
        bj = bool(is_full_year[j])

        if (not ai) and (not bj):
            edges_by_mode[0].append((i, j, wt))
        elif ai and bj:
            edges_by_mode[1].append((i, j, wt))
        elif ai and (not bj):
            edges_by_mode[2].append((i, j, wt))
        else:
            edges_by_mode[3].append((i, j, wt))

    packed = {}
    for mode in (0, 1, 2, 3):
        if not edges_by_mode[mode]:
            packed[mode] = (np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float64))
            continue
        ii, jj, ww = zip(*edges_by_mode[mode])
        packed[mode] = (np.array(ii, dtype=np.int32), np.array(jj, dtype=np.int32), np.array(ww, dtype=np.float64))

    fixed_rows = {idx[name]: fixed_conflicts[name] for name in fixed_names}

    return {
        "all_names": all_names,
        "idx": idx,
        "is_full_year": is_full_year,
        "edges": packed,
        "fixed_rows": fixed_rows,
    }

def calculate_energy(schedule: Dict[str, List[Optional[str]]], energy_ctx: dict) -> float:
    idx = energy_ctx["idx"]
    is_full_year = energy_ctx["is_full_year"]
    edges = energy_ctx["edges"]
    fixed_rows = energy_ctx["fixed_rows"]
    n = len(energy_ctx["all_names"])

    counts = np.zeros((n, 8), dtype=np.int16)

    for row in schedule.values():
        for p, cname in enumerate(row):
            if cname is None:
                continue
            counts[idx[cname], p] += 1

    for i, p in fixed_rows.items():
        counts[i, p] = 1

    s = np.where(is_full_year, counts[:, 0:4].sum(axis=1), counts.sum(axis=1)).astype(np.float64)
    s_safe = np.where(s > 0, s, 1.0)

    p8 = counts / s_safe[:, None]
    p4 = counts[:, 0:4] / s_safe[:, None]
    p4_any = (counts[:, 0:4] + counts[:, 4:8]) / s_safe[:, None]

    total = 0.0

    ii, jj, ww = edges[0]
    if ww.size:
        overlaps = np.sum(p8[ii] * p8[jj], axis=1)
        total += float(np.sum(ww * overlaps))

    ii, jj, ww = edges[1]
    if ww.size:
        overlaps = np.sum(p4[ii] * p4[jj], axis=1)
        total += float(np.sum(ww * overlaps))

    ii, jj, ww = edges[2]
    if ww.size:
        overlaps = np.sum(p4[ii] * p4_any[jj], axis=1)
        total += float(np.sum(ww * overlaps))

    ii, jj, ww = edges[3]
    if ww.size:
        overlaps = np.sum(p4_any[ii] * p4[jj], axis=1)
        total += float(np.sum(ww * overlaps))

    return total


# ============================================================
# 10) MCMC / SIMULATED ANNEALING (HARD-CONSTRAINT SAFE)
# ============================================================

def weighted_choice(rng: random.Random, items, weights):
    if not items:
        raise ValueError("weighted_choice called with empty items")
    total = float(sum(weights))
    if total <= 0:
        return rng.choice(items)
    r = rng.random() * total
    acc = 0.0
    for it, w in zip(items, weights):
        acc += float(w)
        if r <= acc:
            return it
    return items[-1]

def seat_penalty(offered: Counter, courses: Dict[str, Course], waste_weight: float = 0.0) -> Tuple[float, int, int]:
    total_short = 0
    total_waste = 0
    for cname, c in courses.items():
        if c.demand <= 0:
            continue
        off = offered.get(cname, 0)
        seats = off * c.capacity
        total_short += max(0, c.demand - seats)
        total_waste += max(0, seats - c.demand)
    return float(total_short) + waste_weight * float(total_waste), int(total_short), int(total_waste)

def assert_required_targets(offered: Counter, courses: Dict[str, Course], targets: Dict[str, int]) -> None:
    bad = []
    for cname, c in courses.items():
        if c.demand <= 0 or not c.is_required:
            continue
        off = int(offered.get(cname, 0))
        targ = int(targets.get(cname, 0))
        if off < targ:
            bad.append((cname, off, targ, c.demand, c.capacity))

    if bad:
        msg = "\n".join(
            f"  - {cname}: offered={off} < target={targ} (demand={dem}, cap={cap})"
            for cname, off, targ, dem, cap in sorted(bad)
        )
        raise RuntimeError("HARD CONSTRAINT VIOLATION: required course(s) under target:\n" + msg)

def mcmc_optimize(
    schedule: Dict[str, List[Optional[str]]],
    teachers: Dict[str, Teacher],
    courses: Dict[str, Course],
    forced_off: Dict[str, Dict[str, Set[int]]],
    energy_ctx: dict,
    floors: Dict[str, int],
    targets: Dict[str, int],
    seed: int = 0,
    iterations: int = 30_000,
    temp_start: float = 1200.0,
    temp_end: float = 40.0,
    p_course_move: float = 0.50,
    seat_weight: float = 6.0,
    waste_weight: float = 0.0,
    log_every: int = 5000
) -> Dict[str, List[Optional[str]]]:

    rng = random.Random(seed)
    eligible = build_eligibility(courses, teachers)

    cur = {t: row[:] for t, row in schedule.items()}
    cur_conf = calculate_energy(cur, energy_ctx)
    offered_cur = compute_offered_sections(cur, courses)
    cur_seat_pen, cur_short, cur_waste = seat_penalty(offered_cur, courses, waste_weight=waste_weight)
    cur_obj = cur_conf + seat_weight * cur_seat_pen

    best = {t: row[:] for t, row in cur.items()}
    best_obj = cur_obj

    tnames = list(teachers.keys())

    # Identify the teacher holding Studio Art, so we don't break the hard bundle.
    art_teacher = None
    if ART_FALL in courses and courses[ART_FALL].demand > 0:
        for tname, row in cur.items():
            if ART_FALL in row[0:4]:
                art_teacher = tname
                break

    def is_linked(row8: List[Optional[str]], b: int) -> bool:
        f = row8[b]
        s = row8[b + 4]
        if f is None or s is None:
            return False
        return courses[f].is_full_year and f == s

    def cell_mutable(row8: List[Optional[str]], tname: str, sem: str, b: int) -> bool:
        p = b if sem == "F" else b + 4
        cname = row8[p]
        if cname is None:
            return False
        if is_linked(row8, b):
            return False
        if cname == ART_FALL:
            return False
        if art_teacher == tname and sem == "S" and cname == ART_SPRING:
            return False
        return True

    def donor_weight(cname: str) -> float:
        off = offered_cur.get(cname, 0)
        slack = max(0, off - floors.get(cname, 0))
        c = courses[cname]
        waste = max(0, off * c.capacity - c.demand)
        return 1.0 + 3.0 * float(slack) + 0.02 * float(waste)

    def pick_want_course() -> Optional[str]:
        names = []
        weights = []
        for cname, c in courses.items():
            if c.demand <= 0:
                continue
            if c.is_full_year:
                continue
            if cname in PAIRED_NEXT_SEMESTER:
                continue

            off = offered_cur.get(cname, 0)
            seats = off * c.capacity
            short = max(0, c.demand - seats)
            if short <= 0:
                continue
            names.append(cname)
            weights.append(short)

        if not names:
            return None
        return weighted_choice(rng, names, weights)

    for it in range(iterations):
        frac = it / max(1, iterations - 1)
        T = temp_start * (1.0 - frac) + temp_end * frac

        # Move A: period swaps (counts unchanged)
        if rng.random() > p_course_move:
            tname = rng.choice(tnames)
            t = teachers[tname]
            row = cur[tname]

            blocks = [0, 1, 2] if t.coach else [0, 1, 2, 3]
            if len(blocks) < 2:
                continue
            b1, b2 = rng.sample(blocks, 2)

            r = rng.random()
            if r < 0.45:
                move = "BOTH"
            elif r < 0.725:
                move = "FALL"
            else:
                move = "SPRING"

            if move != "BOTH" and (is_linked(row, b1) or is_linked(row, b2)):
                continue

            f_off = set(forced_off.get(tname, {}).get("F", set()))
            s_off = set(forced_off.get(tname, {}).get("S", set()))
            if t.coach:
                f_off |= {3}
                s_off |= {3}

            if move == "FALL" and (b1 in f_off or b2 in f_off):
                continue
            if move == "SPRING" and (b1 in s_off or b2 in s_off):
                continue
            if move == "BOTH" and ((b1 in f_off or b2 in f_off) or (b1 in s_off or b2 in s_off)):
                continue

            new_row = row[:]
            if move == "BOTH":
                new_row[b1], new_row[b2] = new_row[b2], new_row[b1]
                new_row[b1 + 4], new_row[b2 + 4] = new_row[b2 + 4], new_row[b1 + 4]
            elif move == "FALL":
                new_row[b1], new_row[b2] = new_row[b2], new_row[b1]
            else:
                new_row[b1 + 4], new_row[b2 + 4] = new_row[b2 + 4], new_row[b1 + 4]

            if not validate_teacher_row(t, new_row, courses, forced_off):
                continue

            new = cur.copy()
            new[tname] = new_row

            new_conf = calculate_energy(new, energy_ctx)
            new_obj = new_conf + seat_weight * cur_seat_pen
            dE = new_obj - cur_obj

            if dE <= 0 or rng.random() < math.exp(-dE / max(1e-9, T)):
                cur = new
                cur_conf = new_conf
                cur_obj = new_obj
                if cur_obj < best_obj:
                    best_obj = cur_obj
                    best = {k: v[:] for k, v in cur.items()}

        # Move B: section-count shifts (changes offered counts but respects floors)
        else:
            want = pick_want_course()
            if want is None:
                continue

            if courses[want].term == "F":
                sem = "F"
            elif courses[want].term == "S":
                sem = "S"
            else:
                sem = rng.choice(["F", "S"])

            if not courses[want].allowed_in_semester(sem):
                continue

            cell1 = []
            w1 = []
            for t1 in eligible.get(want, []):
                row1 = cur[t1]
                for b1 in range(4):
                    if not cell_mutable(row1, t1, sem, b1):
                        continue
                    p1 = b1 if sem == "F" else b1 + 4
                    old1 = row1[p1]
                    if old1 == want:
                        continue
                    cell1.append((t1, b1))
                    w1.append(donor_weight(old1))

            if not cell1:
                continue

            t1, b1 = weighted_choice(rng, cell1, w1)
            row1 = cur[t1]
            p1 = b1 if sem == "F" else b1 + 4
            old1 = row1[p1]
            if old1 is None or old1 == want:
                continue

            # Direct relabel if old1 can be reduced without violating floors
            if offered_cur.get(old1, 0) > floors.get(old1, 0):
                new_row1 = row1[:]
                new_row1[p1] = want
                if not validate_teacher_row(teachers[t1], new_row1, courses, forced_off):
                    continue

                offered_new = offered_cur.copy()
                offered_new[old1] -= 1
                offered_new[want] += 1

                new = cur.copy()
                new[t1] = new_row1

                new_conf = calculate_energy(new, energy_ctx)
                new_seat_pen, new_short, new_waste = seat_penalty(offered_new, courses, waste_weight=waste_weight)
                new_obj = new_conf + seat_weight * new_seat_pen
                dE = new_obj - cur_obj

                if dE <= 0 or rng.random() < math.exp(-dE / max(1e-9, T)):
                    cur = new
                    offered_cur = offered_new
                    cur_conf = new_conf
                    cur_seat_pen, cur_short, cur_waste = new_seat_pen, new_short, new_waste
                    cur_obj = new_obj
                    if cur_obj < best_obj:
                        best_obj = cur_obj
                        best = {k: v[:] for k, v in cur.items()}

            else:
                # 2-cell chain to preserve old1 at floor
                sem2_opts = ["F", "S"] if courses[old1].term == "EITHER" else [courses[old1].term]
                rng.shuffle(sem2_opts)

                made_move = False
                for sem2 in sem2_opts:
                    if not courses[old1].allowed_in_semester(sem2):
                        continue

                    cell2 = []
                    w2 = []
                    for t2 in eligible.get(old1, []):
                        row2 = cur[t2]
                        for b2 in range(4):
                            if t2 == t1 and sem2 == sem and b2 == b1:
                                continue
                            if not cell_mutable(row2, t2, sem2, b2):
                                continue
                            p2 = b2 if sem2 == "F" else b2 + 4
                            old2 = row2[p2]
                            if old2 is None:
                                continue
                            if old2 in (want, old1):
                                continue
                            if offered_cur.get(old2, 0) <= floors.get(old2, 0):
                                continue
                            cell2.append((t2, b2, sem2))
                            w2.append(donor_weight(old2))

                    if not cell2:
                        continue

                    t2, b2, sem2 = weighted_choice(rng, cell2, w2)
                    row2 = cur[t2]
                    p2 = b2 if sem2 == "F" else b2 + 4
                    old2 = row2[p2]

                    if old2 is None or old2 in (want, old1):
                        continue

                    if t1 == t2:
                        new_row = row1[:]
                        new_row[p1] = want
                        new_row[p2] = old1
                        if not validate_teacher_row(teachers[t1], new_row, courses, forced_off):
                            continue

                        offered_new = offered_cur.copy()
                        offered_new[want] += 1
                        offered_new[old2] -= 1
                        if offered_new.get(old2, 0) < floors.get(old2, 0):
                            continue

                        new = cur.copy()
                        new[t1] = new_row

                    else:
                        new_row1 = row1[:]
                        new_row2 = row2[:]
                        new_row1[p1] = want
                        new_row2[p2] = old1
                        if not validate_teacher_row(teachers[t1], new_row1, courses, forced_off):
                            continue
                        if not validate_teacher_row(teachers[t2], new_row2, courses, forced_off):
                            continue

                        offered_new = offered_cur.copy()
                        offered_new[want] += 1
                        offered_new[old2] -= 1
                        if offered_new.get(old2, 0) < floors.get(old2, 0):
                            continue

                        new = cur.copy()
                        new[t1] = new_row1
                        new[t2] = new_row2

                    new_conf = calculate_energy(new, energy_ctx)
                    new_seat_pen, new_short, new_waste = seat_penalty(offered_new, courses, waste_weight=waste_weight)
                    new_obj = new_conf + seat_weight * new_seat_pen
                    dE = new_obj - cur_obj

                    if dE <= 0 or rng.random() < math.exp(-dE / max(1e-9, T)):
                        cur = new
                        offered_cur = offered_new
                        cur_conf = new_conf
                        cur_seat_pen, cur_short, cur_waste = new_seat_pen, new_short, new_waste
                        cur_obj = new_obj
                        if cur_obj < best_obj:
                            best_obj = cur_obj
                            best = {k: v[:] for k, v in cur.items()}
                        made_move = True
                        break

                if not made_move:
                    continue

        if log_every and (it + 1) % log_every == 0:
            print(
                f"Iter {it+1}/{iterations} "
                f"obj={cur_obj:.2f} conf={cur_conf:.2f} "
                f"seatShort={cur_short} wasteSeats={cur_waste} T={T:.1f} "
                f"| best_obj={best_obj:.2f}"
            )

    if not validate_hard(best, teachers, courses, floors, forced_off):
        raise RuntimeError("BUG: best schedule violates hard constraints after MCMC.")

    return best


# ============================================================
# 11) REPORTING (single schedule)
# ============================================================

def print_schedule(schedule: Dict[str, List[Optional[str]]], teachers: Dict[str, Teacher]) -> None:
    header = ["Teacher"] + PERIODS
    print("\n=== MASTER SCHEDULE ===")
    # Use fixed width for table formatting
    col_widths = [20] * 9
    col_widths[0] = 20
    
    header_out = [f"{h:<{col_widths[i]}}" for i, h in enumerate(header)]
    print(" | ".join(header_out))
    print("-" * (sum(col_widths) + 8 * 3)) # Approximate separator length

    for tname in sorted(teachers.keys()):
        row = schedule[tname]
        out = [tname] + [(c[:20] if c else "OFF") for c in row]
        row_out = [f"{x:<{col_widths[i]}}" for i, x in enumerate(out)]
        print(" | ".join(row_out))

def print_schedule_long(schedule: Dict[str, List[Optional[str]]], teachers: Dict[str, Teacher]) -> None:
    """
    ISEF-friendly printing: full course names (no truncation).
    This is intentionally verbose (teachers × 8 lines).
    """
    print("\n=== MASTER SCHEDULE (FULL NAMES) ===")
    for tname in sorted(teachers.keys()):
        row = schedule[tname]
        print(f"\nTeacher: {tname}")
        for p, period in enumerate(PERIODS):
            cname = row[p] if row[p] is not None else "OFF"
            print(f"  {period}: {cname}")


def _sample_objective(
    sample: "ScheduleSample",
    courses: Dict[str, Course],
    seat_weight: float,
    waste_weight: float,
) -> Tuple[float, float, int, int]:
    """
    Returns (objective, energy, seat_short, waste_seats).

    objective = energy + seat_weight * (seat_short + waste_weight * waste_seats)

    ISEF clarity:
      - energy = expected conflict cost from the request-matrix overlap model (lower is better)
      - seat_short = total missing seats across courses (lower is better)
      - waste_seats = total extra seats offered above demand (reported; optional penalty)
    """
    seat_pen, seat_short, waste_seats = seat_penalty(sample.offered, courses, waste_weight=waste_weight)
    obj = float(sample.energy) + float(seat_weight) * float(seat_pen)
    return obj, float(sample.energy), int(seat_short), int(waste_seats)


def print_best_and_worst_bank_schedules(
    samples: List["ScheduleSample"],
    courses: Dict[str, Course],
    teachers: Dict[str, Teacher],
    seat_weight: float,
    waste_weight: float,
    top_k: int = 5,
    bottom_k: int = 5,
    style: str = "long",
    export_txt: bool = False,
    export_prefix: str = "bank_extremes",
) -> None:
    """
    Prints the most/least effective schedules in the bank (based on the SAME objective used in MCMC).
    """
    if not samples:
        print("\n(No schedules in bank; cannot print best/worst.)")
        return

    scored = []
    for s in samples:
        obj, E, short, waste = _sample_objective(s, courses, seat_weight, waste_weight)
        scored.append((obj, E, short, waste, s.seed, s))

    scored.sort(key=lambda x: x[0])  # lowest objective = best

    top_k = max(0, min(int(top_k), len(scored)))
    bottom_k = max(0, min(int(bottom_k), len(scored)))

    best = scored[:top_k]
    worst = scored[-bottom_k:][::-1]  # highest objective first

    def _print_sched(sample: "ScheduleSample") -> None:
        # --- MODIFIED LOGIC START ---
        if style == "table":
            print_schedule(sample.schedule, teachers)
        else: # style == "long"
            print_schedule_long(sample.schedule, teachers)
        # --- MODIFIED LOGIC END ---

    def _export(sample: "ScheduleSample", tag: str, rank: int, seed: int) -> None:
        if not export_txt:
            return
        from io import StringIO
        import contextlib
        buf = StringIO()
        with contextlib.redirect_stdout(buf):
            _print_sched(sample)
        path = f"{export_prefix}_{tag}_{rank:02d}_seed_{seed}.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(buf.getvalue())

    print("\n=== TOP (BEST) MASTER SCHEDULE SAMPLES IN BANK ===")
    for rank, (obj, E, short, waste, seed, s) in enumerate(best, start=1):
        print(f"\n--- BEST #{rank}/{top_k} | seed={seed} | obj={obj:.2f} | energy={E:.2f} | seatShort={short} | wasteSeats={waste} ---")
        _print_sched(s)
        _export(s, "best", rank, seed)

    print("\n=== BOTTOM (WORST) MASTER SCHEDULE SAMPLES IN BANK ===")
    for rank, (obj, E, short, waste, seed, s) in enumerate(worst, start=1):
        print(f"\n--- WORST #{rank}/{bottom_k} | seed={seed} | obj={obj:.2f} | energy={E:.2f} | seatShort={short} | wasteSeats={waste} ---")
        _print_sched(s)
        _export(s, "worst", rank, seed)

def print_section_summary(courses: Dict[str, Course], offered: Counter, targets: Dict[str, int]) -> None:
    print("\n=== SECTION SUMMARY (Demand vs Offered) ===")
    rows = []
    for cname in sorted(courses.keys()):
        c = courses[cname]
        if c.demand <= 0:
            continue
        off = offered.get(cname, 0)
        targ = targets.get(cname, 0)
        seats = off * c.capacity
        short = max(0, c.demand - seats)
        rows.append((short, -c.demand, cname, c.demand, c.capacity, targ, off, seats))

    rows.sort()
    print(f"{'Course':<28} {'Demand':>6} {'Cap':>4} {'Target':>6} {'Offered':>6} {'Seats':>6} {'SeatShort':>9}")
    for short, _negd, cname, dem, cap, targ, off, seats in rows:
        print(f"{cname[:28]:<28} {dem:>6} {cap:>4} {targ:>6} {off:>6} {seats:>6} {short:>9}")


# ============================================================
# 12) SCHEDULE BANK (PICKLE STEP) — NO FOCUS COURSES
# ============================================================

def compute_course_seat_probabilities(offered: Counter, courses: Dict[str, Course]) -> Dict[str, float]:
    p = {}
    for cname, c in courses.items():
        if c.demand <= 0:
            continue
        off = offered.get(cname, 0)
        seats = off * c.capacity
        p[cname] = float(min(1.0, seats / float(c.demand)))
    return p

def precompute_course_option_masks(
    schedule: Dict[str, List[Optional[str]]],
    courses: Dict[str, Course],
) -> Tuple[Dict[str, List[int]], List[int]]:
    """
    Build per-course placement options as 8-slot bitmasks.
      - One-term sections => 1-bit mask in the appropriate semester slot.
      - Full-year sections => 2-bit mask (fall block + matching spring block).
    Also builds Studio Art bundle masks enforcing SAME TEACHER across semesters.
    """
    opts = defaultdict(set)

    art_fall_blocks_by_teacher = defaultdict(set)
    tal_spring_blocks_by_teacher = defaultdict(set)

    for tname, row in schedule.items():
        for b in range(4):
            f = row[b]
            s = row[b + 4]

            if f is not None and f in courses and courses[f].is_full_year and s == f:
                mask = (1 << b) | (1 << (b + 4))
                opts[f].add(mask)
                continue

            if f is not None:
                opts[f].add(1 << b)
                if f == ART_FALL:
                    art_fall_blocks_by_teacher[tname].add(b)

            if s is not None:
                opts[s].add(1 << (b + 4))
                if s == ART_SPRING:
                    tal_spring_blocks_by_teacher[tname].add(b)

    bundle_masks = set()
    for tname, bfs in art_fall_blocks_by_teacher.items():
        bss = tal_spring_blocks_by_teacher.get(tname, set())
        for bf in bfs:
            for bs in bss:
                bundle_masks.add((1 << bf) | (1 << (bs + 4)))

    return {c: sorted(list(ms)) for c, ms in opts.items()}, sorted(list(bundle_masks))

@dataclass
class ScheduleSample:
    seed: int
    schedule: Dict[str, List[Optional[str]]]
    energy: float
    offered: Counter
    seat_p: Dict[str, float]
    course_option_masks: Dict[str, List[int]]
    art_bundle_option_masks: List[int]


_WORKER_CTX = None
_WORKER_MCMC_ITERS = None
_WORKER_SEAT_WEIGHT = None
_WORKER_WASTE_WEIGHT = None

def _init_sample_bank_worker(mcmc_iters: int, seat_weight: float, waste_weight: float):
    global _WORKER_CTX, _WORKER_MCMC_ITERS, _WORKER_SEAT_WEIGHT, _WORKER_WASTE_WEIGHT

    courses = parse_request_matrix(CO_REQUEST_TEXT)
    teachers = init_teachers()
    floors, targets = compute_hard_floors(courses)
    energy_ctx = build_energy_context(courses, FIXED_PERIOD_MAP)

    _WORKER_CTX = (courses, teachers, floors, targets, energy_ctx)
    _WORKER_MCMC_ITERS = int(mcmc_iters)
    _WORKER_SEAT_WEIGHT = float(seat_weight)
    _WORKER_WASTE_WEIGHT = float(waste_weight)

def _worker_generate_one_sample(seed: int) -> Optional[ScheduleSample]:
    courses, teachers, floors, targets, energy_ctx = _WORKER_CTX

    teach, _section_counts, msg = allocate_counts_and_teachers(
        courses=courses, teachers=teachers, floors=floors, targets=targets, seed=seed
    )
    if teach is None:
        return None

    schedule0, forced_off, msg2 = build_initial_schedule(
        teach=teach, teachers=teachers, courses=courses, floors=floors, seed=seed
    )
    if schedule0 is None:
        return None

    best = mcmc_optimize(
        schedule=schedule0,
        teachers=teachers,
        courses=courses,
        forced_off=forced_off,
        energy_ctx=energy_ctx,
        floors=floors,
        targets=targets,
        seed=seed,
        iterations=_WORKER_MCMC_ITERS,
        temp_start=2000.0,
        temp_end=40.0,
        p_course_move=0.55,
        seat_weight=_WORKER_SEAT_WEIGHT,
        waste_weight=_WORKER_WASTE_WEIGHT,
        log_every=0,
    )

    E = calculate_energy(best, energy_ctx)
    offered = compute_offered_sections(best, courses)

    # Extra safety for required courses (ISEF: show hard guarantees)
    assert_required_targets(offered, courses, targets)

    seat_p = compute_course_seat_probabilities(offered, courses)
    course_opts, art_bundle_opts = precompute_course_option_masks(best, courses)

    return ScheduleSample(
        seed=seed,
        schedule=best,
        energy=E,
        offered=offered,
        seat_p=seat_p,
        course_option_masks=course_opts,
        art_bundle_option_masks=art_bundle_opts,
    )

def build_schedule_sample_bank_parallel(
    num_samples: int,
    seed0: int = 0,
    mcmc_iters: int = 50_000,
    max_attempts: int = 10_000,
    n_workers: Optional[int] = None,
    inflight: Optional[int] = None,
    verbose_every: int = 10,
    seat_weight: float = 6.0,
    waste_weight: float = 0.0,
) -> Tuple[List[ScheduleSample], Dict[str, Course], Dict[str, Teacher], Dict[str, int], Dict[str, int]]:
    from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

    courses = parse_request_matrix(CO_REQUEST_TEXT)
    teachers = init_teachers()
    floors, targets = compute_hard_floors(courses)

    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 1) // 1)

    if inflight is None:
        inflight = max(2, 2 * n_workers)

    samples: List[ScheduleSample] = []
    attempts = 0
    seed_iter = iter(range(seed0, seed0 + max_attempts))

    ex = ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_sample_bank_worker,
        initargs=(mcmc_iters, seat_weight, waste_weight),
    )

    futures = set()
    try:
        for _ in range(min(inflight, max_attempts)):
            s = next(seed_iter, None)
            if s is None:
                break
            futures.add(ex.submit(_worker_generate_one_sample, s))

        while futures and len(samples) < num_samples and attempts < max_attempts:
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            for fut in done:
                attempts += 1
                try:
                    res = fut.result()
                except Exception:
                    res = None

                if res is not None:
                    samples.append(res)
                    if verbose_every and (len(samples) % verbose_every == 0):
                        print(f"[sample bank] {len(samples)}/{num_samples} samples (seed={res.seed}, E={res.energy:.2f})")

                if len(samples) < num_samples:
                    s = next(seed_iter, None)
                    if s is not None:
                        futures.add(ex.submit(_worker_generate_one_sample, s))

        if len(samples) < num_samples:
            print(f"WARNING: only generated {len(samples)}/{num_samples} schedules after {attempts} attempts.")

        samples.sort(key=lambda x: x.seed)
        return samples, courses, teachers, floors, targets

    finally:
        ex.shutdown(wait=True, cancel_futures=True)

def save_schedule_bank(path: str, payload: dict) -> None:
    with open(path, "wb") as f:
        pickle.dump(payload, f)

def load_schedule_bank(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ============================================================
# 13) BANK-LEVEL STATISTICS (PRINT + CSV)
# ============================================================

def _course_period_presence_mask(schedule: Dict[str, List[Optional[str]]]) -> Dict[str, int]:
    masks = defaultdict(int)
    for row in schedule.values():
        for p, cname in enumerate(row):
            if cname is None:
                continue
            masks[cname] |= (1 << p)
    return masks

def compute_bank_overall_summary(samples: List[ScheduleSample], courses: Dict[str, Course]) -> dict:
    energies = [s.energy for s in samples]
    shorts = []
    wastes = []
    for s in samples:
        _pen, short, waste = seat_penalty(s.offered, courses, waste_weight=0.0)
        shorts.append(short)
        wastes.append(waste)

    def _mean(xs: List[float]) -> float:
        return float(sum(xs) / max(1, len(xs)))

    def _min(xs: List[float]) -> float:
        return float(min(xs)) if xs else 0.0

    def _max(xs: List[float]) -> float:
        return float(max(xs)) if xs else 0.0

    return {
        "n_samples": len(samples),
        "energy_mean": _mean(energies),
        "energy_min": _min(energies),
        "energy_max": _max(energies),
        "seat_short_mean": _mean(shorts),
        "seat_short_min": _min(shorts),
        "seat_short_max": _max(shorts),
        "waste_seats_mean": _mean(wastes),
        "waste_seats_min": _min(wastes),
        "waste_seats_max": _max(wastes),
    }

def compute_bank_course_stats(
    samples: List[ScheduleSample],
    courses: Dict[str, Course],
    floors: Dict[str, int],
    targets: Dict[str, int],
) -> Tuple[List[dict], Dict[str, Dict[int, int]], Dict[str, List[int]]]:
    """
    Returns:
      rows: list of per-course summary rows (printable / CSV)
      dist_offered: course -> {sections_offered: count_over_samples}
      period_any_counts: course -> [count_of_samples_where_course_is_available_in_period_p]
    """
    n = len(samples)
    dist_offered: Dict[str, Counter] = {cname: Counter() for cname in courses if courses[cname].demand > 0}
    period_any_counts: Dict[str, List[int]] = {cname: [0] * 8 for cname in courses if courses[cname].demand > 0}

    for s in samples:
        for cname, c in courses.items():
            if c.demand <= 0:
                continue
            k = int(s.offered.get(cname, 0))
            dist_offered[cname][k] += 1

        pres = _course_period_presence_mask(s.schedule)
        for cname, c in courses.items():
            if c.demand <= 0:
                continue
            mask = pres.get(cname, 0)
            for p in range(8):
                if (mask >> p) & 1:
                    period_any_counts[cname][p] += 1

    rows: List[dict] = []
    for cname in sorted([c for c in courses if courses[c].demand > 0]):
        c = courses[cname]
        dist = dist_offered[cname]

        mean_off = sum(k * cnt for k, cnt in dist.items()) / max(1, n)
        min_off = min(dist.keys()) if dist else 0
        max_off = max(dist.keys()) if dist else 0

        exp_short = sum(max(0, c.demand - k * c.capacity) * cnt for k, cnt in dist.items()) / max(1, n)
        exp_waste = sum(max(0, k * c.capacity - c.demand) * cnt for k, cnt in dist.items()) / max(1, n)

        p_seat_ok = sum(cnt for k, cnt in dist.items() if (k * c.capacity) >= c.demand) / max(1, n)
        targ = int(targets.get(cname, 0))
        flr = int(floors.get(cname, 0))
        p_meet_target = sum(cnt for k, cnt in dist.items() if k >= targ) / max(1, n)

        common = dist.most_common(4)
        hist_str = "; ".join(f"{k} sec: {100.0*cnt/max(1,n):.1f}%" for k, cnt in common) if common else ""

        period_probs = [period_any_counts[cname][p] / max(1, n) for p in range(8)]
        top_periods = sorted(range(8), key=lambda p: -period_probs[p])[:3]
        top_periods_str = ", ".join(f"{PERIODS[p]}:{100.0*period_probs[p]:.1f}%" for p in top_periods)

        rows.append({
            "course": cname,
            "demand": c.demand,
            "capacity": c.capacity,
            "target_sections": targ,
            "floor_sections": flr,
            "mean_offered_sections": mean_off,
            "min_offered_sections": min_off,
            "max_offered_sections": max_off,
            "p_meet_target": p_meet_target,
            "p_seats_ok": p_seat_ok,
            "expected_seat_short": exp_short,
            "expected_waste_seats": exp_waste,
            "most_common_offered": hist_str,
            "top_periods": top_periods_str,
        })

    dist_offered_out = {c: dict(dist_offered[c]) for c in dist_offered}
    return rows, dist_offered_out, period_any_counts

def print_bank_course_stats(rows: List[dict]) -> None:
    print("\n=== COURSE STATISTICS ACROSS MASTER-SCHEDULE SAMPLES ===")
    print("Each line includes: demand/cap/target/floor + offered-section distribution (%) + seat feasibility")

    for r in rows:
        print(
            f"- {r['course']}: "
            f"demand={r['demand']}, cap={r['capacity']}, target={r['target_sections']}, floor={r['floor_sections']} | "
            f"meanOff={r['mean_offered_sections']:.2f} (min={r['min_offered_sections']}, max={r['max_offered_sections']}) | "
            f"P(seats>=demand)={100.0*r['p_seats_ok']:.1f}% | "
            f"E[seatShort]={r['expected_seat_short']:.1f}, E[wasteSeats]={r['expected_waste_seats']:.1f} | "
            f"offeredDist: {r['most_common_offered']} | "
            f"topPeriods: {r['top_periods']}"
        )

def write_bank_course_stats_csv(path: str, rows: List[dict]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    if not fieldnames:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

def write_bank_course_period_availability_csv(
    path: str,
    period_any_counts: Dict[str, List[int]],
    n_samples: int,
) -> None:
    header = ["course"] + [f"P_any_{p}" for p in PERIODS]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for cname in sorted(period_any_counts.keys()):
            probs = [period_any_counts[cname][i] / max(1, n_samples) for i in range(8)]
            w.writerow([cname] + probs)


# ============================================================
# 14) STUDENT OUTCOME SIMULATOR (Monte Carlo)
# ============================================================

def _popcount(mask: int) -> int:
    try:
        return mask.bit_count()
    except AttributeError:
        return bin(mask).count("1")

def _slots_from_mask(mask: int) -> List[int]:
    return [p for p in range(8) if (mask >> p) & 1]

def normalize_course_list(xs: List[str]) -> List[str]:
    out = []
    for x in xs:
        if x is None:
            continue
        s = str(x).strip()
        if not s:
            continue
        out.append(normalize_name(s))
    return out

def validate_course_names(names: Iterable[str], courses: Dict[str, Course]) -> None:
    bad = [n for n in names if (n not in courses and n not in FIXED_PERIOD_MAP)]
    if bad:
        raise ValueError(
            "Unknown course name(s): " + ", ".join(sorted(set(bad))) +
            "\nFix spelling OR add alias in ALIASES OR add to request matrix."
        )

@dataclass(frozen=True)
class PlacementOption:
    mask: int
    slot_to_course: Tuple[Tuple[int, str], ...]

    @property
    def courses_obtained(self) -> Set[str]:
        return {c for _p, c in self.slot_to_course}

@dataclass
class StudentSpec:
    top_slots: List[str]
    backup_slots: List[str]

    top: List[str]
    backups: List[str]

    top_set: Set[str]
    backup_set: Set[str]
    weights: Dict[str, float]

@dataclass
class StudentSolveResult:
    filled_mask: int
    obtained_courses: Set[str]
    missing_top: Set[str]
    used_backups: Set[str]
    off_slots: int
    got_all_top: bool
    failed: bool
    assignment: Optional[Dict[str, str]] = None

def prepare_student_spec(
    top8: List[str],
    backups: List[str],
    courses: Dict[str, Course],
    top_weight: float = 100.0,
    backup_weights: Tuple[float, float, float, float] = (30.0, 20.0, 10.0, 5.0),
) -> StudentSpec:
    top_slots = normalize_course_list(top8)
    backup_slots = normalize_course_list(backups)

    validate_course_names(top_slots + backup_slots, courses)

    # Enforce: full-year requested => must appear twice in top/backups
    top_cnt = Counter(top_slots)
    bk_cnt = Counter(backup_slots)
    for cname, c in courses.items():
        if not c.is_full_year:
            continue
        ct = top_cnt.get(cname, 0)
        cb = bk_cnt.get(cname, 0)
        if ct not in (0, 2):
            raise ValueError(f"Full-year course '{cname}' must appear exactly twice in top8 (found {ct}).")
        if cb not in (0, 2):
            raise ValueError(f"Full-year course '{cname}' must appear exactly twice in backups (found {cb}).")

    # Unique top list
    top_u = []
    seen = set()
    for c in top_slots:
        if c not in seen:
            top_u.append(c)
            seen.add(c)

    # Unique backups excluding top
    b_u = []
    seen_b = set(top_u)
    for c in backup_slots:
        if c in seen_b:
            continue
        if c not in b_u:
            b_u.append(c)

    weights = Counter()
    for c in top_slots:
        weights[c] += float(top_weight)

    for i, c in enumerate(backup_slots):
        if c in set(top_u):
            continue
        w = backup_weights[i] if i < len(backup_weights) else backup_weights[-1]
        weights[c] += float(w)

    return StudentSpec(
        top_slots=top_slots,
        backup_slots=backup_slots,
        top=top_u,
        backups=b_u,
        top_set=set(top_u),
        backup_set=set(b_u),
        weights=dict(weights),
    )

def _build_options_for_course(sample: ScheduleSample, cname: str, courses: Dict[str, Course]) -> List[PlacementOption]:
    # Fixed-period pseudo-courses (Dual Enrollment / Early Release)
    if cname in FIXED_PERIOD_MAP:
        p = FIXED_PERIOD_MAP[cname]
        mask = 1 << p
        return [PlacementOption(mask=mask, slot_to_course=((p, cname),))]

    c = courses.get(cname)
    if c is None:
        # Unknown teachable course name shouldn't happen if validate_course_names() passed
        return []

    masks = sample.course_option_masks.get(cname, [])
    out: List[PlacementOption] = []

    for m in masks:
        pc = _popcount(m)

        if c.is_full_year:
            # Must occupy exactly one fall block + matching spring block
            if pc != 2:
                continue
            fall_mask = m & 0x0F
            spring_mask = (m >> 4) & 0x0F
            if fall_mask == 0 or spring_mask == 0:
                continue
            if fall_mask != spring_mask:
                # Reject weird FY masks like F1+S3 (not the same block)
                continue

        else:
            # One-semester courses must occupy exactly ONE slot total
            if pc != 1:
                continue
            # Enforce term-only courses stay in their semester
            if c.term == "F" and (m & 0xF0):
                continue
            if c.term == "S" and (m & 0x0F):
                continue

        slots = _slots_from_mask(m)
        if not slots:
            continue
        pairs = tuple(sorted((p, cname) for p in slots))
        out.append(PlacementOption(mask=m, slot_to_course=pairs))

    return out

def _build_options_for_art_bundle(sample: ScheduleSample) -> List[PlacementOption]:
    out = []
    for m in sample.art_bundle_option_masks:
        slots = _slots_from_mask(m)
        if len(slots) != 2:
            continue
        fall_slots = [p for p in slots if p < 4]
        spring_slots = [p for p in slots if p >= 4]
        if len(fall_slots) != 1 or len(spring_slots) != 1:
            continue
        pf = fall_slots[0]
        ps = spring_slots[0]
        out.append(PlacementOption(
            mask=m,
            slot_to_course=tuple(sorted(((pf, ART_FALL), (ps, ART_SPRING))))
        ))
    return out

def solve_student_for_sample(
    sample: ScheduleSample,
    courses: Dict[str, Course],
    spec: StudentSpec,
    rng: random.Random,
    seat_model: str = "none",   # "none" or "course_lottery"
    fill_bonus: float = 1000.0,
    return_assignment: bool = False,
) -> StudentSolveResult:
    need_avail = set(spec.top) | set(spec.backups)
    if ART_FALL in need_avail:
        need_avail.add(ART_SPRING)

    avail = {}
    if seat_model == "none":
        for c in need_avail:
            avail[c] = True
    elif seat_model == "course_lottery":
        for c in need_avail:
            if c in FIXED_PERIOD_MAP:
                avail[c] = True
                continue
            p = sample.seat_p.get(c, 0.0)
            avail[c] = (rng.random() < float(p))
    else:
        raise ValueError("seat_model must be 'none' or 'course_lottery'")

    items: List[Tuple[str, float, List[PlacementOption]]] = []

    for cname in list(spec.top) + list(spec.backups):
        if cname == ART_FALL:
            if not avail.get(ART_FALL, False) or not avail.get(ART_SPRING, False):
                options = []
            else:
                options = _build_options_for_art_bundle(sample)
            w = float(spec.weights.get(ART_FALL, 0.0)) + float(spec.weights.get(ART_SPRING, 0.0))
            items.append((ART_FALL, w, options))
            continue

        if not avail.get(cname, False):
            items.append((cname, float(spec.weights.get(cname, 0.0)), []))
            continue

        options = _build_options_for_course(sample, cname, courses)
        items.append((cname, float(spec.weights.get(cname, 0.0)), options))

    NEG = -1e18
    dp: Dict[Tuple[int, int], float] = {(0, 0): 0.0}
    back: Dict[Tuple[int, int], Optional[Tuple[Tuple[int, int], str, PlacementOption]]] = {(0, 0): None}

    items.sort(key=lambda it: (len(it[2]) if it[2] else 10_000, it[0]))

    for key, base_w, options in items:
        new_dp = dict(dp)
        new_back = dict(back)

        for (mask, ta), score in dp.items():
            for opt in options:
                if mask & opt.mask:
                    continue

                opt_courses = opt.courses_obtained
                takes_tal_art = (ART_SPRING in opt_courses)
                if ta == 1 and takes_tal_art:
                    continue

                nmask = mask | opt.mask
                nta = 1 if (ta == 1 or takes_tal_art) else 0

                val = score + base_w + fill_bonus * float(_popcount(opt.mask))
                if val > new_dp.get((nmask, nta), NEG):
                    new_dp[(nmask, nta)] = val
                    new_back[(nmask, nta)] = ((mask, ta), key, opt)

        dp = new_dp
        back = new_back

    best_state = max(dp.items(), key=lambda kv: kv[1])[0]
    best_mask, _best_ta = best_state

    chosen: List[Tuple[str, PlacementOption]] = []
    cur = best_state
    while True:
        bp = back.get(cur)
        if bp is None:
            break
        prev, key, opt = bp
        chosen.append((key, opt))
        cur = prev
    chosen.reverse()

    slot_to_course_raw: Dict[int, str] = {}
    for _key, opt in chosen:
        for p, cn in opt.slot_to_course:
            slot_to_course_raw[p] = cn

    # ---- DUPLICATE COURSE GUARD ----
    # Rule:
    #   - non-full-year courses: max 1 slot total
    #   - full-year courses: exactly 2 slots total (we allow up to 2 here; masks are already validated)
    clean_slot_to_course: Dict[int, str] = {}
    course_counts = Counter()
    had_illegal_duplicate = False

    for p in range(8):
        cn = slot_to_course_raw.get(p)
        if cn is None:
            continue

        if cn in FIXED_PERIOD_MAP:
            # Fixed pseudo-courses should also not duplicate
            if course_counts[cn] >= 1:
                had_illegal_duplicate = True
                continue
            course_counts[cn] += 1
            clean_slot_to_course[p] = cn
            continue

        cobj = courses.get(cn)
        if cobj is None:
            # Shouldn't happen; keep it, but don't crash
            clean_slot_to_course[p] = cn
            continue

        limit = 2 if cobj.is_full_year else 1
        if course_counts[cn] >= limit:
            had_illegal_duplicate = True
            continue

        course_counts[cn] += 1
        clean_slot_to_course[p] = cn

    slot_to_course = clean_slot_to_course
    obtained = set(slot_to_course.values())

    filled = 0
    for p in slot_to_course:
        filled |= (1 << p)

    off_slots = 8 - _popcount(filled)
   
    # If we had to drop duplicates, that can create OFF slots -> treat as failure (realistic)
    # (Even if off_slots==0, dropping duplicates means the original solution was invalid.)
    # Force-fail on had_illegal_duplicate keeps Monte Carlo honest so I have it on
    # --------------------------------------

    implied = set()
    if ART_FALL in obtained:
        implied.add(ART_SPRING)

    missing_top = set(spec.top_set) - obtained
    used_backups = (obtained & spec.backup_set) - implied
    got_all_top = (len(missing_top) == 0)

    # LAST RESORT FOR THE BUG fix it still works without had_illegal_duplicate I think
    failed = (off_slots > 0) or had_illegal_duplicate

    assignment = None
    if return_assignment:
        assignment = {PERIODS[p]: (slot_to_course.get(p, "OFF")) for p in range(8)}

    return StudentSolveResult(
        filled_mask=filled,
        obtained_courses=obtained,
        missing_top=missing_top,
        used_backups=used_backups,
        off_slots=off_slots,
        got_all_top=got_all_top,
        failed=failed,
        assignment=assignment,
    )

def monte_carlo_student(
    samples: List[ScheduleSample],
    courses: Dict[str, Course],
    top8: List[str],
    backups: List[str],
    iterations: int = 20_000,
    seed: int = 0,
    seat_model: str = "none",
    top_scenarios: int = 12,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    spec = prepare_student_spec(top8, backups, courses)

    scen = Counter()
    n_all_top = 0
    n_fail = 0
    top_got_hist = Counter()

    for _ in range(iterations):
        s = rng.choice(samples)
        res = solve_student_for_sample(
            sample=s,
            courses=courses,
            spec=spec,
            rng=rng,
            seat_model=seat_model,
            fill_bonus=1000.0,
            return_assignment=False,
        )

        if res.got_all_top:
            n_all_top += 1
        if res.failed:
            n_fail += 1

        got_top_n = sum(1 for c in spec.top_slots if c in res.obtained_courses)
        top_got_hist[got_top_n] += 1

        key = (tuple(sorted(res.missing_top)), tuple(sorted(res.used_backups)), bool(res.failed))
        scen[key] += 1

    scen_list = []
    for (miss, used, failed), cnt in scen.most_common(top_scenarios):
        scen_list.append({
            "p": cnt / iterations,
            "count": cnt,
            "failed": failed,
            "missing_top": list(miss),
            "used_backups": list(used),
        })

    return {
        "iterations": iterations,
        "seat_model": seat_model,
        "top": spec.top_slots,
        "backups": spec.backup_slots,
        "p_all_top": n_all_top / iterations,
        "p_failed": n_fail / iterations,
        "top_got_hist": {k: v / iterations for k, v in sorted(top_got_hist.items())},
        "top_scenarios": scen_list,
    }

def print_student_report(result: Dict[str, Any]) -> None:
    print("\n=== STUDENT OUTCOME MONTE CARLO ===")
    print(f"Iterations: {result['iterations']}")
    print(f"Seat model: {result['seat_model']}")
    print(f"Top requests ({len(result['top'])}): {', '.join(result['top'])}")
    print(f"Backups ({len(result['backups'])}): {', '.join(result['backups'])}")

    print(f"\nP(get ALL top requests): {100.0 * result['p_all_top']:.2f}%")
    print(f"FAILED % (even backups can't fill a conflict-free 4×4): {100.0 * result['p_failed']:.2f}%")

    print("\nDistribution: #Top obtained")
    for k, p in result["top_got_hist"].items():
        print(f"  {k}: {100.0*p:.2f}%")

    print("\nTop fallback scenarios (includes multi-backup combos):")
    for s in result["top_scenarios"]:
        miss = s["missing_top"]
        used = s["used_backups"]
        print(
            f"  p={100.0*s['p']:.2f}% | failed={s['failed']} | "
            f"missing_top={miss if miss else '[]'} | used_backups={used if used else '[]'}"
        )


# ============================================================
# 15) MAIN (two modes)
# ============================================================


def main_build_bank() -> None:
    # Prevent BLAS oversubscription in parallel workers (important for Ryzen / many cores)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    print("=== PICKLE STEP: building schedule bank (run once) ===")
    print(
        "ISEF clarity: Every schedule saved to the bank satisfies ALL hard constraints.\n"
        "We then rank schedules by the same objective used in simulated annealing:\n"
        "  objective = conflict_energy + seat_weight * seat_shortage (+ optional waste penalty)\n"
    )

    t0 = time.time()

    samples, courses, teachers, floors, targets = build_schedule_sample_bank_parallel(
        num_samples=BANK_NUM_SAMPLES,
        seed0=BANK_SEED0,
        mcmc_iters=BANK_MCMC_ITERS,
        max_attempts=BANK_MAX_ATTEMPTS,
        n_workers=BANK_N_WORKERS,
        inflight=BANK_INFLIGHT,
        verbose_every=BANK_VERBOSE_EVERY,
        seat_weight=BANK_SEAT_WEIGHT,
        waste_weight=BANK_WASTE_WEIGHT,
    )

    summary = compute_bank_overall_summary(samples, courses)
    rows, dist_offered, period_any_counts = compute_bank_course_stats(samples, courses, floors, targets)

    print("\n=== BANK OVERALL SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print_bank_course_stats(rows)

    # Print the best/worst schedules in the bank
    print_best_and_worst_bank_schedules(
        samples=samples,
        courses=courses,
        teachers=teachers,
        seat_weight=BANK_SEAT_WEIGHT,
        waste_weight=BANK_WASTE_WEIGHT,
        top_k=BANK_PRINT_TOP_K,
        bottom_k=BANK_PRINT_BOTTOM_K,
        style=BANK_PRINT_SCHEDULE_STYLE, # This now uses "table" or "long"
        export_txt=BANK_EXPORT_EXTREME_SCHEDULES,
        export_prefix="bank_extremes",
    )

    # CSVs for graphs
    write_bank_course_stats_csv("bank_course_stats.csv", rows)
    write_bank_course_period_availability_csv("bank_course_period_availability.csv", period_any_counts, len(samples))

    payload = {
        "version": 4,
        "created_unix": time.time(),
        "build_params": {
            "BANK_NUM_SAMPLES": BANK_NUM_SAMPLES,
            "BANK_SEED0": BANK_SEED0,
            "BANK_MCMC_ITERS": BANK_MCMC_ITERS,
            "BANK_MAX_ATTEMPTS": BANK_MAX_ATTEMPTS,
            "BANK_N_WORKERS": BANK_N_WORKERS,
            "BANK_INFLIGHT": BANK_INFLIGHT,
            "BANK_SEAT_WEIGHT": BANK_SEAT_WEIGHT,
            "BANK_WASTE_WEIGHT": BANK_WASTE_WEIGHT,
        },
        "summary": summary,
        "course_stats_rows": rows,
        "dist_offered_sections": dist_offered,
        "period_any_counts": period_any_counts,
        "samples": samples,
    }
    save_schedule_bank(BANK_PATH, payload)

    print(f"\nSaved: {BANK_PATH}")
    print(f"Time: {time.time() - t0:.1f} sec")

def main_run_monte_carlo_only() -> None:
    print("=== MONTE CARLO ONLY MODE: loading schedule bank ===")
    payload = load_schedule_bank(BANK_PATH)
    samples: List[ScheduleSample] = payload["samples"]

    # Rebuild courses from the request matrix (fast & keeps single source of truth)
    courses = parse_request_matrix(CO_REQUEST_TEXT)

    # YOU CAN EDIT LINES 2728-2732 IF YOU'RE USING MONTE CARLO MODE
    # Make sure top8 is 8 classes only. Backups can be any number of classes, but our school only allowed 4.
    # Full year classes (AP Bio, AP Calc, and APUSH) must be inputted twice
    # Make sure the wording of the class is exactly the same as used everywhere else
    # Example single-student Monte Carlo:
    top8 = [
        "AP Psychology", "AP European Hist", "AP English III (11th/12th Grade)", "AP Statistics",
        "DE Intro to Engnr", "AP Biology (full year)", "AP Biology (full year)", "AP Env Sci 1C"
    ]
    backups = ["Robotics (LSU)", "Int Comp Think H", "Afr American H", "AP Calculus AB", "AP Calculus AB"]

    r_time = monte_carlo_student(samples, courses, top8, backups, iterations=20_000, seed=1, seat_model="none")
    print_student_report(r_time)

    r_seats = monte_carlo_student(samples, courses, top8, backups, iterations=20_000, seed=1, seat_model="course_lottery")
    print_student_report(r_seats)

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()

    if BUILD_SCHEDULE_BANK_ONCE:
        main_build_bank()
    else:
        main_run_monte_carlo_only()