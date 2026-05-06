"""
config.py
---------
Central registry for ALL projectile / object types.

Each entry has physics parameters, detection hints, simulation presets,
and a stroke classifier. To add a NEW object → add one dict entry here.
Nothing else changes.
"""

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# HSV colour ranges for detection
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    "white":  {"lower": np.array([0,   0, 180]), "upper": np.array([180, 60, 255])},
    "orange": {"lower": np.array([5,  100, 100]), "upper": np.array([25, 255, 255])},
    "yellow": {"lower": np.array([20,  80, 100]), "upper": np.array([40, 255, 255])},
    "green":  {"lower": np.array([35,  60,  60]), "upper": np.array([85, 255, 255])},
    "red":    {"lower": np.array([0,  120,  70]), "upper": np.array([10, 255, 255])},
    "brown":  {"lower": np.array([5,   50,  50]), "upper": np.array([20, 200, 180])},
    "black":  {"lower": np.array([0,    0,   0]), "upper": np.array([180,255, 60])},
    "blue":   {"lower": np.array([100, 80,  50]), "upper": np.array([130,255,255])},
}

# ─────────────────────────────────────────────────────────────────────────────
# Classifiers
# ─────────────────────────────────────────────────────────────────────────────
def _cls_tt(V0, rps):
    if V0 > 8 and abs(rps) > 15: return "Top Spin" if rps > 0 else "Heavy Push"
    elif V0 < 6: return "Push"
    return "Counter Attack"

def _cls_tennis(V0, rps):
    if V0 > 40: return "Serve / Smash"
    elif V0 > 20: return "Topspin Drive"
    elif rps < -5: return "Slice"
    return "Rally Shot"

def _cls_football(V0, rps):
    if V0 > 25: return "Power Shot"
    elif abs(rps) > 8: return "Banana / Curve"
    elif V0 < 10: return "Pass"
    return "Driven Shot"

def _cls_basketball(V0, rps):
    if V0 > 10: return "Full-Court Pass"
    elif V0 > 6: return "Jump Shot"
    return "Free Throw"

def _cls_cricket(V0, rps):
    if V0 > 35: return "Fast Delivery"
    elif abs(rps) > 15: return "Spin Delivery"
    return "Medium Pace"

def _cls_baseball(V0, rps):
    if V0 > 38: return "Fastball"
    elif abs(rps) > 30: return "Curveball / Slider"
    elif V0 < 30: return "Changeup"
    return "Breaking Ball"

def _cls_volleyball(V0, rps):
    if V0 > 20: return "Spike"
    elif abs(rps) > 10: return "Float Serve"
    return "Rally Hit"

def _cls_mortar(V0, rps):
    if V0 > 200: return "High Velocity"
    elif V0 > 100: return "Medium Velocity"
    return "Low Velocity"

def _cls_generic(V0, rps):
    if V0 > 30: return "High Speed"
    elif V0 > 10: return "Medium Speed"
    return "Low Speed"


# ─────────────────────────────────────────────────────────────────────────────
# Main registry
# ─────────────────────────────────────────────────────────────────────────────
OBJECTS = {

    "table_tennis": {
        "display_name": "Table Tennis Ball",
        "mass_kg":      0.0027,
        "radius_m":     0.020,
        "drag_coeff":   0.45,
        "has_spin":     True,
        "colors":       ["white", "orange"],
        "min_radius_px": 3, "max_radius_px": 30,
        "scene_width_m": 2.74,
        "classify":     _cls_tt,
        "presets": {
            "topspin": dict(V0=13.0, angle_deg=8.0,  omega=35.0,  t_end=0.45),
            "push":    dict(V0=3.5,  angle_deg=3.0,  omega=-12.0, t_end=0.60),
            "counter": dict(V0=7.0,  angle_deg=10.0, omega=15.0,  t_end=0.55),
        },
    },

    "tennis": {
        "display_name": "Tennis Ball",
        "mass_kg":      0.0577,
        "radius_m":     0.033,
        "drag_coeff":   0.55,
        "has_spin":     True,
        "colors":       ["yellow", "green"],
        "min_radius_px": 1, "max_radius_px": 40,
        "scene_width_m": 23.77,
        "classify":     _cls_tennis,
        "presets": {
            "serve":         dict(V0=55.0, angle_deg=-5.0, omega=80.0,  t_end=0.55),
            "topspin_drive": dict(V0=25.0, angle_deg=8.0,  omega=50.0,  t_end=0.80),
            "slice":         dict(V0=20.0, angle_deg=5.0,  omega=-20.0, t_end=0.90),
        },
    },

    "football": {
        "display_name": "Football (Soccer)",
        "mass_kg":      0.430,
        "radius_m":     0.110,
        "drag_coeff":   0.47,
        "has_spin":     True,
        "colors":       ["white", "black"],
        "min_radius_px": 8, "max_radius_px": 60,
        "scene_width_m": 40.0,
        "classify":     _cls_football,
        "presets": {
            "power_shot": dict(V0=30.0, angle_deg=15.0, omega=0.0,  t_end=1.20),
            "banana":     dict(V0=22.0, angle_deg=12.0, omega=10.0, t_end=1.10),
            "pass":       dict(V0=8.0,  angle_deg=3.0,  omega=2.0,  t_end=1.50),
        },
    },

    "basketball": {
        "display_name": "Basketball",
        "mass_kg":      0.620,
        "radius_m":     0.120,
        "drag_coeff":   0.54,
        "has_spin":     True,
        "colors":       ["orange"],
        "min_radius_px": 10, "max_radius_px": 80,
        "scene_width_m": 28.0,
        "classify":     _cls_basketball,
        "presets": {
            "jump_shot":  dict(V0=7.5, angle_deg=52.0, omega=3.0, t_end=1.00),
            "free_throw": dict(V0=6.8, angle_deg=55.0, omega=2.5, t_end=0.95),
            "full_court": dict(V0=14.0,angle_deg=40.0, omega=4.0, t_end=1.80),
        },
    },

    "cricket": {
        "display_name": "Cricket Ball",
        "mass_kg":      0.156,
        "radius_m":     0.036,
        "drag_coeff":   0.40,
        "has_spin":     True,
        "colors":       ["red", "brown"],
        "min_radius_px": 4, "max_radius_px": 35,
        "scene_width_m": 20.0,
        "classify":     _cls_cricket,
        "presets": {
            "fast":   dict(V0=40.0, angle_deg=1.0, omega=20.0, t_end=0.50),
            "spin":   dict(V0=22.0, angle_deg=2.0, omega=40.0, t_end=0.70),
            "medium": dict(V0=30.0, angle_deg=1.5, omega=10.0, t_end=0.55),
        },
    },

    "baseball": {
        "display_name": "Baseball",
        "mass_kg":      0.145,
        "radius_m":     0.037,
        "drag_coeff":   0.35,
        "has_spin":     True,
        "colors":       ["white"],
        "min_radius_px": 4, "max_radius_px": 35,
        "scene_width_m": 18.5,
        "classify":     _cls_baseball,
        "presets": {
            "fastball":  dict(V0=42.0, angle_deg=1.5,  omega=20.0,  t_end=0.45),
            "curveball": dict(V0=32.0, angle_deg=3.0,  omega=-60.0, t_end=0.52),
            "changeup":  dict(V0=26.0, angle_deg=2.0,  omega=15.0,  t_end=0.58),
        },
    },

    "volleyball": {
        "display_name": "Volleyball",
        "mass_kg":      0.270,
        "radius_m":     0.105,
        "drag_coeff":   0.50,
        "has_spin":     True,
        "colors":       ["white", "yellow"],
        "min_radius_px": 8, "max_radius_px": 60,
        "scene_width_m": 18.0,
        "classify":     _cls_volleyball,
        "presets": {
            "spike":       dict(V0=22.0, angle_deg=-20.0, omega=15.0, t_end=0.40),
            "float_serve": dict(V0=14.0, angle_deg=10.0,  omega=0.5,  t_end=0.80),
            "rally":       dict(V0=10.0, angle_deg=20.0,  omega=8.0,  t_end=0.90),
        },
    },

    "mortar": {
        "display_name": "Mortar / Artillery Shell",
        "mass_kg":      4.2,
        "radius_m":     0.040,
        "drag_coeff":   0.30,
        "has_spin":     True,
        "colors":       ["black", "brown"],
        "min_radius_px": 2, "max_radius_px": 20,
        "scene_width_m": 500.0,
        "classify":     _cls_mortar,
        "presets": {
            "high_velocity": dict(V0=250.0, angle_deg=45.0, omega=5.0, t_end=35.0),
            "medium":        dict(V0=150.0, angle_deg=60.0, omega=3.0, t_end=28.0),
            "low":           dict(V0=80.0,  angle_deg=70.0, omega=1.0, t_end=18.0),
        },
    },

    "generic": {
        "display_name": "Generic / Custom Object",
        "mass_kg":      0.100,
        "radius_m":     0.030,
        "drag_coeff":   0.47,
        "has_spin":     False,
        "colors":       ["white", "orange", "yellow"],
        "min_radius_px": 3, "max_radius_px": 80,
        "scene_width_m": 30.0,
        "classify":     _cls_generic,
        "presets": {
            "high_angle": dict(V0=20.0, angle_deg=45.0, omega=0.0, t_end=2.50),
            "low_fast":   dict(V0=35.0, angle_deg=15.0, omega=0.0, t_end=1.20),
            "lob":        dict(V0=12.0, angle_deg=60.0, omega=0.0, t_end=2.00),
        },
    },
}


def get_object(name):
    key = name.lower().replace(" ", "_").replace("-", "_")
    if key not in OBJECTS:
        raise ValueError(f"Unknown object '{name}'. "
                         f"Available: {', '.join(OBJECTS.keys())}")
    return OBJECTS[key]


def list_objects():
    print("\n  Available object types:")
    print("  " + "-"*50)
    for k, v in OBJECTS.items():
        presets = ", ".join(v["presets"].keys())
        print(f"  {k:<16} → {v['display_name']}")
        print(f"  {'':16}   presets : {presets}")
        print(f"  {'':16}   mass    : {v['mass_kg']*1000:.0f} g  |  "
              f"radius: {v['radius_m']*100:.1f} cm  |  "
              f"Cd: {v['drag_coeff']}")
    print()
