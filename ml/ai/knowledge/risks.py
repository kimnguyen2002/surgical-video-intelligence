"""
Phase-indexed anatomical risk map.

The Anatomy Identification Agent answers "what is at risk *right now*". That
question is phase-dependent: the common bile duct is the structure that matters
during hepatocystic dissection and largely irrelevant during port placement, so
a flat per-specialty list of structures would bury the one thing worth
surfacing.

Each entry records the structure, why it is at risk during this phase, and the
manoeuvre that conventionally protects it. Severity drives the colour of the
overlay tile:

``critical``  injury here is typically the reported major complication
``caution``   injury causes meaningful morbidity or conversion
``watch``     worth keeping oriented to, rarely injured

This is **educational reference content** describing what is commonly taught.
It is not a clinical protocol, it is not exhaustive, institutional practice
varies, and it is derived from a static map rather than from anything the
model sees in the frame. The agent labels it as such in every response.
"""

from __future__ import annotations

from typing import Optional


def _risk(structure: str, severity: str, why: str, protect: str) -> dict:
    return {"structure": structure, "severity": severity, "why": why, "protect": protect}


# (specialty, phase) -> list of at-risk structures
DANGER_ZONES: dict[str, dict[str, list[dict]]] = {
    "General Surgery": {
        "Preparation": [
            _risk("small bowel", "caution",
                  "Blind Veress or first-trocar entry can perforate adherent bowel.",
                  "Open (Hasson) entry in a previously operated abdomen."),
            _risk("aorta / iliac vessels", "critical",
                  "Retroperitoneal great vessels sit close to the umbilicus in thin patients.",
                  "45° entry angle toward the pelvis; confirm insufflation pressures early."),
        ],
        "Exposure": [
            _risk("liver capsule", "watch",
                  "Fundal retraction can tear the capsule and obscure the field with bleeding.",
                  "Grasp the gallbladder, not the liver edge."),
        ],
        "Dissection": [
            _risk("common bile duct", "critical",
                  "The classic injury: the CBD is mistaken for the cystic duct and divided.",
                  "Stay high on the gallbladder; dissect only within the hepatocystic triangle."),
            _risk("right hepatic artery", "critical",
                  "Runs immediately behind the triangle and can be taken with the cystic artery.",
                  "Identify the artery entering the gallbladder wall before clipping."),
            _risk("cystic artery", "caution",
                  "Retraction avulses it; bleeding then obscures the very field being dissected.",
                  "Control with clips, never blind cautery near the triangle."),
            _risk("duodenum", "caution",
                  "Adherent in inflammation; thermal spread causes a delayed perforation.",
                  "Sharp dissection over energy when tissue planes are inflamed."),
        ],
        "Critical View of Safety": [
            _risk("common bile duct", "critical",
                  "This phase exists specifically to exclude CBD misidentification.",
                  "Two and only two structures entering the gallbladder, lower third off the cystic plate."),
            _risk("aberrant right posterior duct", "critical",
                  "A low-inserting sectoral duct can look exactly like the cystic duct.",
                  "If a third tubular structure appears, stop and image."),
        ],
        "Clipping": [
            _risk("common bile duct", "critical",
                  "A clip placed across the CBD is often not recognised intraoperatively.",
                  "Clip only after the critical view is documented."),
        ],
        "Gallbladder Separation": [
            _risk("liver bed vessels", "caution",
                  "Middle hepatic vein branches lie just deep to the cystic plate.",
                  "Stay in the plane on the gallbladder side of the cystic plate."),
        ],
        "Hemostasis": [
            _risk("cystic artery stump", "caution",
                  "A retracted stump can bleed after pneumoperitoneum is released.",
                  "Inspect at reduced insufflation pressure before closing."),
        ],
    },
    "Cardiac Surgery": {
        "Sternotomy": [
            _risk("right ventricle", "critical",
                  "Adherent to the sternum in a redo; the saw enters it directly.",
                  "Oscillating saw with elevation, or peripheral cannulation first in a redo."),
            _risk("innominate vein", "caution",
                  "Sits immediately behind the manubrium.",
                  "Control saw depth at the upper sternum."),
        ],
        "Conduit Harvest": [
            _risk("internal mammary artery", "critical",
                  "Intimal injury or skeletonisation trauma compromises the best graft available.",
                  "Handle the pedicle, never the artery; low-power cautery."),
            _risk("pleura", "watch",
                  "Entry causes a post-operative effusion and pain.",
                  "Stay medial to the pleural reflection where possible."),
        ],
        "Cannulation": [
            _risk("ascending aorta", "critical",
                  "Cannulation of a calcified aorta causes dissection or embolic stroke.",
                  "Epiaortic scanning; cannulate a soft segment."),
            _risk("right atrium", "caution",
                  "Tears at the purse-string bleed briskly.",
                  "Adequate purse-strings before the atriotomy."),
        ],
        "Anastomosis": [
            _risk("coronary artery posterior wall", "critical",
                  "A back-wall bite occludes the vessel the graft was meant to supply.",
                  "Visualise the lumen on every bite."),
        ],
        "Hemostasis": [
            _risk("graft anastomoses", "critical",
                  "Bleeding here after chest closure requires reopening.",
                  "Inspect at systemic pressure before closing."),
        ],
    },
    "Neurosurgery": {
        "Craniotomy": [
            _risk("superior sagittal sinus", "critical",
                  "A burr hole over the midline sinus causes torrential venous bleeding and air embolism.",
                  "Plan burr holes off the midline; navigation before drilling."),
            _risk("dura mater", "caution",
                  "Stripping tears bridging veins.",
                  "Dural elevation under direct vision at each burr hole."),
        ],
        "Dural Opening": [
            _risk("cortical bridging veins", "critical",
                  "Avulsion causes venous infarction of the territory drained.",
                  "Open the dura in a flap based away from major veins."),
        ],
        "Microsurgical Dissection": [
            _risk("middle cerebral artery branches", "critical",
                  "Perforators taken during arachnoid dissection cause a dense deficit.",
                  "Sharp arachnoid dissection; preserve every perforator."),
            _risk("eloquent cortex", "critical",
                  "Retraction or resection here converts a survivable operation into a disabling one.",
                  "Mapping and intraoperative monitoring guide the resection limit."),
        ],
        "Resection": [
            _risk("tumour-brain interface", "caution",
                  "Aggressive resection past the gliotic plane injures functional tissue.",
                  "Ultrasonic aspirator at the interface, monitoring for change."),
        ],
        "Hemostasis": [
            _risk("resection cavity wall", "critical",
                  "A post-operative haematoma in a closed cavity is rapidly fatal.",
                  "Valsalva to systolic pressure before closing."),
        ],
    },
    "Thoracic Surgery": {
        "Hilar Dissection": [
            _risk("pulmonary artery", "critical",
                  "The thin-walled PA tears easily and bleeds beyond thoracoscopic control.",
                  "Dissect the artery last where possible; keep a sponge stick available."),
            _risk("phrenic nerve", "critical",
                  "Runs on the pericardium anterior to the hilum; injury causes hemidiaphragm paralysis.",
                  "Identify and keep it on the pericardium during anterior dissection."),
            _risk("recurrent laryngeal nerve", "caution",
                  "At risk in left-sided and subaortic nodal dissection.",
                  "Avoid energy in the aortopulmonary window."),
        ],
        "Bronchial Division": [
            _risk("bronchial stump", "critical",
                  "A devascularised or long stump becomes a bronchopleural fistula.",
                  "Minimal peribronchial stripping; short stump."),
        ],
        "Lymphadenectomy": [
            _risk("thoracic duct", "caution",
                  "Injury causes chylothorax, often recognised only post-operatively.",
                  "Clip rather than divide tissue in the lower posterior mediastinum."),
        ],
    },
    "Vascular Surgery": {
        "Exposure": [
            _risk("hypoglossal nerve", "caution",
                  "Crosses the distal carotid; injury causes tongue deviation and dysphagia.",
                  "Identify before dividing the digastric or ansa."),
            _risk("vagus nerve", "caution",
                  "Lies posterolateral in the carotid sheath and is clamped with the artery.",
                  "Open the sheath and identify before applying clamps."),
        ],
        "Proximal Control": [
            _risk("plaque at the clamp site", "critical",
                  "Clamping across plaque fractures it and embolises distally.",
                  "Clamp a soft segment; palpate before applying."),
        ],
        "Flow Restoration": [
            _risk("cerebral circulation", "critical",
                  "Unclamping in the wrong order directs debris into the brain.",
                  "External carotid first, so debris flushes away from the internal."),
        ],
    },
    "Gynecology": {
        "Ligament Division": [
            _risk("ureter", "critical",
                  "Runs immediately beneath the infundibulopelvic ligament at the pelvic brim.",
                  "Open the retroperitoneum and see the ureter before dividing the ligament."),
            _risk("ovarian vessels", "caution",
                  "Retract into the retroperitoneum when avulsed.",
                  "Seal before division; do not divide under tension."),
        ],
        "Vascular Control": [
            _risk("ureter", "critical",
                  "Passes under the uterine artery about 2 cm lateral to the cervix — 'water under the bridge'.",
                  "Skeletonise the artery at the level of the cervix and confirm the ureter is lateral."),
            _risk("uterine artery", "caution",
                  "A retracted stump bleeds into the broad ligament.",
                  "Secure at the cervix, not laterally."),
        ],
        "Colpotomy": [
            _risk("bladder", "critical",
                  "The bladder is adherent anteriorly, particularly after caesarean section.",
                  "Develop the vesicouterine plane sharply before entering."),
            _risk("rectum", "caution",
                  "At risk posteriorly with endometriosis or adhesions.",
                  "Identify the rectovaginal plane before the posterior cut."),
        ],
        "Cuff Closure": [
            _risk("ureter", "caution",
                  "A lateral suture can kink or ligate the distal ureter.",
                  "Full-thickness bites confined to the cuff; consider cystoscopy."),
        ],
    },
    "Urology": {
        "Bladder Neck Dissection": [
            _risk("ureteric orifices", "critical",
                  "A posterior bladder-neck cut too close obstructs both ureters.",
                  "Identify the orifices before completing the posterior division."),
        ],
        "Pedicle Control": [
            _risk("neurovascular bundle", "critical",
                  "Thermal energy on the pedicle destroys the nerves controlling continence and potency.",
                  "Athermal control with clips; no cautery near the bundle."),
        ],
        "Nerve Sparing": [
            _risk("neurovascular bundle", "critical",
                  "Traction alone causes neuropraxia even without division.",
                  "Countertraction on the prostate, not on the bundle."),
        ],
        "Apical Dissection": [
            _risk("external urethral sphincter", "critical",
                  "Apical over-dissection is the dominant cause of persistent incontinence.",
                  "Divide the urethra sharply at the apex; preserve maximal length."),
            _risk("dorsal venous complex", "caution",
                  "Bleeds heavily and obscures the apex.",
                  "Secure the DVC before apical division."),
        ],
        "Anastomosis": [
            _risk("posterior urethral wall", "caution",
                  "A back-wall bite narrows the anastomosis and causes a stricture.",
                  "Visualise the lumen on posterior sutures."),
        ],
    },
    "Robotic Surgery (SurgVU)": {
        "Suturing": [
            _risk("needle control", "caution",
                  "A lost needle in a closed cavity requires a formal search.",
                  "Needle always in view or under the driver's control."),
            _risk("tissue tearing", "watch",
                  "Robotic wrists apply force without haptic feedback.",
                  "Judge tension visually; the robot will not tell you it is tearing."),
        ],
        "Uterine Horn": [
            _risk("ureter", "critical",
                  "Adjacent to the horn and the suspensory ligament.",
                  "Identify before applying energy."),
        ],
        "Rectal Artery/Vein": [
            _risk("rectal wall", "critical",
                  "Thermal injury during vessel dissection presents late as a leak.",
                  "Sharp dissection close to the vessel, away from the wall."),
        ],
        "Retraction and Collision Avoidance": [
            _risk("out-of-view instruments", "critical",
                  "A robotic arm outside the endoscopic view can injure tissue unobserved.",
                  "Move only instruments you can see; re-establish view before repositioning."),
        ],
    },
}

#: Fallback when a specialty has no phase-specific map.
GENERIC_RISKS: list[dict] = [
    _risk("out-of-view instruments", "critical",
          "Tissue injury by an instrument outside the visual field is unobserved and unrecorded.",
          "Move only instruments you can see."),
    _risk("adjacent vasculature", "caution",
          "Major vessels run close to most dissection planes.",
          "Identify and preserve before applying energy."),
    _risk("thermal spread", "caution",
          "Energy devices injure tissue several millimetres beyond the visible effect.",
          "Keep active electrodes clear of structures you intend to preserve."),
]

SEVERITY_ORDER = {"critical": 0, "caution": 1, "watch": 2}


def risks_for(specialty: str, phase: Optional[str]) -> list[dict]:
    """
    At-risk structures for a specialty and phase, most severe first.

    Falls back to the specialty's most severe entries when the phase is
    unknown, and to a generic set when the specialty has no map — an empty
    answer here would read as "nothing is at risk", which is never true.
    """
    specialty_map = DANGER_ZONES.get(specialty)
    if not specialty_map:
        return list(GENERIC_RISKS)

    if phase:
        for name, risks in specialty_map.items():
            if name.lower() == phase.lower():
                return sorted(risks, key=lambda r: SEVERITY_ORDER.get(r["severity"], 3))

    # No phase, or a phase with no specific entry: surface this specialty's
    # critical items so the answer is still specialty-appropriate.
    pooled: list[dict] = []
    seen: set[str] = set()
    for risks in specialty_map.values():
        for risk in risks:
            if risk["severity"] == "critical" and risk["structure"] not in seen:
                pooled.append(risk)
                seen.add(risk["structure"])
    return pooled[:5] or list(GENERIC_RISKS)


def phases_with_risks(specialty: str) -> list[str]:
    return list(DANGER_ZONES.get(specialty, {}))


def coverage() -> dict:
    """How much of the risk map is populated — shown in the console."""
    return {
        "specialties": len(DANGER_ZONES),
        "phases": sum(len(v) for v in DANGER_ZONES.values()),
        "entries": sum(len(r) for v in DANGER_ZONES.values() for r in v.values()),
    }
