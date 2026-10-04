"""
Specialty ontologies.

Each supported specialty carries its own structured description of the domain:
the operative phases and the order they normally follow, the anatomy in play,
the instruments used, the events worth flagging, and the concepts a learner is
expected to pick up. Selecting a specialty in the UI reconfigures terminology,
timeline vocabulary, retrieval filters, and the assistant's framing from this
data — there is no single generic model pretending to cover every operation.

These are *educational* reference structures describing how procedures are
commonly taught. They are not clinical protocols, they are not exhaustive, and
institutional practice varies. Nothing here should be read as guidance for a
real operation.
"""

from __future__ import annotations

from typing import Optional


def _phase(name: str, description: str, landmarks: list[str] | None = None) -> dict:
    return {"name": name, "description": description, "landmarks": landmarks or []}


ONTOLOGIES: dict[str, dict] = {
    "General Surgery": {
        "description": (
            "Abdominal and gastrointestinal procedures, most commonly performed "
            "laparoscopically. Laparoscopic cholecystectomy is the canonical "
            "teaching case."
        ),
        "phases": [
            _phase("Preparation", "Port placement, insufflation, and initial survey of the abdomen.", ["umbilicus", "peritoneum"]),
            _phase("Exposure", "Retraction of the gallbladder fundus to open the operative field.", ["gallbladder fundus", "liver edge"]),
            _phase("Dissection", "Clearing fat and fibrous tissue from the hepatocystic triangle.", ["hepatocystic triangle", "cystic plate"]),
            _phase("Critical View of Safety", "Confirming exactly two structures enter the gallbladder before any division.", ["cystic duct", "cystic artery"]),
            _phase("Clipping", "Securing the cystic duct and cystic artery with clips.", ["cystic duct", "cystic artery"]),
            _phase("Division", "Dividing the clipped structures.", ["cystic duct"]),
            _phase("Gallbladder Separation", "Separating the gallbladder from the liver bed.", ["liver bed", "cystic plate"]),
            _phase("Hemostasis", "Controlling bleeding and inspecting the operative field.", ["liver bed"]),
            _phase("Extraction", "Removing the specimen in a retrieval bag.", []),
            _phase("Closure", "Desufflation, port removal, and fascial closure.", ["port sites"]),
        ],
        "anatomy": [
            "gallbladder", "cystic duct", "cystic artery", "common bile duct",
            "common hepatic duct", "hepatocystic triangle", "cystic plate",
            "liver bed", "duodenum", "omentum", "peritoneum", "falciform ligament",
        ],
        "instruments": [
            "monopolar curved scissors", "maryland dissector", "bipolar forceps",
            "clip applier", "grasping forceps", "laparoscope", "suction irrigator",
            "specimen retrieval bag", "needle driver",
        ],
        "events": ["bleeding", "bile spillage", "smoke from electrocautery", "clip misfire", "adhesiolysis"],
        "concepts": [
            "Critical View of Safety reduces bile duct injury by requiring positive identification, not inference.",
            "Bile duct injury most often follows misidentification, not technical failure of an instrument.",
            "Electrocautery near the hepatocystic triangle risks thermal injury to the bile duct.",
        ],
    },
    "Cardiac Surgery": {
        "description": "Open and minimally invasive procedures on the heart and great vessels.",
        "phases": [
            _phase("Positioning", "Patient positioning, prep, and drape.", []),
            _phase("Sternotomy", "Median sternotomy and sternal retraction.", ["sternum", "manubrium"]),
            _phase("Conduit Harvest", "Harvesting the internal mammary artery or saphenous vein.", ["internal mammary artery", "saphenous vein"]),
            _phase("Pericardiotomy", "Opening the pericardium to expose the heart.", ["pericardium"]),
            _phase("Cannulation", "Aortic and venous cannulation for cardiopulmonary bypass.", ["ascending aorta", "right atrium"]),
            _phase("Cardioplegia", "Aortic cross-clamp and delivery of cardioplegic arrest.", ["aortic root", "coronary sinus"]),
            _phase("Anastomosis", "Constructing distal and proximal graft anastomoses.", ["left anterior descending", "aorta"]),
            _phase("Weaning", "Rewarming, de-airing, and separation from bypass.", []),
            _phase("Decannulation", "Removing cannulae and reversing anticoagulation.", []),
            _phase("Hemostasis", "Securing suture lines and inspecting for bleeding.", []),
            _phase("Closure", "Sternal wiring and layered closure.", ["sternum"]),
        ],
        "anatomy": [
            "ascending aorta", "aortic root", "right atrium", "left ventricle",
            "left anterior descending artery", "circumflex artery",
            "right coronary artery", "internal mammary artery", "saphenous vein",
            "coronary sinus", "pericardium", "pulmonary artery",
        ],
        "instruments": [
            "sternal saw", "sternal retractor", "vascular clamp", "needle driver",
            "DeBakey forceps", "coronary scissors", "cannula", "suction",
            "electrocautery", "internal mammary retractor",
        ],
        "events": ["bleeding", "cross-clamp application", "bypass initiation", "de-airing", "graft flow check"],
        "concepts": [
            "Cross-clamp and bypass times are primary determinants of myocardial recovery.",
            "The internal mammary artery to LAD graft has the best long-term patency of any conduit.",
            "De-airing before cross-clamp removal reduces the risk of air embolism.",
        ],
    },
    "Neurosurgery": {
        "description": "Cranial and spinal procedures, typically under an operating microscope.",
        "phases": [
            _phase("Positioning", "Head fixation, neuronavigation registration, and planning.", ["skull landmarks"]),
            _phase("Craniotomy", "Skin flap, burr holes, and bone flap elevation.", ["skull", "burr holes"]),
            _phase("Dural Opening", "Opening the dura and tacking it back.", ["dura mater"]),
            _phase("Brain Exposure", "Cortical exposure and identification of the approach corridor.", ["cortex", "sylvian fissure"]),
            _phase("Microsurgical Dissection", "Microscope-assisted arachnoid and tissue dissection.", ["arachnoid", "cerebral vessels"]),
            _phase("Resection", "Lesion or tumour resection.", ["tumour margin"]),
            _phase("Hemostasis", "Bipolar coagulation and haemostatic agents.", []),
            _phase("Dural Closure", "Watertight dural closure.", ["dura mater"]),
            _phase("Bone Flap Replacement", "Replacing and fixing the bone flap.", ["bone flap"]),
            _phase("Closure", "Galea and skin closure.", ["scalp"]),
        ],
        "anatomy": [
            "cortex", "dura mater", "arachnoid", "falx cerebri", "sylvian fissure",
            "middle cerebral artery", "anterior cerebral artery", "cranial nerves",
            "ventricles", "corpus callosum", "brainstem", "tentorium",
        ],
        "instruments": [
            "bipolar forceps", "microscissors", "suction", "ultrasonic aspirator",
            "craniotome", "high-speed drill", "brain retractor", "dural hook",
            "microdissector", "needle driver",
        ],
        "events": ["bleeding", "brain relaxation", "CSF release", "neuromonitoring change", "tumour margin reached"],
        "concepts": [
            "Brain relaxation from CSF release widens the corridor and reduces retraction injury.",
            "Retraction pressure and duration correlate with post-operative deficit.",
            "Eloquent cortex mapping guides how aggressive a resection can safely be.",
        ],
    },
    "Thoracic Surgery": {
        "description": "Lung, pleural, and mediastinal procedures, increasingly by VATS.",
        "phases": [
            _phase("Positioning", "Lateral decubitus positioning and single-lung ventilation.", []),
            _phase("Port Placement", "Placing VATS ports or performing thoracotomy.", ["intercostal space"]),
            _phase("Exploration", "Inspecting the pleural space and lung.", ["pleura", "lung lobes"]),
            _phase("Hilar Dissection", "Dissecting hilar vessels and bronchus.", ["pulmonary artery", "pulmonary vein", "bronchus"]),
            _phase("Vessel Division", "Stapling and dividing vessels.", ["pulmonary artery"]),
            _phase("Bronchial Division", "Stapling and dividing the bronchus.", ["bronchus"]),
            _phase("Lymphadenectomy", "Nodal sampling or dissection.", ["mediastinal nodes"]),
            _phase("Specimen Extraction", "Removing the specimen.", []),
            _phase("Air Leak Test", "Testing the bronchial stump and parenchyma.", []),
            _phase("Closure", "Chest drain placement and closure.", ["intercostal space"]),
        ],
        "anatomy": [
            "lung lobes", "pulmonary artery", "pulmonary vein", "bronchus",
            "pleura", "mediastinum", "phrenic nerve", "vagus nerve",
            "azygos vein", "thoracic duct", "diaphragm",
        ],
        "instruments": [
            "endoscopic stapler", "thoracoscope", "ring forceps", "harmonic scalpel",
            "lung retractor", "suction", "clip applier", "chest drain",
        ],
        "events": ["air leak", "bleeding", "fissure dissection", "nodal sampling"],
        "concepts": [
            "Vein-first versus artery-first sequencing is a deliberate choice with oncologic rationale.",
            "An incomplete fissure raises the risk of prolonged post-operative air leak.",
        ],
    },
    "Vascular Surgery": {
        "description": "Open and endovascular procedures on arteries and veins.",
        "phases": [
            _phase("Exposure", "Exposing the target vessel.", ["target artery"]),
            _phase("Proximal Control", "Obtaining proximal vascular control.", ["proximal artery"]),
            _phase("Distal Control", "Obtaining distal vascular control.", ["distal artery"]),
            _phase("Heparinisation", "Systemic anticoagulation before clamping.", []),
            _phase("Arteriotomy", "Opening the vessel.", ["arterial wall"]),
            _phase("Endarterectomy or Bypass", "Plaque removal or graft anastomosis.", ["plaque", "graft"]),
            _phase("Closure of Arteriotomy", "Patch or primary closure.", ["arterial wall"]),
            _phase("Flow Restoration", "Unclamping in a controlled sequence.", []),
            _phase("Hemostasis", "Securing suture lines.", []),
            _phase("Closure", "Layered wound closure.", []),
        ],
        "anatomy": [
            "carotid artery", "femoral artery", "popliteal artery", "aorta",
            "iliac artery", "great saphenous vein", "vagus nerve", "hypoglossal nerve",
        ],
        "instruments": [
            "vascular clamp", "Potts scissors", "DeBakey forceps", "needle driver",
            "shunt", "Fogarty catheter", "Doppler probe", "patch graft",
        ],
        "events": ["clamping", "unclamping", "bleeding", "shunt placement", "flow check"],
        "concepts": [
            "Proximal control before distal control is the rule that keeps a bleed manageable.",
            "Unclamping order is chosen to direct any debris away from the brain.",
        ],
    },
    "Orthopedic Surgery": {
        "description": "Procedures on bone, joints, and soft tissue, open and arthroscopic.",
        "phases": [
            _phase("Positioning", "Positioning, traction, and tourniquet application.", []),
            _phase("Approach", "Skin incision and interval development.", ["muscle interval"]),
            _phase("Exposure", "Exposing the joint or fracture.", ["joint capsule", "fracture site"]),
            _phase("Reduction", "Reducing the fracture or preparing the joint.", ["fracture fragments"]),
            _phase("Fixation or Implantation", "Placing implants, screws, or prosthesis.", ["bone cortex"]),
            _phase("Verification", "Imaging and range-of-motion checks.", []),
            _phase("Irrigation", "Washout of debris.", []),
            _phase("Closure", "Layered closure and dressing.", []),
        ],
        "anatomy": [
            "femur", "tibia", "humerus", "acetabulum", "rotator cuff",
            "meniscus", "cruciate ligaments", "joint capsule", "periosteum",
        ],
        "instruments": [
            "oscillating saw", "drill", "reamer", "bone rongeur", "osteotome",
            "arthroscope", "shaver", "screwdriver", "retractor", "mallet",
        ],
        "events": ["reduction achieved", "implant seating", "fluoroscopy check", "bleeding"],
        "concepts": [
            "Anatomic reduction quality predicts functional outcome more than implant choice.",
            "Thermal necrosis from unirrigated drilling compromises screw purchase.",
        ],
    },
    "Plastic Surgery": {
        "description": "Reconstructive and microsurgical procedures on soft tissue.",
        "phases": [
            _phase("Marking", "Pre-operative marking and planning.", []),
            _phase("Incision", "Incision along planned lines.", []),
            _phase("Flap Elevation", "Raising the flap and identifying its pedicle.", ["vascular pedicle"]),
            _phase("Pedicle Dissection", "Skeletonising the pedicle.", ["artery", "vein"]),
            _phase("Recipient Preparation", "Preparing recipient vessels.", ["recipient artery"]),
            _phase("Microanastomosis", "Arterial and venous anastomosis under the microscope.", ["anastomosis"]),
            _phase("Perfusion Check", "Confirming flap perfusion.", ["flap"]),
            _phase("Inset", "Positioning and securing the flap.", []),
            _phase("Closure", "Layered closure and drain placement.", []),
        ],
        "anatomy": [
            "vascular pedicle", "perforator", "fascia", "subcutaneous tissue",
            "dermis", "recipient artery", "recipient vein",
        ],
        "instruments": [
            "microscissors", "jeweller's forceps", "microvascular clamp",
            "needle driver", "bipolar forceps", "operating microscope", "doppler probe",
        ],
        "events": ["pedicle identified", "anastomosis complete", "perfusion confirmed", "venous congestion"],
        "concepts": [
            "Venous congestion, not arterial insufficiency, is the more common early flap failure.",
            "Pedicle geometry matters: a kinked or twisted pedicle fails a technically perfect anastomosis.",
        ],
    },
    "Ophthalmology": {
        "description": "Microsurgical procedures on the eye, most commonly cataract extraction.",
        "phases": [
            _phase("Incision", "Clear corneal or limbal incision.", ["cornea", "limbus"]),
            _phase("Capsulorhexis", "Creating a continuous curvilinear capsulotomy.", ["anterior capsule"]),
            _phase("Hydrodissection", "Separating the nucleus from the cortex.", ["lens cortex"]),
            _phase("Phacoemulsification", "Ultrasonic emulsification of the nucleus.", ["lens nucleus"]),
            _phase("Cortical Removal", "Irrigation and aspiration of residual cortex.", ["lens cortex"]),
            _phase("Lens Implantation", "Inserting the intraocular lens into the capsular bag.", ["capsular bag"]),
            _phase("Wound Closure", "Hydrating the wound and confirming a seal.", ["cornea"]),
        ],
        "anatomy": [
            "cornea", "limbus", "anterior chamber", "iris", "anterior capsule",
            "lens nucleus", "lens cortex", "capsular bag", "zonules", "retina",
        ],
        "instruments": [
            "keratome", "capsulorhexis forceps", "phaco handpiece",
            "irrigation aspiration probe", "IOL injector", "chopper", "viscoelastic cannula",
        ],
        "events": ["capsule tear", "zonular stress", "anterior chamber collapse", "IOL centration"],
        "concepts": [
            "A continuous, well-centred rhexis is what makes every later step controllable.",
            "Posterior capsule rupture changes the entire remaining plan of the operation.",
        ],
    },
    "ENT": {
        "description": "Otolaryngology — head and neck, endoscopic sinus, and otologic procedures.",
        "phases": [
            _phase("Endoscopic Survey", "Inspecting the nasal cavity or airway.", ["nasal cavity"]),
            _phase("Access", "Creating the surgical corridor.", ["middle meatus"]),
            _phase("Dissection", "Dissecting the target region.", ["ethmoid air cells"]),
            _phase("Resection", "Removing diseased tissue.", []),
            _phase("Hemostasis", "Controlling bleeding.", []),
            _phase("Packing or Closure", "Packing, stenting, or closure.", []),
        ],
        "anatomy": [
            "nasal septum", "middle turbinate", "maxillary sinus", "ethmoid air cells",
            "sphenoid sinus", "lamina papyracea", "skull base", "facial nerve",
            "tympanic membrane", "ossicles", "vocal cords",
        ],
        "instruments": [
            "endoscope", "microdebrider", "through-cutting forceps", "sickle knife",
            "suction elevator", "bipolar forceps", "drill",
        ],
        "events": ["bleeding", "orbital fat exposure", "CSF leak", "packing placement"],
        "concepts": [
            "The lamina papyracea and skull base are the boundaries that define safe dissection.",
            "Clear visualisation depends on haemostasis; blind dissection in a bloody field is how injuries happen.",
        ],
    },
    "Gynecology": {
        "description": (
            "Procedures on the female reproductive tract, frequently robotic or "
            "laparoscopic. The SurgVU dataset's porcine tasks are modelled on "
            "this domain."
        ),
        "phases": [
            _phase("Port Placement", "Docking and port placement.", ["abdominal wall"]),
            _phase("Exploration", "Surveying the pelvis.", ["pelvic peritoneum"]),
            _phase("Uterine Manipulation", "Positioning the uterus for exposure.", ["uterus", "uterine horn"]),
            _phase("Ligament Division", "Dividing round and suspensory ligaments.", ["suspensory ligament", "round ligament"]),
            _phase("Vascular Control", "Securing the uterine vessels.", ["uterine artery", "ureter"]),
            _phase("Colpotomy", "Circumferential vaginal incision.", ["vaginal cuff"]),
            _phase("Specimen Removal", "Extracting the specimen.", []),
            _phase("Cuff Closure", "Suturing the vaginal cuff.", ["vaginal cuff"]),
            _phase("Hemostasis", "Final inspection and haemostasis.", []),
        ],
        "anatomy": [
            "uterus", "uterine horn", "fallopian tube", "ovary", "round ligament",
            "suspensory ligament", "uterine artery", "ureter", "rectum",
            "bladder", "vaginal cuff", "rectal artery", "rectal vein",
        ],
        "instruments": [
            "needle driver", "cadiere forceps", "prograsp forceps",
            "monopolar curved scissors", "bipolar forceps", "vessel sealer",
            "tip-up fenestrated grasper", "clip applier", "suction irrigator",
        ],
        "events": ["bleeding", "ureter identification", "colpotomy entry", "suturing", "smoke from electrocautery"],
        "concepts": [
            "The ureter runs immediately beneath the uterine artery — identifying it before division is the safety step.",
            "Vaginal cuff closure technique is the main determinant of dehiscence risk.",
        ],
    },
    "Urology": {
        "description": "Procedures on the urinary tract and male reproductive organs, often robotic.",
        "phases": [
            _phase("Port Placement", "Docking and port placement.", ["abdominal wall"]),
            _phase("Exposure", "Developing the retropubic or retroperitoneal space.", ["Retzius space"]),
            _phase("Bladder Neck Dissection", "Separating the bladder neck from the prostate.", ["bladder neck", "prostate"]),
            _phase("Pedicle Control", "Controlling the vascular pedicles.", ["prostatic pedicle"]),
            _phase("Nerve Sparing", "Preserving the neurovascular bundles.", ["neurovascular bundle"]),
            _phase("Apical Dissection", "Dissecting the apex and dividing the urethra.", ["urethra", "prostate apex"]),
            _phase("Lymphadenectomy", "Pelvic node dissection when indicated.", ["obturator nodes"]),
            _phase("Anastomosis", "Vesicourethral anastomosis.", ["urethra", "bladder neck"]),
            _phase("Leak Test", "Testing the anastomosis.", []),
            _phase("Closure", "Specimen removal and closure.", []),
        ],
        "anatomy": [
            "prostate", "bladder", "bladder neck", "urethra", "ureter",
            "seminal vesicles", "neurovascular bundle", "obturator nerve",
            "kidney", "renal artery", "renal vein", "Retzius space",
        ],
        "instruments": [
            "needle driver", "prograsp forceps", "monopolar curved scissors",
            "bipolar forceps", "vessel sealer", "clip applier", "suction irrigator",
            "grasping retractor",
        ],
        "events": ["bleeding", "nerve bundle release", "anastomosis suturing", "leak test", "clip application"],
        "concepts": [
            "Athermal dissection of the neurovascular bundles preserves post-operative function.",
            "A watertight vesicourethral anastomosis determines how early the catheter can come out.",
        ],
    },
    "Robotic Surgery (SurgVU)": {
        "description": (
            "The task vocabulary of the SurgVU dataset (arXiv:2501.09209v1): "
            "da Vinci robotic training and porcine-lab exercises. This is the "
            "specialty to select when reviewing SurgVU footage or a model "
            "trained on it, because the labels match exactly."
        ),
        "phases": [
            _phase("Suturing", "Needle driving, tissue approximation, and knot tying.", ["tissue edges"]),
            _phase("Uterine Horn", "Dissection and manipulation of the uterine horn.", ["uterine horn"]),
            _phase("Rectal Artery/Vein", "Dissection around the rectal vessels.", ["rectal artery", "rectal vein"]),
            _phase("Suspensory Ligaments", "Division of the suspensory ligaments.", ["suspensory ligament"]),
            _phase("Skills Application", "General skills exercises on the training model.", []),
            _phase("Retraction and Collision Avoidance", "Retraction and instrument-arm collision management.", []),
            _phase("Range of Motion", "Instrument range-of-motion exercises.", []),
            _phase("Other", "Segments that do not match a defined task label.", []),
        ],
        "anatomy": [
            "uterine horn", "rectal artery", "rectal vein", "suspensory ligament",
            "peritoneum", "bowel", "abdominal wall",
        ],
        "instruments": [
            "needle driver", "cadiere forceps", "prograsp forceps",
            "monopolar curved scissors", "bipolar forceps", "stapler",
            "force bipolar", "vessel sealer", "cautery hook", "clip applier",
            "tip-up fenestrated grasper", "grasping retractor",
        ],
        "events": ["tool exchange", "camera movement", "bleeding", "smoke from electrocautery", "collision"],
        "concepts": [
            "SurgVU tool labels are derived from robot instrument-installation logs, so a tool is labelled present while it is installed on an arm — even when it is out of the endoscopic view.",
            "That labelling makes tool presence a weak-supervision signal rather than a visual ground truth, which is exactly why the task is framed as classification rather than detection.",
            "Task boundaries in the dataset are coarse; short transitions between tasks are frequently unlabelled.",
        ],
    },
}

#: Ordered list used to populate the specialty selector.
SPECIALTIES: list[str] = list(ONTOLOGIES.keys())

_DEFAULT_KEY = "General Surgery"


def get_ontology(specialty: Optional[str]) -> dict:
    """Look up a specialty ontology, tolerating case and partial names."""
    if not specialty:
        return ONTOLOGIES[_DEFAULT_KEY]
    if specialty in ONTOLOGIES:
        return ONTOLOGIES[specialty]
    lowered = specialty.strip().lower()
    for name, ontology in ONTOLOGIES.items():
        if name.lower() == lowered:
            return ontology
    for name, ontology in ONTOLOGIES.items():
        if lowered in name.lower() or name.lower().split()[0] == lowered:
            return ontology
    return ONTOLOGIES[_DEFAULT_KEY]
