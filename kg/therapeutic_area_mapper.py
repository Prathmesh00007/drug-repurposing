"""Automated therapeutic area classification using medical ontologies."""

import logging
import httpx
from typing import Optional, Dict, List
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class TherapeuticAreaMapper:
    """Map diseases to therapeutic areas using medical ontologies."""
    
    # Ontology-based therapeutic area classification
    THERAPEUTIC_AREAS = {
    # -------------------------
    # CORE AREAS (original + expanded)
    # -------------------------
        "hematological": {
            "mesh_trees": ["C15"],  # Hemic and Lymphatic Diseases
            "efo_ancestors": ["EFO:0005803", "EFO:0000540"],  # hematological system disease (populate more as needed)
            "mondo_patterns": [
                "anemia", "blood", "hemoglobin", "iron", "hematologic", "hematological",
                "leukemia", "lymphoma", "myeloma", "thrombocytopenia", "hemophilia",
                "sickle cell", "coagulopathy", "myelodysplastic"
            ],
            "doid_branches": ["DOID:2355"],
            "subdomains": ["anemias", "coagulation disorders", "hematologic malignancies", "bone marrow failure"],
            "common_biomarkers": ["CBC (Hb, Hct, WBC, Platelets)", "reticulocyte count", "ferritin", "LDH", "BCR-ABL"],
            "example_indications": ["aplastic anemia", "acute myeloid leukemia", "immune thrombocytopenia"],
            "typical_endpoints": ["overall survival", "remission rate", "hemoglobin increase", "bleeding events"],
            "favored_formulations": ["oral", "IV", "subcutaneous (biologics)"],
            "assay_types": ["flow cytometry", "colony forming unit assays", "coagulation panels", "ELISA for cytokines"],
            "preclinical_models": ["xenograft leukemia models", "transgenic mice", "in vitro hematopoietic colony assays"],
            "repurposing_priority": "high",
            "regulatory_notes": "Opportunities for orphan designations common; safety thresholds differ for cytotoxic vs supportive agents.",
            "notes": "High unmet need in certain rare hematologic disorders — repurposing of immunomodulators and epigenetic drugs is common."
        },

        "respiratory": {
            "mesh_trees": ["C08"],
            "efo_ancestors": ["EFO:0003785", "EFO:0000684"],
            "mondo_patterns": ["respiratory", "lung", "pulmonary", "bronch", "asthma", "copd", "pneumonia", "fibrosis"],
            "doid_branches": ["DOID:3226"],
            "subdomains": ["obstructive lung disease", "interstitial lung disease", "infectious pneumonias", "pulmonary hypertension"],
            "common_biomarkers": ["FEV1", "DLCO", "oxygen saturation", "BNP (for PH)", "sputum culture", "exhaled NO"],
            "example_indications": ["asthma", "COPD", "idiopathic pulmonary fibrosis", "COVID-19 pneumonia"],
            "typical_endpoints": ["change in FEV1", "time to exacerbation", "6-minute walk distance", "mortality"],
            "favored_formulations": ["inhalation", "oral", "IV", "nebulized"],
            "assay_types": ["spirometry", "lung function tests", "histology for fibrosis", "viral/bacterial cultures"],
            "preclinical_models": ["bleomycin lung fibrosis mouse", "ovalbumin/hyperresponsiveness models", "ex vivo lung slices"],
            "repurposing_priority": "high",
            "regulatory_notes": "Delivery route (inhaled vs systemic) changes safety/regulatory footprint significantly.",
            "notes": "Respiratory agents often benefit from reformulation (inhalers, nebulizers) for targeted exposure and reduced systemic toxicity."
        },

        "cardiovascular": {
            "mesh_trees": ["C14"],
            "efo_ancestors": ["EFO:0000319"],
            "mondo_patterns": ["cardiovascular", "heart", "cardiac", "vascular", "hypertension", "arterial", "ischemia"],
            "doid_branches": ["DOID:1287"],
            "subdomains": ["ischemic heart disease", "heart failure", "arrhythmia", "hypertension", "peripheral vascular disease"],
            "common_biomarkers": ["BP", "troponin", "BNP/NT-proBNP", "LDL-C", "CRP", "ejection fraction"],
            "example_indications": ["heart failure", "angina", "atrial fibrillation", "hypertension"],
            "typical_endpoints": ["MACE composite", "hospitalization for HF", "BP reduction", "CV mortality"],
            "favored_formulations": ["oral", "transdermal", "IV for acute care"],
            "assay_types": ["echocardiography", "ECG", "hemodynamic monitoring", "biomarker assays"],
            "preclinical_models": ["pressure-overload HF models", "myocardial infarction rodent models", "isolated heart prep"],
            "repurposing_priority": "high",
            "regulatory_notes": "Cardiac safety (QT prolongation, proarrhythmia) is a major gating factor; thorough QT studies often required.",
            "notes": "High commercial value; generics and biosimilars common. Repurposing often targets symptomatic relief or comorbidity management."
        },

        "neurological": {
            "mesh_trees": ["C10"],
            "efo_ancestors": ["EFO:0000618"],
            "mondo_patterns": ["neurological", "brain", "neural", "alzheimer", "parkinson", "dementia", "epilepsy", "stroke"],
            "doid_branches": ["DOID:863"],
            "subdomains": ["neurodegenerative", "epilepsy", "stroke/ischemia", "peripheral neuropathy"],
            "common_biomarkers": ["CSF tau/amyloid", "EEG patterns", "neuroimaging markers (MRI)", "alpha-synuclein (research)"],
            "example_indications": ["Alzheimer's disease", "Parkinson's disease", "epilepsy", "multiple sclerosis"],
            "typical_endpoints": ["cognitive scales (MMSE, ADAS-Cog)", "seizure frequency", "EDSS (MS)", "motor scales"],
            "favored_formulations": ["oral", "intrathecal (rare)", "transdermal", "IV for emergencies"],
            "assay_types": ["behavioral assays", "EEG", "neuroimaging", "motor function tests"],
            "preclinical_models": ["transgenic mouse models (APP, tau, alpha-synuclein)", "6-OHDA/ MPTP PD models", "kindling epilepsy models"],
            "repurposing_priority": "medium-high",
            "regulatory_notes": "Biomarker-driven approvals increasingly possible; slow disease progression means long trials and surrogate endpoints are valuable.",
            "notes": "CNS penetration (BBB) is a critical PK filter for small molecules; peripheral immunomodulators occasionally repositioned for CNS disorders."
        },

        "metabolic": {
            "mesh_trees": ["C18"],
            "efo_ancestors": ["EFO:0000589"],
            "mondo_patterns": ["metabolic", "diabetes", "obesity", "lipid", "insulin", "thyroid"],
            "doid_branches": ["DOID:0014667"],
            "subdomains": ["diabetes mellitus (T1/T2)", "obesity", "dyslipidemia", "thyroid disorders"],
            "common_biomarkers": ["HbA1c", "fasting glucose", "insulin", "lipid panel", "HOMA-IR"],
            "example_indications": ["type 2 diabetes", "non-alcoholic fatty liver disease", "hypercholesterolemia"],
            "typical_endpoints": ["HbA1c reduction", "weight loss", "LDL reduction", "NASH fibrosis score"],
            "favored_formulations": ["oral", "injectable (GLP-1)", "transdermal"],
            "assay_types": ["glucose tolerance tests", "metabolic panels", "liver biopsy (NAFLD/NASH)"],
            "preclinical_models": ["db/db mice", "diet-induced obesity models", "hepatic steatosis rodent models"],
            "repurposing_priority": "high",
            "regulatory_notes": "Cardio-renal outcomes increasingly required for diabetes drugs; weight/ metabolic endpoints attractive in payer evaluations.",
            "notes": "Metabolic diseases are attractive for repurposing due to large patient populations; safety in chronic dosing is essential."
        },

        "oncology": {
            "mesh_trees": ["C04"],
            "efo_ancestors": ["EFO:0000311"],
            "mondo_patterns": ["cancer", "tumor", "carcinoma", "neoplasm", "metastasis", "oncogenic"],
            "doid_branches": ["DOID:162"],
            "subdomains": ["solid tumors", "hematologic malignancies", "rare cancers", "precision oncology / biomarker-defined subgroups"],
            "common_biomarkers": [
                "tumor mutational burden", "PD-L1 expression", "HER2", "EGFR mutations", "ALK fusions", "BRAF V600E"
            ],
            "example_indications": ["non-small cell lung cancer", "breast cancer (HER2+)", "melanoma (BRAF)"],
            "typical_endpoints": ["overall survival", "progression-free survival", "objective response rate"],
            "favored_formulations": ["oral targeted agents", "IV chemotherapeutics", "injectable biologics (mAbs)"],
            "assay_types": ["IHC", "NGS panels", "ctDNA assays", "tumor biopsy histology"],
            "preclinical_models": ["PDX models", "cell-line xenografts", "organoids", "syngeneic immunocompetent models"],
            "repurposing_priority": "high",
            "regulatory_notes": "Biomarker-driven label expansions common; accelerated approvals possible with surrogate endpoints.",
            "notes": "High complexity and high commercial value; combination therapy strategies are frequent in repurposing efforts."
        },

        "infectious": {
            "mesh_trees": ["C01", "C02"],
            "efo_ancestors": ["EFO:0005741"],
            "mondo_patterns": ["infection", "viral", "bacterial", "hiv", "hepatitis", "tuberculosis", "parasitic", "fungal"],
            "doid_branches": ["DOID:0050117"],
            "subdomains": ["viral infections", "bacterial infections", "tuberculosis", "parasitic diseases", "antifungal"],
            "common_biomarkers": ["viral load", "CRP", "procalcitonin", "culture positivity", "serology titers"],
            "example_indications": ["HIV", "HBV/HCV", "tuberculosis", "malaria", "COVID-19"],
            "typical_endpoints": ["viral clearance", "time to symptom resolution", "microbiological cure", "mortality"],
            "favored_formulations": ["oral", "IV", "topical (antifungal)", "inhaled (some antivirals)"],
            "assay_types": ["PCR", "culture & sensitivity", "MIC assays", "in vitro viral inhibition"],
            "preclinical_models": ["cellular viral replication assays", "mouse/ferret viral challenge models", "mycobacterial models"],
            "repurposing_priority": "high",
            "regulatory_notes": "Antimicrobial resistance (AMR) and safety in immunocompromised hosts are key gating factors.",
            "notes": "Rapid preclinical screening possible; high public health impact for successful repurposings."
        },

        "immunological": {
            "mesh_trees": ["C20"],
            "efo_ancestors": ["EFO:0000540"],
            "mondo_patterns": ["autoimmune", "inflammatory", "rheumatoid", "lupus", "psoriasis", "immune complex", "cytokine storm"],
            "doid_branches": ["DOID:2914"],
            "subdomains": ["autoimmune diseases", "immune deficiency", "inflammatory conditions", "cytokine release syndromes"],
            "common_biomarkers": ["CRP", "ESR", "autoantibodies (ANA, RF)", "cytokine panels (IL-6, TNF)"],
            "example_indications": ["rheumatoid arthritis", "systemic lupus erythematosus", "psoriasis", "IBD (overlap)"],
            "typical_endpoints": ["ACR response", "disease activity index", "flares frequency", "steroid-sparing effects"],
            "favored_formulations": ["injectable biologics", "oral small molecules", "topicals (derm)"],
            "assay_types": ["immune cell phenotyping", "cytokine ELISA", "functional immunoassays"],
            "preclinical_models": ["collagen-induced arthritis", "IL-10 KO colitis", "psoriasis xenograft skin models"],
            "repurposing_priority": "high",
            "regulatory_notes": "Immunosuppression-associated infection risk requires careful monitoring; combination/regimen patents common.",
            "notes": "Immunomodulators and small-molecule kinase inhibitors frequently repurposed across autoimmune indications."
        },

        "gastrointestinal": {
            "mesh_trees": ["C06"],
            "efo_ancestors": ["EFO:0010282"],
            "mondo_patterns": ["gastrointestinal", "digestive", "gastric", "intestinal", "colitis", "crohn", "ibd", "hepatic"],
            "doid_branches": ["DOID:77"],
            "subdomains": ["IBD", "IBS", "peptic ulcer disease", "liver disease", "pancreatitis"],
            "common_biomarkers": ["ALT/AST", "fecal calprotectin", "albumin", "bilirubin", "amylase/lipase"],
            "example_indications": ["ulcerative colitis", "Crohn's disease", "NASH", "peptic ulcer"],
            "typical_endpoints": ["mucosal healing", "clinical remission", "liver fibrosis score"],
            "favored_formulations": ["oral", "rectal (enemas/suppositories)", "IV (acute)"],
            "assay_types": ["endoscopy with biopsy", "liver biopsy", "stool tests", "histopathology"],
            "preclinical_models": ["TNBS/DSS colitis models", "diet-induced NASH models", "bile duct ligation"],
            "repurposing_priority": "medium-high",
            "regulatory_notes": "Local delivery (rectal) may reduce systemic exposure and allow different safety tradeoffs.",
            "notes": "Microbiome-targeting repurposing approaches are an active area (probiotics, small molecules altering microbiota)."
        },

        "urological": {
            "mesh_trees": ["C12", "C13"],
            "efo_ancestors": ["EFO:0009690"],
            "mondo_patterns": ["urological", "urinary", "bladder", "kidney", "renal", "prostate", "nephrolithiasis"],
            "doid_branches": ["DOID:18"],
            "subdomains": ["CKD/renal failure", "UTIs", "BPH", "urologic oncology"],
            "common_biomarkers": ["creatinine/GFR", "urinalysis", "PSA", "proteinuria"],
            "example_indications": ["chronic kidney disease", "benign prostatic hyperplasia", "urinary tract infection"],
            "typical_endpoints": ["eGFR decline", "symptom scores (IPSS)", "microbiological cure"],
            "favored_formulations": ["oral", "intravesical (bladder instillation)", "IV (acute kidney injury)"],
            "assay_types": ["renal function tests", "urine culture", "urodynamic studies"],
            "preclinical_models": ["5/6 nephrectomy CKD model", "stone formation models", "UTI mouse models"],
            "repurposing_priority": "medium",
            "regulatory_notes": "CKD introduces PK changes requiring dose adjustments and renal dosing studies.",
            "notes": "Renal safety and clearance are major filters for candidate molecules."
        },

        "musculoskeletal": {
            "mesh_trees": ["C05"],
            "efo_ancestors": ["EFO:0009688"],
            "mondo_patterns": ["musculoskeletal", "bone", "joint", "arthritis", "osteoporosis", "sarcopenia"],
            "doid_branches": ["DOID:17"],
            "subdomains": ["osteoarthritis", "rheumatoid arthritis (overlap w/ immunology)", "osteoporosis", "muscle wasting"],
            "common_biomarkers": ["bone turnover markers (CTX, P1NP)", "CRP (inflammatory)", "BMD via DEXA"],
            "example_indications": ["osteoarthritis", "post-menopausal osteoporosis", "rheumatoid arthritis"],
            "typical_endpoints": ["pain scores", "joint function", "fracture incidence", "BMD change"],
            "favored_formulations": ["oral", "intra-articular injections", "topical NSAIDs", "IV bisphosphonates"],
            "assay_types": ["DEXA", "joint imaging (MRI/XR)", "gait analysis"],
            "preclinical_models": ["OVA arthritis models", "ovariectomy-induced osteoporosis", "sarcopenia rodent models"],
            "repurposing_priority": "medium",
            "regulatory_notes": "Pain endpoints are subjective — regulatory agencies look for functional improvements and QoL measures.",
            "notes": "Local delivery (intra-articular) is a path for repositioning with reduced systemic AEs."
        },

        # -------------------------
        # NEW / ADDITIONAL AREAS (expanded heavily)
        # -------------------------
        "dermatological": {
            "mesh_trees": ["C17"],  # Skin and Connective Tissue Diseases
            "efo_ancestors": [],  # lookup_required
            "mondo_patterns": ["dermatology", "skin", "psoriasis", "eczema", "dermatitis", "acne", "melanoma"],
            "doid_branches": [],
            "subdomains": ["inflammatory dermatoses", "infectious skin disease", "cutaneous oncology", "wound healing"],
            "common_biomarkers": ["TEWL (transepidermal water loss)", "skin biopsy histology", "cytokine profiles"],
            "example_indications": ["psoriasis", "atopic dermatitis", "acne vulgaris", "melanoma"],
            "typical_endpoints": ["PASI score (psoriasis)", "EASI (eczema)", "lesion count", "wound closure rate"],
            "favored_formulations": ["topical creams/ointments", "oral", "injectable biologics", "phototherapy adjuncts"],
            "assay_types": ["skin histology", "patch testing", "UV challenge", "microbiome swabs"],
            "preclinical_models": ["imiquimod psoriasis mouse", "atopic dermatitis mouse models", "skin explants"],
            "repurposing_priority": "high",
            "regulatory_notes": "Topical formulations can often be repurposed with simpler bridging studies; systemic safety still applies for oral/biologics.",
            "notes": "Fast feedback from topical proof-of-concept trials (small patient numbers) accelerates iteration."
        },

        "ophthalmology": {
            "mesh_trees": ["C11"],  # Eye Diseases
            "efo_ancestors": [],
            "mondo_patterns": ["eye", "retina", "glaucoma", "macular degeneration", "uveitis", "ocular"],
            "doid_branches": [],
            "subdomains": ["retinal disease", "glaucoma", "dry eye", "ocular infections", "uveitis"],
            "common_biomarkers": ["intraocular pressure", "visual acuity", "OCT retinal thickness", "angiography"],
            "example_indications": ["age-related macular degeneration (AMD)", "glaucoma", "diabetic retinopathy"],
            "typical_endpoints": ["visual acuity change", "OCT metrics", "retinal thickness", "time to vision loss"],
            "favored_formulations": ["eye drops", "intravitreal injection", "topical gels", "ocular implants"],
            "assay_types": ["OCT", "fluorescein angiography", "visual field tests"],
            "preclinical_models": ["laser-induced CNV in rodents (AMD)", "ocular hypertension glaucoma models"],
            "repurposing_priority": "medium",
            "regulatory_notes": "Local ocular delivery reduces systemic exposure but requires specialized CMC and sterility controls.",
            "notes": "Drug delivery innovations (sustained-release implants) are a high-value repurposing route."
        },

        "psychiatric": {
            "mesh_trees": ["F-series (mental disorders) as cross-mapped"],  # not strictly MeSH code mapping here
            "efo_ancestors": [],
            "mondo_patterns": ["psychiatric", "depression", "anxiety", "bipolar", "schizophrenia", "ptsd", "ocd"],
            "doid_branches": [],
            "subdomains": ["mood disorders", "psychotic disorders", "anxiety disorders", "substance use disorders"],
            "common_biomarkers": ["questionnaire scores (HAM-D, HDRS, PANSS)", "neuroimaging patterns (research)", "inflammatory markers (emerging)"],
            "example_indications": ["major depressive disorder", "generalized anxiety disorder", "schizophrenia"],
            "typical_endpoints": ["symptom scale reduction", "time to relapse", "functional outcomes"],
            "favored_formulations": ["oral", "long-acting injectables (LAIs)", "transdermal (limited)"],
            "assay_types": ["clinical rating scales", "cognitive batteries", "EEG (research)"],
            "preclinical_models": ["chronic stress models", "prepulse inhibition (PPI) for antipsychotics", "learned helplessness"],
            "repurposing_priority": "medium",
            "regulatory_notes": "Subjective endpoints and placebo effects are large — robust trial design required.",
            "notes": "Psychedelic-assisted therapy and neuromodulators are modern repurposing vectors; careful regulatory and safety designs required."
        },

        "endocrinology": {
            "mesh_trees": ["C18 overlap"],
            "efo_ancestors": [],
            "mondo_patterns": ["endocrine", "hormone", "thyroid", "adrenal", "pituitary", "gonadal"],
            "doid_branches": [],
            "subdomains": ["thyroid disease", "adrenal disorders", "pituitary disorders", "reproductive endocrinology"],
            "common_biomarkers": ["TSH/T4/T3", "cortisol", "ACTH", "sex hormones", "IGF-1"],
            "example_indications": ["hypothyroidism", "Cushing's syndrome", "PCOS (endocrine aspect)"],
            "typical_endpoints": ["hormone normalization", "symptom relief", "biochemical remission"],
            "favored_formulations": ["oral", "injectable hormones", "transdermal"],
            "assay_types": ["hormone assays", "dynamic endocrine testing"],
            "preclinical_models": ["endocrine knockout models", "hormone challenge models"],
            "repurposing_priority": "low-medium",
            "regulatory_notes": "Hormone replacement therapies need careful long-term safety databases.",
            "notes": "Hormone-modulating drugs have systemic effects and often limited repurposing windows due to safety."
        },

        "renal_nephrology": {
            "mesh_trees": ["C12 overlap"],
            "efo_ancestors": [],
            "mondo_patterns": ["renal", "kidney", "nephro", "nephritis", "glomerulonephritis", "proteinuria"],
            "doid_branches": [],
            "subdomains": ["acute kidney injury", "chronic kidney disease", "glomerular disease", "dialysis-related conditions"],
            "common_biomarkers": ["serum creatinine", "eGFR", "proteinuria/albuminuria", "urinalysis"],
            "example_indications": ["CKD", "FSGS", "nephrotic syndrome"],
            "typical_endpoints": ["eGFR slope", "time to dialysis", "proteinuria reduction"],
            "favored_formulations": ["oral", "IV (inpatient)"],
            "assay_types": ["renal function tests", "biopsy histology"],
            "preclinical_models": ["5/6 nephrectomy", "adriamycin nephropathy"],
            "repurposing_priority": "medium",
            "regulatory_notes": "Renal dosing and accumulation of metabolites are major safety concerns.",
            "notes": "Many cardio/metabolic drugs have renal effects — cross-domain repurposing often fruitful."
        },

        "hepatology": {
            "mesh_trees": ["C06 hepatic overlap"],
            "efo_ancestors": [],
            "mondo_patterns": ["liver", "hepatic", "hepatitis", "cirrhosis", "NASH", "fibrosis"],
            "doid_branches": [],
            "subdomains": ["viral hepatitis", "NASH/NAFLD", "cirrhosis", "drug-induced liver injury (DILI)"],
            "common_biomarkers": ["ALT", "AST", "bilirubin", "FibroScan", "albumin"],
            "example_indications": ["chronic hepatitis", "NASH", "cirrhosis complications"],
            "typical_endpoints": ["ALT/AST normalization", "fibrosis stage improvement", "liver-related events"],
            "favored_formulations": ["oral", "IV"],
            "assay_types": ["liver biopsy", "elastography", "viral load assays"],
            "preclinical_models": ["diet-induced NASH models", "CCl4 fibrosis models"],
            "repurposing_priority": "medium",
            "regulatory_notes": "Liver safety is a leading cause of drug attrition; DILI signals are a major blocker for repurposing.",
            "notes": "Repurposing for NASH is commercially attractive but scientifically challenging."
        },

        "women_health_obgyn": {
            "mesh_trees": ["C13/C12 overlap"],
            "efo_ancestors": [],
            "mondo_patterns": ["obstetrics", "gynecology", "menopause", "pcos", "endometriosis", "fertility"],
            "doid_branches": [],
            "subdomains": ["fertility", "menstrual disorders", "pregnancy complications", "gynecologic oncology"],
            "common_biomarkers": ["hCG", "AMH", "estrogen/progesterone levels", "FSH/LH"],
            "example_indications": ["PCOS", "uterine fibroids", "preeclampsia management"],
            "typical_endpoints": ["live birth rate (fertility)", "symptom scores", "pregnancy outcome metrics"],
            "favored_formulations": ["oral", "intrauterine devices (IUDs - device+drug combos)", "injectables"],
            "assay_types": ["hormonal assays", "transvaginal ultrasound", "pregnancy monitoring"],
            "preclinical_models": ["rodent fertility & implantation models", "endometriosis models"],
            "repurposing_priority": "medium",
            "regulatory_notes": "Pregnancy exposure requires extremely conservative approaches; teratogenic risk critical.",
            "notes": "High impact but high regulatory barriers if pregnancy/pediatric exposure possible."
        },

        "pediatrics": {
            "mesh_trees": ["special population flag"],
            "efo_ancestors": [],
            "mondo_patterns": ["pediatric", "child", "neonate", "infant"],
            "doid_branches": [],
            "subdomains": ["neonatology", "pediatric oncology", "infectious diseases in children", "congenital disorders"],
            "common_biomarkers": ["growth metrics", "pediatric-specific lab normals", "developmental scales"],
            "example_indications": ["pediatric asthma", "pediatric epilepsy", "congenital metabolic disorders"],
            "typical_endpoints": ["growth and development outcomes", "age-adjusted symptom scales", "survival for pediatric cancer"],
            "favored_formulations": ["oral suspensions", "dispersible tablets", "age-appropriate dosing forms"],
            "assay_types": ["developmental assessments", "age-normed lab tests"],
            "preclinical_models": ["juvenile animal models", "perinatal exposure studies"],
            "repurposing_priority": "medium-high",
            "regulatory_notes": "Pediatric investigation plans and age-stratified safety are required; palatable formulations are crucial.",
            "notes": "Significant unmet need in paediatrics; many adult drugs lack pediatric labels—opportunity for repurposing with dosing studies."
        },

        "geriatrics": {
            "mesh_trees": ["special population flag"],
            "efo_ancestors": [],
            "mondo_patterns": ["geriatric", "elderly", "frailty", "polypharmacy"],
            "doid_branches": [],
            "subdomains": ["frailty", "polypharmacy management", "falls prevention", "cognitive decline"],
            "common_biomarkers": ["functional status scales", "polypharmacy indices", "nutritional markers"],
            "example_indications": ["frailty-associated sarcopenia", "drug repurposing for improved mobility"],
            "typical_endpoints": ["falls reduction", "improved ADL (activities of daily living) scores", "functional measures"],
            "favored_formulations": ["oral", "low-dose regimens", "transdermal to avoid swallowing issues"],
            "assay_types": ["gait analysis", "frailty indices"],
            "preclinical_models": ["aged rodent models", "sarcopenia models"],
            "repurposing_priority": "medium",
            "regulatory_notes": "Polypharmacy increases interaction risk; dose adjustments and PK studies in elderly necessary.",
            "notes": "High translational value for quality-of-life repurposings; safety trade-offs differ in geriatric populations."
        },

        "rare_diseases": {
            "mesh_trees": ["cross-cutting"],
            "efo_ancestors": [],
            "mondo_patterns": ["rare", "orphan", "genetic", "congenital", "ultra-rare"],
            "doid_branches": [],
            "subdomains": ["genetic metabolic disorders", "rare oncology", "rare neuromuscular diseases"],
            "common_biomarkers": ["disease-specific enzyme assays", "genetic mutation confirmation", "specialized biomarkers"],
            "example_indications": ["Gaucher disease", "spinal muscular atrophy (SMA)", "rare lysosomal storage disorders"],
            "typical_endpoints": ["biomarker normalization", "clinical meaningfulness in small n studies", "patient-reported outcomes"],
            "favored_formulations": ["IV enzyme replacement", "oral small molecules", "gene therapy adjuncts"],
            "assay_types": ["genetic testing", "enzyme activity assays", "specialized functional assays"],
            "preclinical_models": ["knockout mouse models", "patient-derived cell lines", "iPSC disease models"],
            "repurposing_priority": "high",
            "regulatory_notes": "Orphan designation & accelerated pathways available; small n trial designs accepted.",
            "notes": "Repurposing often fastest route to therapy because full de novo discovery timelines may be impractical."
        },

        "pain_palliative": {
            "mesh_trees": ["symptom-based"],
            "efo_ancestors": [],
            "mondo_patterns": ["pain", "analgesic", "neuropathic", "palliativecare", "opioid-sparing"],
            "doid_branches": [],
            "subdomains": ["acute pain", "chronic pain", "neuropathic pain", "palliative symptom control"],
            "common_biomarkers": ["pain scales (VAS)", "opioid use metrics", "quality-of-life scores"],
            "example_indications": ["neuropathic pain", "cancer pain", "postoperative pain management"],
            "typical_endpoints": ["pain reduction", "opioid-sparing effect", "functional improvement"],
            "favored_formulations": ["oral", "transdermal (fentanyl patches)", "topical", "IV for acute"],
            "assay_types": ["pain scoring", "quantitative sensory testing"],
            "preclinical_models": ["neuropathic pain rodent models (CCI)", "inflammatory pain models"],
            "repurposing_priority": "high",
            "regulatory_notes": "Opioid-related liabilities; non-opioid repurposing is a major public health priority.",
            "notes": "Analgesic repurposing should consider abuse potential and regulatory scheduling."
        },

        "toxicology_overdose": {
            "mesh_trees": ["safety"],
            "efo_ancestors": [],
            "mondo_patterns": ["overdose", "poisoning", "toxicity", "antidote"],
            "doid_branches": [],
            "subdomains": ["drug overdose antidotes", "toxin neutralizers", "organ-specific toxicities"],
            "common_biomarkers": ["serum drug levels", "organ injury labs (LFTs, creatinine)", "specific toxin assays"],
            "example_indications": ["acetaminophen overdose (NAC)", "opioid overdose (naloxone) - examples of successful repurposing/antidotes"],
            "typical_endpoints": ["survival", "reversal of toxicity markers", "organ function recovery"],
            "favored_formulations": ["IV (emergency)","intranasal (naloxone)"],
            "assay_types": ["toxin assays", "organ function tests"],
            "preclinical_models": ["toxicity rodent models", "organoid toxicity screens"],
            "repurposing_priority": "medium",
            "regulatory_notes": "Emergency indications can justify compassionate/expanded access pathways.",
            "notes": "Antidote repurposing is high-impact but requires rapid PK/PD understanding."
        },

        "transplantation_immunosuppression": {
            "mesh_trees": ["C20 overlaps"],
            "efo_ancestors": [],
            "mondo_patterns": ["transplant", "graft", "immunosuppression", "GVHD"],
            "doid_branches": [],
            "subdomains": ["solid organ transplant", "bone marrow transplant", "graft-versus-host disease"],
            "common_biomarkers": ["donor-specific antibodies", "drug trough levels (tacrolimus)", "graft function metrics"],
            "example_indications": ["prevention of rejection", "treatment of acute rejection", "chronic allograft dysfunction"],
            "typical_endpoints": ["graft survival", "acute rejection episodes", "immunosuppressive drug levels"],
            "favored_formulations": ["oral", "IV (induction agents)", "topical (eye graft)"],
            "assay_types": ["HLA typing", "immunologic assays", "graft biopsy"],
            "preclinical_models": ["allograft models in rodents", "GVHD models"],
            "repurposing_priority": "medium",
            "regulatory_notes": "Immunosuppressive label expansions need robust infection risk mitigation strategies.",
            "notes": "Repurposing frequently focuses on steroid-sparing or organ-protective adjuncts."
        },

        "dental_oral_health": {
            "mesh_trees": ["C09"],  # Oral Diseases overlap
            "efo_ancestors": [],
            "mondo_patterns": ["dental", "oral", "periodontal", "tooth", "gingivitis", "caries"],
            "doid_branches": [],
            "subdomains": ["periodontal disease", "oral infections", "dental hypersensitivity", "oral mucosal disease"],
            "common_biomarkers": ["plaque indices", "microbial cultures", "inflammatory cytokines in saliva"],
            "example_indications": ["periodontitis", "oral candidiasis", "dry mouth (xerostomia)"],
            "typical_endpoints": ["clinical periodontal indices", "microbiological clearance", "oral pain reduction"],
            "favored_formulations": ["topical gels", "mouthwash", "local delivery systems (periodontal pockets)"],
            "assay_types": ["microbial assays", "saliva diagnostic tests"],
            "preclinical_models": ["periodontitis rodent models", "oral candidiasis models"],
            "repurposing_priority": "low-medium",
            "regulatory_notes": "Many dental products are regulated as medical devices or OTC — different regulatory paths possible.",
            "notes": "Localized delivery is common and often simplifies safety profiling."
        },

        "allergy": {
            "mesh_trees": ["C20/C17 overlap"],
            "efo_ancestors": [],
            "mondo_patterns": ["allergy", "anaphylaxis", "allergic rhinitis", "urticaria"],
            "doid_branches": [],
            "subdomains": ["food allergy", "allergic rhinitis", "urticaria", "anaphylaxis management"],
            "common_biomarkers": ["IgE", "skin prick testing results", "specific IgE panels"],
            "example_indications": ["allergic rhinitis", "food allergy adjuncts", "chronic urticaria"],
            "typical_endpoints": ["symptom reduction", "challenge test outcomes", "quality-of-life"],
            "favored_formulations": ["intranasal sprays", "oral antihistamines", "epinephrine auto-injectors (rescue)"],
            "assay_types": ["skin prick tests", "IgE serology"],
            "preclinical_models": ["allergen sensitization models", "anaphylaxis rodent models"],
            "repurposing_priority": "medium",
            "regulatory_notes": "Immunotherapies and desensitization approaches have distinct regulatory considerations.",
            "notes": "Adjuncts that reduce allergic inflammation or mast cell activation are often explored for repurposing."
        },

        "addiction_substance_use": {
            "mesh_trees": ["behavioral health overlap"],
            "efo_ancestors": [],
            "mondo_patterns": ["addiction", "substance use", "opioid dependence", "alcohol use disorder", "nicotine"],
            "doid_branches": [],
            "subdomains": ["opioid use disorder", "alcohol dependence", "smoking cessation"],
            "common_biomarkers": ["drug screening panels", "craving scores", "liver enzymes (AUD)"],
            "example_indications": ["opioid dependence (buprenorphine, methadone existing therapies) - repurposing options adjunctive"],
            "typical_endpoints": ["abstinence rates", "retention in treatment", "overdose rates"],
            "favored_formulations": ["oral", "sublingual", "injectable depot formulations"],
            "assay_types": ["toxicology screens", "behavioral assays"],
            "preclinical_models": ["self-administration rodent models", "conditioned place preference"],
            "repurposing_priority": "medium",
            "regulatory_notes": "Addiction therapies face high scrutiny due to potential for misuse; combination with psychosocial support recommended.",
            "notes": "Novel adjuncts that reduce craving or withdrawal symptoms are key repurposing targets."
        },

        "oncology_supportive_care": {
            "mesh_trees": ["supportive"],
            "efo_ancestors": [],
            "mondo_patterns": ["nausea", "anemia of cancer", "cachexia", "chemotherapy-induced neuropathy"],
            "doid_branches": [],
            "subdomains": ["antiemetics", "growth factors", "pain control", "neuroprotective agents"],
            "common_biomarkers": ["weight loss metrics", "nausea scales", "hemoglobin levels"],
            "example_indications": ["chemotherapy-induced nausea and vomiting", "cancer cachexia"],
            "typical_endpoints": ["symptom control", "reduced hospitalization", "QoL improvements"],
            "favored_formulations": ["oral", "IV", "patches (antiemetic)"],
            "assay_types": ["patient-reported outcomes", "biomarker panels"],
            "preclinical_models": ["chemotherapy-induced neuropathy", "cachexia models"],
            "repurposing_priority": "high",
            "regulatory_notes": "Supportive care approvals often rely on symptom-based trials with clear QoL improvements.",
            "notes": "High value for patient quality-of-life; repurposing can be faster with smaller trials."
        },

        # -------------------------
        # METADATA / UTILITIES (helps automation & matching)
        # -------------------------
        "_metadata": {
            "version": "2025-12-14.expand-v1",
            "description": "Expanded therapeutic area ontology-like mapping for multiagent repurposing pipelines. \
                            Fields: mesh_trees, efo_ancestors, mondo_patterns (keyword patterns), doid_branches, \
                            subdomains, common_biomarkers, example_indications, typical_endpoints, favored_formulations, \
                            assay_types, preclinical_models, repurposing_priority, regulatory_notes, notes.",
            "usage_examples": [
                "Use mondo_patterns for fuzzy keyword matching of disease names",
                "Use mesh_trees for high-level MeSH anchoring",
                "Populate efo_ancestors/doid_branches with authoritative ontology IDs when available",
                "repurposing_priority is a quick heuristic (low/medium/high) for triage scoring"
            ],
            "recommended_actions": [
                "Add exact EFO/DOID/SNOMED codes for each area via ontology lookup",
                "Create helper functions: match_to_area(disease_name) and score_candidate(candidate, area_rules)",
                "Extend per-area 'exclusion_keywords' if you want stronger filtering (e.g., pregnancy, pediatric-only)"
            ],
            "caveats": [
                "This mapping is a starting taxonomy — for regulatory, IP, and clinical decisions, consult domain experts.",
                "Ontology IDs should be validated programmatically against live services where possible."
            ]
        }
    }


    def __init__(self):
        self.timeout = 10.0

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def _query_mesh_tree(self, disease_name: str) -> Optional[List[str]]:
        """
        Query NCBI MeSH for disease classification tree numbers.
        Tree numbers like "C08.381" indicate Respiratory Tract Diseases.
        """
        try:
            # Search MeSH
            search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            search_params = {
                "db": "mesh",
                "term": disease_name,
                "retmode": "json",
                "retmax": 3
            }
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                search_resp = await client.get(search_url, params=search_params)
                if search_resp.status_code != 200:
                    return None
                
                search_data = search_resp.json()
                mesh_ids = search_data.get("esearchresult", {}).get("idlist", [])
                if not mesh_ids:
                    return None
                
                # Get MeSH tree numbers
                fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                fetch_params = {
                    "db": "mesh",
                    "id": ",".join(mesh_ids),
                    "retmode": "json"
                }
                
                fetch_resp = await client.get(fetch_url, params=fetch_params)
                if fetch_resp.status_code != 200:
                    return None
                
                fetch_data = fetch_resp.json()
                results = fetch_data.get("result", {})
                
                # ✅ CORRECT: Extract from ds_idxlinks list of dicts
                tree_numbers = []
                for mesh_id in mesh_ids:
                    if mesh_id not in results:
                        continue
                    record = results[mesh_id]
                    
                    idxlinks = record.get("ds_idxlinks", [])
                    if isinstance(idxlinks, list):
                        for link in idxlinks:
                            if isinstance(link, dict):
                                tree_num = link.get("treenum", "")
                                if tree_num and not tree_num.startswith("@"):  # Skip supplemental records
                                    tree_numbers.append(tree_num)
                    
                    # Only check first MeSH ID (usually most relevant)
                    if tree_numbers:
                        break
                
                logger.info(f"✓ MeSH tree numbers for '{disease_name}': {tree_numbers[:5]}")
                return tree_numbers if tree_numbers else None
                
        except Exception as e:
            logger.debug(f"MeSH tree query failed: {e}")
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def _query_efo_ancestors(self, disease_name: str) -> Optional[List[str]]:
        """
        Query EBI OLS for EFO/MONDO ancestors to determine therapeutic area.
        """
        try:
            url = "https://www.ebi.ac.uk/ols4/api/search"
            params = {
                "q": disease_name,
                "ontology": "efo,mondo,doid",
                "type": "class",
                "rows": 5,
                "exact": "false"
            }
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, params=params)
                if response.status_code != 200:
                    return None
                
                data = response.json()
                docs = data.get("response", {}).get("docs", [])
                if not docs:
                    return None
                
                # Get best match
                best_match = docs[0]
                iri = best_match.get("iri")
                if not iri:
                    return None
                
                # Query for ancestors
                ancestor_url = f"https://www.ebi.ac.uk/ols4/api/ontologies/{best_match.get('ontology_name')}/terms/{httpx.QueryParams({'iri': iri}).get('iri')}/ancestors"
                ancestor_resp = await client.get(ancestor_url)
                if ancestor_resp.status_code != 200:
                    return None
                
                ancestor_data = ancestor_resp.json()
                terms = ancestor_data.get("_embedded", {}).get("terms", [])
                ancestor_ids = [term.get("obo_id") for term in terms if term.get("obo_id")]
                
                logger.info(f"✓ EFO/MONDO ancestors for '{disease_name}': {ancestor_ids[:5]}")
                return ancestor_ids
                
        except Exception as e:
            logger.debug(f"EFO ancestor query failed: {e}")
            return None

    def _classify_by_tree_numbers(self, tree_numbers: List[str]) -> Optional[str]:
        """
        Classify therapeutic area based on MeSH tree numbers.
        ✅ FIX #2: PRIORITIZE SPECIFIC OVER GENERAL (C15 > C18)
        """
        if not tree_numbers:
            return None
        
        # ✅ Priority scoring: Prefer more specific classifications
        PRIORITY_ORDER = {
            "hematological": 10,  # C15 - Blood disorders (HIGHEST PRIORITY)
            "oncology": 9,         # C04 - Cancer (very specific)
            "infectious": 8,       # C01/C02 - Infections (specific)
            "cardiovascular": 7,   # C14 - Heart diseases
            "respiratory": 7,      # C08 - Lung diseases
            "neurological": 7,     # C10 - Brain diseases
            "gastrointestinal": 7, # C06 - Digestive diseases
            "urological": 7,       # C12/C13 - Urinary diseases
            "immunological": 6,    # C20 - Immune diseases
            "musculoskeletal": 6,  # C05 - Bone diseases
            "metabolic": 5         # C18 - Metabolic (LOWEST - too broad!)
        }
        
        matches = []
        for area, config in self.THERAPEUTIC_AREAS.items():
            mesh_trees = config.get("mesh_trees", [])
            for tree in tree_numbers:
                if any(tree.startswith(prefix) for prefix in mesh_trees):
                    priority = PRIORITY_ORDER.get(area, 0)
                    matches.append((area, priority, tree))
        
        if not matches:
            return None
        
        # Return area with HIGHEST priority
        best_match = max(matches, key=lambda x: x[1])
        logger.info(f"✓ Classified as '{best_match[0]}' via MeSH tree: {best_match[2]} (priority: {best_match[1]})")
        return best_match[0]

    def _classify_by_ancestors(self, ancestor_ids: List[str]) -> Optional[str]:
        """Classify therapeutic area based on ontology ancestors."""
        if not ancestor_ids:
            return None
        
        for area, config in self.THERAPEUTIC_AREAS.items():
            efo_ancestors = config.get("efo_ancestors", [])
            doid_branches = config.get("doid_branches", [])
            
            # Check EFO/MONDO ancestors
            for ancestor in ancestor_ids:
                if ancestor in efo_ancestors or ancestor in doid_branches:
                    logger.info(f"✓ Classified as '{area}' via ancestor: {ancestor}")
                    return area
        
        return None

    def _classify_by_keywords(self, disease_name: str) -> Optional[str]:
        """Fallback: Classify by keyword matching."""
        disease_lower = disease_name.lower()
        
        # Score each therapeutic area
        scores = {}
        for area, config in self.THERAPEUTIC_AREAS.items():
            patterns = config.get("mondo_patterns", [])
            score = sum(1 for pattern in patterns if pattern in disease_lower)
            if score > 0:
                scores[area] = score
        
        if not scores:
            return None
        
        # Return area with highest score
        best_area = max(scores.items(), key=lambda x: x[1])
        logger.info(f"✓ Classified as '{best_area[0]}' via keywords (score: {best_area[1]})")
        return best_area[0]

    async def classify(self, disease_name: str) -> Optional[str]:
        """
        Classify disease into therapeutic area using multi-source approach.
        
        Priority:
        1. MeSH tree numbers (most reliable) - WITH PRIORITY SCORING ✅
        2. EFO/MONDO ancestors (good coverage)
        3. Keyword matching (fallback)
        
        Returns:
            Therapeutic area name or None
        """
        logger.info(f"🏥 Classifying therapeutic area for: {disease_name}")
        
        # Method 1: MeSH tree classification
        tree_numbers = await self._query_mesh_tree(disease_name)
        if tree_numbers:
            area = self._classify_by_tree_numbers(tree_numbers)
            if area:
                return area
        
        # Method 2: Ontology ancestor classification
        ancestors = await self._query_efo_ancestors(disease_name)
        if ancestors:
            area = self._classify_by_ancestors(ancestors)
            if area:
                return area
        
        # Method 3: Keyword fallback
        area = self._classify_by_keywords(disease_name)
        if area:
            return area
        
        logger.warning(f"⚠️ Could not classify therapeutic area for: {disease_name}")
        return None


# Global instance
_mapper = None

def get_therapeutic_area_mapper() -> TherapeuticAreaMapper:
    """Get singleton therapeutic area mapper."""
    global _mapper
    if _mapper is None:
        _mapper = TherapeuticAreaMapper()
    return _mapper

async def classify_disease_therapeutic_area(disease_name: str) -> Optional[str]:
    """
    Public API: Classify disease into therapeutic area.
    
    Usage:
        area = await classify_disease_therapeutic_area("Iron Deficiency Anemia")
        # Returns: "hematological" ✅
    """
    mapper = get_therapeutic_area_mapper()
    return await mapper.classify(disease_name)
