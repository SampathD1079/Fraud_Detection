import re
import binascii
import pikepdf
from PyPDF2 import PdfReader
from PyPDF2.generic import ContentStream, NullObject
import numpy as np
import sqlite3
import os

# ================= ML ADDITION: new imports =================
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier  # noqa: F401  (used in train_ml_model.py, imported here for type clarity)
# ================= END ML ADDITION =================


class Invoice_Data:

    def extract_pdf_forensics(self,path):
        # 1. Physical Layer & Header Analysis
        with open(path, "rb") as f:
            raw_bytes = f.read()
        
        header_match = re.search(rb"%PDF-(\d\.\d)", raw_bytes[:64])
        pdf_version = header_match.group(1).decode() if header_match else "unknown"
        
        # 2. Metadata & Trailer
        meta_info = {
            "Creation Date": None, "Modification Date": None,
            "producer": None, "creator": None, "trailer_id_array": []
        }

        try:
            with pikepdf.Pdf.open(path) as pdf:
                info = pdf.docinfo or {}
                meta_info["Creation Date"] = str(info.get("/CreationDate", ""))
                meta_info["Modification Date"] = str(info.get("/ModDate", ""))
                meta_info["producer"] = str(info.get("/Producer", ""))
                meta_info["creator"] = str(info.get("/Creator", ""))
                
                if "/ID" in pdf.trailer:
                    id_obj = pdf.trailer.get("/ID")
                    for part in id_obj:
                        try:
                            raw = part.read_bytes() if hasattr(part, "read_bytes") else bytes(str(part), "utf-8")
                        except Exception:
                            raw = bytes(str(part), "utf-8")
                        meta_info["trailer_id_array"].append(binascii.hexlify(raw).decode())
        except Exception as e:
            print(f"Metadata Error: {e}")

        # 3. Content Stream Analysis (Per Page)
        reader = PdfReader(path, strict=False)
        pages_data = []
        stream_broken_count = 0

        total_pages = len(reader.pages)
        pages_to_analyze = range(total_pages - 1) if total_pages > 1 else range(total_pages)

        for pno in pages_to_analyze:
            page_stats = {
                "page": pno,
                "BT_count": 0,
                "Tj_count": 0,
                "TJ_count": 0,
                "empty_Tj": 0,
                "empty_TJ": 0,
                "font_changes": {},
                "color_ops": {},
                "anomalies":[]
            }
            
            try:
                page = reader.pages[pno]
                contents = page.get("/Contents")
                if contents is None or isinstance(contents, NullObject):
                    page_stats["anomalies"].append("missing_or_null_contents")
                    pages_data.append(page_stats)
                    continue
                
                content_stream = ContentStream(contents.get_object(), reader)
                in_text_block = False
                has_text = False

                for operands, operator in content_stream.operations:
                    op = operator.decode() if isinstance(operator, bytes) else str(operator)
                    
                    if op == "BT":
                        page_stats["BT_count"] += 1
                        in_text_block = True
                        has_text = False
                    elif op == "ET":
                        if in_text_block and not has_text:
                            page_stats["anomalies"].append("empty_BT_block")
                        in_text_block = False
                    
                    elif op == "Tj":
                        page_stats["Tj_count"] += 1
                        has_text = True
                        if operands and str(operands[0]).strip() == "":
                            page_stats["empty_Tj"] += 1
                    elif op == "TJ":
                        page_stats["TJ_count"] += 1
                        has_text = True
                        if operands and all(str(x).strip() == "" for x in operands[0] if not isinstance(x, (int, float))):
                            page_stats["empty_TJ"] += 1
                    
                    elif op == "Tf" and operands:
                        f_name = str(operands[0])
                        page_stats["font_changes"][f_name] = page_stats["font_changes"].get(f_name, 0) + 1
                    elif op in ("g", "G", "rg", "RG"):
                        page_stats["color_ops"][op] = page_stats["color_ops"].get(op, 0) + 1

            except Exception:
                stream_broken_count += 1
                page_stats["anomalies"].append("stream_broken")
            
            pages_data.append(page_stats)

        return {
            "pdf_version": pdf_version,
            "producer": meta_info["producer"],
            "creator": meta_info["creator"],
            "xref_table_text": bool(re.search(rb"\nxref\b", raw_bytes)),
            "xref_table_stream": b"/XRef" in raw_bytes or b"/XRefStm" in raw_bytes,
            "start_xref_count": len(re.findall(b"startxref", raw_bytes)),
            "Creation Date": meta_info["Creation Date"],
            "Modification Date": meta_info["Modification Date"],
            "trailer_id_array": meta_info["trailer_id_array"],
            "stream_broken_total": stream_broken_count,
            "pages": pages_data
        }
    

    def font_analysis(self,results):
        font_score = 0

        bt_count = results['pages'][0]['BT_count']
        Tj_count = results['pages'][0]['Tj_count']
        empty_TJ_count=results['pages'][0]['empty_Tj']

        #empty Tj count
        if empty_TJ_count>50:
            print('Empty_Tj')
            font_score+=2

        total = bt_count + Tj_count

        #No text obj
        if total==0:
            font_score=5
            return font_score

        font_dict = results['pages'][0]['font_changes']

        if len(font_dict) < 2 and total>300:
            font_score+=1
            return font_score

        fonts = list(font_dict.keys())
        counts = np.array(list(font_dict.values()))
        
        total_usage = np.sum(counts)
        mean_usage = np.mean(counts)
        
        low_usage_fonts = {}
        
        for i in range(len(counts)):
            # Calculate what % of the document this font represents
            participation_percentage = counts[i] / total_usage
            
            # LOGIC: A font is suspicious if:
            # 1. It represents less than 5% of the total text
            # 2. OR its count is significantly lower than the average (e.g., < 1/5th of mean)
            # 3. AND it has a low absolute count (e.g., used less than 10 times)
            
            is_tiny_fraction = participation_percentage < 0.05
            is_far_from_mean = counts[i] < (mean_usage * 0.50)
            
            if is_tiny_fraction or is_far_from_mean:
                low_usage_fonts[fonts[i]] = counts[i]
        print(low_usage_fonts)

        # Scoring Logic

        if low_usage_fonts and total>300:
            font_score+=3
            return font_score
        elif low_usage_fonts:
            font_score+=1
            return font_score
        elif total>300 or total==0:
            font_score+=2
            return font_score

        return font_score

    def calculate_scores(self,results):
        creator = 0
        modification_creation = 0
        pdf_version = 0
        trailer_id_match = 0
        match_xref = 0
        
        #Merge Creator and Producer
        creator_name = results.get('creator', '').lower()
        if any(keyword.lower() in creator_name for keyword in ['microsoft', 'ios']):
            creator=1

        producer_name = results.get('producer', '').lower()
        if any(keyword.lower() in producer_name for keyword in ['microsoft', 'ios']):
            creator=1


        if results['Modification Date'] != results['Creation Date'] and results['Modification Date']:
            modification_creation = 1

        if results['pdf_version'] != '1.7':
            pdf_version = 1

        if len(results['trailer_id_array'])>=2:

            if results['trailer_id_array'][0] != results['trailer_id_array'][1]:
                trailer_id_match = 1
                

        if (results['xref_table_stream'] == True and results['xref_table_text'] == False) or (results['xref_table_stream'] == True and results['xref_table_text'] == True) :
            match_xref = 1

        font_score = self.font_analysis(results)

        return {
            "creator": creator,
            "modification_creation": modification_creation,
            "pdf_version": pdf_version,
            "trailer_id_match": trailer_id_match,
            "match_xref": match_xref,
            "font_score": font_score
        }

    def assign_weightage(self,scores,file_structure):
        print(file_structure)
        final_score=0
        creator=scores['creator']
        modification_creation=scores['modification_creation']
        pdf_version=scores['pdf_version']
        trailer_id_match=scores['trailer_id_match']
        match_xref=scores['match_xref']
        font_score=scores['font_score']

        final_score=(creator*20)+(modification_creation*40)+(pdf_version*10)+(trailer_id_match*10)+(match_xref*40)+(font_score*10)
        
        #For UBER CASE
        print(file_structure.get('trailer_id_array'))
        if (
            (font_score!=0 and
            creator == 0 and
            modification_creation == 0 and
            trailer_id_match == 0 and
            match_xref == 0) and (not file_structure.get('trailer_id_array')) and (final_score>20)):
            print('lower')
            print(final_score)
            final_score=final_score-20
            


        return final_score

    # ================= ML ADDITION: feature vector helper =================
    def prepare_ml_features(self, scores: dict) -> "pd.DataFrame":
        """
        Turns the existing rule-based `scores` dict (the output of
        calculate_scores()) into a single-row DataFrame in the exact
        column order the ML model expects (FEATURE_COLUMNS).

        This does NOT re-run or duplicate any forensic analysis -- it is
        purely a reshaping step on top of the already-computed scores.
        """
        feature_row = {col: scores.get(col, 0) for col in FEATURE_COLUMNS}
        return pd.DataFrame([feature_row], columns=FEATURE_COLUMNS)
    # ================= END ML ADDITION =================


# ================= ML ADDITION: configuration & model wrapper =================
# All ML-related config lives in one place so weights/thresholds/paths are
# easy to tune without hunting through the rest of the file.

FEATURE_COLUMNS = [
    "creator",
    "modification_creation",
    "pdf_version",
    "trailer_id_match",
    "match_xref",
    "font_score",
]

MODEL_PATH = "fraud_model.pkl"

# Weighting between the existing rule-based score and the new ML score
# when combining them into final_score. Configurable, not hardcoded inline.
RULE_WEIGHT = 0.5
ML_WEIGHT = 0.5

# Thresholds applied to the final combined score (0-100 scale).
HIGH_RISK_THRESHOLD = 70
SUSPICIOUS_THRESHOLD = 40


class MLFraudModel:
    """
    Thin wrapper around the trained RandomForestClassifier.

    - Loading happens once (e.g. one instance per batch run), not per file.
    - If fraud_model.pkl is missing or fails to load, is_available() is
      False and the rest of the pipeline falls back to rule-based-only
      scoring -- it never crashes because the model isn't there.
    - This class only does INFERENCE. Training lives in train_ml_model.py
      and is run manually/explicitly, never as a side effect of processing
      an invoice.
    """

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self._load()

    def _load(self):
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
            except Exception as e:
                print(f"[ML] Failed to load model at '{self.model_path}': {e}")
                self.model = None
        else:
            print(
                f"[ML] No trained model found at '{self.model_path}'. "
                f"ML scoring will be skipped -- pipeline continues with "
                f"rule-based scoring only. Run train_ml_model.py to train one."
            )
            self.model = None

    def is_available(self) -> bool:
        return self.model is not None

    def predict_proba(self, feature_df: "pd.DataFrame"):
        """
        Returns the fraud probability (float in [0, 1]) for a single-row
        feature DataFrame, or None if the model isn't available / prediction
        fails for any reason.
        """
        if self.model is None:
            return None
        try:
            proba = self.model.predict_proba(feature_df[FEATURE_COLUMNS])[0][1]
            return float(proba)
        except Exception as e:
            print(f"[ML] Prediction failed, falling back to rule-based only: {e}")
            return None


def classify_final_score(final_score) -> str:
    """
    Configurable risk-bucket classification on top of the combined
    final_score (0-100 scale). Deliberately uses risk language
    (Low Risk / Suspicious / High Risk) rather than a hard fraud/not-fraud
    claim, since this is a probabilistic estimate, not a verdict.
    """
    if final_score is None:
        return "UNKNOWN"
    if final_score >= HIGH_RISK_THRESHOLD:
        return "HIGH RISK / FRAUD"
    elif final_score >= SUSPICIOUS_THRESHOLD:
        return "SUSPICIOUS"
    else:
        return "LOW RISK / NORMAL"
# ================= END ML ADDITION =================


# ──────────────────────────────────────────────
# DATABASE LAYER
# ──────────────────────────────────────────────

class DB_main:

    def pdf_reader(self,pdf_path):
        reader = PdfReader(pdf_path)
        text = reader.pages[0].extract_text() or ""     
        return text 

    def detect_vendor(self,text) -> str | None:       
        text_lower = text.lower()
        for vendor_name, keywords in VENDOR_CORPUS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return vendor_name
        return None  # Unknown vendor — flag for manual review
    
 
# ──────────────────────────────────────────────
# INVOICE NUMBER EXTRACTION
# ──────────────────────────────────────────────
 
    VENDOR_INVOICE_PATTERNS = {
        "Zomato":     r'Invoice\s*No\.?\s*[:\-]?\s*([A-Z0-9]{10,30})',
        "Swiggy":     r'Order\s*ID[:\s]*([A-Z0-9\-]{6,25})',
        "Amazon":     r'Invoice\s*Number[:\s]*([\w\-]+)',
        "Flipkart":   r'Invoice\s*No[.:\s]*([\w\-]+)',
        "MakeMyTrip": r'Booking\s*(?:ID|Reference)[:\s]*([\w\-]+)',
        "Uber":       r'Trip\s*(?:ID|Invoice)[:\s]*([\w\-]+)',
        "Ola":        r'(?:Booking|Ride)\s*ID[:\s]*([\w\-]+)',
        "IRCTC":      r'PNR\s*(?:No)?[.:\s]*(\d{10})',
    }
    
    GENERIC_INVOICE_PATTERNS = [
        r'Invoice\s*(?:No|Number|#|Num|No\.)[:\s#]*([A-Z0-9][\w\-/]{4,30})',
        r'(?:Bill|Tax\s*Invoice|Inv)\s*(?:No|Number|#)?[:\s]*([A-Z0-9][\w\-/]{4,30})',
        r'(?:Order|Receipt|Ref)\s*(?:No|Number|ID)[:\s#]*([A-Z0-9][\w\-/]{4,30})',
        r'\b([A-Z]{2,6}[-/]\d{5,15})\b',
    ]
    def extract_invoice_number(self,text: str, vendor: str = None) -> str | None:
        """Extract invoice number using vendor-specific then generic patterns."""
        if vendor and vendor in DB_main.VENDOR_INVOICE_PATTERNS:
            match = re.search(DB_main.VENDOR_INVOICE_PATTERNS[vendor], text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        for pattern in DB_main.GENERIC_INVOICE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def create_db(self):
        with sqlite3.connect('invoice_db.db') as conn:
            cursor=conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS invoices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                invoice_number TEXT,
                vendor_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(invoice_number, vendor_name)
            )
            """)

            conn.commit()

            # ================= ML ADDITION: backward-compatible schema extension =================
            # Adds the new score/classification columns if they aren't
            # already there. Wrapped in try/except per-column so running
            # this against either a brand-new DB or an existing (old-schema)
            # DB is always safe and never breaks the existing invoices table.
            new_columns = {
                "rule_based_score": "REAL",
                "ml_probability":   "REAL",
                "ml_score":         "REAL",
                "final_score":      "REAL",
                "classification":   "TEXT",
            }
            for col_name, col_type in new_columns.items():
                try:
                    cursor.execute(f"ALTER TABLE invoices ADD COLUMN {col_name} {col_type}")
                except sqlite3.OperationalError:
                    # Column already exists on this DB -- nothing to do.
                    pass
            conn.commit()
            # ================= END ML ADDITION =================
    
    def insert_invoice(self, invoice_number: str, vendor_name: str,
                        # ================= ML ADDITION: optional score columns =================
                        rule_based_score: float = None,
                        ml_probability: float = None,
                        ml_score: float = None,
                        final_score: float = None,
                        classification: str = None
                        # ================= END ML ADDITION =================
                        ) -> bool:
        """
        Insert invoice with its fraud score.
        Returns True if inserted, False if duplicate.

        The new keyword arguments are optional and default to None, so any
        existing call site that only passes (invoice_number, vendor_name)
        keeps working exactly as before.
        """
        with sqlite3.connect("invoice_db.db") as conn:
            try:
                conn.execute(
                    """INSERT INTO invoices
                       (invoice_number, vendor_name, rule_based_score,
                        ml_probability, ml_score, final_score, classification)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (invoice_number, vendor_name, rule_based_score,
                     ml_probability, ml_score, final_score, classification)
                )
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False
    
        
 
    def get_all_invoices(self):
            """Fetch all stored invoices."""
            with sqlite3.connect("invoice_db.db") as conn:
                return conn.execute("SELECT * FROM invoices").fetchall()
# ──────────────────────────────────────────────
# VENDOR CORPUS
# ──────────────────────────────────────────────
 
VENDOR_CORPUS = {
    # Food Delivery
    "Zomato":       ["zomato", "eternal limited", "formerly known as zomato", "zomato limited"],
    "Swiggy":       ["swiggy", "bundl technologies", "scootsy"],
    "Dunzo":        ["dunzo", "dunzo digital"],
    "Zepto":        ["zepto", "kiranakart technologies"],
    "Blinkit":      ["blinkit", "grofers"],
    "BigBasket":    ["bigbasket", "innovative retail"],
    # Travel
    "MakeMyTrip":   ["makemytrip", "mmt", "ibibo"],
    "Goibibo":      ["goibibo"],
    "Cleartrip":    ["cleartrip"],
    "IRCTC":        ["irctc", "indian railway catering"],
    "Yatra":        ["yatra online"],
    "OYO":          ["oyo", "oravel stays"],
    # Cab / Mobility
    "Uber":         ["uber", "uber india"],
    "Ola":          ["ola", "ani technologies"],
    "Rapido":       ["rapido", "roppen transportation"],
}
 

# ──────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────
 
def process_single_invoice(pdf_path: str, db: DB_main, analyzer: Invoice_Data,
                            # ================= ML ADDITION =================
                            ml_model: "MLFraudModel" = None
                            # ================= END ML ADDITION =================
                            ) -> dict:
    """
    Full pipeline for one PDF:
      1. Extract text  → detect vendor + invoice number
      2. Run forensics → compute rule-based fraud score
      3. (ML ADDITION) Run ML model on top of the rule-based features
         → ML fraud probability → combined final score → classification
      4. Check DB for duplicate
      5. Store if new, skip if duplicate
    Returns a result summary dict.
    """
    result = {
        "file":           os.path.basename(pdf_path),
        "vendor":         None,
        "invoice_number": None,
        "status":         None,
        "score":          None,   # kept for backward compatibility (== rule_based_score)
        # ================= ML ADDITION =================
        "rule_based_score": None,
        "ml_probability":   None,
        "ml_score":          None,
        "final_score":       None,
        "classification":    None,
        # ================= END ML ADDITION =================
    }
 
    # ── Step 1: Text extraction
    try:
        text=db.pdf_reader(pdf_path)
        vendor=db.detect_vendor(text)
        invoice=db.extract_invoice_number(text,vendor)
        result['vendor']=vendor
        result['invoice_number']=invoice

    except Exception as e:
        result["status"] = f"ERROR (text extraction): {e}"
        return result
    
    # ── Step 2: Forensics (unchanged rule-based logic)
    try:
        forensics = analyzer.extract_pdf_forensics(pdf_path)
        scores = analyzer.calculate_scores(forensics)
        rule_based_score = analyzer.assign_weightage(scores, forensics)
        result["score"] = rule_based_score
        result["rule_based_score"] = rule_based_score
    except Exception as e:
        result["status"] = f"ERROR (forensics): {e}"
        return result

    # ================= ML ADDITION: Step 2b - ML scoring layer =================
    ml_probability = None
    ml_score = None
    final_score = rule_based_score  # default: rule-based only, if ML unavailable

    if ml_model is not None and ml_model.is_available():
        try:
            ml_features = analyzer.prepare_ml_features(scores)
            ml_probability = ml_model.predict_proba(ml_features)
            if ml_probability is not None:
                ml_score = ml_probability * 100
                final_score = (rule_based_score * RULE_WEIGHT) + (ml_score * ML_WEIGHT)
        except Exception as e:
            print(f"[ML] Scoring error, falling back to rule-based only: {e}")
            ml_probability = None
            ml_score = None
            final_score = rule_based_score

    result["ml_probability"] = ml_probability
    result["ml_score"] = ml_score
    result["final_score"] = final_score
    result["classification"] = classify_final_score(final_score)
    # ================= END ML ADDITION =================
 
    if not result["invoice_number"]: #or result["invoice"].isnumeric():
        result["status"] = "ERROR: invoice number not found"
        return result
 
    if not result["vendor"]:
        result["vendor"] = "Unknown"
    
    inserted = db.insert_invoice(
        result["invoice_number"], result["vendor"],
        # ================= ML ADDITION =================
        rule_based_score=result["rule_based_score"],
        ml_probability=result["ml_probability"],
        ml_score=result["ml_score"],
        final_score=result["final_score"],
        classification=result["classification"],
        # ================= END ML ADDITION =================
    )
    result["status"] = " Stored" if inserted else "DUPLICATE"

    print(result)
    return result
 
 

#BATCH
def process_pdfs_in_folder(root_folder: str) -> list[dict]:
    """Walk a folder tree and process every PDF found."""
    db       = DB_main()
    analyzer = Invoice_Data()
    db.create_db()

    # ================= ML ADDITION =================
    # Loaded once per batch run, not once per file.
    ml_model = MLFraudModel()
    # ================= END ML ADDITION =================
 
    output_data = []
 
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(".pdf"):
                file_path = os.path.join(root, file)
                result    = process_single_invoice(file_path, db, analyzer, ml_model)  # ML ADDITION: pass ml_model
                output_data.append(result)
                print(
                    f"[{result['status']}] {result['file']} | "
                    f"Vendor: {result['vendor']} | "
                    f"Invoice: {result['invoice_number']} | "
                    f"Rule Score: {result['rule_based_score']} | "     # ML ADDITION
                    f"ML Score: {result['ml_score']} | "               # ML ADDITION
                    f"Final Score: {result['final_score']} | "         # ML ADDITION
                    f"Classification: {result['classification']}"      # ML ADDITION
                )

    
 
    return output_data
 
 

if __name__ == "__main__":
     # ── Option A: Process a single PDF
     single_pdf = r"C:/Users/saisa/Documents/fake_invoice/withML/test.pdf"
 
     db       = DB_main()
     analyzer = Invoice_Data()
     db.create_db()
     ml_model = MLFraudModel()          # ================= ML ADDITION =================
 
     result = process_single_invoice(single_pdf, db, analyzer, ml_model)
     print("\n── Single Invoice Result ──")
     for k, v in result.items():
         print(f"  {k}: {v}")
 
#     # ── Option B: Batch-process a folder (uncomment to use)
#     # root_folder = r"C:\Users\nikhil.rana\OneDrive - Nangia & Co LLP\Invoice_Dataset"
#     # data = process_pdfs_in_folder(root_folder)
#     #
#     # import pandas as pd
#     # df = pd.DataFrame(data)
#     # df.to_csv("pdf_scores.csv", index=False)
#     # print("\n✅ Results saved to pdf_scores.csv")
 
#     # ── Print all DB records
#     print("\n── All stored invoices ──")
#     for row in db.get_all_invoices():
#         print(row)