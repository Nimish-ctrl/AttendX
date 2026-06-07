import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore")
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
from dotenv import load_dotenv
load_dotenv()
from deepface import DeepFace
import numpy as np
import faiss
import tempfile
from datetime import datetime
from supabase import create_client, Client
MODEL_NAME    = "Facenet512"
EMBEDDING_DIM = 512
THRESHOLD     = 0.60
INDEX_PATH    = "face_index.faiss"
BUCKET        = "faiss"
FAISS_OBJECT  = "face_index.faiss"
_sb: Client = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_KEY"],
)
_index = None
_meta: dict[int, dict] = {}


def get_embedding(image_path: str) -> np.ndarray:
    result = DeepFace.represent(
        img_path          = image_path,
        model_name        = MODEL_NAME,
        enforce_detection = True,
        detector_backend  = "retinaface",
    )
    if len(result) == 0:
        raise ValueError("No face detected.")
    if len(result) > 1:
        raise ValueError("Multiple faces detected — use a solo photo.")
    vec  = np.array(result[0]["embedding"], dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("Zero-norm embedding — image may be corrupt.")
    return vec / norm


def _download_index() -> bool:
    try:
        data = _sb.storage.from_(BUCKET).download(FAISS_OBJECT)
        with open(INDEX_PATH, "wb") as f:
            f.write(data)
        return True
    except Exception:
        return False
    

def _upload_index() -> None:
    with open(INDEX_PATH, "rb") as f:
        data = f.read()
    _sb.storage.from_(BUCKET).upload(
        FAISS_OBJECT, data,
        {"content-type": "application/octet-stream", "upsert": "true"},
    )


def _build_meta_from_db() -> None:
    global _meta
    rows = _sb.table("students").select("roll_no, name, faiss_id, enrolled_at").execute()
    _meta = {
        row["faiss_id"]: {"roll_no": row["roll_no"], "name": row["name"], "enrolled_at": row["enrolled_at"]}
        for row in rows.data
    }


def load_or_create_index() -> None:
    global _index, _meta
    found = _download_index()
    if found:
        _index = faiss.read_index(INDEX_PATH)
        _build_meta_from_db()
        print(f"[model] Loaded index — {_index.ntotal} enrolled faces.")
    else:
        _index = faiss.IndexFlatIP(EMBEDDING_DIM)
        _meta  = {}
        print("[model] Fresh FAISS index created.")



def enroll(roll_no: str, name: str, image_path: str) -> dict:
    existing = _sb.table("students").select("roll_no").eq("roll_no", roll_no).execute()
    if existing.data:
        raise ValueError(f"Roll no. '{roll_no}' is already enrolled.")
    vec       = get_embedding(image_path)
    faiss_id  = _index.ntotal
    enrolled_at = datetime.now().isoformat()
    _index.add(vec.reshape(1, -1))
    _meta[faiss_id] = {"roll_no": roll_no, "name": name, "enrolled_at": enrolled_at}
    faiss.write_index(_index, INDEX_PATH)
    _upload_index()
    _sb.table("students").insert({
        "roll_no"    : roll_no,
        "name"       : name,
        "faiss_id"   : faiss_id,
        "enrolled_at": enrolled_at,
    }).execute()
    print(f"[enroll] '{name}' ({roll_no}) enrolled — faiss_id={faiss_id}")
    return {"status": "enrolled", "roll_no": roll_no, "name": name, "enrolled_at": enrolled_at}


def recognize(image_path: str) -> dict:
    if _index.ntotal == 0:
        return {"status": "error", "message": "No students enrolled yet."}
    try:
        vec = get_embedding(image_path)
    except ValueError as e:
        return {"status": "error", "message": str(e)}
    similarities, indices = _index.search(vec.reshape(1, -1), k=1)
    best_sim = float(similarities[0][0])
    best_idx = int(indices[0][0])
    if best_sim >= THRESHOLD:
        match      = _meta[best_idx]
        marked_now = log_attendance(match["roll_no"], match["name"], round(best_sim, 4))
        return {
            "status"        : "identified",
            "roll_no"       : match["roll_no"],
            "name"          : match["name"],
            "similarity"    : round(best_sim, 4),
            "enrolled_at"   : match["enrolled_at"],
            "marked_at"     : datetime.now().isoformat(),
            "already_marked": not marked_now,
        }
    print(f"[recognize] No match (best_sim={best_sim:.3f} < {THRESHOLD})")
    return {"status": "not_identified", "similarity": round(best_sim, 4)}


def log_attendance(roll_no: str, name: str, similarity: float) -> bool:
    today = datetime.now().date().isoformat()
    existing = (
        _sb.table("attendance")
        .select("id")
        .eq("roll_no", roll_no)
        .gte("marked_at", f"{today}T00:00:00")
        .lte("marked_at", f"{today}T23:59:59")
        .execute()
    )
    if existing.data:
        print(f"[attendance] {roll_no} already marked today.")
        return False
    _sb.table("attendance").insert({
        "roll_no"   : roll_no,
        "name"      : name,
        "similarity": similarity,
        "marked_at" : datetime.now().isoformat(),
    }).execute()
    print(f"[attendance] Marked: {name} ({roll_no})")
    return True


def list_enrolled() -> list[dict]:
    rows = _sb.table("students").select("*").order("enrolled_at").execute()
    return rows.data


def get_attendance(date: str = None) -> list[dict]:
    q = _sb.table("attendance").select("*").order("marked_at", desc=True)
    if date:
        q = q.gte("marked_at", f"{date}T00:00:00").lte("marked_at", f"{date}T23:59:59")
    return q.execute().data


def delete_student(roll_no: str) -> dict:
    global _index
    target_id = None
    for fid, meta in _meta.items():
        if meta["roll_no"] == roll_no:
            target_id = fid
            break
    if target_id is None:
        raise ValueError(f"Roll no. '{roll_no}' not found.")
    remaining = [fid for fid in _meta if fid != target_id]
    new_index = faiss.IndexFlatIP(EMBEDDING_DIM)
    new_meta  = {}
    for new_id, old_id in enumerate(remaining):
        vec = _index.reconstruct(old_id)
        new_index.add(vec.reshape(1, -1))
        new_meta[new_id] = _meta[old_id]
    _index = new_index
    _meta.clear()
    _meta.update(new_meta)
    faiss.write_index(_index, INDEX_PATH)
    _upload_index()
    _sb.table("students").delete().eq("roll_no", roll_no).execute()
    print(f"[delete] Removed '{roll_no}'. Index now has {_index.ntotal} faces.")
    return {"status": "deleted", "roll_no": roll_no}
load_or_create_index()
if __name__ == "__main__":
    import sys
    print("\n=== Attendance Model — Quick Test ===\n")
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "enroll" and len(sys.argv) == 5:
        print(enroll(sys.argv[2], sys.argv[3], sys.argv[4]))
    elif cmd == "recognize" and len(sys.argv) == 3:
        print(recognize(sys.argv[2]))
    elif cmd == "list":
        students = list_enrolled()
        print(f"{len(students)} enrolled student(s):")
        for s in students: print(f"  {s['roll_no']} — {s['name']} — enrolled {s['enrolled_at']}")
    elif cmd == "attendance":
        date = sys.argv[2] if len(sys.argv) == 3 else None
        records = get_attendance(date)
        print(f"{len(records)} record(s):")
        for r in records: print(f"  {r['roll_no']} — {r['name']} — {r['marked_at']}")
    elif cmd == "delete" and len(sys.argv) == 3:
        print(delete_student(sys.argv[2]))
    else:
        print("Usage:")
        print("  python model.py enroll <roll_no> <name> <image_path>")
        print("  python model.py recognize <image_path>")
        print("  python model.py list")
        print("  python model.py attendance [YYYY-MM-DD]")
        print("  python model.py delete <roll_no>")