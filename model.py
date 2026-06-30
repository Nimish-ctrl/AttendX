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
from datetime import datetime
from supabase import create_client, Client

MODEL_NAME    = "Facenet512"
EMBEDDING_DIM = 512
THRESHOLD     = 0.60
BUCKET        = "faiss"

_sb: Client = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_KEY"],
)

# changed: indexes and meta are now per-classroom, keyed by classroom_id
# _indexes  = { classroom_id: faiss.IndexFlatIP }
# _metas    = { classroom_id: {faiss_id: {roll_no, name, enrolled_at}} }
_indexes: dict[str, faiss.IndexFlatIP] = {}
_metas: dict[str, dict[int, dict]] = {}


def get_embedding(image_path: str) -> np.ndarray:
    result = DeepFace.represent(
        img_path          = image_path,
        model_name        = MODEL_NAME,
        enforce_detection = True,
        detector_backend  = "opencv",
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


def _index_path(classroom_id: str) -> str:
    # changed: one local temp path per classroom
    return f"/tmp_{classroom_id}.faiss" if os.name != "nt" else f"index_{classroom_id}.faiss"


def _object_name(classroom_id: str) -> str:
    return f"{classroom_id}.faiss"


def _download_index(classroom_id: str) -> bool:
    try:
        data = _sb.storage.from_(BUCKET).download(_object_name(classroom_id))
        with open(_index_path(classroom_id), "wb") as f:
            f.write(data)
        return True
    except Exception:
        return False


def _upload_index(classroom_id: str) -> None:
    with open(_index_path(classroom_id), "rb") as f:
        data = f.read()
    _sb.storage.from_(BUCKET).upload(
        _object_name(classroom_id), data,
        {"content-type": "application/octet-stream", "upsert": "true"},
    )


def _build_meta_from_db(classroom_id: str) -> dict[int, dict]:
    rows = (
        _sb.table("classroom_students")
        .select("roll_no, faiss_id, enrolled_at, students(name)")
        .eq("classroom_id", classroom_id)
        .execute()
    )
    meta = {}
    for row in rows.data:
        meta[row["faiss_id"]] = {
            "roll_no": row["roll_no"],
            "name": row["students"]["name"],
            "enrolled_at": row["enrolled_at"],
        }
    return meta


def _get_index(classroom_id: str) -> faiss.IndexFlatIP:
    # changed: lazy-load — only loads a classroom's index into RAM on first use
    if classroom_id in _indexes:
        return _indexes[classroom_id]

    found = _download_index(classroom_id)
    if found:
        _indexes[classroom_id] = faiss.read_index(_index_path(classroom_id))
        _metas[classroom_id] = _build_meta_from_db(classroom_id)
        print(f"[model] Loaded index for classroom {classroom_id} — {_indexes[classroom_id].ntotal} faces.")
    else:
        _indexes[classroom_id] = faiss.IndexFlatIP(EMBEDDING_DIM)
        _metas[classroom_id] = {}
        print(f"[model] Fresh index created for classroom {classroom_id}.")

    return _indexes[classroom_id]


def enroll(classroom_id: str, roll_no: str, name: str, image_path: str) -> dict:
    index = _get_index(classroom_id)
    meta  = _metas[classroom_id]

    # changed: duplicate check is now scoped to this classroom only
    existing = (
        _sb.table("classroom_students")
        .select("roll_no")
        .eq("classroom_id", classroom_id)
        .eq("roll_no", roll_no)
        .execute()
    )
    if existing.data:
        raise ValueError(f"'{roll_no}' is already enrolled in this classroom.")

    vec         = get_embedding(image_path)
    faiss_id    = index.ntotal
    enrolled_at = datetime.now().isoformat()

    index.add(vec.reshape(1, -1))
    meta[faiss_id] = {"roll_no": roll_no, "name": name, "enrolled_at": enrolled_at}

    faiss.write_index(index, _index_path(classroom_id))
    _upload_index(classroom_id)

    # changed: ensure student exists in global students table (identity only)
    student_exists = _sb.table("students").select("roll_no").eq("roll_no", roll_no).execute()
    if not student_exists.data:
        _sb.table("students").insert({"roll_no": roll_no, "name": name}).execute()

    # changed: link student to this classroom
    _sb.table("classroom_students").insert({
        "classroom_id": classroom_id,
        "roll_no": roll_no,
        "faiss_id": faiss_id,
        "enrolled_at": enrolled_at,
    }).execute()

    print(f"[enroll] '{name}' ({roll_no}) enrolled in classroom {classroom_id} — faiss_id={faiss_id}")
    return {"status": "enrolled", "roll_no": roll_no, "name": name, "enrolled_at": enrolled_at}


def recognize(classroom_id: str, image_path: str) -> dict:
    index = _get_index(classroom_id)
    meta  = _metas[classroom_id]

    if index.ntotal == 0:
        return {"status": "error", "message": "No students enrolled in this classroom yet."}

    try:
        vec = get_embedding(image_path)
    except ValueError as e:
        return {"status": "error", "message": str(e)}

    similarities, indices = index.search(vec.reshape(1, -1), k=1)
    best_sim = float(similarities[0][0])
    best_idx = int(indices[0][0])

    if best_sim >= THRESHOLD:
        match      = meta[best_idx]
        marked_now = log_attendance(classroom_id, match["roll_no"], match["name"], round(best_sim, 4))
        return {
            "status"        : "identified",
            "roll_no"       : match["roll_no"],
            "name"          : match["name"],
            "similarity"    : round(best_sim, 4),
            "enrolled_at"   : match["enrolled_at"],
            "marked_at"     : datetime.now().isoformat(),
            "already_marked": not marked_now,
        }

    print(f"[recognize] No match in classroom {classroom_id} (best_sim={best_sim:.3f} < {THRESHOLD})")
    return {"status": "not_identified", "similarity": round(best_sim, 4)}


def log_attendance(classroom_id: str, roll_no: str, name: str, similarity: float) -> bool:
    today = datetime.now().date().isoformat()

    existing = (
        _sb.table("attendance")
        .select("id")
        .eq("classroom_id", classroom_id)
        .eq("roll_no", roll_no)
        .gte("marked_at", f"{today}T00:00:00")
        .lte("marked_at", f"{today}T23:59:59")
        .execute()
    )
    if existing.data:
        print(f"[attendance] {roll_no} already marked today in classroom {classroom_id}.")
        return False

    _sb.table("attendance").insert({
        "classroom_id": classroom_id,
        "roll_no": roll_no,
        "name": name,
        "similarity": similarity,
        "marked_at": datetime.now().isoformat(),
    }).execute()

    print(f"[attendance] Marked: {name} ({roll_no}) in classroom {classroom_id}")
    return True


def list_enrolled(classroom_id: str) -> list[dict]:
    rows = (
        _sb.table("classroom_students")
        .select("roll_no, enrolled_at, students(name)")
        .eq("classroom_id", classroom_id)
        .order("enrolled_at")
        .execute()
    )
    return [
        {"roll_no": r["roll_no"], "name": r["students"]["name"], "enrolled_at": r["enrolled_at"]}
        for r in rows.data
    ]


def get_attendance(classroom_id: str, date: str = None) -> list[dict]:
    q = _sb.table("attendance").select("*").eq("classroom_id", classroom_id).order("marked_at", desc=True)
    if date:
        q = q.gte("marked_at", f"{date}T00:00:00").lte("marked_at", f"{date}T23:59:59")
    return q.execute().data


def delete_student(classroom_id: str, roll_no: str) -> dict:
    index = _get_index(classroom_id)
    meta  = _metas[classroom_id]

    target_id = None
    for fid, m in meta.items():
        if m["roll_no"] == roll_no:
            target_id = fid
            break
    if target_id is None:
        raise ValueError(f"'{roll_no}' not found in this classroom.")

    remaining = [fid for fid in meta if fid != target_id]
    new_index = faiss.IndexFlatIP(EMBEDDING_DIM)
    new_meta  = {}
    for new_id, old_id in enumerate(remaining):
        vec = index.reconstruct(old_id)
        new_index.add(vec.reshape(1, -1))
        new_meta[new_id] = meta[old_id]

    _indexes[classroom_id] = new_index
    _metas[classroom_id] = new_meta

    faiss.write_index(new_index, _index_path(classroom_id))
    _upload_index(classroom_id)

    _sb.table("classroom_students").delete().eq("classroom_id", classroom_id).eq("roll_no", roll_no).execute()

    print(f"[delete] Removed '{roll_no}' from classroom {classroom_id}. Index now has {new_index.ntotal} faces.")
    return {"status": "deleted", "roll_no": roll_no, "classroom_id": classroom_id}


def delete_classroom_index(classroom_id: str) -> None:
    # changed: new function — cleans up RAM + storage when a classroom is deleted
    _indexes.pop(classroom_id, None)
    _metas.pop(classroom_id, None)
    try:
        _sb.storage.from_(BUCKET).remove([_object_name(classroom_id)])
    except Exception:
        pass
    local_path = _index_path(classroom_id)
    if os.path.exists(local_path):
        os.remove(local_path)
    print(f"[delete] Cleaned up index for classroom {classroom_id}.")


def get_student_classrooms(roll_no: str) -> list[dict]:
    # changed: new function — used by student dashboard to show all enrolled classes
    rows = (
        _sb.table("classroom_students")
        .select("classroom_id, enrolled_at, classrooms(name, join_code, teachers(name))")
        .eq("roll_no", roll_no)
        .execute()
    )
    return [
        {
            "classroom_id": r["classroom_id"],
            "classroom_name": r["classrooms"]["name"],
            "join_code": r["classrooms"]["join_code"],
            "teacher_name": r["classrooms"]["teachers"]["name"],
            "enrolled_at": r["enrolled_at"],
        }
        for r in rows.data
    ]