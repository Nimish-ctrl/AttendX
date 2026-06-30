import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

import shutil
import tempfile
import random
import string
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr
import model
import auth
from supabase import create_client

app = FastAPI(title="Attendance Portal", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

def _save_upload(file: UploadFile):
    suffix = os.path.splitext(file.filename)[-1] or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    with tmp as f:
        shutil.copyfileobj(file.file, f)
    return tmp.name

def _generate_join_code() -> str:
    # changed: 6-char alphanumeric code, like Google Classroom
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))

def _resolve_join_code(join_code: str) -> dict:
    # changed: looks up a classroom by its join_code, used by student enroll flow
    result = _sb.table("classrooms").select("*").eq("join_code", join_code.upper()).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Invalid join code.")
    return result.data[0]

@app.get("/health")
def root():
    return {"status": "ok"}


class RegisterBody(BaseModel):
    name: str
    email: EmailStr
    password: str

class LoginBody(BaseModel):
    email: EmailStr
    password: str

class ClassroomBody(BaseModel):
    name: str


@app.post("/teachers/register")
def register_teacher(body: RegisterBody):
    existing = _sb.table("teachers").select("id").eq("email", body.email).execute()
    if existing.data:
        raise HTTPException(status_code=400, detail="Email already registered.")

    hashed = auth.hash_password(body.password)
    result = _sb.table("teachers").insert({
        "name": body.name,
        "email": body.email,
        "password_hash": hashed,
    }).execute()

    teacher = result.data[0]
    token = auth.create_token(teacher["id"], teacher["email"])
    return {"token": token, "name": teacher["name"], "email": teacher["email"]}


@app.post("/teachers/login")
def login_teacher(body: LoginBody):
    result = _sb.table("teachers").select("*").eq("email", body.email).execute()
    if not result.data:
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    teacher = result.data[0]
    if not auth.verify_password(body.password, teacher["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    token = auth.create_token(teacher["id"], teacher["email"])
    return {"token": token, "name": teacher["name"], "email": teacher["email"]}


# ── classroom CRUD — teacher-only ─────────────────────────────────────────
@app.post("/classrooms")
def create_classroom(body: ClassroomBody, teacher=Depends(auth.get_current_teacher)):
    # changed: generate a unique join code, retry on collision
    code = _generate_join_code()
    while _sb.table("classrooms").select("id").eq("join_code", code).execute().data:
        code = _generate_join_code()

    result = _sb.table("classrooms").insert({
        "name": body.name,
        "teacher_id": teacher["teacher_id"],
        "join_code": code,
    }).execute()
    return result.data[0]


@app.get("/classrooms")
def list_classrooms(teacher=Depends(auth.get_current_teacher)):
    result = (
        _sb.table("classrooms")
        .select("*")
        .eq("teacher_id", teacher["teacher_id"])
        .order("created_at", desc=True)
        .execute()
    )
    return {"classrooms": result.data}


@app.delete("/classrooms/{classroom_id}")
def delete_classroom(classroom_id: str, teacher=Depends(auth.get_current_teacher)):
    owned = (
        _sb.table("classrooms")
        .select("id")
        .eq("id", classroom_id)
        .eq("teacher_id", teacher["teacher_id"])
        .execute()
    )
    if not owned.data:
        raise HTTPException(status_code=404, detail="Classroom not found.")

    _sb.table("classrooms").delete().eq("id", classroom_id).execute()
    model.delete_classroom_index(classroom_id)
    return {"status": "deleted", "classroom_id": classroom_id}


# ── student-facing: join a classroom via code ─────────────────────────────
@app.post("/enroll")
def enroll_student(
    join_code: str = Form(...),   # changed: students type a join code, not a UUID
    roll_no: str = Form(...),
    name: str = Form(...),
    photo: UploadFile = File(...),
):
    classroom = _resolve_join_code(join_code)
    tmp_path = _save_upload(photo)
    try:
        result = model.enroll(classroom["id"], roll_no, name, tmp_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        os.unlink(tmp_path)
    result["classroom_name"] = classroom["name"]
    return result


@app.post("/recognize")
def recognize_face(
    classroom_id: str = Form(...),   # attend.html still uses classroom_id directly (machine is pre-configured per room)
    photo: UploadFile = File(...),
):
    tmp_path = _save_upload(photo)
    try:
        result = model.recognize(classroom_id, tmp_path)
    finally:
        os.unlink(tmp_path)
    return result


# ── student dashboard: see all classrooms they're enrolled in ─────────────
@app.get("/students/{roll_no}/classrooms")
def student_classrooms(roll_no: str):
    # changed: new endpoint — Google-Classroom-style "my classes" view
    classrooms = model.get_student_classrooms(roll_no)
    return {"roll_no": roll_no, "total": len(classrooms), "classrooms": classrooms}


@app.get("/my-attendance")
def my_attendance(roll_no: str = Query(...), classroom_id: str = Query(...)):
    all_records = model.get_attendance(classroom_id)
    records = [r for r in all_records if r["roll_no"] == roll_no]
    if not records:
        return {"roll_no": roll_no, "total": 0, "records": []}
    records.sort(key=lambda r: r["marked_at"], reverse=True)
    return {
        "roll_no": roll_no,
        "name": records[0]["name"],
        "total": len(records),
        "records": records,
    }


# ── teacher-facing: attendance + student management, scoped + owned check ─
def _verify_classroom_ownership(classroom_id: str, teacher_id: str):
    owned = (
        _sb.table("classrooms")
        .select("id, name")
        .eq("id", classroom_id)
        .eq("teacher_id", teacher_id)
        .execute()
    )
    if not owned.data:
        raise HTTPException(status_code=404, detail="Classroom not found.")
    return owned.data[0]


@app.get("/attendance")
def attendance(
    classroom_id: str = Query(...),
    date: str = Query(...),
    teacher=Depends(auth.get_current_teacher),
):
    _verify_classroom_ownership(classroom_id, teacher["teacher_id"])

    enrolled = model.list_enrolled(classroom_id)
    present_records = model.get_attendance(classroom_id, date)
    present_roll_nos = {r["roll_no"] for r in present_records}
    report = []
    for student in enrolled:
        roll_number = student["roll_no"]
        attended = roll_number in present_roll_nos
        entry = {
            "roll_no": roll_number,
            "name": student["name"],
            "enrolled_at": student["enrolled_at"],
            "present": attended,
        }
        if attended:
            for r in present_records:
                if r["roll_no"] == roll_number:
                    rec = r
                    break
            entry["marked_at"] = rec["marked_at"]
            entry["similarity"] = rec["similarity"]
        report.append(entry)
    report.sort(key=lambda x: (not x["present"], x["name"]))
    return {
        "date": date,
        "total": len(enrolled),
        "attended": len(present_roll_nos),
        "absent": len(enrolled) - len(present_roll_nos),
        "report": report,
    }


@app.get("/classrooms/{classroom_id}/students")
def list_students(classroom_id: str, teacher=Depends(auth.get_current_teacher)):
    _verify_classroom_ownership(classroom_id, teacher["teacher_id"])
    students = model.list_enrolled(classroom_id)
    return {"total": len(students), "students": students}


@app.delete("/classrooms/{classroom_id}/students/{roll_no}")
def delete_student(classroom_id: str, roll_no: str, teacher=Depends(auth.get_current_teacher)):
    _verify_classroom_ownership(classroom_id, teacher["teacher_id"])
    try:
        return model.delete_student(classroom_id, roll_no)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


app.mount("/", StaticFiles(directory=".", html=True), name="static")