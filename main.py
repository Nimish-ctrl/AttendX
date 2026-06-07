import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")


from dotenv import load_dotenv
load_dotenv()

import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles 
import model

app = FastAPI(title="Attendance Portal", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ADMIN_KEY = os.getenv("ADMIN_KEY", "IITH2026")

def verify_admin(key: str = Query(..., description="Admin secret key")):
    if key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Invalid admin key.")

def _save_upload(file: UploadFile):
    suffix = os.path.splitext(file.filename)[-1] or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    with tmp as f:
        shutil.copyfileobj(file.file, f)
    return tmp.name

@app.get("/health")  # changed: was '/' — avoids conflict with StaticFiles serving index.html
def root():
    return {"status": "ok", "enrolled": model._index.ntotal}

@app.post("/enroll")
def enroll_student(
    roll_no: str = Form(...),
    name: str = Form(...),
    photo: UploadFile = File(...),
):
    tmp_path = _save_upload(photo)
    try:
        result = model.enroll(roll_no, name, tmp_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        os.unlink(tmp_path)
    return result

@app.post("/recognize")
def recognize_face(photo: UploadFile = File(...)):
    tmp_path = _save_upload(photo)
    try:
        result = model.recognize(tmp_path)
    finally:
        os.unlink(tmp_path)
    return result

@app.get("/my-attendance")
def my_attendance(roll_no: str = Query(...)):
    all_records = model.get_attendance()
    records = [r for r in all_records if r["roll_no"] == roll_no]
    if not records:
        enrolled = model.list_enrolled()
        known = any(s["roll_no"] == roll_no for s in enrolled)
        if not known:
            raise HTTPException(status_code=404, detail=f"Roll no. '{roll_no}' not found.")
        return {"roll_no": roll_no, "total": 0, "records": []}
    records.sort(key=lambda r: r["marked_at"], reverse=True)
    return {
        "roll_no": roll_no,
        "name": records[0]["name"],
        "total": len(records),
        "records": records,
    }

@app.get("/attendance")
def attendance(
    date: str = Query(...),
    _: None = Depends(verify_admin),
):
    enrolled = model.list_enrolled()
    present_records = model.get_attendance(date)
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

@app.get("/students")
def list_students(_: None = Depends(verify_admin)):
    students = model.list_enrolled()
    return {"total": len(students), "students": students}

@app.delete("/students/{roll_no}")
def delete_student(roll_no: str, _: None = Depends(verify_admin)):
    try:
        return model.delete_student(roll_no)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


app.mount("/", StaticFiles(directory=".", html=True), name="static")