# AttendX

**Face-recognition attendance, without the hardware.**

AttendX replaces roll calls and ID-card swipes with on-device face recognition. An organizer creates a space, shares a join code or QR, and from then on attendance is a five-second face scan — on a kiosk, a teacher's laptop, or a student's own phone.

🔗 **Live demo:** https://nimish-ctrl-attendx.hf.space/

---

## Why this exists

Manual attendance is slow, easy to fake, and a genuine point of friction in classrooms, workshops, and recurring events. Card/QR-only systems solve the speed problem but not the fraud problem — anyone can carry someone else's card.

AttendX uses **1:N face recognition** (not 1:1 verification): the system is never told who to expect, it has to identify the person from the enrolled roster itself. No claimed identity, no card, no PIN to share.

---

## Features

- **Spaces, not just classrooms** — organizers create any number of independent spaces (a class, a team, an event), each with its own auto-generated join code and isolated student roster.
- **One-time face enrollment** — students join with a code, ID, name, and a single photo. No re-enrollment per session.
- **True face recognition, not verification** — scans search the full enrolled roster for a match; nothing is typed in to confirm identity beforehand.
- **QR-based scan stations** — every space gets a shareable scan-station link and QR code. Students scan on their own device — no dedicated kiosk hardware required per room.
- **Per-space isolation** — each space has its own FAISS similarity index. A face enrolled in one space is never matched against another.
- **Live daily reports** — present/absent breakdown per date, with match confidence and timestamps, scoped to the organizer who owns the space.
- **Student self-service dashboard** — students can look up every space they've joined and their full attendance history, no login required.
- **JWT-authenticated organizer accounts** — register/login, bcrypt-hashed passwords, scoped access (organizers can only see and manage their own spaces).

---

## How it works

1. **Organizer** creates a space → gets a 6-character join code.
2. **Student** enters the code + ID + name + a face photo on `student.html` → a 512-dim face embedding (Facenet512) is generated and added to that space's FAISS index, stored in Supabase Storage.
3. **Scan** — at `attend.html?id=<space_id>` (opened directly or via QR), a face is captured, embedded, and matched (cosine similarity) against that space's index only. A match above threshold marks attendance for that day; below threshold, no match.
4. **Organizer** reviews attendance per date in `classroom.html`, or removes students (which also purges their face data from the index).

---

## Tech stack

| Layer | Tech |
|---|---|
| API | FastAPI (Python) |
| Auth | JWT (python-jose) + bcrypt password hashing (passlib) |
| Face embeddings | DeepFace — Facenet512 model, OpenCV detector backend |
| Similarity search | FAISS (`IndexFlatIP`, one index per space) |
| Database | Supabase (Postgres) |
| File/index storage | Supabase Storage |
| Frontend | Vanilla HTML/CSS/JS, no framework or build step |
| Deployment | Docker → Hugging Face Spaces |

---

## Project structure

```
.
├── main.py              # FastAPI app, routes
├── model.py              # Face embedding, FAISS indexing, attendance logic
├── auth.py                # JWT issuing/verification, password hashing
├── warmup.py              # Pre-caches model weights at build time
├── index.html              # Landing page
├── login.html               # Organizer register/login
├── dashboard.html             # Organizer: list/create spaces
├── classroom.html               # Organizer: daily report, members, QR code
├── student.html                  # Student: join a space, view attendance
├── attend.html                     # Scan station (per-space)
├── requirements.txt
├── Dockerfile
└── .dockerignore / .gitignore
```

---

## Getting started

### Prerequisites
- Python 3.10+
- A [Supabase](https://supabase.com) project with:
  - Tables: `teachers`, `classrooms`, `students`, `classroom_students`, `attendance`
  - A Storage bucket named `faiss`

### Local setup

```bash
git clone https://github.com/Nimish-ctrl/AttendX.git
cd AttendX

pip install -r requirements.txt
python warmup.py        # pre-downloads the Facenet512 weights, one-time

uvicorn main:app --host 0.0.0.0 --port 7860 --reload
```

Create a `.env` file in the project root:

```env
SUPABASE_URL=https://yourproject.supabase.co
SUPABASE_KEY=your-supabase-key
JWT_SECRET=a-long-random-string
```

Then open `http://localhost:7860/index.html`.

### Docker

```bash
docker build -t attendx .
docker run --env-file .env -p 7860:7860 attendx
```

### Deploying to Hugging Face Spaces

1. Create a Docker-type Space.
2. Push this repo to the Space's git remote.
3. Add `SUPABASE_URL`, `SUPABASE_KEY`, and `JWT_SECRET` under **Settings → Variables and secrets** (not `.env` — Spaces don't read it).

---

## API overview

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/teachers/register` | Create an organizer account |
| `POST` | `/teachers/login` | Organizer login → JWT |
| `POST` | `/classrooms` | Create a space (auth) |
| `GET` | `/classrooms` | List organizer's spaces (auth) |
| `DELETE` | `/classrooms/{id}` | Delete a space + its face index (auth) |
| `POST` | `/enroll` | Student joins a space via join code + photo |
| `POST` | `/recognize` | Scan a face, mark attendance for a space |
| `GET` | `/attendance` | Daily report for a space (auth) |
| `GET` | `/classrooms/{id}/students` | List enrolled students (auth) |
| `DELETE` | `/classrooms/{id}/students/{roll_no}` | Remove a student + their face data (auth) |
| `GET` | `/students/{roll_no}/classrooms` | All spaces a student has joined |
| `GET` | `/my-attendance` | A student's attendance history in one space |

---

## Roadmap

- [ ] Time-windowed scan sessions (prevent scanning outside class hours)
- [ ] Custom domain for the deployed app
- [ ] Bulk student import/export
- [ ] Analytics across multiple spaces for an organizer

---

Built by [Nimish Wadhwa](https://github.com/Nimish-ctrl)
