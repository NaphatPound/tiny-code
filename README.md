# tinycoder

ผู้ช่วยเขียนโค้ดในเทอร์มินอลแบบเดียวกับ Claude Code / Codex CLI / Gemini CLI
แต่ออกแบบมาเพื่อให้ **LLM ตัวเล็กๆ ที่รัน local** (เช่น Qwen2.5-Coder 1.5B, Llama
3.2 3B, Phi-3 mini) สามารถ **เขียนโค้ดลงไฟล์จริง** ได้

## ทำไมโมเดลเล็กถึงเขียนไฟล์ไม่ได้ — และเราแก้ยังไง

โมเดลใหญ่ๆ (Claude, GPT-4, Gemini Pro) ทำ **function calling / tool use** ได้
เพราะมันเก่งพอจะ output JSON ที่ถูก schema ตลอดเวลา โมเดลเล็กทำไม่ได้ มันมัก
ตอบเป็นข้อความธรรมดา, markdown ที่ฟอร์แมตเพี้ยน, หรือ JSON ที่ syntax ผิด

**tinycoder แก้ปัญหานี้ด้วย constrained decoding**: เราส่ง JSON schema ให้ inference
backend (Ollama, llama.cpp, vLLM, LM Studio) ซึ่งจะ **บังคับ token-level**
ให้โมเดลออก output ที่ตรง schema 100% — แม้แต่โมเดล 1.5B ก็ทำได้

## สถาปัตยกรรม

```
user prompt
    │
    ▼
┌──────────┐    JSON schema     ┌──────────────┐
│  Agent   │ ─────────────────▶ │ Backend      │
│  loop    │                    │ (Ollama /    │
│          │ ◀───── JSON ────── │  llama.cpp / │
└──────────┘                    │  openai)     │
    │                           └──────────────┘
    │ parse → Pydantic Action       constrained
    ▼                                decoding
┌──────────┐
│Executor  │  write_file / read_file / edit_file
│(sandbox) │  list_files / run_command / finish
└──────────┘
    │ result string
    └───── feed back to agent ─────┐
                                   │
                                   └─▶ next turn
```

หัวใจคือ **one-action-per-turn** + **constrained JSON output**:
โมเดลเล็กไม่ต้องคิดเยอะ ไม่ต้องวางแผนยาว ไม่ต้อง format อะไร แค่ตัดสินใจ
"ตาต่อไปทำอะไร" แล้วเรา execute ให้ ผลลัพธ์ส่งกลับเข้าไปเป็น context ของรอบถัดไป

## ติดตั้ง

```bash
pip install -e .
```

ต้องมี Python 3.10+

## ใช้งาน

### กับ Ollama (แนะนำสำหรับเริ่มต้น)

```bash
ollama pull qwen2.5-coder:1.5b
tinycoder --backend ollama --model qwen2.5-coder:1.5b \
    "create a fastapi hello world in app.py"
```

### กับ llama.cpp server

```bash
# รัน llama.cpp server แยกก่อน:
# ./llama-server -m qwen2.5-coder-1.5b.gguf --port 8080

tinycoder --backend llamacpp --model qwen2.5-coder --base-url http://localhost:8080 \
    "write a python script that fetches example.com and prints the title"
```

### กับ LM Studio / vLLM / OpenAI-compat ใดๆ

```bash
tinycoder --backend openai \
    --model your-model-id \
    --base-url http://localhost:1234/v1 \
    "create a CLI todo app in todo.py"
```

### Interactive mode

ไม่ใส่ prompt = เปิด REPL

```bash
tinycoder --backend ollama --model qwen2.5-coder:1.5b
you> create hello.py that prints hi
→ write_file  Creating the requested hello.py.
  ✓ wrote 11 bytes to hello.py
→ finish  File created.
you> /quit
```

### ตัวเลือกที่ใช้บ่อย

| flag | คำอธิบาย |
| --- | --- |
| `--workspace DIR` | โฟลเดอร์ที่ agent มีสิทธิ์อ่าน/เขียน (default `./workspace`) |
| `--max-steps N` | จำกัด iteration ต่อ 1 คำขอ (default 20) |
| `--allow-commands` | เปิดให้ agent รัน shell command (ปิดโดยปริยายเพื่อความปลอดภัย) |
| `--planner-model NAME` | เปิดโหมด big-AI planner (e.g. `qwen3-coder:cloud`) — วางแผน + รีวิว + intervene |
| `--scaffold` | สแกฟโฟลด์โหมด: big AI เขียนโครงไฟล์ครบทุกบรรทัด เหลือแค่ `TODO(small-ai):` หลุมเล็กๆ ให้ small AI เติม แล้ว big AI รัน verify + ซ่อมเอง (ต้องใช้คู่กับ `--planner-model`) |
| `--log-dir DIR` | บันทึก session ทุก action/intervention/review ลง JSON 1 ไฟล์ต่อ 1 คำขอ — ส่งให้ AI ตัวใหญ่กว่าวิเคราะห์ปรับ workflow ได้ |

หรือใช้ env var: `TINYCODER_BACKEND`, `TINYCODER_MODEL`, `TINYCODER_BASE_URL`,
`TINYCODER_WORKSPACE`, `TINYCODER_SCAFFOLD`, `TINYCODER_LOG_DIR`

## Scaffold mode (big AI guide → small AI fill → big AI verify)

ปัญหา: ปล่อยให้โมเดล 1.5B–3B เขียน React/Vite app ตั้งแต่ศูนย์มัน "เกือบ" จะรันได้
เสมอ — package.json ขาด dep, index.html อยู่ผิดที่, JSX ใช้ .js, ฯลฯ. มันออกแบบ
โครงสร้างไฟล์ไม่เก่งพอ

ทางแก้: `--scaffold` flag ให้ big AI (ผ่าน `--planner-model`) ทำ 3 อย่าง:

1. **SCAFFOLD** — เขียนไฟล์ทั้งหมดครบโครงสร้าง (imports/exports/scripts ครบ
   ถูก ทุกบรรทัด) เหลือไว้แค่หลุม `TODO(small-ai): …` เล็กๆ ตรงที่ logic ของ
   feature ต้องไป
2. **FILL** (วนทีละหลุม) — ส่งคำสั่งแคบๆ ให้ small AI: "ในไฟล์นี้ เปลี่ยน
   บรรทัด `// TODO(small-ai): xyz` เป็น <implementation>" — งานง่ายลงมหาศาล
3. **VERIFY** — big AI สั่ง `npm run dev` (หรือ run command ของ project) เอง
   ถ้าพัง → big AI takeover เขียนไฟล์ที่พังใหม่ทันที (ไม่ยอมให้ small AI ลองอีก)

```bash
# ตัวอย่าง: Qwen 1.5B เป็น executor, Qwen3-coder (cloud) เป็น scaffolder/verifier
tinycoder \
    --backend ollama --model qwen2.5-coder:1.5b \
    --planner-backend ollama --planner-model qwen3-coder:cloud \
    --scaffold \
    "make a tetris game with arrow-key controls"
```

Trade-off: big AI ทำงานหนักขึ้น (เขียนสแกฟโฟลด์ทั้งโปรเจกต์) → ใช้ token มากกว่า
plan-only โหมดประมาณ 2x ต่อ request แต่แลกกับ pass-rate ที่สูงขึ้นมากใน
framework projects ที่ small AI เคยพังบ่อย

## Session logging (สำหรับวิเคราะห์ workflow ด้วย AI ตัวใหญ่)

ตั้ง `--log-dir ./logs` (หรือ env `TINYCODER_LOG_DIR=./logs`) เพื่อบันทึก
ทุก event ของ 1 user request เป็นไฟล์ JSON 1 ไฟล์ ตั้งชื่อ
`session-YYYYMMDD-HHMMSS-<id>.json`. โครงไฟล์:

```json
{
  "session_id": "20260515-021430-a1b2c3",
  "user_request": "create xo game with react typescript",
  "mode": "planner | scaffold | plain",
  "executor":  {"backend": "ollama", "model": "gemma4:e2b"},
  "planner":   {"backend": "ollama", "model": "qwen3-coder-next:cloud"},
  "summary_header": {
    "finished": true,
    "event_counts": {"action": 12, "intervention": 3, "review": 1, ...},
    "intervention_kinds": {"takeover": 2, "guide": 1},
    "review_rounds": 1
  },
  "events": [ /* ทุก action, result, intervention, takeover_write, ... */ ]
}
```

ใช้กับ AI ตัวใหญ่กว่า: ส่งไฟล์ใน `./logs/` ให้ Claude / Opus / o1 อ่าน แล้วถามว่า
"ดู session ที่ failed ในรอบนี้: ตรงไหนที่ small AI ติด, intervention type ไหนช่วย
ได้บ่อยสุด, ที่ planner prompt ควรเสริมเรื่องอะไรเพิ่ม"

```bash
tinycoder \
    --backend ollama --model gemma4:e2b \
    --planner-model qwen3-coder-next:cloud \
    --log-dir ./logs \
    --commands yes \
    "create a fastapi todo app"

# ดูผล:
ls ./logs/
# session-20260515-021430-a1b2c3.json
```

## โมเดลที่แนะนำสำหรับ local

| โมเดล | ขนาด | คะแนน (ของเราเอง) |
| --- | --- | --- |
| `qwen2.5-coder:1.5b` | 1.5B | เริ่มต้นได้ เขียน Python/JS ง่ายๆ ผ่าน |
| `qwen2.5-coder:3b` | 3B | ดีขึ้นเยอะ แก้ไฟล์เก่าได้ |
| `qwen2.5-coder:7b` | 7B | ใช้งานจริงได้สบาย |
| `llama3.2:3b` | 3B | follow instruction ดี แต่ coding กลางๆ |
| `phi3:mini` | 3.8B | reasoning ดี เขียนโค้ดพอใช้ |

## ข้อจำกัด

- โมเดลเล็กยังคิดผิดบ่อย โดยเฉพาะ task ที่ต้องวางแผนหลายขั้น — ลอง prompt
  เป็น sub-task เล็กๆ
- `edit_file` ต้องการ `search` ที่ match พอดี 1 ครั้ง ถ้า fuzzy match พลาด
  agent จะลอง read แล้วแก้ใหม่
- ไม่มี streaming output ในเวอร์ชันนี้ (รอผลทั้งก้อนก่อน parse)
- `run_command` ปิดโดยปริยาย เปิดด้วย `--allow-commands` เท่านั้น

## โครงสร้างโค้ด

```
src/tinycoder/
├── cli.py            # entry point + REPL
├── agent.py          # agent loop
├── schemas.py        # Pydantic action schema (the JSON contract)
├── prompts.py        # system prompt + few-shot examples
├── workspace.py      # sandboxed file ops
└── backends/
    ├── base.py       # abstract Backend
    ├── ollama.py     # Ollama (format=schema)
    ├── llamacpp.py   # llama.cpp server (response_format json_schema)
    └── openai_compat.py  # generic OpenAI-compat
```

## License

MIT
