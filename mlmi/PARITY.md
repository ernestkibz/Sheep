# Parity: what Sheep's apps do, feature by feature

Ernest's rule (2026-09-30): **ours must do what theirs can**, and every ✅ is
proven by a test that opens the result — not by reading the code. This file is
the tick table for that, written from the code on 2026-10-07.

- ✅ built, and a test or a screenshot proves it
- 🔨 built, not yet proven by its own test
- ❌ not built
- ✗ deliberately not ours to build

**Nothing here is copied.** Our own names, our own UI, our own formula
evaluator, our own motion maths; .docx/.xlsx/.pptx are open standards we read
and write. See the plan's section 7 for the full rule.

**Proofs** are `mlmi/tests/test_ecosystem.py` (13), `test_studio.py` (12),
`test_tools.py`, `test_shepherd.py` (28), `test_attachments.py`, and the
orientation suite (`mlmi/backend/orient.py`, 800 graded tasks that open what was
made). A row says which one covers it.

---

## Sheep Words (.sdoc, imports and exports .docx)

| Feature | Status | Proof / note |
|---|---|---|
| Styles and headings | ✅ | ecosystem; `words_create`, `words_add` kinds |
| Bullet and numbered lists | ✅ | orientation words area; `words_add_list` |
| Tables, and editing one cell | ✅ | ecosystem; `words_add_table`, `words_set_cell` |
| Pictures | 🔨 | `words_add_image` |
| Find and replace | ✅ | ecosystem; `words_replace` |
| Page breaks | 🔨 | `words_add_page_break` |
| Alignment, size, colour, bold, italic | ✅ | ecosystem; `words_format` |
| Import and export .docx | ✅ | ecosystem; `import_file` / `export_file` |
| Move and delete blocks | ✅ | ecosystem; `words_move`, `words_delete` |
| Read back as numbered blocks | ✅ | every test reads the file it wrote |
| PDF export | ✅ | `pdf_export.doc_pdf`, read back with pypdf |
| Headers, footers, page numbers | ❌ | after Oct 30 |
| Table of contents | ❌ | after Oct 30 |
| Footnotes | ❌ | after Oct 30 |
| Comments, track changes | ❌ | after Oct 30 |
| Columns | ❌ | after Oct 30 |
| Spell check | ❌ | after Oct 30 |

## Sheep Sheets (.ssheet, imports and exports .xlsx and .csv)

| Feature | Status | Proof / note |
|---|---|---|
| Formulas | ✅ | our own evaluator, 33 functions: ABS AND AVERAGE AVERAGEIF CONCAT CONCATENATE COUNT COUNTA COUNTIF FALSE IF INT LEFT LEN LOWER MAX MIN MOD NOT OR PI POWER RIGHT ROUND ROUNDDOWN ROUNDUP SQRT SUM SUMIF TODAY TRIM TRUE UPPER |
| Values show without Excel | ✅ | ecosystem; `sheets_read` prints value and formula |
| Formatting: bold, fill, number format, width, alignment | ✅ | ecosystem; `sheets_format` |
| Charts | 🔨 | `sheets_add_chart` |
| Sorting a range | ✅ | orientation sheets area; `sheets_sort` |
| Several sheets (tabs) | ✅ | ecosystem; `sheets_add_sheet` |
| Whole table in one call, with a total row | ✅ | orientation; `sheets_table` |
| CSV in and out | ✅ | ecosystem |
| Import and export .xlsx | ✅ | ecosystem |
| Filters | ❌ | after Oct 30 |
| Freeze panes | ❌ | after Oct 30 |
| Conditional formatting | ❌ | after Oct 30 |
| Pivot tables | ❌ | after Oct 30 |
| VLOOKUP / XLOOKUP / INDEX-MATCH | ❌ | after Oct 30 |
| Data validation | ❌ | after Oct 30 |
| Date functions beyond TODAY | ❌ | after Oct 30 |

## Sheep Presentations (.sdeck, imports and exports .pptx)

| Feature | Status | Proof / note |
|---|---|---|
| Layouts | ✅ | ecosystem; title and content slides |
| Themes | ✅ | `slides_set_theme`, our own palettes |
| Bullets, with indent | ✅ | orientation slides area |
| Pictures | 🔨 | `slides_add_image` |
| Free text boxes | 🔨 | `slides_add_text` |
| Speaker notes | ✅ | orientation; `slides_set` notes |
| Reorder and delete slides | ✅ | ecosystem; `slides_move`, `slides_delete` |
| Import and export .pptx | ✅ | ecosystem |
| Presenting full screen | 🔨 | `slides.js` present(); needs a screenshot |
| PDF export | ✅ | `pdf_export.deck_pdf` |
| Transitions and animations | ❌ | after Oct 30 |
| Shapes | ❌ | after Oct 30 |
| Charts on a slide | ❌ | after Oct 30 |
| Master slides | ❌ | after Oct 30 |

## Sheep Studio — video and motion in one app (.sstudio → .mp4)

| Feature | Status | Proof / note |
|---|---|---|
| One timeline, layers with start and length | ✅ | studio tests |
| Trim, cut, split | ✅ | studio tests |
| Titles and text, with entrance and exit | ✅ | studio tests; `studio_add_text` |
| Keyframes with easing | ✅ | studio tests; ease_out, ease_back |
| Shapes | ✅ | studio tests; `studio_add_shape` |
| Pictures and clips | ✅ | studio tests; `studio_add_media` |
| Music, with ducking and fades | ✅ | studio tests |
| Quick templates | ✅ | studio tests; title-intro, logo-reveal, outro, 9:16, lower third |
| 16:9, 9:16 and shorts export | ✅ | studio tests; ffprobe reads the result |
| Render on the GPU (NVENC), libx264 fallback | ✅ | studio tests |
| Live preview in the app | 🔨 | needs a screenshot |
| Voice-over (text to speech) in Studio | ✅ | `test_studio_audio`: `studio_voice` lays the .wav on the timeline as an audio layer |
| Transcribe a clip in Studio | ✅ | `test_studio_audio`: `studio_transcribe` writes down what a clip says |
| Captions and subtitles on the timeline | ✅ | `test_studio_audio`: caption blocks, one per sentence, over the clip's own time |
| Transitions between clips | ❌ | after Oct 30 |
| Clip speed | ❌ | after Oct 30 |
| Colour grading | ❌ | after Oct 30 |
| Several video tracks | ❌ | after Oct 30 |
| Masks, motion paths, particles | ❌ | after Oct 30; herders cover the long tail |

## Sheep Text & Code (.txt, .md, code files)

| Feature | Status | Proof / note |
|---|---|---|
| Editor for text, markdown and code | ✅ | screenshot |
| Markdown preview | ✅ | screenshot |
| Check (the Checker's parse rules) | ✅ | `test_checker` |
| Run, sandboxed | ✅ | herder runner |
| Unsaved drafts | ✅ | screenshot (2026-10-07) |
| Any extension written by tool | ✅ | orientation code area; `write_text_file` |

## Across the apps

| Feature | Status | Proof / note |
|---|---|---|
| A click and a model's tool call are the same operation | ✅ | the editors call the same tools |
| Undo, with every version kept | ✅ | `test_tools`; the workspace journal |
| The model writes, Sheep makes the file | ✅ | `test_shepherd` 28, orientation |
| Herders: tools the model writes itself | ✅ | `test_tools`; 3 built in |
| Attachments: picture, sound, video, office, pdf | ✅ | `test_attachments` |
| Reading a picture, on this machine | ✅ | `test_vision`: a red circle, a square, a blue shape, 6 s each on the CPU; Sheep's own engine and weights (SmolVLM quantized by Sheep) |
| Email: read, draft, send (asks first), over IMAP/SMTP | ✅ | `test_mail` (18): local IMAP and SMTP servers |
| Calendar: add and list events, as .ics | ✅ | `test_mail`: RFC 5545 file, undo |
| Web search with sources | ✅ | Tavily when keyed, else Bing then Wikipedia |
| Fonts: open licence only | ✅ | Inter, Source Sans, JetBrains Mono, Carlito |
| ffmpeg called as a separate program | ✅ | never linked; NOTICE records it |

**Open before Oct 30, in order:** the 🔨 rows
given their own test or screenshot (pictures in Words and Slides, page breaks,
charts, free text boxes, presenting, the live preview).
