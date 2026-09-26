# CLAUDE_CONTINUE 2026-09-26 — pre_block WMMA-Analyse abgeschlossen (nur Analyse, kein Commit)

Übergabe, weil die Cross-Session-Nachricht an die Cloud-Session zweimal abgelaufen ist
(2026-09-26). Inhaltlich identisch mit dem Abschlussbericht dieser Session.

## Auftrag (erledigt): was tut der Filter pro Tensor-Core-Call?

**Aussage: pro v_wmma_f32_16x16x16_f16-Call der Hauptstufe (Sites 1–4) effektiv eine
1×1-Faltung über 16 interne Features → 16 Ausgangskanäle.** Von den drei
Auftrags-Hypothesen trifft „echter 16-Kanal-1×1-Mix" zu, nicht 4×4 und nicht 8×2.

### Messgrundlage (rein Emulator)
- Instrumentierung: `rdna2/emu/trace_wmma_pre.py` (Modi: `mat`/`loops`/`probe`/`taps`/
  `tapcount`/`sweep`, `IMP_AMP`-Env für Linearity-Probes) + `stop_nth` in
  `rdna2/emu/gfx11emu.py` (default-off). Harness = difftest_pre.py-Fixture, H=W=16,
  grid 2×2, seed 1, RGB-Input in regions[1] auf 0, Impuls = 12 Byte = 1.0.
- **A** (v80:87): M = 16 Pixel der 4×4-Region ↔ Lane (4·y_rel+x_rel, Dup bei +16),
  K = 16 Features ↔ 8 Regs × 2 f16-Hälften.
- **B** (v16:23): statische 16×16-Gewichtsmatrix aus LDS, pro Call/Iteration neu.
- **D**: M = Pixel ↔ Reg/LaneHalf, N = 16 Kanäle ↔ lane%16; Sites 1,2 → v8:15,
  Sites 3,4 → v48:55 = 32 Kanäle (passt „32" im Kernel-Namen).
- **Gegen räumliche Taps**: 9×9-Impulsraster — Impuls (4,4) ändert bei Site 1 nur die
  Spalte des eigenen Pixels (lane0 in w6/w7), alle 16 Features, kein Nachbar, kein Halo
  (Reihe 8/Spalte 8 → null). 4×4-Region = reines Workgroup-Tiling, kein Filter.

### Gewichtsblob-Abgleich (WEIGHTS_HT_FINDINGS.md)
- B: **768/768 exakt e4m3** (Sites 1, 2, 6 geprüft).
- A: **2560/2560 e4m3** (Sites 1–5, beide Waves).
- → Operanden sind f16-gehaltene e4m3-Werte; aligniert mit dem FP8-e4m3-Fund der
  Wide-Load-Segmente im Blob. Shapes bleiben wie dort dokumentiert per Size/Statistik
  argumentiert.

### Weitere Struktur
- Hauptschleife: **4 Iterationen, A identisch, B wechselt** (4 verschiedene 16×16-
  Matrices) → pro Pixel 4×16 Kanal-Mixes derselben 16 Features.
- Sites 3/4 lesen andere, abgeleitete e4m3-Features (Werte bis 30), gleiche 1×1-Struktur.
- **Linearität**: Amp 0.5 → keine exakte Halbierung, aber alle Δ-Muster vereinbar mit
  linearer RGB→Feature-Karte + e4m3-Quantisierung (Grid-Crossings; Endpunkte e4m3).
  11/16 Δ-Zeilen positionsidentisch (Ecke vs. Mitte), 5/16 verschieden — E vermutlich
  positionsunabhängig, **nicht final getrennt** (offene Messung).
- Sites 5,6: Cross-Region-Austausch-Regel **gelöst** (2026-09-26, eigene Datei):
  Iteration t liest die t-Region in Row-Major-Reihenfolge über die drei Regionen
  OHNE die diagonal-gegenüberliegende; Slot l=4·y_rel+x_rel trackt die Position
  („fester lane0" war Widerspruch); B überall statisch. Details:
  `handover/pre_block-sites56-swap-rule.md`.
- Sites 7–11: vom RGB-Impuls unabhängig (A-Positionskreuz 17/17 null, B/D bei zwei
  Impulsen null) → fremder Tensor (slot2/RNG-Kandidat).
- Site 12: A statisch, B bewegt → vertauschte Operanden. Sites 13,14: Post-Divergence,
  Global-Normalization überdeckt das Signal.

### Rewrite-Verdikt
1. **Naiver FMA-Rewrite verliert**: dichter 16³-GEMM = 4096 MAC; ≈64×`v_dot2c`
   (64×2×32 = 4096, volle Auslastung) + Overhead ≈ 99 Instr. vs ≥128 reine
   FMA-Instruktionen. Zusätzlich: Es gibt in diesen Calls **gar kein räumliches
   im2col** zu ersetzen — die Auftragsprämisse greift für Sites 1–4 nicht.
2. **Gewinn nur über Struktur**: A,B rein e4m3; A = linear aus 3 RGB-Werten +
   Positions-Bias, quantisiert. Positions-Unabhängigkeit von E bestätigt (rn 16/16):
   Stage kollabiert pro Pixel/Call auf `G = Eᵀ·B` (3×16, einmal/vorab berechenbar)
   → **256→48 MAC pro Pixel (≈5,3×)** bzw. 4096→768 pro Call (gleiches Verhältnis),
   plus Bias-Adds. (Die frühere „~85×"-Zahl war eine Einheiten-Mischung 4096/48,
   korrigiert in `pre_block-rewrite-sketch.md` §4.) Messung bestätigt: rn 16/16 shared.
3. Sites 5–14 nicht Teil der Kalkulation (anderer Tensor bzw. Late-Stage).

## Offene Punkte (Session 8a54df)
- **attention2-Commit wartet auf Freigabe**: Update 30 in FRAME_STATE geschrieben,
  nicht committet. Substitute-Gate erfüllt: GPU-out_sha256-Identität +
  Arena-Hash `d54273b81de7cf88` + Encoder 0/65536. Installierter Stand:
  attention2 gesplicet (21949e3f) + expand2 + qkv2, contract2 unverändert.
- ~~Optionale nächste Messung: E-Positions-Unabhängigkeit~~ — erledigt 2026-09-26:
  rn 16/16 shared, rd 16/16, rz 15/16, ru 11/16, 0 Modell-Refutationen,
  336/336 Ziele e4m3-exakt → E positionsunabhängig; Lücke (a) des Rewrite-Sketches
  geschlossen. Rundungsmodus (Lücke e) bleibt offen: rn und rd passen beide.
- 4 abgelehnte rotierte-Layout-Sites: dokumentiert, vor Berührung fragen.
- Timing-Nachmessung unter sauberen Bedingungen (Wallpaper Engine aus).

Regeln nach wie bindend: .co vor Ersetzen sichern; nie committen ohne GPU-Bestätigung
(difftest 0 mismatches + Arena-Hash exakt `d54273b81de7cf88`); bei Abweichung sofort
rollback; bei Unsicherheit stoppen/dokumentieren/fragen.
