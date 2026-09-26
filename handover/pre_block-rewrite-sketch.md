# pre_block-Rewrite-Sketch — Kollaps-Rechnung für Sites 1–4 (Analyse, kein Commit)

Skizze zum Rewrite des Haupt-Tensor-Core-Pfads in `k_pre_block_1h_32_fp8`
(Sites 1–4, 4 Iterationen). Basis: abgeschlossene Emulator-Messungen
(`trace_wmma_pre.py`, `CLAUDE_CONTINUE_2026-09-26.md`, `WEIGHTS_HT_FINDINGS.md`).
Alles Nicht-bestätigte ist als **BEDINGT/open** markiert mit der schließenden Messung.

## 1. Verdikt und Chance

Die Stage rechnet pro Call effektiv eine 1×1-Faltung über 16 interne Features →
16 Ausgangskanäle (32 über Sites 1/2 + 3/4), ohne räumliche Taps — die 4×4-Region ist
reines Workgroup-Tiling. Ein naives dichtes Lowering des 16³-GEMM verliert:
4096 MAC/Call ≈ 64×`v_dot2c` + Overhead ≈ 99 Instr., nicht besser als eine reine
FMA-Kette (≥128). Der Gewinn steckt in der Struktur: A ist affine in den 3 RGB-Werten
vor der e4m3-Quantisierung (`A = q_e4m3(b(pos) + E·RGB)`), B ist statische e4m3-Matrix —
bei positions-unabhängigem E kollabiert der ganze Call auf
`G = Eᵀ·B_k` (3×16 pro Iteration, einmal pro Dispatch vorberechenbar):
**48 MAC pro Pixel pro Call statt 256 (dicht pro Pixel) bzw. 4096 (Call-gesamt)**.
Das ist der einzige Rewrite-Pfad, der die WMMA-Sites 1–4 ohne Verlust ersetzt;
Sites 5–14 (Cross-Region, fremder Tensor, swapped, post-divergence) sind nicht Teil.

## 2. Exacte Mathematik des Kollapses

Index-Konventionen (wie in den Messungen):
- **E**: 16 Features × 3 RGB — Koeffizienten der affinen Karte vor Quantisierung.
- **b(pos)**: 16-Vektor Positions-Bias, `A[p][i] = q_e4m3(b(pos_p)[i] + Σ_m E[i][m]·RGB[p][m])`.
- **B_k**: 16×16 LDS-Gewichtsmatrix der Iteration k ∈ {1..4}, 4 verschiedene Matrices,
  pro Call/Iteration neu gelesen, 768/768 e4m3-exakt.
- **G_k = Eᵀ·B_k**: 3×16 pro Iteration (48 MAC je, 4×192 MAC gesamt pro Dispatch).
- Ausgang: `out[p][c] = b_part(p,c) + Σ_{i∈{R,G,B}} G_k[i][c]·RGB[p][i]`,
  p = Pixel (16 pro 4×4-Region, lane = 4·y_rel+x_rel), c = Kanal der Iteration
  (Sites 1/2 → v8:15, Sites 3/4 → v48:55).

Requantisierung: **open** — ob der Original-Call nach dem MMAD auf e4m3 zurück-
requantisiert (und mit welchem Scale/Mode), ist nicht sichtbar. Indiz: Sites 3/4 lesen
abgeleitete e4m3-Features mit Werten bis 30 → vermutete Scale-Kette zwischen den
Stages, die wir noch nicht vollständig sehen. Solange offen, muss der Kollaps-Pfad die
Ausgabe bit-exakt so hinterlassen, wie der Downstream-Schritt es liest (siehe Risiken).

## 3. Unbewiesen — schließende Messung je Lücke

| # | Lücke | Status | Schließende Messung |
|---|-------|--------|---------------------|
| a | E positions-unabhängig? | **GESCHLOSSEN, ja** (unter rn oder rd) | Messung 2026-09-26: 3 Positionen (0,0)/(0,4)/(4,4) × 7 Amps, 336/336 Ziele e4m3-exakt, 0 Modell-Refutationen. rn: 16/16 Zeilen mit shared w; rd: 16/16; rz: 15/16 (Zeile 6); ru: 11/16. E ist positionsunabhängig, b(pos) trägt die Positionsabhängigkeit |
| b | Struktur von b(pos): nur 4×4-Region-rel oder absolut? | **open** | Ein Impuls-Sweep bei fester Amp über das 9×9-Kreuz **bei Iteration 1**: wiederholt sich das Bias-Muster pro Tile (region-rel), unterscheidet es sich pro Tile-Lage (absolut)? |
| c | Downstream: e4m3 oder f32 nach dieser Stage? | **open** | Trace der Ausgaberegister von Sites 1/2 bis zum A-Zugriff von Sites 3/4: `cvt`-/Pack-Instruktionen suchen, Zielformat der abgeleiteten Features bestimmen |
| d | B_k pro-Tile-Konstante oder pro-Frame? | **open** | Bestehende Dumps nachschlagen: identisches B über Waves/Blocks/hinweg? Wenn ja → pro-Frame, G einmal/dispatch gültig; sonst pro-tile, G pro Tile (48 statt 3072 MAC, weiterhin billig) |
| e | Quantisierungsmodus der e4m3-Cast (rn/rz/ru/rd)? | **open** | Kontrollierter Probe: RGB so wählen, dass `b+E·RGB` exakt zwischen zwei e4m3-Gridpunkten liegt; beobachtete Rundung → Mode; alternativ `cvt`-Instruktion im Disassemblierungsbild um die Cast-Stelle identifizieren |

## 4. Instruktionsbudget (Dense vs. Kollabiert)

Annahmen (explizit): „Call" = 1×1-Stage-Durchlauf wie in den Messungen geführt
(16³ = 4096 MAC, 64×`v_dot2c`, ≈99 Instr. gesamt); die Schleife lauft 4× mit
identischer A, wechselndem B_k → ×4 für den Gesamt-Call. Lanes halten Pixel
(16 eindeutig, Duplizierung bei +16); 1 `v_dot2` = 2 MAC/lane; G ist über alle Pixel
geteilt, d.h. eine Wellen-Instruktion verarbeitert alle 16 Pixel parallel.

| Pfad | MAC/Call | Instr./Call (abgeleitet) |
|------|----------|---------------------------|
| Dense (Lowering des 16³-GEMM) | 4096 | 64×`v_dot2c` (64×2×32=4096, volle ALU-Auslastung) + Overhead ≈ **99** |
| Dense, Schleife 1–4 | ≈16384 | ≈**396** |
| Kollabiert, `v_mac`-Pfad | 768 (16 px × 48) | 48×`v_mac_f32` + 16×`v_add` (Bias) ≈ **64** |
| Kollabiert, `v_dot2`-Pfad (paarig) | 768 | 24×`v_dot2` + 16×`v_add` ≈ **40** |
| Kollabiert, strikt (3-Term-Dot pro Kanal: 2+1) | 768 | 16×`v_dot2` + 16×`v_mac` + 16×`v_add` = **48** |
| G-Vorberechnung (EᵀB, k=1..4) | 4×3×16×16 = 3072 | ≈96×`v_dot2`, **1×/Dispatch** — amortisiert vernachlässigbar |

Arithmetik-Kontrolle: dicht pro Pixel = 16 Features × 16 Kanäle = 256 MAC; kollabiert
pro Pixel = 3 RGB × 16 Kanäle = 48 MAC → **5,3× weniger MAC pro Pixel**; das in den
Messungen notierte „≈85×" ist 4096/48 (Call-gesamt ÷ pro-Pixel) — beide Größen sind
oben nach Einheit getrennt ausgewiesen. Instruktionen: ≈99 → ≈40–64 pro Call
(≈1,5–2,5×), zuzüglich Entfall des LDS-B-Matrix-Loads pro Iteration (G in Konstanten).

Wo der Einsparung landet: der Dense-Pfad ist ALU/Dot2-bound (jede `v_dot2c` füllt die
ALU-Pipe, Overhead ~35 Instr.). Der kollabierte Pfad ist danach **latency-/overhead-
bound**: Bias-Adds, Konstanten-Zugriff auf G, etwaige Requant-cvt. Der Gewinn ist
primär weniger Instruktionsdurchsatz + entfallener LDS-Traffic, nicht weniger Latency
pro Instruktion — bei einem von der WMMA abhängigen Taktbild (RDNA2 ohne Tensor-Pfad)
genau das, was die Stage freilegt.

## 5. Risiken — was den Rewrite tötet

1. **E positions-abhängig** (Messung a läuft): dann wird G positionsspezifisch,
   die 3×16-Tabelle wächst pro Position auf die Auswertung selbst zurück
   (E_pos·RGB pro Pixel + B·…), Vorberechnung amortisiert nicht → Ersparnis fällt
   etwa auf null. Killt den Kollaps komplett; Dense-Pfad bleibt die einzige Option.
2. **Nicht-affine Features**: die Linearity-Evidenz ist „vereinbar mit affine + e4m3",
   nicht bewiesen. Ein nicht-linearer Term (z.B. RGB-Multiplikationen, Pre-Activation)
   macht `G = EᵀB` ungültig → Difftest-Mismatch, Rollback.
3. **Requant-Unsicherheit** (Punkte 2 + e): Downstream-Sites 3/4 lesen e4m3-Werte bis 30 über
   eine nicht sichtbare Scale-Kette. Unbekannter Rundungsmodus/scale → der kollabierte
   f32-Pfad kann die Ausgabe nicht bit-exakt reproduzieren → Gate (difftest
   0 mismatches + Arena-Hash) nicht erreichbar, Rewrite stoppt.
4. **b(pos) absolut** (Messung b): Bias muss pro Position nachgeladen/berechnet
   werden — nicht fatal (16 Adds bleiben), verschlechtert aber die Overhead-Bilanz des
   bereits latency-bound Pfads.

Offene Punkte der Reihe nach schließen: ~~a~~ (erledigt, ja unter rn/rd), b, e, c, d — dann erst Rewrite.
Regeln nach wie bindend: .co vor Ersetzen sichern; nie committen ohne GPU-Bestätigung;
bei Abweichung sofort rollback; bei Unsicherheit stoppen/dokumentieren/fragen.