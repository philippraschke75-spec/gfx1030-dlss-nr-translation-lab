# pre_block Sites 5 & 6 — Cross-Region-Austausch-Regel (Analyse, kein Commit)

Ergebnis des Subagent-Tracers 2026-09-26 (wmma_n=4,5). Rein Emulator-Messung;
rdna2/ unberührt, nichts committet. Sites 5/6 sind **nicht** Teil des
Rewrite-Scope (Sites 1–4) — dieses Dokument schließt nur die offene
"Regel ungelöst"-Lücke in CLAUDE_CONTINUE_2026-09-26.md.

## Regel (identisch für Site 5 und Site 6)

Workgroup (wx=0,wy=0) deckt 8×8 Pixel = vier 4×4-Regionen:
R00=w0,w1 · R04=w2,w3 · R40=w4,w5 · R44=w6,w7.
Jede Wellen-Paar-P erfürt 3 Iterationen; in Iteration t hält deren A-Operand
die Daten einer Region, gewählt als die t-zeile in **Row-Major-Reihenfolge der
vier Regionen OHNE die diagonal-gegenüberliegende Region** von P:

| Paar P | A-Regionen in Iter. 1,2,3 | ausgeschlossen |
|---|---|---|
| w0,w1 (R00) | R00, R04, R40 | R44 |
| w2,w3 (R04) | R00, R04, R44 | R40 |
| w4,w5 (R40) | R00, R40, R44 | R04 |
| w6,w7 (R44) | R04, R40, R44 | R00 |

Quellzentrisch: ein Impuls in Region R erscheint in Iterations-Menge
R00→{1}, R04→{1,2}, R40→{2,3}, R44→{3} — genau 6 von 8 Wellen, alle Paare
außer flip(R) (flip = diagonal tauschen, 0↔4 in beiden Koordinaten).

- **Slot l = 4·y_rel+x_rel** trackt die positions-relative Lage (A-Lanes {l,l+16},
  D-Zeile l = v8+floor(l/2), Halbebene l%2). „Fester lane0"-Eindruck war
  Widerspruch — er entstand nur, weil bislang nur region-rel (0,0) getestet war.
- **Iteration = f(Konsumentenpaar, Region)**, nicht f(Region) allein (R04 wird
  von Paar R44 in Iter.1 gelesen, von R00/R04 in Iter.2).
- **Rückfluss in die eigene Region: ja** — jedes Paar liest seine eigene Region
  genau einmal (t_own = 1 + popcount(Regionsbits): R00@1, R04@2, R40@2, R44@3);
  nur das diagonal-gegenüberliegende Paar sieht sie nie.
- **Wellen-Mengen und Iterationen sind pro Region fix** (≥5 Positionen/Region
  getestet, 25 Positionen total, inkl. Rand-/mittleren Lagen), unabhängig von
  der Position innerhalb der Region. Positionen mit Koordinate 8 ((4,8),(8,4))
  erzeugen keinen Delta — außerhalb des Workgroup-Fensters.
- **B ändert sich nie**: 0 B-Deltas über alle 25 Positionen × 3 Iterationen ×
  8 Wellen an beiden Sites. B ist statische pro-Welle-Gewichtsmatrix;
  Datenfluss Impuls → A → D. Site5-B (v80:87) liegt im selben Registerbereich
  wie Site6-A (v80:87) und wird zwischen den Besuchen überschrieben
  (Gewicht → Payload); site-6-Besuch n läuft konstant 13368 Schritte VOR
  site 5 (Callee-Rotation in prepare()).

## Wert-Level-Anomalien (Regel unberührt)

- Drei Einzelzellen mit exakt-null-Delta trotz identischem A-Delta in der
  jeweils anderen Welle (site6 iter1 (0,0) w1 v8 lane14; site6 (7,0) w7 v14
  lane1; site6 (4,6) w7 v9 lane1) → exakte Kollision/Kontrolle 0 in der
  Wellen-spezifischen B für diesen Ausgang.
- A-Register-Submengen variieren pro Position: mat zeigt K-Spalten mit
  netto-null-Impuls, keine fehlenden Lanes (l und D-Zeile l immer intakt).

## Beweislage

- 21-Positions-Sweeps (site5.txt/site6.txt) + 4 R00-Zusatzpositionen
  ((1,1),(2,1),(3,0),(3,3)) + 6 Wert-Level-`mat`-Probes — beide Sites
  strukturell identisch.
- Direkter Ausschluss-Probe: site6 iter1 Impuls (0,0) auf wave6 (R44-Paar)
  → kein A-, B- oder D-Delta, während w0,w1 identische volle A-Deltas haben.
- Register-Bereiche aus run_stop: site5 D=v8:15, A=v96:103, B=v80:87;
  site6 D=v8:15, A=v80:87, B=v16:23; beide stoppen nach 3 Iterationen.

Skripte/rohe Daten: `$CLAUDE_JOB_DIR/tmp/` (sweep_sites.py, site5.txt,
site6.txt, site5x.txt, site6x.txt, matprobes.txt) — job-lokal, nach
Joblöschung weg; die Regel oben ist das, was überleben muss.