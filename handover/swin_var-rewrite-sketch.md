# `k_swin_var` WMMA-Analyse und Rewrite-Skizze — 2026-09-27

## 1. Verdikt

**Kein Rewrite freigegeben.** Der `<256,false>`-Pfad enthält 23 statische
WMMA-Instruktionen. Der neue Emulator-Tracer wurde mit der realen Stage-4-
Aktivierung und den echten `block15.bin`-Gewichten betrieben. Impulse ändern
WMMA-Eingangsfragmente und Ausgaben in mehreren Sitegruppen. Sites 18–21 zeigen
insbesondere datenabhängige Operand- und Akkumulatorpfade. Ein statisches,
kleindimensionales Gewichtsmuster, das einen bitgenauen Collapse rechtfertigt,
ist nicht nachgewiesen. Deshalb gibt es weder Lowering-Code noch einen
Korrektheits- oder Performanceclaim.

Die Strukturen von 128 und 256 teilen grobe Gruppen, aber 64, 32-false und
32-true weichen sowohl bei Sitezahl als auch bei Impulsantwort ab. Die Rewrite-
Methodik ist nicht pauschal auf alle fünf Varianten übertragbar.

| Variante | Statische WMMA-Sites | Im Emulator dynamisch erreicht* |
|---|---:|---:|
| `<256,false>` | 23 | 3–21 |
| `<128,false>` | 23 | 3–21 |
| `<64,false>` | 29 | 3–25 |
| `<32,false>` | 18 | 3–16 |
| `<32,true>` | 18 | 3–16 |

\* Flags=1, je ein repräsentativer Einzel-Workgroup-Lauf. Die nicht erreichten
Sites 1–2 und die letzten Sites sind bedingte Pfade, keine global toten Sites.
Die dynamischen Site-Nummern bezeichnen jeweils nur die statische Reihenfolge
innerhalb derselben Variante.

## 2. Fixture und Instrumentierung

`rdna2/emu/trace_wmma_var.py` instrumentiert `k_swin_var` ohne Änderungen an
Übersetzer oder Kernel. Es kann alle Sites eines Emulatorlaufs vor/nach WMMA
erfassen, Operandenregister A/B/C gegen D-Ergebnisse vergleichen und die
erreichten Waves zählen. Input- und Gewichtsdateien, Feature-/Positionsimpuls,
Shape, Offsets und Grid lassen sich explizit angeben. Checkpoints von 1 MiB
werden auf die Shape-spezifische Aktivierungsgröße begrenzt.

Der 256er Stage-4-Input wurde rekonstruiert: gespeicherter realer
`block13_checkpoint.npy` als Aktivierung für `block14.bin`, H=W=16, flags=4,
Offsets (-4,-4), Grid 3×3. Der fused gepoolte Output aus Arena-Offset +0x38
ist 1 MiB; die Stage-4-Aktivierung belegt darin die ersten 65,536 Bytes für
C=256, H=W=8. Das Ergebnis liegt unter
`rdna2/build/emu_var/swin_var_stage4_input.bin`.

Für Varianten-Sweeps wurden vorhandene Checkpoints aus
`rdna2/build/import-real-20260922/rdna2/build` und zugehörige echte Gewichte
verwendet:

| Kernel | Fixtureinput | Gewicht | H×W | Flags | Herkunft |
|---|---|---|---:|---:|---|
| 32-true | generisches e4m3-Muster | block0.bin | 64×64 | 1 | synthetischer Kontrolllauf |
| 32-false | generisches e4m3-Muster | block1.bin | 64×64 | 1 | synthetischer Kontrolllauf |
| 64-false | block4_checkpoint.npy | block5.bin | 32×32 | 1 | echter Stage-2-Eingang |
| 128-false | block8_checkpoint.npy | block9.bin | 16×16 | 1 | echter Stage-3-Eingang |
| 256-false | pooled4_checkpoint.npy | block15.bin | 8×8 | 1 | echter Stage-4-Eingang |

Hinweis: Der 32er-Stage-1-Input ist nicht unter den gespeicherten Checkpoints;
die 32er Sweep-Daten sind deshalb ausdrücklich synthetisch. Die Checkpoints
und Gewichtsdateien wurden gelesen, nicht verändert. Der Tracer nutzt für
`k_swin_var` den expliziten Kernarg-Pointer in SGPR 0:1. Er filtert Helper-
Instruktionen vor Sitezählung heraus und liest D-Ergebnisse nach der WMMA,
während C davor als Akkumulator erfasst wird.

## 3. Mathe und Sitegruppen

Der Emulator definiert jede Instruktion als `D = A · Bᵀ + C` mit 16×16-
Fragmenten. Das benennt noch keine Tensorrolle. Die folgende Einordnung stützt
sich auf statische Registerfolgen sowie auf einen e4m3-Byte-Impuls `0x38` oder
`0x40` in der Aktivierung. Es sind Strukturgruppen, keine bestätigten Namen wie
Q, K, V, Score oder Output-Projektion.

| 256er Sites | Beobachtung (realer Stage-4-Fixturelauf) | vorsichtige Einordnung |
|---|---|---|
| 1–2 | unter flags=1 in der getesteten Workgroup nicht besucht | bedingter Pfad; unklassifiziert |
| 3–6 | besucht; Feature-0- und Feature-255-Impulse änderten erfasste WMMA-Register nicht | Prep-/Tabellenkandidat; Nullbefund ist kein Beweis für Inputunabhängigkeit |
| 7–13 | Inputimpulse ändern vor allem A und D; Antworten sitzen in wenigen Wave/Lane-Fragmenten und propagieren über benachbarte Sites | inputabhängige Matmul-Sequenz; projektion-/räumlich nicht abschließend getrennt |
| 14–17 | Feature-255-Impuls ändert A, teils B/C/D; Feature-0-Impuls in gleicher Position weitgehend ohne Änderung | gemischte, kanalabhängige Fragmente; keine verlässliche semantische Zuordnung |
| 18–19 | Feature-0-Impuls ändert B und D an Site 18; C/D propagieren an Site 19. Feature-255 ändert zusätzlich A/B/D an Site 18 und A/C/D an Site 19 | nicht als statische Gewichtsprojektion freigegeben; attention-/interaktionsartiger Kandidat, noch kein bewiesenes QKᵀ |
| 20–21 | Inputabhängige A/D; Site 21 übernimmt verändertes C und propagiert auf D | inputabhängige Matmul-/Akkumulationsfolge; nicht kollabierbar belegt |
| 22–23 | unter flags=1 nicht besucht | bedingter Pfad; unklassifiziert |

Die Feature-0-Änderung bei Site 18 wurde zusätzlich mit einem unabhängigen
Feature-255-Impuls ergänzt. Dass dort **beide** Eingangsoperanden A und B auf
Änderungen in der Quellaktivierung reagieren, schließt die einfache Form
`X·W` mit während dieses Dispatch fixem W für diesen beobachteten Pfad aus.
Es beweist allein noch keine quadratische Q·Kᵀ-Interaktion: Dafür muss die
gemeinsame Antwort zweier unabhängiger Eingangsänderungen gegen die Summe der
Einzelantworten getestet werden, inklusive Dekodierung der Fragmente und der
vorherigen Skalierungs-/Quantisierungsstufen. Solange dies offen ist, ist
Site 18/19 für Kollaps ausgeschlossen.

Impulsantworten der anderen Varianten (Feature 0, mittlere Position, flags=1):

| Variante | Antwortbild |
|---|---|
| 32-true | Sites 7–8 reagieren in A/D; Sites 11–12 in B; Sites 15–16 breit in A/D/C. Letzte zwei Sites unerreicht. |
| 32-false | Ähnlich im frühen Bereich, aber Sites 15–16 anders verteilt/breiter; letzte zwei unerreicht. |
| 64-false | Sites 3–25 erreicht, aber Feature 0, 31 und 63 bei (16,16) änderten in den erfassten ersten Sitebesuchen kein A/B/C/D-Fragment. Nicht als Unabhängigkeitsbeweis lesen. |
| 128-false | sichtbare Gruppen 7–13 / 18–19 / 20–21, mit inputabhängiger B/D-Reaktion in Site 18. |
| 256-false | Gruppen 7–13 / 14–17 / 18–19 / 20–21; Site 18 reagiert in A und B bei mehreren Features. |

Die Ähnlichkeit von 128 und 256 ist auf Gruppenebene interessant, aber nicht
ausreichend für einen gemeinsamen Rewrite. 64er Sitezahl und Impulsbild sind
anders; die 32er Templates unterscheiden sich bereits voneinander.

## 4. Kollapsprüfung und Ausschlüsse

Ein Call ist nur für einen Collapse geeignet, wenn der eingehende A-Tensor
nach allen vorherigen nichtlinearen, requantisierten und positionsabhängigen
Schritten affin in wenigen echten Eingangswerten ist, B dispatch-konstant ist
und die Ausgabe exakt einschließlich Rundung/Quantisierung reproduziert wird.
Keiner dieser vollständigen Nachweise liegt für eine Sitegruppe vor.

Site 18/19 sind explizit ausgeschlossen: beide A und B ändern sich bei
Eingabeimpulsen in Site 18. Das widerspricht einer festen-B-Projektion. Eine
Q·Kᵀ-artige bilineare Attention-Matmul wäre deshalb plausibel, bleibt aber
Hypothese. Für Bestätigung fehlen Kreuzterm-/Doppelpuls-Tests und die
Fragmentdekodierung in mathematische Fensterpositionen. Auch Sites 7–13 und
20–21 sind nicht freigegeben: A-Reaktion alleine zeigt weder Affinität der
Vorgeschichte noch Kollaps über Requantisierung hinweg. Sites ohne beobachtete
Delta gelten ebenfalls nicht als statisch, bis weitere Features, Positionen,
Flags und Besuche ausgeschlossen sind.

## 5. Offene Messungen

1. 256er Site-18 Doppelpuls/Kreuzterm: zwei unabhängig gewählte räumliche
   Positionen oder Kanäle einzeln und gemeinsam perturbieren; A/B-Matrizen und
   D numerisch dekodieren und additive gegen bilineare Vorhersage prüfen.
2. Sitegruppen 7–13 und 20–21: räumliches Raster über ein 4×4-Tile, Tilegrenze
   und überlappende Ursprünge; aus Lane-/Fragmentantwort die Raum-/Kanalachsen
   rekonstruieren.
3. 64er Nullantwort erklären: weiter entfernte Positionen, Tile-origin und
   mehrere Zielsites/-Besuche testen; bisher wurden Feature 0 und 31 nur bei
   der mittleren Position geprüft.
4. Für beide 32er Varianten realen Stage-1-Input mit block0/1-Gewichten
   bereitstellen und dieselben Scans erneut fahren.
5. Dynamische Sites über reale Launchgrids, Modus-Origins (0/−4) und erste/
   letzte Stageflags abdecken. Die Einzel-Workgroup flags=1-Ausführung zählt
   nicht alle deployed Sites.
6. Gewichtsabhängigkeit und Requantisierung zwischen Sitegruppen aus Speicher-
   und Konvertierungsinstruktionen gegen gemessene Fragmente prüfen.
7. Erst nach vollständiger Mathematik einen Rewrite-Kandidaten bauen und jeden
   betroffenen Pfad gegen Referenz und Full-Frame-Artefakt validieren.

## 6. Reproduktion

Interpreter:

```
C:\Users\<user>\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe
```

Stage-4-Aktivierung erstellen (nutzt den gespeicherten Stage-3-Checkpoint und
`block14.bin`):

```
python rdna2/emu/trace_wmma_var.py prepare-stage4-input
```

Beispiel kompletter 256er Site-Scan mit realem Input und Gewichten; Variablen
werden in PowerShell mit denselben Namen gesetzt:

```
$env:TRACE_INPUT_FILE='rdna2/build/emu_var/swin_var_stage4_input.bin'
$env:TRACE_WEIGHT_FILE='rdna2/build/weights/block15.bin'
$env:TRACE_HEIGHT='8'; $env:TRACE_WIDTH='8'
$env:TRACE_POS_Y='4'; $env:TRACE_POS_X='4'; $env:TRACE_FEATURE='0'
python rdna2/emu/trace_wmma_var.py analyze 256_0 1
```

Varianten-API für Checkpoint-gestützte Messungen: `analyze_all(key, flags,
H, W, pos=..., input_file=..., weight_file=..., feature=..., amp=...)` aus
`trace_wmma_var.py` aufrufen. Featureänderungen erfolgen relativ zur echten
Checkpoint-Aktivierung, nicht zu Null.

## 7. Korrektheits- und Commit-Gates

Es wurde kein Kernel geändert. Daher wurde kein GPU-Difftest für einen Rewrite
und kein Full-Frame-Arena-Hash gemessen. Der bekannte Sollhash
`d54273b81de7cf88` ist **nicht erneut bestätigt**. Instrumentierungs- und
Analysearchitekturen sind emulatorische Befunde und kein Ersatz für diese
Gates.

Vor jeder Kerneländerung gelten:

1. Vollständiger `k_swin_var`-Chain-Difftest aller betroffenen Sites/Varianten
   mit 0 Mismatches gegen die Referenz.
2. Unveränderter Full-Frame-Arena-Hash `d54273b81de7cf88`.
3. Erst dann Commit des Lowerings. Analyseartefakte können unabhängig davon
   versioniert werden.

Status dieses Handover: die Analyse hat reale 64/128/256-Fixtures, beide
synthetische 32er Varianten und ein neues Tracingwerkzeug. Die mathematische
Semantik ist teilweise eingegrenzt, aber keine Sitegruppe ist für Lowering
freigegeben.
