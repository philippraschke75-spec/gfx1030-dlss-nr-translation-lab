# Performance-Analyse `k_pre_block` / `k_post_block` (gfx1030)

Nur statisch, ohne GPU. An `.s`-Dateien, Translator und Runner wurde **nichts** geändert.

## 0. Was diese Analyse kann und was nicht

Die übersetzten `.s`-Dateien (`rdna2/build/kernels-hw-scratch/*.s`) und
`analysis/gfx1100-disassembly.txt` sind **nicht** in diesem Repo, und das ist Absicht (README:
„no disassembly“). Deshalb kann hier **keine Zeile der Kernel zitiert** werden. Das Dokument
trennt klar zwischen drei Arten von Aussagen:

| Markierung | Bedeutung |
|---|---|
| **[fest]** | folgt aus Architekturdaten (RDNA2/RDNA3) plus den im Auftrag genannten Zahlen (210 VGPR, ~64 KB LDS, 256 Threads, 8×8-Tiles) |
| **[Repo]** | steht mit Stelle im Repo (`VARPARAMS_HOST_CONTRACT.md`, `emu/`) |
| **[Hypothese]** | plausibel, aber erst mit dem Skript aus §8 auf den echten Dateien prüfbar |

Die offenen Punkte beantwortet `perf/kernel_profile.py` in Sekunden lokal (siehe §8). Das Skript
gibt nur Zahlen und Mnemonics aus, keinen Code, damit die Ergebnisse hier eingetragen werden können.

Zur Einordnung, welcher Kernel gemessen wurde: Laut Contract (`VARPARAMS_HOST_CONTRACT.md:60-73`)
gibt es zwei Prolog-Varianten, `k_pre_block_1h_32_fp8` (PreParams) und `k_swin_var<32,true>`
(VarParams). Welche davon live ist, hängt an einem Host-Boolean. `k_swin_var<32,true>` läuft im
Repo mit **15 616 B LDS** (`emu/difftest_var.py:19`), nicht mit 64 KB. Die ~64 KB aus dem Auftrag
gehören also zu `k_pre_block_1h_32_fp8` / `k_post_block_1h_32_fp8`. Davon geht das Folgende aus.

---

## 1. Die Belegungsrechnung: Das ist das Kernergebnis

**[fest]** RDNA2 (gfx1030), Wave32:
- pro SIMD 1024 VGPRs, Allokation in 8er-Schritten, maximal 16 Waves
- 1 CU = 2 SIMD32; 1 WGP = 2 CUs = 4 SIMD32
- LDS: 64 KB pro CU (CU-Modus) bzw. 128 KB pro WGP (WGP-Modus); maximal 64 KB pro Workgroup

| | Rechnung | Ergebnis |
|---|---|---|
| VGPR-Allokation | 210 → aufgerundet 216 | 216 |
| Waves/SIMD durch VGPR | ⌊1024 / 216⌋ | **4** |
| Waves pro Workgroup | 256 / 32 | 8 |
| Workgroups/CU durch VGPR | 2 SIMD × 4 / 8 | **1** |
| Workgroups/CU durch LDS | ⌊64 KB / ~64 KB⌋ | **1** |
| (WGP-Modus) | 4×4/8 = 2 bzw. 128/64 = 2 pro WGP | **2 pro WGP**, also ebenfalls 1 pro CU |

**Schlussfolgerung 1 [fest]: VGPR- und LDS-Grenze greifen gleichzeitig und auf demselben Wert.**
Wenn nur eine der beiden sinkt, ändert sich an der Belegung **gar nichts**.

| Ziel | braucht **beides** |
|---|---|
| 2 WG/CU (8 Waves/SIMD), CU-Modus | VGPR ≤ **128** **und** LDS ≤ **32 KB** |
| 3 WG/WGP, WGP-Modus | VGPR ≤ **168** **und** LDS ≤ **42.5 KB** |

Von 210 auf 128 VGPRs und gleichzeitig von 64 auf 32 KB LDS ist bei einem 8×8-Swin-Tile mit
C=32 **kein Aufräumen mehr, sondern ein anderer Kernel**.

**Schlussfolgerung 2 [fest]: Auf gfx1100 lief das Original mit derselben Belegung.**
RDNA3 (gfx1100/1101) hat 1536 VGPRs pro SIMD (Granularität 24). Mit ~210 VGPRs wären dort 7 Waves
pro SIMD möglich. Das LDS-Limit ist aber identisch (64 KB pro Workgroup, 128 KB pro WGP), also sind
es dort ebenfalls **2 WG/WGP = 4 Waves/SIMD**. Das Original war schon LDS-gebunden, und zwar genau
auf dem Niveau, das gfx1030 jetzt hat.

> Die Belegung ist deshalb **sehr wahrscheinlich nicht** der Grund, warum der Kernel auf gfx1030
> langsamer ist als auf gfx1100. Der Unterschied muss aus der **Menge an ausgeführten
> Instruktionen** kommen (§3).

Einschränkung: Das gilt nur, wenn die übersetzte Datei dieselbe LDS-Größe deklariert wie das
Original. `kernel_profile.py --orig` gibt beide Werte aus.

Zum Begriff „keine Latenzverdeckung“: Es laufen 4 Waves pro SIMD, also gibt es Latenzverdeckung.
Was fehlt, ist eine *zweite Workgroup*, die rechnet, während die erste an einem `s_barrier` oder
auf Global-Loads wartet. Wie viel das kostet, hängt an der Zahl der Barriers pro Workgroup, die das
Skript ebenfalls zählt.

---

## 2. Zeitbudget: VALU-gebunden oder latenzgebunden?

**[fest]** 1792×1024 bei Grid (⌈W/8⌉, ⌈H/8⌉) = 224 × 128 = **28 672 Workgroups**. Eine RX 6900 XT
hat 80 CUs und schafft 1 WG/CU gleichzeitig, das sind ≈ 358 Durchläufe.

- 111 ms / 358 ≈ **310 µs pro Workgroup** ≈ **~700 000 Takte** bei ~2.25 GHz
- Eine CU hat 2 SIMDs mit je ≤1 VALU-Instruktion pro Takt (Wave32). Das sind ≈ 1,4 Mio. Issue-Slots
  pro Workgroup, also ≈ **175 000 Instruktionen pro Wave**, falls der Kernel rein VALU-gebunden ist.

**Entscheidungsregel**, ohne GPU messbar (§8, Schritte 2+3): `est_gfx1030_executed_per_wave` aus
`kernel_profile.py --exec-hist` gibt die ausgeführten Instruktionen pro Wave.
- **≳ 100 000** pro Wave: Der Kernel ist issue-gebunden. Hebel sind **weniger Instruktionen**
  (Übersetzungs-Expansion, §3). Die Belegung spielt keine Rolle.
- **≪ 50 000** pro Wave: Die Zeit geht in Latenz verloren (Global-Loads, Barriers, LDS-Konflikte).
  Dann hilft Belegung, die aber nach §1 nicht billig zu haben ist.

Nebenbei **[Repo]**: Der per-Workgroup-Scratch in globalem Speicher (`ctx+0x180`, 8 KiB × Grid,
`VARPARAMS_HOST_CONTRACT.md:63`) macht bei 28 672 WGs ≈ 235 MB. Selbst bei doppelter Bandbreite
(Schreiben + Lesen) sind das bei ~500 GB/s unter 1 ms, also **kein** relevanter Anteil an 111 ms.

---

## 3. Frage 3: Ineffizienzen der Übersetzung

### 3a. WMMA-Emulation: wahrscheinlich der Haupttreiber [Repo + Hypothese]

**[Repo]** gfx1100 hat `v_wmma_f32_16x16x16_f16`, gfx1030 hat keine Matrix-Instruktionen. Der
Contract spricht von „the same software lowering“ für WMMA (`VARPARAMS_HOST_CONTRACT.md:553-554`,
`1134-1135`). WMMA-haltige Kernel sind z. B. `k_qkv` (4 WMMA) und `k_contract2` (4 WMMA).

**[fest]** Größenordnung: Eine WMMA 16×16×16 sind 4096 MACs pro Wave. Auf gfx1100 ist das **eine**
Instruktion. In Software sind es pro Lane mindestens 128 MACs (8 Ausgaben × 16 K), also
**≥ 64** `v_dot2`/`v_pk_fma`-artige oder **≥ 128** `v_fma_f32`-Instruktionen. Dazu kommt der
Operandenaustausch zwischen Lanes: Im WMMA-Layout hält Lane *i* nur Zeile *i mod 16* von A, jede
Lane braucht aber alle 16 A-Zeilen. Das geht über `v_readlane`, DPP oder LDS-Roundtrips.
Realistisch sind **150 bis 400 Instruktionen pro WMMA**.

**[Hypothese]** Liegt eine WMMA in einer Schleife über Tokens oder Kanäle, dominiert dieser Faktor
die Laufzeit. Das würde auch die ~89 000 Zeilen pro Datei erklären (ein Teil davon sind
`// original:`-Kommentare). **Prüfen:** Die Tabelle `expansion.by_original_opcode` zeigt Faktor
und Befehlsmix von `v_wmma_*`. `dynamic.top_contributors_per_wave` zeigt, ob WMMA die Laufzeit
dominiert.

**Eingriff:** nicht an den `.s`-Dateien, sondern im Lowering in `translate_kernels.py`, und danach
Neuverifikation per Difftest. Mögliche Richtungen, abhängig vom gemessenen Befehlsmix:
- A-Operanden einmal pro Wave in LDS legen und per `ds_read_b128` mit Broadcast lesen, statt einzeln
  per `v_readlane` pro Element holen.
- f16-Paare mit gepackter Arithmetik (`v_pk_fma_f16` bzw. `v_fma_mix_f32`, die auf gfx1030
  existieren) statt skalarem f32-FMA, und nur dort, wo die Rundung erlaubt ist.

**Risiko: hoch für „0 Mismatches“.** Jede Änderung der Akkumulationsreihenfolge oder -präzision
ändert f32-Ergebnisse in der letzten Stelle. Der Contract zeigt, dass 1-ULP-Abweichungen an
e4m3-Quantisierungsgrenzen schon zu Byte-Flips führen (`VARPARAMS_HOST_CONTRACT.md:1125-1143`).
Bit-exakt bleibt nur ein Lowering, das **dieselbe Operationsfolge** billiger umsetzt, also z. B.
effizienteren Operandentransport bei unveränderter FMA-Kette. Das ist die einzige Variante mit
mittlerem statt hohem Risiko.

### 3b. VOPD (`v_dual_*`) [fest, kein Defekt]

gfx1100 führt zwei VALU-Operationen in einer `v_dual_*`-Instruktion aus, gfx1030 braucht zwei. Das
verdoppelt die Kosten dieser Instruktionen und ist **architekturbedingt**. Daran lässt sich nichts
optimieren. `original.vopd` zeigt, wie viele es sind.

### 3c. Übersetzungs-Overhead ohne Architekturgrund [Hypothese, prüfbar]

Dafür gibt das Skript direkte Kennzahlen:

| Kennzahl | Was sie verrät |
|---|---|
| `expansion.ratio` gesamt, pro Opcode | Wie viel Code die Übersetzung erzeugt. Ein Faktor > 1 bei Opcodes, die auf gfx1030 direkt existieren, ist echter Overhead. |
| `reg_to_reg_v_mov_b32`, `self_moves` (übersetzt vs. `original`) | zusätzliche Kopien |
| `overwrite_before_read_same_bb` | Register, die geschrieben und im selben Basisblock vor jedem Lesen überschrieben werden (tote Werte; EXEC-Wechsel und Branches werden konservativ als Grenze behandelt) |
| `foreign units inside expansions` | z. B. `ds in v_wmma…` heißt: Das Lowering benutzt LDS. `scratch in …` hieße: Das Lowering spillt in Scratch-Speicher, was sehr teuer ist. |
| `descriptor.scratch_bytes`, `memory.scratch_ops` | Der Ordnername `kernels-hw-scratch` legt nahe, dass Hardware-Scratch aktiv ist. Tauchen Scratch-Zugriffe **nur** in der Übersetzung auf, sind das eingeführte Spills, und die kosten pro Zugriff Global-Memory-Latenz. |

Risiko je Fund: Tote Kopien oder Writes aus den `.s` zu streichen ist **mittel riskant** (EXEC-,
VCC- und Teilregister-Semantik). Nach §1 bringt es für die **Belegung auch nichts**, solange LDS bei
64 KB bleibt. Es spart nur die Issue-Slots dieser Instruktionen.

---

## 4. Frage 1: Sind 210 VGPRs nötig?

**[fest]** Für die Laufzeit ist die Zahl **unerheblich, solange sie über 128 liegt** (CU-Modus) bzw.
über 168 (WGP-Modus). Mit 64 KB LDS ist es sogar ganz egal (§1). Eine VGPR-Reduktion **allein**
bringt keinen Takt.

Was das Skript trotzdem zeigt, falls die LDS-Frage (§5) ein anderes Ergebnis liefert:
- `holes_below_alloc`: Register unterhalb von `.amdhsa_next_free_vgpr`, die **nie** benutzt werden.
  Ist `max_index_used + 1 < declared`, dann ist die Deklaration zu groß. Die Korrektur beträfe nur
  den **Descriptor**, keinen Code: **niedriges Risiko**, aber nur sinnvoll, wenn damit eine Schwelle
  aus §1 unterschritten wird.
- `vgprs_only_in_translation` (mit `--orig`): Register, die im Original nie vorkommen, also
  Temporärregister des Translators. Liegt `original.max_vgpr_index` deutlich unter 210, hat die
  Übersetzung Druck hinzugefügt (typisch für WMMA-Lowering).
- `written_never_read`: statisch tote Register über den ganzen Kernel.

---

## 5. Frage 2: Sind ~64 KB LDS nötig?

**[Repo]** Die LDS-Größe stammt sehr wahrscheinlich aus dem **Original** und nicht aus der
Übersetzung. Der Swin-Kernel des Originals arbeitet mit einer `SwinLDS`-Struktur
(`_Z10swin_layerR7SwinLDSPKhRK10BlobLayouti`, `emu/run_emu.py:8`) und wird mit **62 592 B** LDS
gestartet (`emu/run_emu.py:36`). Das sind ~61 KB. Sie tragen das 8×8-Fenster plus Halo, Q/K/V und
Zwischenergebnisse, also genau das Sliding-Window-Tile aus dem Auftrag.

Prüfen, ob die Übersetzung LDS **zusätzlich** braucht oder überdimensioniert ist:
1. `descriptor.lds_bytes` der `.s` mit dem Original-Descriptor vergleichen (`difftest_var.group_size`
   liest ihn aus dem Code-Objekt).
2. `memory.ds_max_static_offset` ist nur eine Untergrenze, weil Adressen dynamisch sind.
3. `exec_hist.py` liefert `lds_highest_nonzero_byte`, das höchste im Emulator beschriebene Byte
   (ebenfalls eine Untergrenze).
4. `foreign units inside expansions` enthält `ds in …`: Dann nutzt das Lowering LDS selbst, und die
   Frage ist, ob es dafür Platz *zusätzlich* reserviert hat.

Wenn die Deklaration zum Original passt: **kein Optimierungspotential ohne Algorithmusänderung.**
Falls der Descriptor größer ist als der tatsächlich genutzte Bereich, ließe er sich ohne Codeänderung
verkleinern (**niedriges Risiko**). Das hilft aber nur zusammen mit VGPR ≤ 128/168 (§1).

---

## 6. Frage 4: Kleinere Workgroup (z. B. 128 Threads) per Grid-Umkonfiguration?

**Nein, aus drei unabhängigen Gründen.**

1. **[fest] Es bringt keine Belegung.** VGPRs sind pro Thread vergeben. Bei 210 VGPRs bleiben es
   4 Waves pro SIMD, egal wie diese Waves auf Workgroups verteilt sind. Zwei Workgroups à 4 Waves
   belegen eine CU genauso wie eine à 8. Der LDS-Bedarf pro Workgroup ist im Descriptor fest
   (`group_segment_fixed_size`) und schrumpft nicht mit der Threadzahl.
2. **[Repo] Die Aufteilung der Arbeit steht im Code.** Grid = (⌈W/8⌉, ⌈H/8⌉) mit 256 Threads
   (`VARPARAMS_HOST_CONTRACT.md:63`, `emu/net_frame_full.py:230`). Jede Workgroup rechnet genau ein
   8×8-Tile, und welcher Thread welches Pixel/welchen Kanal rechnet, wird aus `v0` (Thread-ID)
   berechnet. Mit 128 Threads würde die halbe Arbeit pro Tile schlicht **nicht gemacht**. Mit einem
   2×-Grid in x würden Tiles doppelt gerechnet bzw. falsch adressiert.
3. **[fest] Die Waves arbeiten zusammen.** Fenster-Attention braucht alle 64 Tokens des Fensters.
   Die Waves tauschen sie über LDS und `s_barrier` aus. Eine Aufteilung auf mehrere Workgroups
   bräche diese Kommunikation.

**Risiko: sicher falsch.** Es wäre ein neuer Kernel, keine Konfiguration.

---

## 7. Empfehlungen, nach Nutzen/Risiko geordnet

| # | Maßnahme | Nutzen | Risiko |
|---|---|---|---|
| 1 | Skripte aus §8 laufen lassen, Zahlen hier eintragen | entscheidet zwischen §2 „issue-gebunden“ und „latenzgebunden“, ohne GPU | keins |
| 2 | Bei WMMA-dominierter Laufzeit: WMMA-Lowering im **Translator** verbessern (Operandentransport), **FMA-Reihenfolge unverändert** lassen, dann per Difftest neu verifizieren | vermutlich der einzige große Hebel | mittel (bit-exakt möglich) |
| 3 | dasselbe mit anderer Akkumulation (`v_pk_fma_f16`, `v_dot2`-artig) | größer | **hoch**: bricht „0 Mismatches“, bräuchte eine Toleranz statt Bit-Gleichheit |
| 4 | Scratch-Spills, die nur in der Übersetzung existieren, im Translator vermeiden | groß, falls vorhanden | mittel |
| 5 | Descriptor-Werte (`next_free_vgpr`, `group_segment_fixed_size`) auf den tatsächlichen Bedarf senken | **null**, solange nicht **beide** Schwellen aus §1 unterschritten werden | niedrig |
| 6 | Tote Kopien/Writes aus den `.s` streichen | klein (nur Issue-Slots) | mittel |
| 7 | Kleinere Workgroup | keiner | Korrektheit sicher verletzt |

---

## 8. Anleitung zum lokalen Ausführen

Die Pfade gelten für das lokale Projekt (`rdna2/…`, `analysis/…`). Die Skripte liegen hier in
`perf/` und können dorthin kopiert werden. Sie brauchen nur die Python-Standardbibliothek;
`exec_hist.py` braucht zusätzlich die normale Emulator-Umgebung.

```sh
# 1. statisch: Descriptor, Belegung, VGPR-/LDS-/Scratch-Kennzahlen, Expansion pro Original-Opcode
python perf/kernel_profile.py \
    rdna2/build/kernels-hw-scratch/_Z21k_pre_block_1h_32_fp89PreParams.s \
    rdna2/build/kernels-hw-scratch/_Z22k_post_block_1h_32_fp810PostParams.s \
    --orig analysis/gfx1100-disassembly.txt --json perf_static.json
#    Bei zwei Dateien mit --orig wird der Symbolname aus dem Dateinamen genommen; ruft der Kernel
#    Hilfsfunktionen, diese per --orig-sym zusätzlich angeben (dann eine Datei pro Aufruf).

# 2. dynamisch: ausgeführte ORIGINAL-Instruktionen pro Wave, im Emulator, 1 Workgroup
python perf/exec_hist.py hist_pre.json rdna2/emu/difftest_preblock.py 1 8 8

# 3. beides kombinieren: geschätzte ausgeführte gfx1030-Instruktionen pro Wave, größte Verursacher
python perf/kernel_profile.py rdna2/build/kernels-hw-scratch/_Z21k_pre_block_1h_32_fp89PreParams.s \
    --orig analysis/gfx1100-disassembly.txt --exec-hist hist_pre.json
```

Hinweise zur Auswertung:
- `annotation mode` zeigt, wie das Skript die `// original:`-Kommentare gelesen hat (`header` =
  Kommentarzeile vor der Expansion, `trailing` = am Befehl). Steht dort `none`, fehlen die
  Expansions-Tabelle und die dynamische Schätzung. Dann das Kommentarformat prüfen.
- `exec_hist.py` misst das **Original** im Emulator. Die gfx1030-Zahl ist **Original × gemessener
  Expansionsfaktor pro Opcode**, also eine Schätzung. Datenabhängige Schleifen im Lowering sind
  darin nicht abgebildet.
- Für `k_post_block` gibt es im Repo kein Difftest-Skript. Jedes Emulator-Skript, das den Kernel
  über `gfx11emu.run_workgroup` laufen lässt, funktioniert mit `exec_hist.py`.

## 9. Ergebnisse (nach lokalem Lauf ausfüllen)

| Kennzahl | k_pre_block | k_post_block |
|---|---|---|
| `.amdhsa_next_free_vgpr` / `max_index_used` / Original-Max | | |
| LDS deklariert / Original / `lds_highest_nonzero_byte` | | |
| Scratch-Bytes / Scratch-Ops (übersetzt / Original) | | |
| WMMA / VOPD im Original | | |
| Expansion gesamt / für `v_wmma_*` | | |
| ausgeführt pro Wave: Original / gfx1030 (geschätzt) | | |
| Vergleich mit ~175 000 (§2) → issue- oder latenzgebunden | | |
