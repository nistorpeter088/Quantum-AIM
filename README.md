# Fraud on a Quantum Budget — Variational Quantum Fraud Detection

Proiect pentru **Task 2**: detectarea tranzacțiilor frauduloase din **ULB Credit Card Fraud** cu un *Variational Quantum Classifier* (VQC) implementat în PennyLane, sub *depolarizing noise*, cu atenție la AUPRC și costul de execuție pe hardware cuantic.

> **Soluția principală documentată:** `quantic_mlp.py`, encoder MLP `30 → 8 → 4` + circuit **Reverse2**. Codul conține și baseline-ul PCA + ZZ. `quantic_linear.py` și `quantic_kan_spsa.py` reprezintă experimente/comparații separate. KAN/SPSA nu este descris mai jos ca rezultat verificat, deoarece codul său și logurile complete nu au fost furnizate aici pentru audit.

## 1. Cerințe ale competiției

- Dataset ULB: 30 features (`V1`–`V28`, `Time`, `Amount`), etichetă `Class`.
- Același split stratificat **60% TRAIN / 20% VALIDATION / 20% TEST**; indicii sunt în `split_indices.npz`.
- Preprocesarea/encoderul se antrenează numai pe TRAIN; validation alege checkpoint-ul și threshold-ul; TEST este rezervat evaluării finale.
- Quantum în **PennyLane**, componentele neuronale în **PyTorch**.
- Noise depolarizing pe porți din setul `{RX, RY, RZ, Rot, H, CNOT, CZ}`, la `p = 0`, `0.001`, `0.01`, `0.05`.
- Threshold ales pe **validation la același noise**, optimizând F1.
- Raportare: **AUPRC** (principal), AUROC, Spearman (clean/noisy), precision, recall, F1, MCC; medie ± deviație standard pe **5 seed-uri**.
- Comparație cu **baseline VQC** și minimum **Logistic Regression pe aceleași features/unghiuri**.
- *Cost ledger*: qubiți, porți, parametri, locații de noise, execuții estimate și shots.

Consultați documentul original `Fraud on a Quantum Budget - Task.pdf` și notebookul `task1_guided_vqc_fraud_detection.ipynb` pentru protocolul oficial.

## 2. Soluția principală: MLP + Reverse2

```text
creditcard.csv (30 features)
  → StandardScaler (fit doar pe TRAIN)
  → MLP: Linear(30,8) → ReLU → Linear(8,4)
  → π × sigmoid → 4 unghiuri
  → VQC Reverse2, 4 qubiți
  → readout ⟨Z₀⟩ → scor (1 − ⟨Z₀⟩)/2
  → threshold ales pe VALIDATION → predicție fraudă/legitim
```

**De ce acest design:** PCA nu folosește etichetele de fraudă; MLP-ul învață o proiecție neliniară supravegheată. Circuitul păstrează numai 4 qubiți și folosește rotații `Rot` combinate (*gate fusion*). CNOT-urile sunt orientate `3→2, 2→1, 1→0`, repetate de două ori, pentru ca toate cele patru intrări să poată influența măsurarea `Z₀`.

| Componentă | Baseline Task 1 | MLP + Reverse2 |
|---|---:|---:|
| Reducere dimensională | PCA, 30 → 4 | MLP, 30 → 8 → 4 |
| Encoding | ZZ feature map ×2 | Rotații de unghi integrate în `Rot` |
| Qubiți | 4 | 4 |
| Parametri cuantici | 32 | 17 |
| Porți, modelul de concurs | 93 | 15 |
| Locații de noise | 126 | 21 |
| Ansatz | RY/RZ, 3 straturi CNOT | 2 lanțuri CNOT inversate, rotații comasate |

### Antrenare

1. **Pretraining encoder** pe TRAIN complet: `BCEWithLogitsLoss(pos_weight=...)`, cu un clasificator temporar, Adam, selecția epocii după AUPRC pe validation.
2. **Training cuantic clean**: encoderul este fix; ~590 tranzacții TRAIN balansate (toate fraudele TRAIN + același număr de legitime); sunt actualizați parametrii VQC prin BCE + Adam.
3. **Joint fine-tuning**: MLP-ul și parametrii cuantici pot fi actualizați împreună, cu learning rate mai mic.
4. **Evaluare noise**: checkpoint-ul ales este evaluat la cele 4 valori `p` fără reantrenare noisy completă; pentru fiecare `p`, threshold-ul se stabilește pe validation înainte de folosirea TEST.

**Important:** valorile de AUPRC obținute în `quick` sau la validation subeșantionat/ponderat sunt estimări de dezvoltare. Nu se raportează drept rezultat oficial pe TEST.

## 3. Rezultate MLP + Reverse2 — TEST complet, 5 seed-uri

Valorile de mai jos sunt din rularea furnizată pentru `vqc_runs/mlp_test1`:

| Nivel de noise | AUPRC medie ± std | Spearman medie ± std |
|---|---|---|
| 0 | 0.7269 ± 0.0193 | 1.0000 ± 0.0000 |
| 0.001 | 0.7285 ± 0.0207 | ~1.0000 |
| 0.01 | 0.7282 ± 0.0212 | 0.9953 ± 0.0042 |
| 0.05 | **0.7267 ± 0.0190** | **0.8998 ± 0.0791** |

Referința **Logistic Regression** pe aceleași unghiuri MLP: **0.7213 ± 0.0082 AUPRC** pe TEST. Avantajul mediu VQC este modest; nu constituie dovada unui avantaj cuantic general. AUPRC aproape constant sub noise **nu** înseamnă ranking identic (de aceea raportăm Spearman).

Rezultatele baseline se citesc din `vqc_runs/baseline_original/test_results.csv`; înainte de predare, verificați că acoperă **toți cei 5 seed-uri × toate cele 4 valori p**. Acest README nu inventează statistici baseline care nu au fost furnizate integral.

## 4. Costul cuantic estimat

`estimate_cost()` din `quantic_mlp.py` estimează execuțiile de hardware pe baza regulii **parameter-shift**, nu timpul CPU/GPU observat și nu numărul real de shots rulate de simulatorul exact.

Pentru configurațiile folosite în experiment, *per seed*, incluzând validările de checkpoint:

| Resursă | Baseline (30 epoci, full validation la fiecare epocă) | Reverse2 MLP (12 clean + 2 joint, validation redus) |
|---|---:|---:|
| Training, execuții estimate | 1.150.500 | 298.540 |
| Validation în timpul trainingului | 1.708.860 | 30.594 |
| **Total** | **2.859.360** | **329.134** |
| Shots ipotetice, 1.024 / execuție | 2.927.984.640 | 337.033.216 |

Reducerea estimată este ~**88,5%** pentru training **plus selecția checkpoint-urilor**. Comparația reflectă **și numărul diferit de epoci/validări**, nu numai simplificarea circuitului. Pretraining-ul MLP este calcul clasic și nu intră în acest ledger. Evaluarea completă ulterioară pe noise/TEST adaugă execuții și trebuie raportată separat dacă este inclusă în bugetul total.

Pentru valori pe seed și configurația reală, folosiți `vqc_runs/*/cost.csv`.

## 5. Fișierele sursă

- **`quantic_mlp.py`** — soluția principală: MLP integrat, baseline PCA+ZZ, `improved`, `reverse2`, `pruned`; training, validation, evaluare noisy, Logistic Regression, cost.
- **`quantic_linear.py`** — experiment de comparație cu encoder liniar `30→4` integrat; nu este necesar pentru a executa `quantic_mlp.py`.
- **`quantic_kan_spsa.py`** — experiment separat KAN/SPSA; verificați dependențele și comenzile proprii înainte de a afirma că este reproductibil. Rezultatele aferente sunt `kan_vqc_*.csv`.
- **`linear.py`** — implementare istorică a encoderului liniar, păstrată pentru trasabilitate și compatibilitate cu experimente anterioare. Nu este importată de `quantic_mlp.py`.
- **`task1_guided_vqc_fraud_detection.ipynb`** — notebookul ghidat/baseline al competiției.
- **`split_indices.npz`** — split-ul comun. Păstrați exact fișierul folosit în experimente; nu-l regenerați inutil.

## 6. Instalare și date

Necesită Python 3.12 (versiunea folosită în logurile locale) și pachetele din `requirements.txt`. Versiunile exacte ale mediului care a produs rezultatele trebuie păstrate separat cu `python -m pip freeze > requirements-lock.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data
# Obțineți separat datasetul ULB și puneți-l în data/creditcard.csv
```

Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud sau OpenML dataset 42175. **`data/creditcard.csv` nu este inclus în Git** (fișier mare / proveniență externă). `split_indices.npz` trebuie să corespundă aceluiași dataset și ordinii rândurilor.

## 7. Comenzi reproductibile

### Verificare inițială

```bash
python quantic_mlp.py sanity
```

### Antrenare MLP + Reverse2 de la zero, 5 seed-uri

```bash
python quantic_mlp.py train --tag mlp_test1 --models reverse2 --all-seeds --hidden-dim 8 --encoder-epochs 10 --epochs 12 --joint-epochs 2 --init small
```

Dacă există checkpoint-uri în tag-ul respectiv, scriptul **nu** le suprascrie implicit. Pentru un experiment complet nou, folosiți alt `--tag`.

### Validare completă, fără retraining

```bash
python quantic_mlp.py eval --tag mlp_test1 --models reverse2 --all-seeds --noise 0 0.001 0.01 0.05 --full-val
```

### TEST oficial (numai după finalizarea selecției pe validation)

```bash
python quantic_mlp.py eval --tag mlp_test1 --models reverse2 --all-seeds --noise 0 0.001 0.01 0.05 --official
```

### Baseline oficial — antrenare și evaluare

```bash
python quantic_mlp.py train --tag baseline_original --models baseline --all-seeds --epochs 30 --joint-epochs 0 --val-negatives 0 --val-every 1
python quantic_mlp.py eval --tag baseline_original --models baseline --all-seeds --noise 0 0.001 0.01 0.05 --official
```

Pentru reproducerea exactă a checkpoint-urilor deja prezente, folosiți scriptul/parametrii cu care au fost antrenate: baseline-ul istoric poate proveni din `quantic_linear.py`. **Nu combinați checkpoint-uri din experimente cu arhitecturi diferite.**

## 8. Cum sunt salvate rezultatele

```text
vqc_runs/mlp_test1/
  mlp_seed0.pt ... mlp_seed4.pt           # encodere preantrenate
  reverse2_seed0.pt ... reverse2_seed4.pt # checkpoint final: VQC + encoderul aferent
  validation_results.csv                 # validation complet (sau ultima validare executată)
  test_results.csv                       # metrici oficiale + LR
  cost.csv                               # cost estimat per seed

vqc_runs/baseline_original/
  baseline_seed0.pt ... baseline_seed4.pt
  test_results.csv
  cost.csv
```

`test_results.csv` conține metricile per seed și noise; poate fi agregat prin `groupby(['model','noise'])`. Fișierul `cost.csv` este o estimare de cost, nu un profil de timp real. Atenție: scriptul scrie rezultatele abia la finalul unei evaluări complete; o întrerupere poate lăsa fișierul anterior neschimbat.

## 9. Ce intră în GitHub

**Cod și protocol:** `quantic_mlp.py`, `quantic_linear.py`, `quantic_kan_spsa.py`, `linear.py`, `split_indices.npz`, notebookul Task 1, `README.md`, `.gitignore`, `requirements.txt`. Includeți PDF-ul competiției **numai dacă redistribuirea este permisă de organizatori**.

**Dovezi și evaluare:** `kan_vqc_{runs,summary,cost}.csv`, rezultatele și costurile din `vqc_runs/mlp_test1`, `vqc_runs/baseline_original`, `vqc_runs/reparat2`. Checkpoint-urile **finale** MLP+Reverse2 și baseline sunt păstrate pentru evaluare fără reantrenare; cele experimentale sunt ignorate.

**Nu urcați:** `data/creditcard.csv`, `.venv/`, `__pycache__/`, fișiere temporare quick/dev, `results/*.joblib`, `vqc_checkpoints/`, `linear_only_seed0.pt`, checkpoint-uri din `fresh1/`, `nou1/`, `reparat2/`, `main.py` și `serban.py` (păstrați-le în arhiva locală până confirmați că nu sunt necesare). Acest README **nu certifică** faptul că KAN nu folosește modulele din urmă, deoarece codul KAN nu a fost inspectat.

## 10. Checklist înainte de predarea Task 2

- [ ] `split_indices.npz` verificat ca fiind cel folosit de Task 1.
- [ ] Baseline `test_results.csv`: câte **5 seed-uri × 4 noise** pentru baseline.
- [ ] MLP+Reverse2 `test_results.csv`: 5 × 4, aceleași split și protocol.
- [ ] Comparație LR cu aceleași 4 inputuri, raportată corect.
- [ ] Tabel cu AUPRC, AUROC, Spearman, precision, recall, F1, MCC: **mean ± std**.
- [ ] Cost ledger: training, checkpoint validation, test/noise dacă este raportat, shots (explicând ipotezele).
- [ ] `requirements-lock.txt` generat din mediul real de execuție și, ideal, versiuni fixate.
- [ ] Prezentare PDF/PPTX cu pipeline, circuit, schimbări, rezultate, cost, experimente nereușite și limitări.

## 11. Limitări și interpretarea rezultatelor

- Datele din `TEST` nu trebuie folosite pentru noi alegeri de arhitectură sau hiperparametri.
- Scăderea Spearman la noise 5% indică schimbări în ranking, chiar dacă AUPRC rămâne aproape neschimbat.
- Avantajul față de Logistic Regression este mic; nu afirmăm automat „quantum advantage”.
- KAN și SPSA au fișiere proprii, dar rezultatele și dependențele lor nu au fost verificate în materialele disponibile aici.
- Numărul de porți/noise sites și economiile de cost sunt specifice gate set-ului și simulării din cerința competiției; hardware-ul real poate descompune `Rot` diferit.
