# EARLY-STOP-A01 diagnostics

## Stopping result

No fold demonstrated convergence. Each fold evaluated 51 eligible checkpoints from 1500 through 4000 completed iterations; every full convergence gate was FAIL. All folds retained zero-based iteration 3999 with `stop_reason = MAX_ITERATIONS`.

### Fold 1

- Stop: 4000 completed / iteration 3999; MAX_ITERATIONS; convergence demonstrated = false.
- Gate sequence: 1500:50:4000 = FAIL x 51; final patience = 0.
- Diagnostic ELBO: loss 0.166704606788; endpoint relative change 0.000108033409114; relative range 0.00553249357903; pass = false.
- Parameters: spatial X 0.00586330388679; spatial Y 0.00902525003555; variance 0.0169289685425; temporal LS 0.0620794231077; beta_time absolute 0.0119429976954; pass = false.
- CVI sites: median 0.091646141456; maximum 0.0985900336204; pass = false.
- Limiting components at 4000: ELBO,PARAMETER,CVI_SITE.

### Fold 2

- Stop: 4000 completed / iteration 3999; MAX_ITERATIONS; convergence demonstrated = false.
- Gate sequence: 1500:50:4000 = FAIL x 51; final patience = 0.
- Diagnostic ELBO: loss 0.148824152109; endpoint relative change 0.00760005335952; relative range 0.011414545344; pass = false.
- Parameters: spatial X 0.0222992606151; spatial Y 0.00130572408671; variance 0.0215798782777; temporal LS 0.066483904615; beta_time absolute 0.0189467277845; pass = false.
- CVI sites: median 0.115516935232; maximum 0.120673950768; pass = false.
- Limiting components at 4000: ELBO,PARAMETER,CVI_SITE.

### Fold 3

- Stop: 4000 completed / iteration 3999; MAX_ITERATIONS; convergence demonstrated = false.
- Gate sequence: 1500:50:4000 = FAIL x 51; final patience = 0.
- Diagnostic ELBO: loss 0.133893540015; endpoint relative change 0.00871001189608; relative range 0.0156414490878; pass = false.
- Parameters: spatial X 0.0069898429135; spatial Y 0.0293936692234; variance 0.010513732986; temporal LS 0.0685555809016; beta_time absolute 0.0207093012627; pass = false.
- CVI sites: median 0.128341697697; maximum 0.131885062221; pass = false.
- Limiting components at 4000: ELBO,PARAMETER,CVI_SITE.

## Runtime

Logged optimization time totaled 11462.198 seconds (3.184 hours). Fold times were 1: 2683.828s, 2: 4040.726s, 3: 4737.643s.

## Prediction assessment

The result is **mixed** relative to TREN-A01 at 1500 iterations: ranking metrics improved in all folds, while probability quality and calibration changed unevenly. The configured stopping rule did not trigger, so these are maximum-budget states, not convergence-selected states.

Macro deltas below are EARLY-STOP-A01 minus TREN-A01:

- pr_auc: 0.169053016397 -> 0.214128900512 (delta +0.0450758841157)
- roc_auc: 0.799628177241 -> 0.835134788846 (delta +0.0355066116054)
- log_loss: 0.232467627089 -> 0.254290778268 (delta +0.0218231511791)
- brier_score: 0.0615711808373 -> 0.0702748612563 (delta +0.00870368041898)
- ece: 0.0547596501007 -> 0.088588839653 (delta +0.0338291895523)
- probability_bias: 0.0282260008509 -> 0.0880589750409 (delta +0.0598329741899)
- calibration_slope: 0.806319103373 -> 1.45342678853 (delta +0.647107685152)

Detailed fold metrics and convergence comparisons are in the adjacent CSV files. Raw stochastic ELBO, gradient/clipping, and minimum site precision are secondary diagnostics and did not control stopping.
