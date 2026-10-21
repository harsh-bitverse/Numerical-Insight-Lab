# Upstream Provenance & Attribution Record

## 1. Upstream Source Information

* **Repository Name:** `pinn-fdm-error-estimation`
* **Repository URL:** `https://github.com/aleks-krasowski/pinn-fdm-error-estimation`
* **Upstream Authors:** Aleksander Krasowski and Aycan Celik (Fraunhofer Heinrich Hertz Institute, Berlin)
* **Associated Paper:**
  * Title: *Building Trust in PINNs: Error Estimation through Finite Difference Methods*
  * Authors: Aleksander Krasowski, René P. Klausen, Aycan Celik, Sebastian Lapuschkin, Wojciech Samek, Jonas Naujoks
  * arXiv: `https://arxiv.org/abs/2603.15526` (cs.LG)
  * Target Venue: Submitted to the XAI World Conference 2026

## 2. Commit & Provenance Details

* **Retrieved Commit SHA:** `460d66bc2a59720883fa6a06e8c5436db4f99020`
* **Commit Date:** Wed Jul 1 22:50:16 2026 -0300
* **Commit Message:** `Create CNAME`
* **Upstream Git Tags:** None (verified with `git tag -l`)
* **Retrieval Timestamp:** 2026-10-20 18:06:22 IST (UTC+05:30)
* **Local Isolation Path:** `upstream/pinn-fdm-error-estimation/`

## 3. Vendored Snapshot Scope

To ensure reproducible, offline-verifiable research and avoid Git submodule synchronization friction among undergraduate team members, the upstream source code was imported as an immutable **vendored snapshot**.

* **Snapshot Scope:** Exactly mirrors upstream repository at commit `460d66bc2a59720883fa6a06e8c5436db4f99020`.
* **Included Components (405 files):**
  * `pyproject.toml`
  * `README.md`
  * `pinn_error/` (Core library, problem definitions, utilities)
  * `examples/` (`run_experiment.py`, `run_all_experiments.py`)
  * `figures/` (Paper figures, problem plots, comparison plots)
  * `notebooks/` (Exploratory evaluation notebooks)
  * `results/` (`all_experiments_results_float64.csv` dataset, 850 KB)
  * `docs/` (Interactive web demo, HTML, JS, data files)
* **Excluded Files:**
  * Upstream `.git/` directory (to avoid nested repository corruption)
  * Upstream macOS metadata (`figures/.DS_Store`)
* **Integrity Verification:**
  * 405 out of 405 files verified via SHA-256 hash comparison between the temporary upstream clone and the local vendored path.
  * Zero hash mismatches.

## 4. Legal & Licensing Status (Strict Separation)

### 4.1 License Stated for the Research Paper
* **Paper License:** Creative Commons Attribution 4.0 International (**CC BY 4.0**).
* **Scope:** Governs the preprint manuscript text, figures, and presentation on arXiv (`arXiv:2603.15526`).
* **Important Notice:** The CC BY 4.0 license applies **strictly to the paper publication**, and does **not** automatically extend to or license the software repository source code.

### 4.2 Licensing Status of the Source Code
* **Code License:** **No license file** is present in the official GitHub repository (no `LICENSE`, `LICENSE.md`, or license header in source files).
* **Default Copyright Status:** Under standard intellectual property law and GitHub Terms of Service, when a repository has no open-source license, **all rights are reserved by default** by the original authors (Aleksander Krasowski, Aycan Celik, and their institution Fraunhofer HHI). No public copyright license (such as MIT, Apache 2.0, or BSD) has been granted for the software.

### 4.3 Upstream Authors' Attribution Request
* The upstream `README.md` and repository landing page explicitly request academic citation for any use of the software:

```bibtex
@misc{krasowski2026buildingtrustpinnserror,
      title={Building Trust in PINNs: Error Estimation through Finite Difference Methods}, 
      author={Aleksander Krasowski and René P. Klausen and Aycan Celik and Sebastian Lapuschkin and Wojciech Samek and Jonas Naujoks},
      year={2026},
      eprint={2603.15526},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.15526}, 
}
```

### 4.4 Unresolved Redistribution & Reuse Restrictions
Because no open-source license was attached to the repository:
1. **Redistribution Rights:** Our team does **not** possess a license to re-license, redistribute, or publish the authors' original source code under a permissive license (such as MIT) or for commercial purposes.
2. **Internal Academic Evaluation:** Use within this undergraduate Numerical Analysis course project at IIT Delhi constitutes non-commercial academic research and fair evaluation/reproduction of published scientific results, accompanied by full attribution.
3. **Public Sharing:** If our team repository is eventually published publicly on GitHub, the upstream code must remain identified as the proprietary copyright of the original authors, or linked externally rather than re-licensed. We do not invent or assume any open-source license for the code.

## 5. Scientific Modification Policy

The files inside `upstream/pinn-fdm-error-estimation/` remain strictly **unmodified** to maintain exact baseline fidelity. Any wrapper scripts, fixes for platform portability (e.g. font loading, missing tests), and extensions must be developed outside `upstream/` in separate project modules.
