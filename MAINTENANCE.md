# Maintenance Log

## 2025-11-17: Week3.ipynb Warning Suppression

**Issue:** Week3 notebook produces warnings/errors that distract from current work on other subprojects (e.g., gcnn_opf_01). These include both runtime warnings and Pylance static analysis diagnostics (reportIndexIssue, reportCallIssue, reportOptionalOperand, etc.) - 30+ errors persisting even after pyrightconfig exclusion.

**Solution:** 
1. Added warning suppression cell at the top of `Week3/Week3.ipynb`:
   ```python
   # pyright: reportAttributeAccessIssue=false, reportIndexIssue=false, ...
   import warnings
   warnings.filterwarnings('ignore')
   ```
2. Excluded `Week3/` from Pylance in multiple config layers:
   - `pyrightconfig.json`: Added `"**/Week3/**"` and `"Week3/Week3.ipynb"` to exclude list
   - `.vscode/settings.json`: Added `python.analysis.exclude` and `python.analysis.ignore` for `**/Week3/**`

**Rationale:** Week3 is not the current focus; multi-layer exclusion ensures Pylance diagnostics don't appear regardless of VS Code configuration loading order.

**Files Modified:**
- `Week3/Week3.ipynb`: Added suppression directives and runtime warning filter at top
- `pyrightconfig.json`: Added Week3 paths to exclude list
- `.vscode/settings.json`: Added python.analysis.exclude and ignore rules

**Impact:** Pylance type-checking errors in Week3 should no longer appear in VS Code Problems panel. Requires window reload to take effect.

**Operational Note:** To avoid any remaining noise, do not open `Week3/Week3.ipynb` during current development on other subprojects. Keep VS Code set to show diagnostics for open files only (already configured).
