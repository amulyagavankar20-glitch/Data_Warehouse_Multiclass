## Summary

<!-- What does this change and why? Link the issue: Closes #123 -->

## Type

- [ ] `feat`: new capability
- [ ] `fix`: bug fix
- [ ] `docs` / `refactor` / `test` / `ci` / `chore`
- [ ] Breaking change (API request/response or data contract)

## Checklist

- [ ] `make lint && make test` pass locally
- [ ] Tests added or updated for the change
- [ ] Docs updated (`docs/`, README) where behaviour changed
- [ ] `CHANGELOG.md` updated under **Unreleased**
- [ ] No real patient data, secrets or `.env` files committed

### Model / data changes only

- [ ] Before/after metrics attached (`uv run dvc metrics diff` or MLflow run links)
- [ ] `models/serving_contract.json` and `reports/` regenerated with `uv run dvc repro`
- [ ] Data contract and model card updated
- [ ] No column from `settings.LEAKAGE_COLUMNS` added to the features

## Metrics (if applicable)

| Metric | Before | After |
| :-- | --: | --: |
| Test macro ROC-AUC | | |
| Test macro F1 | | |
