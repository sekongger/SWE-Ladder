# Stage capability development sample

The sample is a public-input smoke set. It contains no gold source patch,
test patch, test names, or parsed source snapshot.

- Sample rows: 25
- Repositories: astropy/astropy, django/django, matplotlib/matplotlib, mwaskom/seaborn, pallets/flask, psf/requests, pydata/xarray, pylint-dev/pylint
- Sample SHA-256: `14a543fe298f344f9f41e169d63bd2c77dd086ae45161f10c0f7f97c94c2f0ac`
- Public manifest: `stage_capability_eval/manifests/all_swebench_verified.jsonl`
- Private LOC labels: `stage_capability_eval/evaluator_private/loc_labels.jsonl`

Mapping status distribution:

- `full`: 13
- `no_hit`: 1
- `partial`: 10
- `unavailable`: 1

Collection status: **development only**. Full-scale model collection is not
approved until real buggy/fixed preflight and both evaluator entry modes pass.


Run `python -m unittest discover -s stage_capability_eval/tests -p 'test*.py'` before collection.
