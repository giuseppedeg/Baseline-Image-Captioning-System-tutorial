# Sample data

Five images and a five-row CSV, included so that the pipeline can be run end to
end before a real corpus is available. They illustrate the expected schema and
nothing more; no result in this repository is computed from them.

## Provenance and licensing — unresolved

> **This directory is not cleared for redistribution.** The sample images were
> collected for local development and their licensing has not been verified.
> At least one carries a visible third-party watermark.
>
> Before this repository is made public, either replace them with images whose
> licence permits redistribution (public-domain or CC-licensed photographs of
> the same subjects are readily available) or remove them and restore the image
> exclusions in `.gitignore`.

The captions in `data.csv` were written for development and are likewise not
cleared for reuse.

## Known defects in `data.csv`, kept on purpose

The sample is not clean, and the code is expected to survive it:

- every row's `Path` column points at `img/001.jpg`;
- row `004` has a malformed quotation, merging `century` and `caption` into one
  field;
- row `005` carries `century = -1`, a sentinel for "unknown" rather than a
  century.

`CaptionDataset` fails loudly on the first (it verifies every path at
construction), and `parse_century` recovers from the second and third. See
`docs/01_problem_formulation.md`.
