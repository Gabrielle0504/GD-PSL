# Baselines

`pang_2023_baseline.py` is the standalone paper-protocol reproduction of Pang,
Nan, and Ishibuchi (SMC 2023). It embeds the required MATLAB algorithm adapter
and does not import GD-PSL modules.

The authors' supplied reference package was used to audit the reproduction but
is not redistributed here: it contains a complete third-party PlatEMO tree and
historical experiment artifacts under separate copyright terms. The exact
observations carried into this implementation are documented in
`../docs/pang_full_text_reproduction.md`.

The project-level runner `../run_pang_archive_baseline.py` applies the Pang
archive method to the common problem suite and common true-FE protocol used in
the GD-PSL comparison.
