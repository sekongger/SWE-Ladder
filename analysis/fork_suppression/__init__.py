"""Fork-suppression offline analysis package.

Pure offline diagnostic: load a frozen checkpoint, run forwards over fixed
trajectories, compute OPSD token-level uplift / NEP, plot. NO weight updates.

See REPORT.md in this directory for the writeup and DEVIATIONS.md for where the
reproduction departs from the paper it replicates (arXiv:2607.05184).
"""
