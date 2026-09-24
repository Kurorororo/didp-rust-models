# Domain-Independent Dynamic Programming Models in Rust

Models using didp-rs (`dypdl` and `dypdl-heuristic-search`) 0.11.1 and RPID 0.4.0.

Build with Rust 1.90 or later:

```sh
cargo build --release --workspace --bins
```

Every RPID and DyPDL executable supports parallel CABS through `--threads N` (or `-j N`):

```sh
cargo run --release -p optw --bin optw_rpid -- instance.txt --solver cabs --threads 4
cargo run --release -p optw --bin optw_dypdl -- instance.txt --solver cabs --threads 4
```

The default is serial CABS (`--threads 1`); the thread count must be positive.
`--solver astar` remains serial. Both solvers write progress to `--history`
(default: `history.csv`) and accept `--time-limit` in seconds.

WT provides `wt_rpid` and `wt_dypdl`. Both compute total processing time from the
scheduled jobs once per state; the DyPDL model uses a state function.

Run the Rust tests and the CLI regressions with:

```sh
cargo test --workspace
cargo build --workspace --bins
python3 tests/test_models.py
```

The tests enumerate small instances to check optimal costs and compare serial
and parallel CABS across all RPID and DyPDL executables. CLI tests use temporary instance
and history files. Set `MODEL_BIN_DIR` to test binaries in another directory.
