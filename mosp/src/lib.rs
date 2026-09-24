use clap::{Parser, ValueEnum};
use fixedbitset::FixedBitSet;
use rpid::io;
use std::cmp;
use std::error::Error;
use std::fs;

pub fn read_from_file(filename: &str) -> Result<Vec<FixedBitSet>, Box<dyn Error>> {
    let file = fs::read_to_string(filename)?;
    let mut digits = file.split_whitespace();

    let m = digits
        .next()
        .ok_or("failed to parse the number of rows".to_owned())?
        .parse()?;
    let n = digits
        .next()
        .ok_or("failed to parse the number of columns".to_owned())?
        .parse()?;
    let matrix = io::read_matrix::<i8>(&mut digits, m, n)?;

    Ok(matrix
        .into_iter()
        .map(|row| {
            let mut set = FixedBitSet::with_capacity(row.len());
            row.iter().enumerate().for_each(|(i, &x)| {
                if x == 1 {
                    set.insert(i)
                }
            });

            set
        })
        .collect())
}

pub fn transpose(matrix: &[FixedBitSet]) -> Vec<FixedBitSet> {
    if matrix.is_empty() {
        return vec![];
    }

    let m = matrix.len();
    let n = matrix.iter().map(|row| row.len()).max().unwrap();

    (0..n)
        .map(|i| {
            let mut set = FixedBitSet::with_capacity(m);
            matrix.iter().enumerate().for_each(|(j, row)| {
                if row.contains(i) {
                    set.insert(j);
                }
            });

            set
        })
        .collect()
}

pub fn validate(matrix: &[FixedBitSet], schedule: &[usize], cost: i32) -> bool {
    if schedule.len() != matrix.len() {
        println!("Invalid schedule length: {len}", len = schedule.len());

        return false;
    }

    let transposed = transpose(matrix);
    let mut produced = FixedBitSet::with_capacity(matrix.len());
    let mut open = FixedBitSet::with_capacity(transposed.len());
    let mut recomputed_cost = 0;

    for &i in schedule.iter() {
        if i >= matrix.len() {
            println!("Invalid row index: {i}");

            return false;
        }

        if produced.contains(i) {
            println!("Containing row twice: {i}");

            return false;
        }

        produced.insert(i);
        open.union_with(&matrix[i]);
        recomputed_cost = cmp::max(recomputed_cost, open.count_ones(..) as i32);

        let mut closed = FixedBitSet::with_capacity(open.len());
        open.ones().for_each(|i| {
            if transposed[i].is_subset(&produced) {
                closed.insert(i);
            }
        });
        open.difference_with(&closed);
    }

    if recomputed_cost != cost {
        println!("Invalid cost: {cost} != {recomputed_cost}");

        return false;
    }

    true
}

#[derive(Debug, Clone, ValueEnum)]
pub enum SolverChoice {
    Cabs,
    Astar,
}

#[derive(Debug, Parser)]
pub struct Args {
    #[arg(
        short = 'j',
        long,
        default_value = "1",
        help = "Number of threads for CABS"
    )]
    pub threads: std::num::NonZeroUsize,
    #[arg(help = "Input file")]
    pub input_file: String,
    #[arg(short, long, value_enum, default_value_t = SolverChoice::Cabs, help = "Solver")]
    pub solver: SolverChoice,
    #[arg(long, default_value_t = String::from("history.csv"), help = "File to save the history")]
    pub history: String,
    #[arg(short, long, default_value_t = 1800.0, help = "Time limit")]
    pub time_limit: f64,
}
