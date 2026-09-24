use clap::{Parser, ValueEnum};
use fixedbitset::FixedBitSet;
use rpid::io;
use std::cmp;
use std::error::Error;
use std::fs;

#[derive(Clone, Debug)]
pub struct Instance {
    pub node_weights: Vec<i32>,
    pub edge_weights: Vec<Vec<i32>>,
}

impl Instance {
    pub fn read_from_file(filename: &str) -> Result<Self, Box<dyn Error>> {
        let file = fs::read_to_string(filename)?;
        let mut digits = file.split_whitespace();

        let n = digits.next().ok_or("empty file".to_owned())?.parse()?;
        digits.next().ok_or("missing number of edges".to_owned())?;
        let node_weights = io::read_vector(&mut digits, n)?;
        let edge_weights = io::read_matrix(&mut digits, n, n)?;

        Ok(Self {
            node_weights,
            edge_weights,
        })
    }

    pub fn validate(&self, solution: &[usize], cost: i32) -> bool {
        let n = self.node_weights.len();

        if solution.len() != n {
            println!("Invalid solution length: {} != {}", solution.len(), n);

            return false;
        }

        let mut clean = FixedBitSet::with_capacity(n);
        let mut recomputed_cost = 0;

        for &i in solution {
            if i >= self.node_weights.len() {
                println!("Invalid node index: {i}");

                return false;
            }

            if clean.contains(i) {
                println!("Node {i} is cleaned more than once");

                return false;
            }

            let mut n_robots = self.node_weights[i] + self.edge_weights[i].iter().sum::<i32>();

            for j in clean.ones() {
                for k in clean.zeroes() {
                    if k != i {
                        n_robots += self.edge_weights[j][k];
                    }
                }
            }

            recomputed_cost = cmp::max(recomputed_cost, n_robots);
            clean.insert(i);
        }

        if recomputed_cost != cost {
            println!("Invalid cost: {cost} != {recomputed_cost}");

            return false;
        }

        true
    }
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
