use clap::{Parser, ValueEnum};
use rpid::io;
use std::error::Error;
use std::fs;

#[derive(Clone, Debug)]
pub struct Instance {
    pub capacity: i32,
    pub weights: Vec<i32>,
}

impl Instance {
    pub fn read_from_file(filename: &str) -> Result<Self, Box<dyn Error>> {
        let file = fs::read_to_string(filename)?;
        let mut digits = file.split_whitespace();

        let n = digits
            .next()
            .ok_or("failed to parse the number of items".to_owned())?
            .parse()?;
        let capacity = digits
            .next()
            .ok_or("failed to parse the capacity".to_owned())?
            .parse()?;
        let weights = io::read_vector(&mut digits, n)?;

        Ok(Self { capacity, weights })
    }

    pub fn validate(&self, solution: &[usize], cost: i32) -> bool {
        let n = self.weights.len();

        if solution.len() != n {
            println!("Invalid solution length: {} != {}", solution.len(), n);

            return false;
        }

        let mut bins = vec![];
        let mut capacity = 0;
        let mut packed = vec![false; self.weights.len()];
        let mut recomputed_cost = 0;

        for &i in solution {
            if i >= n {
                println!("Invalid item index: {i}");

                return false;
            }

            if packed[i] {
                println!("Item {i} is packed more than once");

                return false;
            }

            if self.weights[i] > capacity {
                bins.push(vec![]);
                capacity = self.capacity;
                recomputed_cost += 1;
            }

            bins.last_mut().unwrap().push(i);
            capacity -= self.weights[i];
            packed[i] = true;
        }

        if recomputed_cost != cost {
            println!("Invalid cost: {cost} != {recomputed_cost}");

            return false;
        }

        true
    }

    pub fn print_solution(&self, solution: &[usize]) {
        let mut bins = vec![];
        let mut capacity = 0;

        for &i in solution {
            if self.weights[i] > capacity {
                bins.push(vec![]);
                capacity = self.capacity;
            }

            bins.last_mut().unwrap().push(i);
            capacity -= self.weights[i];
        }

        println!("Solution: {bins:?}");
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
