use clap::{Parser, ValueEnum};
use rpid::io;
use std::error::Error;
use std::fs;

#[derive(Clone, Debug)]
pub struct Instance {
    pub profits: Vec<i32>,
    pub weights: Vec<i32>,
    pub capacity: i32,
    pub indices: Vec<usize>,
}

impl Instance {
    pub fn read_from_file(filename: &str) -> Result<Self, Box<dyn Error>> {
        let file = fs::read_to_string(filename)?;
        let mut digits = file.split_whitespace();
        let n = digits
            .next()
            .ok_or("failed to parse the number of items".to_owned())?
            .parse::<usize>()?;
        let capacity = digits
            .next()
            .ok_or("failed to parse the capacity".to_owned())?
            .parse::<i32>()?;
        let matrix = io::read_matrix(&mut digits, n, 2)?;
        let profits = matrix.iter().map(|x| x[0]).collect::<Vec<_>>();
        let weights = matrix.iter().map(|x| x[1]).collect::<Vec<_>>();
        let mut indices = (0..n).collect::<Vec<_>>();
        indices.sort_by_key(|&i| (weights[i] as f64 / profits[i] as f64).to_bits());
        let profits = indices.iter().map(|&i| profits[i]).collect();
        let weights = indices.iter().map(|&i| weights[i]).collect();

        Ok(Self {
            profits,
            weights,
            capacity,
            indices,
        })
    }

    pub fn validate(&self, solution: &[usize], profit: i32) -> bool {
        let total_weight = solution.iter().map(|&i| self.weights[i]).sum::<i32>();

        if total_weight > self.capacity {
            println!(
                "Total weight {total_weight} exceeds capacity {capacity}",
                capacity = self.capacity
            );

            return false;
        }

        let recomputed_profit = solution.iter().map(|&i| self.profits[i]).sum::<i32>();

        if recomputed_profit != profit {
            println!("Invalid profit: {profit} != {recomputed_profit}");

            return false;
        }

        true
    }

    pub fn print_solution(&self, solution: &[usize]) {
        let mut solution_indices = solution
            .iter()
            .map(|&i| self.indices[i])
            .collect::<Vec<_>>();
        solution_indices.sort();
        let solution_indices = solution_indices
            .iter()
            .map(|&i| i.to_string())
            .collect::<Vec<_>>()
            .join(" ");

        println!("Packed Items: {solution_indices}");
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
    #[arg(
        short,
        long,
        default_value_t = 1e-6,
        help = "Threshold for floating point values"
    )]
    pub epsilon: f64,
}
