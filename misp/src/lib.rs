use clap::{Parser, ValueEnum};
use itertools::Itertools;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};

#[derive(Clone, Debug)]
pub struct Instance {
    pub n: usize,
    pub adjacency_list: Vec<Vec<usize>>,
}

impl Instance {
    pub fn read_from_file(filename: &str) -> Result<Self, Box<dyn Error>> {
        let file = File::open(filename)?;
        let lines = BufReader::new(file).lines();
        let mut n = None;
        let mut adjacency_list = Vec::new();

        for line in lines {
            let line = line?;
            let mut digits = line.split_whitespace();
            let line_type = digits.next().ok_or("failed to parse the line type")?;

            match line_type {
                "c" => continue,
                "p" => {
                    let text = digits.next().ok_or("failed to parse 'edge'")?;

                    if text != "edge" {
                        return Err("invalid line type".into());
                    }

                    n = Some(
                        digits
                            .next()
                            .ok_or("failed to parse the number of vertices")?
                            .parse::<usize>()?,
                    );
                }
                "e" => {
                    let u = digits
                        .next()
                        .ok_or("failed to parse the first vertex")?
                        .parse::<usize>()?;

                    let v = digits
                        .next()
                        .ok_or("failed to parse the second vertex")?
                        .parse::<usize>()?;

                    if adjacency_list.len() < u || adjacency_list.len() < v {
                        adjacency_list.resize_with(u.max(v), Vec::new);
                    }

                    adjacency_list[u - 1].push(v - 1);
                    adjacency_list[v - 1].push(u - 1);
                }
                _ => return Err("invalid line type".into()),
            }
        }

        let n = n.ok_or("missing the number of vertices")?;

        adjacency_list.iter_mut().for_each(|neighbors| {
            neighbors.sort_unstable();
            neighbors.dedup();
        });

        Ok(Self { n, adjacency_list })
    }

    pub fn validate(&self, independent_set: &[usize]) -> bool {
        for [&i, &j] in independent_set.iter().array_combinations() {
            if i >= self.n {
                println!("Node {i} is out of bounds");

                return false;
            }

            if j >= self.n {
                println!("Node {j} is out of bounds");

                return false;
            }

            if i == j {
                println!("Node {i} is repeated");

                return false;
            }

            if self.adjacency_list[i].contains(&j) {
                println!("Nodes {i} and {j} are adjacent");

                return false;
            }
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
