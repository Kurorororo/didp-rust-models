pub mod dypdl_model;
pub mod rpid_model;

use clap::{Parser, ValueEnum};
use rpid::algorithms;
use std::cmp;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::ops::Add;

#[derive(Clone, Debug)]
pub struct Instance {
    pub vertices: Vec<usize>,
    pub coordinates: Vec<(f64, f64)>,
    pub service_time: Vec<f64>,
    pub profits: Vec<f64>,
    pub opening: Vec<f64>,
    pub closing: Vec<f64>,
}

impl Instance {
    pub fn read_from_file(filename: &str) -> Result<Self, Box<dyn Error>> {
        let file = File::open(filename)?;
        let mut lines = BufReader::new(file).lines();

        let line = lines.next().ok_or("failed to read the first line")??;
        let mut digits = line.split_whitespace();
        digits.next();
        digits.next();
        let n = digits
            .next()
            .ok_or("failed to parse the number of customers")?
            .parse::<usize>()?
            + 1;

        lines.next();

        let mut vertices = Vec::with_capacity(n);
        let mut points = Vec::with_capacity(n);
        let mut service_time = Vec::with_capacity(n);
        let mut profits = Vec::with_capacity(n);
        let mut opening = Vec::with_capacity(n);
        let mut closing = Vec::with_capacity(n);

        for i in 0..n {
            let line = lines
                .next()
                .ok_or(format!("failed to read the {i}-th line"))??;
            let mut digits = line.split_whitespace();
            let v = digits.next().ok_or("failed to parse the vertex")?.parse()?;
            let x = digits
                .next()
                .ok_or("failed to parse the x-coordinate")?
                .parse()?;
            let y = digits
                .next()
                .ok_or("failed to parse the y-coordinate")?
                .parse()?;
            let s = digits
                .next()
                .ok_or("failed to parse the service time")?
                .parse()?;
            let p = digits.next().ok_or("failed to parse the profit")?.parse()?;
            let mut digits = digits.rev();
            let b = digits
                .next()
                .ok_or("failed to parse the closing time")?
                .parse()?;
            let a = digits
                .next()
                .ok_or("failed to parse the opening time")?
                .parse()?;

            vertices.push(v);
            points.push((x, y));
            service_time.push(s);
            profits.push(p);
            opening.push(a);
            closing.push(b);
        }

        Ok(Self {
            vertices,
            coordinates: points,
            service_time,
            profits,
            opening,
            closing,
        })
    }
}

#[derive(Clone, Debug)]
pub struct RoundedInstance {
    pub vertices: Vec<usize>,
    pub distances: Vec<Vec<i32>>,
    pub profits: Vec<i32>,
    pub opening: Vec<i32>,
    pub closing: Vec<i32>,
}

impl RoundedInstance {
    pub fn new(instance: Instance, round_to: u32) -> Self {
        let pow = 10f64.powf(round_to as f64);

        let distances = algorithms::compute_pairwise_euclidean_distances(&instance.coordinates);
        let distances = distances
            .into_iter()
            .zip(instance.service_time)
            .map(|(row, s)| {
                row.into_iter()
                    .map(|d| (s * pow).trunc() as i32 + (d * pow).trunc() as i32)
                    .collect()
            })
            .collect();
        let profits = instance.profits.into_iter().map(|p| p as i32).collect();
        let opening = instance
            .opening
            .into_iter()
            .map(|t| (t * pow).trunc() as i32)
            .collect();
        let closing = instance
            .closing
            .into_iter()
            .map(|t| (t * pow).trunc() as i32)
            .collect();

        Self {
            vertices: instance.vertices,
            distances,
            profits,
            opening,
            closing,
        }
    }

    pub fn validate(&self, solution: &[usize], cost: i32) -> bool {
        let n = self.vertices.len();
        let mut visited = vec![false; n];
        let mut current = 0;
        let mut time = self.opening[0].max(0);
        if time > self.closing[0] {
            return false;
        }
        let mut recomputed_profit = 0;

        for &v in solution {
            if v >= n || v == 0 || visited[v] {
                println!("customer {v} is invalid or already visited");

                return false;
            }

            time = cmp::max(time + self.distances[current][v], self.opening[v]);

            if time > self.closing[v] {
                println!(
                    "customer {v} is visited at time {time} after closing time {closing_time}",
                    closing_time = self.closing[v]
                );

                return false;
            }

            visited[v] = true;
            recomputed_profit += self.profits[v];
            current = v;
        }

        time += self.distances[current][0];

        if time > self.closing[0] {
            println!(
                "the vehicle returns to the depot at time {} after closing time {}",
                time, self.closing[0]
            );

            return false;
        }

        if recomputed_profit != cost {
            println!("Invalid profit {cost} != {recomputed_profit}");

            return false;
        }

        true
    }

    pub fn print_solution(&self, solution: &[usize]) {
        println!(
            "Tour: {tour}",
            tour = solution
                .iter()
                .map(|&i| self.vertices[i].to_string())
                .collect::<Vec<_>>()
                .join(" ")
        );
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
        default_value_t = 1,
        help = "Number of decimal places to round the distances"
    )]
    pub round_to: u32,
    #[arg(
        short,
        long,
        default_value_t = 1e-6,
        help = "Threshold for floating point values"
    )]
    pub epsilon: f64,
}

pub fn compute_pairwise_shortest_path_costs<T>(weights: &[Vec<T>]) -> Vec<Vec<T>>
where
    T: Copy + PartialOrd + Add<Output = T>,
{
    let mut distance = Vec::from(weights);
    let n = distance.len();

    for k in 1..n {
        for i in 0..n {
            if i == k {
                continue;
            }

            for j in 0..n {
                if j == i || j == k {
                    continue;
                }

                if distance[i][k] + distance[k][j] < distance[i][j] {
                    distance[i][j] = distance[i][k] + distance[k][j];
                }
            }
        }
    }

    distance
}
