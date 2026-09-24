use clap::{Parser, ValueEnum};
use rpid::io;
use std::cmp;
use std::error::Error;
use std::fs;

#[derive(Clone)]
pub struct Instance {
    pub a: Vec<i32>,
    pub b: Vec<i32>,
    pub c: Vec<Vec<Option<i32>>>,
}

impl Instance {
    pub fn read_from_file(filename: &str) -> Result<Self, Box<dyn Error>> {
        let file = fs::read_to_string(filename)?;
        let mut digits = file.split_whitespace();

        let n = digits.next().ok_or("empty file".to_owned())?.parse()?;
        let c = io::read_matrix(&mut digits, n, n)?;
        let c = c
            .into_iter()
            .enumerate()
            .map(|(i, row)| {
                row.into_iter()
                    .enumerate()
                    .map(|(j, distance)| if i == j { None } else { Some(distance) })
                    .collect()
            })
            .collect();
        let time_windows = io::read_matrix(&mut digits, n, 2)?;
        let a = time_windows.iter().map(|x| x[0]).collect();
        let b = time_windows.iter().map(|x| x[1]).collect();

        Ok(Self { a, b, c })
    }

    pub fn validate(&self, tour: &[usize], cost: i32) -> bool {
        self.validate_inner(tour, cost, false)
    }

    pub fn validate_makespan(&self, tour: &[usize], makespan: i32) -> bool {
        self.validate_inner(tour, makespan, true)
    }

    fn validate_inner(&self, tour: &[usize], cost: i32, minimize_makespan: bool) -> bool {
        if tour.len() != self.a.len() - 1 {
            println!("Invalid tour length: {len}", len = tour.len());

            return false;
        }

        let mut time = self.a[0].max(0);
        let mut current = 0;
        let mut visited = vec![false; self.a.len()];
        let mut recomputed_cost = 0;

        for &next in tour.iter().chain(std::iter::once(&0)) {
            if next >= self.a.len() {
                println!("Invalid node index: {next}");

                return false;
            }

            if visited[next] {
                println!("Visited node twice: {next}");

                return false;
            }

            if let Some(distance) = self.c[current][next] {
                time = cmp::max(time + distance, self.a[next]);
                recomputed_cost += distance;
            } else {
                println!("Invalid edge: {current} -> {next}");

                return false;
            }

            if time > self.b[next] {
                println!("Time window violation: {next} at {time}");

                return false;
            }

            current = next;
            visited[next] = true;
        }

        if minimize_makespan {
            if cost != time {
                println!("Invalid makespan: {cost} != {time}");

                return false;
            }
        } else if cost != recomputed_cost {
            println!("Invalid cost: {cost} != {recomputed_cost}");

            return false;
        }

        true
    }

    pub fn simplify(&mut self, expensive_detection: bool) {
        self.delete_edges(expensive_detection);

        while self.reduce_time_windows() && self.delete_edges(expensive_detection) {}
    }

    fn delete_edges(&mut self, expensive_detection: bool) -> bool {
        let n = self.a.len();
        let mut deleted = false;

        for (i, row) in self.c.iter_mut().enumerate() {
            for (j, distance) in row.iter_mut().enumerate() {
                if let Some(d) = distance {
                    if self.a[i] + *d > self.b[j]
                        || (expensive_detection
                            && i != 0
                            && j != 0
                            && (1..n).any(|k| self.b[i] < self.a[k] && self.b[k] < self.a[j]))
                    {
                        *distance = None;

                        if !deleted {
                            deleted = true;
                        }
                    }
                }
            }
        }

        deleted
    }

    fn reduce_time_windows(&mut self) -> bool {
        let mut reduced = false;
        let n = self.a.len();

        for k in 1..n {
            let original_a = self.a[k];
            let original_b = self.b[k];
            let mut min_arrival_time = self.b[k];
            let mut max_arrival_time = self.a[k];

            for i in 0..n {
                if let Some(d) = self.c[i][k] {
                    min_arrival_time = cmp::min(min_arrival_time, self.a[i] + d);
                    max_arrival_time = cmp::max(max_arrival_time, self.b[i] + d);
                }
            }

            self.a[k] = cmp::max(self.a[k], min_arrival_time);
            self.b[k] = cmp::min(self.b[k], max_arrival_time);

            let mut min_arrival_time = self.b[k];
            let mut max_arrival_time = self.a[k];

            for j in 0..n {
                if let Some(d) = self.c[k][j] {
                    min_arrival_time = cmp::min(min_arrival_time, self.a[j] - d);
                    max_arrival_time = cmp::max(max_arrival_time, self.b[j] - d);
                }
            }

            self.a[k] = cmp::max(self.a[k], min_arrival_time);
            self.b[k] = cmp::min(self.b[k], max_arrival_time);

            if !reduced && (self.a[k] != original_a || self.b[k] != original_b) {
                reduced = true;
            }
        }

        reduced
    }
}

#[derive(Debug, Clone, ValueEnum)]
pub enum SolverChoice {
    Cabs,
    Astar,
}

#[derive(Debug, Clone, ValueEnum, PartialEq)]
pub enum SimplificationChoice {
    None,
    Cheap,
    Expensive,
}

#[derive(Debug, Parser)]
pub struct Args {
    #[arg(long, help = "Use the MST dual bound in the DyPDL model")]
    pub mst: bool,
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
    #[arg(long, value_enum, default_value_t = SimplificationChoice::None, help = "Level of simplification of the instance in preprocessing")]
    pub simplification_level: SimplificationChoice,
    #[arg(long, short, action, help = "Minimize makespan")]
    pub minimize_makespan: bool,
}
