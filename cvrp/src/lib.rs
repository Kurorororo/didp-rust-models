use clap::{Parser, ValueEnum};
use std::error::Error;
use tsplib_parser::Instance;

#[derive(Clone, Debug)]
pub struct RoundedInstance {
    pub n_vehicles: usize,
    pub nodes: Vec<usize>,
    pub demands: Vec<i32>,
    pub depot: usize,
    pub capacity: i32,
    pub distances: Vec<Vec<Option<i32>>>,
}

impl RoundedInstance {
    pub fn new(instance: Instance, n_vehicles: usize) -> Result<Self, Box<dyn Error>> {
        let distances = instance.get_full_distance_matrix()?;
        let distances = distances
            .into_iter()
            .enumerate()
            .map(|(i, row)| {
                row.into_iter()
                    .enumerate()
                    .map(|(j, d)| if i == j { None } else { Some(d) })
                    .collect()
            })
            .collect();

        if instance.demand_dimension != Some(1) {
            return Err("Only one demand dimension is allowed".into());
        }

        let mut demands = instance
            .demands
            .ok_or("Demands not defined")?
            .into_iter()
            .map(|(node, d)| (node, d[0]))
            .collect::<Vec<_>>();
        demands.sort_by_key(|(i, _)| *i);
        let nodes = demands.iter().map(|(i, _)| *i).collect::<Vec<_>>();
        let demands = demands.into_iter().map(|(_, d)| d).collect();
        let depots = instance.depots.ok_or("Depot not defined")?;

        if depots.len() != 1 {
            return Err("Only one depot is allowed".into());
        }

        let depot = nodes
            .iter()
            .position(|&i| i == depots[0])
            .ok_or("Depot not found")?;
        let capacity = instance.capacity.ok_or("Capacity not found")?;

        Ok(Self {
            n_vehicles,
            nodes,
            demands,
            depot,
            capacity,
            distances,
        })
    }

    pub fn validate(&self, tours: &[Vec<usize>], cost: i32) -> bool {
        if tours.len() > self.n_vehicles {
            println!(
                "Invalid number of vehicles {len} > {n_vehicles}",
                len = tours.len(),
                n_vehicles = self.n_vehicles,
            );

            return false;
        }

        let actual_length = tours.iter().map(|t| t.len()).sum::<usize>();
        let expected_length = self.nodes.len() - 1; // Exclude depot

        if actual_length != expected_length {
            println!("Invalid number of nodes {actual_length} != {expected_length}",);

            return false;
        }

        let mut visited_by = vec![None; self.nodes.len()];
        let mut recomputed_cost = 0;

        for (i, t) in tours.iter().enumerate() {
            let mut current = self.depot;
            let mut load = 0;

            for &node in t {
                if node >= self.nodes.len() {
                    println!("Invalid node {node} visited by route {i}");

                    return false;
                }

                if let Some(j) = visited_by[node] {
                    println!("Node {node} visited twice by routes {j} and {i}");

                    return false;
                }

                if let Some(c) = self.distances[current][node] {
                    recomputed_cost += c;
                } else {
                    println!("Invalid edge {current} -> {node}");

                    return false;
                }

                load += self.demands[node];

                if load > self.capacity {
                    println!(
                        "Vehicle load exceeded {load} > {capacity} at node {node}",
                        capacity = self.capacity,
                    );

                    return false;
                }

                visited_by[node] = Some(i);
                current = node;
            }

            if current != self.depot {
                if let Some(c) = self.distances[current][self.depot] {
                    recomputed_cost += c;
                } else {
                    println!("Invalid edge {current} -> {depot}", depot = self.depot);

                    return false;
                }
            }
        }

        if recomputed_cost != cost {
            println!("Invalid cost {cost} != {recomputed_cost}");

            return false;
        }

        true
    }

    pub fn print_solution(&self, tours: &[Vec<usize>]) {
        for (i, tour) in tours.iter().enumerate() {
            println!(
                "Route {}: {}",
                i + 1,
                tour.iter()
                    .map(|&j| j.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        }
    }

    pub fn reduce_edges(&mut self) {
        for (i, row) in self.distances.iter_mut().enumerate() {
            for (j, d) in row.iter_mut().enumerate() {
                if d.is_some() && self.demands[i] + self.demands[j] > self.capacity {
                    *d = None;
                }
            }
        }
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
    #[arg(short, long, action, help = "Performs edge reduction")]
    pub reduce_edges: bool,
}
