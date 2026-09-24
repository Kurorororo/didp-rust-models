use clap::{Parser, ValueEnum};
use fixedbitset::FixedBitSet;
use itertools::Itertools;
use std::error::Error;
use tsplib_parser::Instance;

#[derive(Clone, Debug)]
pub struct RoundedInstance {
    pub nodes: Vec<usize>,
    pub demand_dimension: usize,
    pub demands: Vec<Vec<i32>>,
    pub capacity: i32,
    pub distances: Vec<Vec<Option<i32>>>,
}

impl TryFrom<Instance> for RoundedInstance {
    type Error = Box<dyn Error>;

    fn try_from(instance: Instance) -> Result<Self, Box<dyn Error>> {
        let distances = instance.get_full_distance_matrix()?;
        let distances = distances
            .into_iter()
            .enumerate()
            .map(|(i, row)| {
                row.into_iter()
                    .enumerate()
                    .map(|(j, d)| if i != j && d >= 0 { Some(d) } else { None })
                    .collect()
            })
            .collect();

        let demand_dimension = instance
            .demand_dimension
            .ok_or("Demand dimension not defined")?;
        let mut demands = instance.demands.ok_or("Demands not defined")?;
        demands.sort_by_key(|(i, _)| *i);
        let nodes = demands.iter().map(|(i, _)| *i).collect();
        let demands = demands.into_iter().map(|(_, d)| d).collect();
        let capacity = instance.capacity.ok_or("Capacity not defined")?;

        Ok(Self {
            nodes,
            demand_dimension,
            demands,
            capacity,
            distances,
        })
    }
}

impl RoundedInstance {
    pub fn validate(&self, tours: &[usize], cost: i32) -> bool {
        let n = self.nodes.len();

        if tours.len() != n - 2 {
            println!(
                "Invalid tour length: {len} != {expected}",
                len = tours.len(),
                expected = n - 2
            );

            return false;
        }

        let mut visited = vec![false; self.nodes.len()];
        let mut current = 0;
        let mut loads = vec![0; self.demand_dimension];
        let mut recomputed_cost = 0;

        for &next in tours {
            if next >= self.nodes.len() - 1 {
                println!("Invalid node index: {next}");

                return false;
            }

            if visited[next] {
                println!("Visited node twice: {next}");

                return false;
            }

            if let Some(d) = self.distances[current][next] {
                recomputed_cost += d;
            } else {
                println!("{current} is not connected to {next}");

                return false;
            }

            let mut total_load = 0;

            for (i, (d, l)) in self.demands[next].iter().zip(loads.iter_mut()).enumerate() {
                *l += d;

                if *l < 0 {
                    println!("Negative load {l} in dimension {i} at {next}");

                    return false;
                }

                total_load += *l;
            }

            if total_load > self.capacity {
                println!(
                    "Capacity violation: {total_load} > {capacity} at {next}",
                    capacity = self.capacity
                );

                return false;
            }

            visited[next] = true;
            current = next;
        }

        let goal = self.nodes.len() - 1;

        if let Some(d) = self.distances[current][goal] {
            recomputed_cost += d;
        } else {
            println!("{current} is not connected to {goal}");

            return false;
        }

        if cost != recomputed_cost {
            println!("Invalid cost: {cost} != {recomputed_cost}");

            return false;
        }

        true
    }

    pub fn print_solution(&self, tour: &[usize]) {
        println!(
            "Tour: {tour}",
            tour = tour
                .iter()
                .map(|&i| self.nodes[i].to_string())
                .collect_vec()
                .join(" ")
        );
    }

    pub fn extract_predecessors_and_filtered_distances(
        &self,
    ) -> (Vec<FixedBitSet>, Vec<Vec<Option<i32>>>) {
        let commodity_edges = self.extract_commodity_edge();
        let (predecessors, successors, precedence_matrix) =
            self.extract_precedence(&commodity_edges);
        let not_inferred_commodity_edges = Self::extract_not_inferred_commodity_edges(
            &predecessors,
            &successors,
            &commodity_edges,
        );
        let filtered_distances =
            self.filter_distances(&precedence_matrix, &not_inferred_commodity_edges);

        (predecessors, filtered_distances)
    }

    fn extract_commodity_edge(&self) -> Vec<Vec<Option<i32>>> {
        let n = self.nodes.len();
        let mut edges = vec![vec![None; n]; n];

        for (i, dis) in self.demands.iter().enumerate() {
            if i != 0 && i != n - 1 {
                edges[0][i] = Some(0);
                edges[i][n - 1] = Some(0);
            }

            for (j, djs) in self.demands.iter().enumerate() {
                if i != j {
                    for (&di, &dj) in dis.iter().zip(djs.iter()) {
                        if di > 0 && dj < 0 {
                            assert_eq!(di, -dj);
                            edges[i][j] = Some(di);

                            break;
                        }
                    }
                }
            }
        }

        edges
    }

    fn dfs_traversal(
        &self,
        i: usize,
        commodity_edges: &[Vec<Option<i32>>],
        visited: &mut [bool],
        result: &mut Vec<usize>,
    ) {
        if !visited[i] {
            visited[i] = true;

            for j in 0..self.nodes.len() {
                if commodity_edges[i][j].is_some() {
                    self.dfs_traversal(j, commodity_edges, visited, result);
                }
            }

            result.push(i);
        }
    }

    fn topological_sort(&self, commodity_edges: &[Vec<Option<i32>>]) -> Vec<usize> {
        let n = self.nodes.len();
        let mut visited = vec![false; n];
        let mut result = Vec::with_capacity(n);

        for i in 0..n {
            self.dfs_traversal(i, commodity_edges, &mut visited, &mut result);
        }

        result.reverse();

        result
    }

    fn extract_precedence(
        &self,
        commodity_edges: &[Vec<Option<i32>>],
    ) -> (Vec<FixedBitSet>, Vec<FixedBitSet>, Vec<Vec<bool>>) {
        let sorted_nodes = self.topological_sort(commodity_edges);

        let n = self.nodes.len();
        let mut predecessors = vec![FixedBitSet::with_capacity(n); n];

        for &i in &sorted_nodes {
            for &j in &sorted_nodes {
                if commodity_edges[i][j].is_some() {
                    let mut p = predecessors[j].clone();
                    p.insert(i);
                    p.union_with(&predecessors[i]);
                    predecessors[j] = p;
                }
            }
        }

        let mut successors = vec![FixedBitSet::with_capacity(n); n];

        for &i in sorted_nodes.iter().rev() {
            for &j in sorted_nodes.iter().rev() {
                if commodity_edges[i][j].is_some() {
                    let mut s = successors[i].clone();
                    s.insert(j);
                    s.union_with(&successors[j]);
                    successors[i] = s;
                }
            }
        }

        let precedence_matrix = (0..n)
            .map(|i| (0..n).map(|j| successors[i].contains(j)).collect())
            .collect();

        (predecessors, successors, precedence_matrix)
    }

    fn extract_not_inferred_commodity_edges(
        predecessors: &[FixedBitSet],
        successors: &[FixedBitSet],
        edges: &[Vec<Option<i32>>],
    ) -> Vec<Vec<Option<i32>>> {
        let mut result_edges = edges.to_vec();

        for (i, ss) in successors.iter().enumerate() {
            for (j, ps) in predecessors.iter().enumerate() {
                if let Some(w_ij) = result_edges[i][j] {
                    let intersection = ss.intersection(ps).collect_vec();

                    if !intersection.is_empty() && w_ij == 0 {
                        result_edges[i][j] = None;
                    } else {
                        for k in intersection {
                            if let (Some(w_ik), Some(w_kj)) =
                                (result_edges[i][k], result_edges[k][j])
                            {
                                result_edges[i][k] = Some(w_ik + w_ij);
                                result_edges[k][j] = Some(w_kj + w_ij);
                                result_edges[i][j] = None;

                                break;
                            }
                        }
                    }
                }
            }
        }

        result_edges
    }

    fn check_edge_by_capacity(
        &self,
        i: usize,
        j: usize,
        commodity_edges: &[Vec<Option<i32>>],
    ) -> bool {
        let n = self.nodes.len();

        let load = (0..n)
            .map(|k| {
                if k != i && k != j {
                    commodity_edges[k][i].unwrap_or(0) + commodity_edges[k][j].unwrap_or(0)
                } else {
                    0
                }
            })
            .sum::<i32>();

        if load > self.capacity {
            return false;
        }

        let load = commodity_edges[i]
            .iter()
            .enumerate()
            .filter_map(|(k, &w)| w.map(|w| (i, k, w)))
            .chain((0..n).filter_map(|k| commodity_edges[k][j].map(|w| (k, j, w))))
            .unique()
            .map(|(_, _, w)| w)
            .sum::<i32>();

        if load > self.capacity {
            return false;
        }

        let load_i = commodity_edges[i]
            .iter()
            .enumerate()
            .map(|(k, &w)| if k != j { w.unwrap_or(0) } else { 0 })
            .sum::<i32>();
        let load_j = commodity_edges[j]
            .iter()
            .enumerate()
            .map(|(k, &w)| if k != i { w.unwrap_or(0) } else { 0 })
            .sum::<i32>();

        load_i + load_j <= self.capacity
    }

    fn filter_distances(
        &self,
        precedence_matrix: &[Vec<bool>],
        not_inferred_commodity_edges: &[Vec<Option<i32>>],
    ) -> Vec<Vec<Option<i32>>> {
        self.distances
            .iter()
            .enumerate()
            .map(|(i, row)| {
                row.iter()
                    .enumerate()
                    .map(|(j, d)| {
                        d.and_then(|d| {
                            if self.check_edge_by_capacity(i, j, not_inferred_commodity_edges)
                                && !precedence_matrix[j][i]
                                && (!precedence_matrix[i][j]
                                    || not_inferred_commodity_edges[i][j].is_some())
                            {
                                Some(d)
                            } else {
                                None
                            }
                        })
                    })
                    .collect()
            })
            .collect()
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
