use clap::Parser;
use cvrp::{Args, RoundedInstance, SolverChoice};
use fixedbitset::FixedBitSet;
use regex::Regex;
use rpid::prelude::*;
use rpid::{algorithms, io, solvers, timer::Timer};
use std::cmp::{self, Ordering};
use tsplib_parser::Instance;

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

#[derive(Clone)]
struct Cvrp {
    instance: RoundedInstance,
    n_vehicles: i32,
    sorted_edges: Vec<(usize, usize, i32)>,
    node_to_sorted_out_edges: Vec<Vec<(usize, i32)>>,
    sorted_edges_to_depot: Vec<(usize, i32)>,
}

impl From<RoundedInstance> for Cvrp {
    fn from(instance: RoundedInstance) -> Self {
        let n_vehicles = instance.n_vehicles as i32;
        let depot = instance.depot;
        let weight_matrix = instance
            .distances
            .iter()
            .enumerate()
            .map(|(i, row)| {
                row.iter()
                    .enumerate()
                    .map(|(j, &w)| {
                        match (
                            w,
                            instance.distances[i][depot],
                            instance.distances[depot][j],
                        ) {
                            (Some(w), Some(w_to_depot), Some(w_from_depot)) => {
                                Some(cmp::min(w, w_to_depot + w_from_depot))
                            }
                            (Some(w), _, _) => Some(w),
                            (_, Some(w_to_depot), Some(w_from_depot)) => {
                                Some(w_to_depot + w_from_depot)
                            }
                            _ => None,
                        }
                    })
                    .collect()
            })
            .collect::<Vec<_>>();
        let sorted_edges = algorithms::sort_weight_matrix_with_option(&weight_matrix);
        let n = instance.nodes.len();
        let mut node_to_sorted_out_edges = vec![Vec::with_capacity(n); n];
        let mut sorted_edges_to_depot = Vec::with_capacity(n);

        for &(i, j, w) in &sorted_edges {
            node_to_sorted_out_edges[i].push((j, w));

            if j == depot {
                sorted_edges_to_depot.push((i, w));
            }
        }

        Self {
            instance,
            n_vehicles,
            sorted_edges,
            node_to_sorted_out_edges,
            sorted_edges_to_depot,
        }
    }
}

#[derive(Clone)]
struct CvrpState {
    unvisited: FixedBitSet,
    current: usize,
    load: i32,
    n_vehicles: i32,
}

impl Cvrp {
    fn check_feasibility(&self, state: &CvrpState) -> bool {
        let remaining_demand = state
            .unvisited
            .ones()
            .map(|i| self.instance.demands[i])
            .sum::<i32>();

        (self.n_vehicles - state.n_vehicles + 1) * self.instance.capacity
            >= (state.load + remaining_demand)
    }
}

impl Dp for Cvrp {
    type State = CvrpState;
    type CostType = i32;
    type Label = usize;

    fn get_target(&self) -> Self::State {
        let depot = self.instance.depot;
        let mut unvisited = FixedBitSet::with_capacity(self.instance.nodes.len());
        unvisited.insert_range(..);
        unvisited.remove(depot);

        CvrpState {
            unvisited,
            current: depot,
            load: 0,
            n_vehicles: 1,
        }
    }

    fn get_successors(
        &self,
        state: &Self::State,
    ) -> impl IntoIterator<Item = (Self::State, Self::CostType, Self::Label)> {
        let mut successors = state
            .unvisited
            .ones()
            .filter_map(|next| {
                if let Some(distance) = self.instance.distances[state.current][next] {
                    let load = state.load + self.instance.demands[next];

                    if load <= self.instance.capacity {
                        let mut unvisited = state.unvisited.clone();
                        unvisited.remove(next);
                        let successor = CvrpState {
                            unvisited,
                            current: next,
                            load,
                            n_vehicles: state.n_vehicles,
                        };

                        if self.check_feasibility(&successor) {
                            Some((successor, distance, next))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        if state.n_vehicles < self.n_vehicles {
            successors.extend(state.unvisited.ones().filter_map(|next| {
                if let (Some(distance_to_depot), Some(distance_from_depot)) = (
                    self.instance.distances[state.current][self.instance.depot],
                    self.instance.distances[self.instance.depot][next],
                ) {
                    let mut unvisited = state.unvisited.clone();
                    unvisited.remove(next);
                    let successor = CvrpState {
                        unvisited,
                        current: next,
                        load: self.instance.demands[next],
                        n_vehicles: state.n_vehicles + 1,
                    };

                    if self.check_feasibility(&successor) {
                        let weight = distance_to_depot + distance_from_depot;

                        Some((successor, weight, self.instance.nodes.len() + next))
                    } else {
                        None
                    }
                } else {
                    None
                }
            }))
        }

        successors
    }

    fn get_base_cost(&self, state: &Self::State) -> Option<Self::CostType> {
        if state.unvisited.is_clear() {
            self.instance.distances[state.current][self.instance.depot]
        } else {
            None
        }
    }
}

impl Dominance for Cvrp {
    type State = CvrpState;
    type Key = (FixedBitSet, usize);

    fn get_key(&self, state: &Self::State) -> Self::Key {
        (state.unvisited.clone(), state.current)
    }

    fn compare(&self, a: &Self::State, b: &Self::State) -> Option<Ordering> {
        if a.load == b.load && a.n_vehicles == b.n_vehicles {
            Some(Ordering::Equal)
        } else if a.load <= b.load && a.n_vehicles <= b.n_vehicles {
            Some(Ordering::Greater)
        } else if a.load >= b.load && a.n_vehicles >= b.n_vehicles {
            Some(Ordering::Less)
        } else {
            None
        }
    }
}

impl Bound for Cvrp {
    type State = CvrpState;
    type CostType = i32;

    fn get_dual_bound(&self, state: &Self::State) -> Option<Self::CostType> {
        let n = state.unvisited.count_ones(..);

        if n == 0 {
            return self.instance.distances[state.current][self.instance.depot];
        }

        let minimum_start = self.node_to_sorted_out_edges[state.current]
            .iter()
            .find_map(|&(i, w)| {
                if state.unvisited.contains(i) {
                    Some(w)
                } else {
                    None
                }
            })?;

        let iter = self
            .sorted_edges
            .iter()
            .filter(|(i, j, _)| (state.unvisited.contains(*i)) && state.unvisited.contains(*j))
            .copied();
        let mst_weight =
            algorithms::compute_minimum_spanning_tree_weight(self.instance.demands.len(), n, iter);

        let minimum_return = self.sorted_edges_to_depot.iter().find_map(|&(i, w)| {
            if state.unvisited.contains(i) {
                Some(w)
            } else {
                None
            }
        })?;

        Some(minimum_start + mst_weight + minimum_return)
    }
}

fn main() {
    let timer = Timer::default();
    let args = Args::parse();

    let filepath = args.input_file;
    let filename = filepath.split('/').next_back().unwrap();

    let re = Regex::new(r".+k(\d+).+").unwrap();
    let n_vehicles = re.captures(filename).unwrap()[1].parse().unwrap();

    let instance = Instance::load(&filepath).unwrap();
    let mut instance = RoundedInstance::new(instance, n_vehicles).unwrap();

    if args.reduce_edges {
        instance.reduce_edges();
    }

    let cvrp = Cvrp::from(instance.clone());

    let parameters = SearchParameters {
        time_limit: Some(args.time_limit),
        ..Default::default()
    };
    let solution = match args.solver {
        SolverChoice::Cabs => {
            let cabs_parameters = CabsParameters::default();
            println!("Preparing time: {time}s", time = timer.get_elapsed_time());
            let mut solver = if args.threads.get() > 1 {
                solvers::create_parallel_cabs(cvrp, parameters, cabs_parameters, args.threads.get())
            } else {
                solvers::create_cabs(cvrp, parameters, cabs_parameters)
            };
            io::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap()
        }
        SolverChoice::Astar => {
            println!("Preparing time: {time}s", time = timer.get_elapsed_time());
            let mut solver = solvers::create_astar(cvrp, parameters);
            io::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap()
        }
    };
    io::print_solution_statistics(&solution);

    if let Some(cost) = solution.cost {
        let mut tours = vec![vec![]];

        for transition in solution.transitions {
            if transition >= instance.nodes.len() {
                tours.push(vec![transition - instance.nodes.len()]);
            } else {
                tours.last_mut().unwrap().push(transition);
            }
        }

        instance.print_solution(&tours);

        if instance.validate(&tours, cost) {
            println!("The solution is valid.");
        } else {
            println!("The solution is invalid.");
        }
    }
}
