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

struct Cvrp {
    instance: RoundedInstance,
    n_vehicles: i32,
    min_to: Vec<i32>,
    min_from: Vec<i32>,
}

impl From<RoundedInstance> for Cvrp {
    fn from(instance: RoundedInstance) -> Self {
        let n_vehicles = instance.n_vehicles as i32;
        let min_to = algorithms::take_column_wise_min_with_option(&instance.distances)
            .map(|x| x.unwrap())
            .collect();
        let min_from = algorithms::take_row_wise_min_with_option(&instance.distances)
            .map(|x| x.unwrap())
            .collect();

        Self {
            instance,
            n_vehicles,
            min_to,
            min_from,
        }
    }
}

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
    ) -> impl IntoIterator<Item = (Self::State, Self::CostType, usize)> {
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
        let bound_to = state.unvisited.ones().map(|i| self.min_to[i]).sum::<i32>()
            + self.min_to[self.instance.depot];
        let bound_from = state
            .unvisited
            .ones()
            .map(|i| self.min_from[i])
            .sum::<i32>()
            + self.min_from[state.current];

        Some(cmp::max(bound_to, bound_from))
    }
}

fn main() {
    let timer = Timer::default();
    let args = Args::parse();

    let filepath = args.input_file;
    let filename = filepath.split('/').last().unwrap();

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
            println!("Preparing time: {}s", timer.get_elapsed_time());
            let cabs_parameters = CabsParameters::default();
            let mut solver = solvers::create_cabs(cvrp, parameters, cabs_parameters);
            io::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap()
        }
        SolverChoice::Astar => {
            println!("Preparing time: {}s", timer.get_elapsed_time());
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
