use clap::Parser;
use knapsack::{Args, Instance, SolverChoice};
use rpid::prelude::*;
use rpid::{algorithms, io, solvers, timer::Timer};
use std::cmp::Ordering;

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

#[derive(Clone)]
struct Knapsack {
    instance: Instance,
    sorted_items: Vec<(usize, i32, i32)>,
    epsilon: f64,
}

impl Knapsack {
    fn new(instance: Instance, epsilon: f64) -> Self {
        let sorted_items =
            algorithms::sort_knapsack_items_by_efficiency(&instance.weights, &instance.profits);

        Self {
            instance,
            sorted_items,
            epsilon,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct KnapsackState {
    current: usize,
    remaining: i32,
}

impl Dp for Knapsack {
    type State = KnapsackState;
    type CostType = i32;
    type Label = bool;

    fn get_target(&self) -> Self::State {
        KnapsackState {
            current: 0,
            remaining: self.instance.capacity,
        }
    }

    fn get_successors(
        &self,
        state: &Self::State,
    ) -> impl IntoIterator<Item = (Self::State, Self::CostType, Self::Label)> {
        let ignore = KnapsackState {
            current: state.current + 1,
            remaining: state.remaining,
        };

        if self.instance.weights[state.current] > state.remaining {
            vec![(ignore, 0, false)]
        } else {
            let pack = KnapsackState {
                current: state.current + 1,
                remaining: state.remaining - self.instance.weights[state.current],
            };

            vec![
                (pack, self.instance.profits[state.current], true),
                (ignore, 0, false),
            ]
        }
    }

    fn get_base_cost(&self, state: &Self::State) -> Option<Self::CostType> {
        if state.current == self.instance.profits.len() {
            Some(0)
        } else {
            None
        }
    }

    fn get_optimization_mode(&self) -> OptimizationMode {
        OptimizationMode::Maximization
    }
}

impl Dominance for Knapsack {
    type State = KnapsackState;
    type Key = usize;

    fn get_key(&self, state: &Self::State) -> Self::Key {
        state.current
    }

    fn compare(&self, a: &Self::State, b: &Self::State) -> Option<Ordering> {
        Some(a.remaining.cmp(&b.remaining))
    }
}

impl Bound for Knapsack {
    type State = KnapsackState;
    type CostType = i32;

    fn get_dual_bound(&self, state: &Self::State) -> Option<Self::CostType> {
        if state.current == self.instance.profits.len() {
            return Some(0);
        }

        let sorted_weight_value_pairs = self.sorted_items.iter().filter_map(|&(i, w, p)| {
            if i >= state.current && p > 0 {
                Some((w, p))
            } else {
                None
            }
        });

        let bound = (algorithms::compute_fractional_knapsack_profit(
            state.remaining,
            sorted_weight_value_pairs,
        ) + self.epsilon)
            .floor();

        Some(bound as i32)
    }
}

fn main() {
    let timer = Timer::default();
    let args = Args::parse();

    let instance = Instance::read_from_file(&args.input_file).unwrap();
    let knapsack = Knapsack::new(instance.clone(), args.epsilon);

    let parameters = SearchParameters {
        time_limit: Some(args.time_limit),
        ..Default::default()
    };

    let solution = match args.solver {
        SolverChoice::Cabs => {
            let cabs_parameters = CabsParameters::default();
            println!("Preparing time: {time}s", time = timer.get_elapsed_time());
            let mut solver = if args.threads.get() > 1 {
                solvers::create_parallel_cabs(
                    knapsack,
                    parameters,
                    cabs_parameters,
                    args.threads.get(),
                )
            } else {
                solvers::create_cabs(knapsack, parameters, cabs_parameters)
            };
            io::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap()
        }
        SolverChoice::Astar => {
            println!("Preparing time: {time}s", time = timer.get_elapsed_time());
            let mut solver = solvers::create_astar(knapsack, parameters);
            io::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap()
        }
    };
    io::print_solution_statistics(&solution);

    if let Some(profit) = solution.cost {
        let packed_items = solution
            .transitions
            .iter()
            .enumerate()
            .filter_map(|(i, &x)| if x { Some(i) } else { None })
            .collect::<Vec<_>>();
        instance.print_solution(&packed_items);

        if instance.validate(&packed_items, profit) {
            println!("The solution is valid.");
        } else {
            println!("The solution is invalid.");
        }
    }
}
