use clap::Parser;
use fixedbitset::FixedBitSet;
use graph_clear::{Args, Instance, SolverChoice};
use rpid::prelude::*;
use rpid::{io, solvers, timer::Timer};
use std::cmp;

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

#[derive(Clone)]
struct GraphClear {
    instance: Instance,
    edge_weight_sum: Vec<i32>,
}

impl From<Instance> for GraphClear {
    fn from(instance: Instance) -> Self {
        let edge_weight_sum = instance
            .edge_weights
            .iter()
            .map(|w| w.iter().sum())
            .collect::<Vec<i32>>();

        Self {
            instance,
            edge_weight_sum,
        }
    }
}

impl Dp for GraphClear {
    type State = FixedBitSet;
    type CostType = i32;
    type Label = usize;

    fn get_target(&self) -> FixedBitSet {
        FixedBitSet::with_capacity(self.instance.node_weights.len())
    }

    fn get_successors(
        &self,
        clean: &Self::State,
    ) -> impl IntoIterator<Item = (Self::State, Self::CostType, Self::Label)> {
        clean.zeroes().map(|i| {
            let mut new_state = clean.clone();
            new_state.insert(i);

            let weight = self.instance.node_weights[i]
                + self.edge_weight_sum[i]
                + clean
                    .ones()
                    .map(|j| {
                        clean
                            .zeroes()
                            .filter_map(|k| {
                                if k != i {
                                    Some(self.instance.edge_weights[j][k])
                                } else {
                                    None
                                }
                            })
                            .sum::<i32>()
                    })
                    .sum::<i32>();

            (new_state, weight, i)
        })
    }

    fn get_base_cost(&self, clean: &Self::State) -> Option<Self::CostType> {
        if clean.is_full() { Some(0) } else { None }
    }

    fn combine_cost_weights(&self, a: Self::CostType, b: Self::CostType) -> Self::CostType {
        cmp::max(a, b)
    }
}

impl Dominance for GraphClear {
    type State = FixedBitSet;
    type Key = FixedBitSet;

    fn get_key(&self, state: &Self::State) -> Self::Key {
        state.clone()
    }
}

impl Bound for GraphClear {
    type State = FixedBitSet;
    type CostType = i32;

    fn get_dual_bound(&self, _: &Self::State) -> Option<Self::CostType> {
        Some(0)
    }
}

fn main() {
    let timer = Timer::default();
    let args = Args::parse();

    let instance = Instance::read_from_file(&args.input_file).unwrap();
    let graph_clear = GraphClear::from(instance.clone());

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
                    graph_clear,
                    parameters,
                    cabs_parameters,
                    args.threads.get(),
                )
            } else {
                solvers::create_cabs(graph_clear, parameters, cabs_parameters)
            };
            io::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap()
        }
        SolverChoice::Astar => {
            println!("Preparing time: {time}s", time = timer.get_elapsed_time());
            let mut solver = solvers::create_astar(graph_clear, parameters);
            io::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap()
        }
    };
    io::print_solution_statistics(&solution);

    if let Some(cost) = solution.cost {
        let schedule = solution
            .transitions
            .iter()
            .map(|t| format!("{t}"))
            .collect::<Vec<_>>()
            .join(" ");
        println!("Schedule: {schedule}");

        if instance.validate(&solution.transitions, cost) {
            println!("The solution is valid.");
        } else {
            println!("The solution is invalid.");
        }
    }
}
