use clap::Parser;
use fixedbitset::FixedBitSet;
use rpid::prelude::*;
use rpid::{io, solvers, timer::Timer};
use std::cmp;
use wt::{Args, Instance, SolverChoice};

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

#[derive(Clone)]
struct Wt {
    instance: Instance,
    predecessors: Vec<FixedBitSet>,
}

impl From<Instance> for Wt {
    fn from(instance: Instance) -> Self {
        let (predecessors, _) = instance.extract_precedence();

        Self {
            instance,
            predecessors,
        }
    }
}

impl Dp for Wt {
    type State = FixedBitSet;
    type CostType = i32;
    type Label = usize;

    fn get_target(&self) -> FixedBitSet {
        FixedBitSet::with_capacity(self.instance.processing_times.len())
    }

    fn get_successors(
        &self,
        scheduled: &Self::State,
    ) -> impl IntoIterator<Item = (Self::State, Self::CostType, Self::Label)> {
        let time = scheduled
            .ones()
            .map(|i| self.instance.processing_times[i])
            .sum::<i32>();

        scheduled.zeroes().filter_map(move |i| {
            if self.predecessors[i].is_subset(scheduled) {
                let mut new_state = scheduled.clone();
                new_state.insert(i);

                let weight = self.instance.weights[i]
                    * cmp::max(
                        0,
                        time + self.instance.processing_times[i] - self.instance.deadlines[i],
                    );

                Some((new_state, weight, i))
            } else {
                None
            }
        })
    }

    fn get_base_cost(&self, scheduled: &Self::State) -> Option<Self::CostType> {
        if scheduled.is_full() { Some(0) } else { None }
    }
}

impl Dominance for Wt {
    type State = FixedBitSet;
    type Key = FixedBitSet;

    fn get_key(&self, state: &Self::State) -> Self::Key {
        state.clone()
    }
}

impl Bound for Wt {
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
    let wt = Wt::from(instance.clone());

    let parameters = SearchParameters {
        time_limit: Some(args.time_limit),
        ..Default::default()
    };

    let solution = match args.solver {
        SolverChoice::Cabs => {
            let cabs_parameters = CabsParameters::default();
            println!("Preparing time: {time}s", time = timer.get_elapsed_time());
            let mut solver = if args.threads.get() > 1 {
                solvers::create_parallel_cabs(wt, parameters, cabs_parameters, args.threads.get())
            } else {
                solvers::create_cabs(wt, parameters, cabs_parameters)
            };
            io::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap()
        }
        SolverChoice::Astar => {
            println!("Preparing time: {time}s", time = timer.get_elapsed_time());
            let mut solver = solvers::create_astar(wt, parameters);
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
