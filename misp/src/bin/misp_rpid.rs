use clap::Parser;
use fixedbitset::FixedBitSet;
use misp::{Args, Instance, SolverChoice};
use rpid::prelude::*;
use rpid::{io, solvers, timer::Timer};

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

#[derive(Clone)]
struct Misp(Instance);

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct MispState {
    candidates: FixedBitSet,
    current: usize,
}

impl Dp for Misp {
    type State = MispState;
    type CostType = i32;
    type Label = bool;

    fn get_target(&self) -> Self::State {
        let mut candidates = FixedBitSet::with_capacity(self.0.n);
        candidates.insert_range(..);

        MispState {
            candidates,
            current: 0,
        }
    }

    fn get_successors(
        &self,
        state: &Self::State,
    ) -> impl IntoIterator<Item = (Self::State, Self::CostType, Self::Label)> {
        if state.candidates.contains(state.current) {
            let mut candidates_exclude = state.candidates.clone();
            candidates_exclude.remove(state.current);

            let mut candidates_include = candidates_exclude.clone();

            for &neighbor in &self.0.adjacency_list[state.current] {
                candidates_include.remove(neighbor);
            }

            let successor_include = MispState {
                candidates: candidates_include,
                current: state.current + 1,
            };
            let successor_exclude = MispState {
                candidates: candidates_exclude,
                current: state.current + 1,
            };

            vec![(successor_include, 1, true), (successor_exclude, 0, false)]
        } else {
            let successor = MispState {
                candidates: state.candidates.clone(),
                current: state.current + 1,
            };

            vec![(successor, 0, false)]
        }
    }

    fn get_base_cost(&self, state: &Self::State) -> Option<Self::CostType> {
        if state.current == self.0.n {
            Some(0)
        } else {
            None
        }
    }

    fn get_optimization_mode(&self) -> OptimizationMode {
        OptimizationMode::Maximization
    }
}

impl Dominance for Misp {
    type State = MispState;
    type Key = MispState;

    fn get_key(&self, state: &Self::State) -> Self::Key {
        state.clone()
    }
}

impl Bound for Misp {
    type State = MispState;
    type CostType = i32;

    fn get_dual_bound(&self, state: &Self::State) -> Option<Self::CostType> {
        Some(state.candidates.count_ones(..) as i32)
    }
}

fn main() {
    let timer = Timer::default();
    let args = Args::parse();

    let instance = Instance::read_from_file(&args.input_file).unwrap();
    let misp = Misp(instance.clone());

    let parameters = SearchParameters {
        time_limit: Some(args.time_limit),
        ..Default::default()
    };

    let solution = match args.solver {
        SolverChoice::Cabs => {
            let cabs_parameters = CabsParameters::default();
            println!("Preparing time: {time}s", time = timer.get_elapsed_time());
            let mut solver = if args.threads.get() > 1 {
                solvers::create_parallel_cabs(misp, parameters, cabs_parameters, args.threads.get())
            } else {
                solvers::create_cabs(misp, parameters, cabs_parameters)
            };
            io::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap()
        }
        SolverChoice::Astar => {
            println!("Preparing time: {time}s", time = timer.get_elapsed_time());
            let mut solver = solvers::create_astar(misp, parameters);
            io::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap()
        }
    };
    io::print_solution_statistics(&solution);

    if let Some(cost) = solution.cost {
        let independent_set = solution
            .transitions
            .iter()
            .enumerate()
            .filter_map(|(i, &x)| if x { Some(i) } else { None })
            .collect::<Vec<_>>();
        println!("Independent set: {independent_set:?}");

        if instance.validate(&independent_set) {
            if independent_set.len() != cost as usize {
                println!("Cost {cost} != {len}", len = independent_set.len());
                println!("The solution is invalid.");
            } else {
                println!("The solution is valid.");
            }
        } else {
            println!("The solution is invalid.");
        }
    }
}
