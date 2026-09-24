use clap::Parser;
use fixedbitset::FixedBitSet;
use golomb_ruler::{Args, KNOWN_OPTIMAL_COSTS, SolverChoice};
use rpid::prelude::*;
use rpid::{io, solvers, timer::Timer};
use std::iter;

#[derive(Clone)]
struct GolomobRuler {
    n: usize,
    lower_bounds: Vec<i32>,
}

impl GolomobRuler {
    fn new(n: usize) -> Self {
        let mut lower_bounds = Vec::with_capacity(n + 1);
        lower_bounds.extend(KNOWN_OPTIMAL_COSTS.iter().map(|&i| i as i32));

        while lower_bounds.len() <= n {
            let last = lower_bounds[lower_bounds.len() - 1];
            lower_bounds.push(last + 1);
        }

        GolomobRuler { n, lower_bounds }
    }
}

#[derive(Clone)]
struct GolomobRulerState {
    mark_set: FixedBitSet,
    distance_set: FixedBitSet,
    last_mark: usize,
    n_marks: usize,
}

impl Dp for GolomobRuler {
    type State = GolomobRulerState;
    type CostType = i32;
    type Label = usize;

    fn get_target(&self) -> Self::State {
        let mut mark_set = FixedBitSet::with_capacity(self.n * self.n + 2);
        mark_set.insert(0);

        GolomobRulerState {
            mark_set,
            distance_set: FixedBitSet::with_capacity(self.n * self.n + 2),
            last_mark: 0,
            n_marks: 1,
        }
    }

    fn get_successors(
        &self,
        state: &Self::State,
    ) -> impl IntoIterator<Item = (Self::State, Self::CostType, usize)> {
        let lb = state.last_mark + 1;

        let ub = if state.n_marks < self.n / 2 {
            (self.n * self.n).div_ceil(2) - self.lower_bounds[self.n / 2 - state.n_marks] as usize
        } else {
            self.n * self.n + 1 - self.lower_bounds[self.n - state.n_marks] as usize
        };

        (lb..=ub).filter_map(move |i| {
            let mut distance_set = state.distance_set.clone();

            for j in state.mark_set.ones() {
                let distance = i - j;

                if state.distance_set.contains(distance) {
                    return None;
                }

                distance_set.insert(distance)
            }

            let mut mark_set = state.mark_set.clone();
            mark_set.insert(i);

            let successor = GolomobRulerState {
                mark_set,
                distance_set,
                last_mark: i,
                n_marks: state.n_marks + 1,
            };

            Some((successor, (i - state.last_mark) as i32, i))
        })
    }

    fn get_base_cost(&self, state: &Self::State) -> Option<Self::CostType> {
        if state.n_marks == self.n {
            Some(0)
        } else {
            None
        }
    }
}

impl Dominance for GolomobRuler {
    type State = GolomobRulerState;
    type Key = FixedBitSet;

    fn get_key(&self, state: &Self::State) -> Self::Key {
        state.mark_set.clone()
    }
}

impl Bound for GolomobRuler {
    type State = GolomobRulerState;
    type CostType = i32;

    fn get_dual_bound(&self, state: &Self::State) -> Option<Self::CostType> {
        Some(self.lower_bounds[self.n - state.n_marks])
    }
}

fn main() {
    let timer = Timer::default();
    let args = Args::parse();

    let n = args.n;
    let golomob_ruler = GolomobRuler::new(n);

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
                    golomob_ruler,
                    parameters,
                    cabs_parameters,
                    args.threads.get(),
                )
            } else {
                solvers::create_cabs(golomob_ruler, parameters, cabs_parameters)
            };
            io::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap()
        }
        SolverChoice::Astar => {
            println!("Preparing time: {time}s", time = timer.get_elapsed_time());
            let mut solver = solvers::create_astar(golomob_ruler, parameters);
            io::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap()
        }
    };
    io::print_solution_statistics(&solution);

    if let Some(cost) = solution.cost {
        let marks = iter::once(0)
            .chain(solution.transitions)
            .collect::<Vec<_>>();
        let marks_str = marks
            .iter()
            .map(|i| format!("{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        println!("Marks: {marks_str}");

        if golomb_ruler::validate(n, &marks, cost as usize) {
            println!("The solution is valid");
        } else {
            println!("The solution is invalid");
        }
    }
}
