use clap::Parser;
use fixedbitset::FixedBitSet;
use rpid::prelude::*;
use rpid::{algorithms, io, solvers, timer::Timer};
use salbp_1::{Args, Instance, SolverChoice};
use std::cmp::{self, Ordering};

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

struct Salbp1(Instance);

struct Salbp1State {
    remaining: i32,
    unscheduled: FixedBitSet,
}

impl Dp for Salbp1 {
    type State = Salbp1State;
    type CostType = i32;
    type Label = usize;

    fn get_target(&self) -> Self::State {
        let mut unscheduled = FixedBitSet::with_capacity(self.0.task_times.len());
        unscheduled.insert_range(..);

        Salbp1State {
            remaining: 0,
            unscheduled,
        }
    }

    fn get_successors(
        &self,
        state: &Self::State,
    ) -> impl IntoIterator<Item = (Self::State, Self::CostType, Self::Label)> {
        let successors = state
            .unscheduled
            .ones()
            .filter_map(|i| {
                let remaining = state.remaining - self.0.task_times[i];

                if remaining >= 0
                    && self.0.predecessors[i].intersection_count(&state.unscheduled) == 0
                {
                    let mut unscheduled = state.unscheduled.clone();
                    unscheduled.remove(i);
                    let successor = Salbp1State {
                        remaining,
                        unscheduled,
                    };

                    Some((successor, 0, i))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        if successors.is_empty() {
            vec![(
                Salbp1State {
                    remaining: self.0.cycle_time,
                    unscheduled: state.unscheduled.clone(),
                },
                1,
                self.0.task_times.len(),
            )]
        } else {
            successors
        }
    }

    fn get_base_cost(&self, state: &Self::State) -> Option<Self::CostType> {
        if state.unscheduled.is_clear() {
            Some(0)
        } else {
            None
        }
    }
}

impl Dominance for Salbp1 {
    type State = Salbp1State;
    type Key = FixedBitSet;

    fn get_key(&self, state: &Self::State) -> Self::Key {
        state.unscheduled.clone()
    }

    fn compare(&self, a: &Self::State, b: &Self::State) -> Option<Ordering> {
        Some(a.remaining.cmp(&b.remaining))
    }
}

impl Bound for Salbp1 {
    type State = Salbp1State;
    type CostType = i32;

    fn get_dual_bound(&self, state: &Self::State) -> Option<Self::CostType> {
        let capacity = self.0.cycle_time;
        let weights = state
            .unscheduled
            .ones()
            .map(|i| self.0.task_times[i])
            .collect::<Vec<_>>();

        let weight_sum = weights.iter().sum::<i32>() - state.remaining;
        let lb1 = weight_sum / capacity + if weight_sum % capacity > 0 { 1 } else { 0 };

        let mut lb2 = algorithms::compute_bin_packing_lb2(capacity, weights.iter().copied()) as i32;

        if 2 * state.remaining >= capacity {
            lb2 -= 1;
        }

        let mut lb3 = algorithms::compute_bin_packing_lb3(capacity, weights.into_iter()) as i32;

        if 3 * state.remaining >= capacity {
            lb3 -= 1;
        }

        Some(cmp::max(cmp::max(lb1, lb2), lb3))
    }
}

fn main() {
    let timer = Timer::default();
    let args = Args::parse();

    let instance = Instance::read_from_file(&args.input_file).unwrap();
    let salbp1 = Salbp1(instance.clone());

    let parameters = SearchParameters {
        time_limit: Some(args.time_limit),
        ..Default::default()
    };
    let solution = match args.solver {
        SolverChoice::Cabs => {
            let cabs_parameters = CabsParameters::default();
            println!("Preparing time: {time}s", time = timer.get_elapsed_time());
            let mut solver = solvers::create_cabs(salbp1, parameters, cabs_parameters);
            io::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap()
        }
        SolverChoice::Astar => {
            println!("Preparing time: {time}s", time = timer.get_elapsed_time());
            let mut solver = solvers::create_astar(salbp1, parameters);
            io::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap()
        }
    };
    io::print_solution_statistics(&solution);

    if let Some(cost) = solution.cost {
        let sequence = solution
            .transitions
            .iter()
            .filter_map(|&i| {
                if i < instance.task_times.len() {
                    Some(i)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        instance.print_solution(&sequence);

        if instance.validate(&sequence, cost) {
            println!("The solution is valid.");
        } else {
            println!("The solution is invalid.");
        }
    }
}
