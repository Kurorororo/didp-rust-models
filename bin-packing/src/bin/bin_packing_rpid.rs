use bin_packing::{Args, Instance, SolverChoice};
use clap::Parser;
use fixedbitset::FixedBitSet;
use rpid::prelude::*;
use rpid::{algorithms, io, solvers, timer::Timer};
use std::cmp::{self, Ordering};

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

#[derive(Clone)]
struct BinPacking(Instance);

#[derive(Clone)]
struct BinPackingState {
    remaining: i32,
    unpacked: FixedBitSet,
    bin_number: usize,
}

impl Dp for BinPacking {
    type State = BinPackingState;
    type CostType = i32;
    type Label = usize;

    fn get_target(&self) -> Self::State {
        let mut unpacked = FixedBitSet::with_capacity(self.0.weights.len());
        unpacked.insert_range(..);

        BinPackingState {
            remaining: 0,
            unpacked,
            bin_number: 0,
        }
    }

    fn get_successors(
        &self,
        state: &Self::State,
    ) -> impl IntoIterator<Item = (Self::State, Self::CostType, Self::Label)> {
        let candidates = state
            .unpacked
            .ones()
            .filter(|&i| state.remaining >= self.0.weights[i])
            .collect::<Vec<_>>();

        if candidates.is_empty() {
            for i in state.unpacked.ones() {
                if state.bin_number <= i && self.0.weights[i] > state.remaining {
                    let mut unpacked = state.unpacked.clone();
                    unpacked.remove(i);
                    let successor = BinPackingState {
                        remaining: self.0.capacity - self.0.weights[i],
                        unpacked,
                        bin_number: state.bin_number + 1,
                    };

                    return vec![(successor, 1, i)];
                }
            }

            vec![]
        } else {
            candidates
                .into_iter()
                .filter_map(|i| {
                    let remaining = state.remaining - self.0.weights[i];

                    if state.bin_number <= i + 1 {
                        let mut unpacked = state.unpacked.clone();
                        unpacked.remove(i);
                        let successor = BinPackingState {
                            remaining,
                            unpacked,
                            bin_number: state.bin_number,
                        };

                        Some((successor, 0, i))
                    } else {
                        None
                    }
                })
                .collect()
        }
    }

    fn get_base_cost(&self, state: &Self::State) -> Option<Self::CostType> {
        if state.unpacked.is_clear() {
            Some(0)
        } else {
            None
        }
    }
}

impl Dominance for BinPacking {
    type State = BinPackingState;
    type Key = FixedBitSet;

    fn get_key(&self, state: &Self::State) -> Self::Key {
        state.unpacked.clone()
    }

    fn compare(&self, a: &Self::State, b: &Self::State) -> Option<Ordering> {
        if a.remaining == b.remaining && a.bin_number == b.bin_number {
            Some(Ordering::Equal)
        } else if a.remaining >= b.remaining && a.bin_number <= b.bin_number {
            Some(Ordering::Greater)
        } else if a.remaining <= b.remaining && a.bin_number >= b.bin_number {
            Some(Ordering::Less)
        } else {
            None
        }
    }
}

impl Bound for BinPacking {
    type State = BinPackingState;
    type CostType = i32;

    fn get_dual_bound(&self, state: &Self::State) -> Option<Self::CostType> {
        let capacity = self.0.capacity;
        let weights = state
            .unpacked
            .ones()
            .map(|i| self.0.weights[i])
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
    let bin_packing = BinPacking(instance.clone());

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
                    bin_packing,
                    parameters,
                    cabs_parameters,
                    args.threads.get(),
                )
            } else {
                solvers::create_cabs(bin_packing, parameters, cabs_parameters)
            };
            io::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap()
        }
        SolverChoice::Astar => {
            println!("Preparing time: {time}s", time = timer.get_elapsed_time());
            let mut solver = solvers::create_astar(bin_packing, parameters);
            io::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap()
        }
    };
    io::print_solution_statistics(&solution);

    if let Some(cost) = solution.cost {
        instance.print_solution(&solution.transitions);

        if instance.validate(&solution.transitions, cost) {
            println!("The solution is valid.");
        } else {
            println!("The solution is invalid.");
        }
    }
}
