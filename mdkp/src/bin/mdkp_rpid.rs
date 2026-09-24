use clap::Parser;
use mdkp::{Args, Instance, SolverChoice};
use rpid::prelude::*;
use rpid::{algorithms, io, solvers, timer::Timer};

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;
#[derive(Clone)]
struct Mdkp {
    instance: Instance,
    sorted_items: Vec<Vec<(usize, i32, i32)>>,
    epsilon: f64,
}

impl Mdkp {
    fn new(instance: Instance, epsilon: f64) -> Self {
        let sorted_items = instance
            .weights
            .iter()
            .map(|ws| algorithms::sort_knapsack_items_by_efficiency(ws, &instance.profits))
            .collect();

        Self {
            instance,
            sorted_items,
            epsilon,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct MdkpState {
    current: usize,
    remaining: Vec<i32>,
}

impl Dp for Mdkp {
    type State = MdkpState;
    type CostType = i32;
    type Label = bool;

    fn get_target(&self) -> Self::State {
        MdkpState {
            current: 0,
            remaining: self.instance.capacities.clone(),
        }
    }

    fn get_successors(
        &self,
        state: &Self::State,
    ) -> impl IntoIterator<Item = (Self::State, Self::CostType, Self::Label)> {
        let mut remaining = state.remaining.clone();

        for (j, (&r, ws)) in state
            .remaining
            .iter()
            .zip(self.instance.weights.iter())
            .enumerate()
        {
            if ws[state.current] > r {
                remaining[..j]
                    .iter_mut()
                    .zip(state.remaining[..j].iter())
                    .for_each(|(x, y)| *x = *y);
                let successor = MdkpState {
                    current: state.current + 1,
                    remaining,
                };

                return vec![(successor, 0, false)];
            }

            remaining[j] -= ws[state.current];
        }

        let successor_1 = MdkpState {
            current: state.current + 1,
            remaining,
        };
        let successor_2 = MdkpState {
            current: state.current + 1,
            remaining: state.remaining.clone(),
        };

        vec![
            (successor_1, self.instance.profits[state.current], true),
            (successor_2, 0, false),
        ]
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

impl Dominance for Mdkp {
    type State = MdkpState;
    type Key = MdkpState;

    fn get_key(&self, state: &Self::State) -> Self::Key {
        state.clone()
    }
}

impl Bound for Mdkp {
    type State = MdkpState;
    type CostType = i32;

    fn get_dual_bound(&self, state: &Self::State) -> Option<Self::CostType> {
        if state.current == self.instance.profits.len() {
            return Some(0);
        }

        let bound = state
            .remaining
            .iter()
            .zip(self.sorted_items.iter())
            .map(|(&r, items)| {
                (algorithms::compute_fractional_knapsack_profit(
                    r,
                    items.iter().filter_map(|&(i, w, p)| {
                        if i >= state.current && p > 0 {
                            Some((w, p))
                        } else {
                            None
                        }
                    }),
                ) + self.epsilon)
                    .floor() as i32
            })
            .chain(std::iter::once(
                self.instance.profits[state.current..]
                    .iter()
                    .map(|&p| p.max(0))
                    .sum(),
            ))
            .min()
            .unwrap();

        Some(bound)
    }
}

fn main() {
    let timer = Timer::default();
    let args = Args::parse();

    let instance = Instance::read_from_file(&args.input_file).unwrap();
    let mdkp = Mdkp::new(instance.clone(), args.epsilon);

    let parameters = SearchParameters {
        time_limit: Some(args.time_limit),
        ..Default::default()
    };

    let solution = match args.solver {
        SolverChoice::Cabs => {
            let cabs_parameters = CabsParameters::default();
            println!("Preparing time: {time}s", time = timer.get_elapsed_time());
            let mut solver = if args.threads.get() > 1 {
                solvers::create_parallel_cabs(mdkp, parameters, cabs_parameters, args.threads.get())
            } else {
                solvers::create_cabs(mdkp, parameters, cabs_parameters)
            };
            io::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap()
        }
        SolverChoice::Astar => {
            println!("Preparing time: {time}s", time = timer.get_elapsed_time());
            let mut solver = solvers::create_astar(mdkp, parameters);
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
        println!(
            "Packed items: {}",
            packed_items
                .iter()
                .map(|&i| i.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        );

        if instance.validate(&packed_items, profit) {
            println!("The solution is valid.");
        } else {
            println!("The solution is invalid.");
        }
    }
}
