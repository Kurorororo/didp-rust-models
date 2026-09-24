use clap::Parser;
use fixedbitset::FixedBitSet;
use rpid::prelude::*;
use rpid::{algorithms, io, solvers, timer::Timer};
use std::cmp::{self, Ordering};
use tsptw::{Args, Instance, SimplificationChoice, SolverChoice};

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

#[derive(Clone)]
struct Tsptw {
    instance: Instance,
    c_star: Vec<Vec<Option<i32>>>,
    min_to: Vec<i32>,
    min_from: Vec<i32>,
    minimize_makespan: bool,
}

impl Tsptw {
    fn new(instance: Instance, minimize_makespan: bool) -> Self {
        let mut c = instance.c.clone();
        c.iter_mut().for_each(|row| {
            row[0] = None;
        });
        let c_star = algorithms::compute_pairwise_shortest_path_costs_with_option(&c);
        let min_to = algorithms::take_column_wise_min_with_option(&instance.c)
            .map(|x| x.unwrap())
            .collect();
        let min_from = algorithms::take_row_wise_min_with_option(&instance.c)
            .map(|x| x.unwrap())
            .collect();

        Self {
            instance,
            c_star,
            min_to,
            min_from,
            minimize_makespan,
        }
    }
}

#[derive(Clone)]
struct TsptwState {
    unvisited: FixedBitSet,
    current: usize,
    time: i32,
}

impl Tsptw {
    fn check_feasibility(&self, state: &TsptwState) -> bool {
        state.unvisited.ones().all(|next| {
            if let Some(distance) = self.c_star[state.current][next] {
                state.time + distance <= self.instance.b[next]
            } else {
                false
            }
        })
    }
}

impl Dp for Tsptw {
    type State = TsptwState;
    type CostType = i32;
    type Label = usize;

    fn get_target(&self) -> Self::State {
        let mut unvisited = FixedBitSet::with_capacity(self.instance.a.len());
        unvisited.insert_range(1..);

        TsptwState {
            unvisited,
            current: 0,
            time: self.instance.a[0].max(0),
        }
    }

    fn get_successors(
        &self,
        state: &Self::State,
    ) -> impl IntoIterator<Item = (Self::State, Self::CostType, Self::Label)> {
        state.unvisited.ones().filter_map(|next| {
            if let Some(distance) = self.instance.c[state.current][next] {
                let time = cmp::max(state.time + distance, self.instance.a[next]);

                if time > self.instance.b[next] {
                    return None;
                }

                let mut unvisited = state.unvisited.clone();
                unvisited.remove(next);

                let successor = TsptwState {
                    unvisited,
                    current: next,
                    time,
                };

                if !self.check_feasibility(&successor) {
                    return None;
                }

                let weight = if self.minimize_makespan {
                    time - state.time
                } else {
                    distance
                };

                Some((successor, weight, next))
            } else {
                None
            }
        })
    }

    fn get_base_cost(&self, state: &Self::State) -> Option<Self::CostType> {
        if state.unvisited.is_clear() {
            self.instance.c[state.current][0]
                .filter(|&distance| state.time + distance <= self.instance.b[0])
                .map(|distance| {
                    distance
                        + if self.minimize_makespan {
                            self.instance.a[0].max(0)
                        } else {
                            0
                        }
                })
        } else {
            None
        }
    }
}

impl Dominance for Tsptw {
    type State = TsptwState;
    type Key = (FixedBitSet, usize);

    fn get_key(&self, state: &Self::State) -> Self::Key {
        (state.unvisited.clone(), state.current)
    }

    fn compare(&self, a: &Self::State, b: &Self::State) -> Option<Ordering> {
        Some(b.time.cmp(&a.time))
    }
}

impl Bound for Tsptw {
    type State = TsptwState;
    type CostType = i32;

    fn get_dual_bound(&self, state: &Self::State) -> Option<Self::CostType> {
        let bound_to = state.unvisited.ones().map(|i| self.min_to[i]).sum::<i32>() + self.min_to[0];
        let bound_from = self.min_from[state.current]
            + state
                .unvisited
                .ones()
                .map(|i| self.min_from[i])
                .sum::<i32>();

        Some(cmp::max(bound_to, bound_from))
    }
}

fn main() {
    let timer = Timer::default();
    let args = Args::parse();

    let mut instance = Instance::read_from_file(&args.input_file).unwrap();

    match args.simplification_level {
        SimplificationChoice::None => {}
        SimplificationChoice::Cheap => {
            instance.simplify(false);
        }
        SimplificationChoice::Expensive => {
            instance.simplify(true);
        }
    }

    let tsptw = Tsptw::new(instance.clone(), args.minimize_makespan);

    let parameters = SearchParameters {
        time_limit: Some(args.time_limit),
        ..Default::default()
    };
    let solution = match args.solver {
        SolverChoice::Cabs => {
            let cabs_parameters = CabsParameters::default();
            println!(
                "Preparing time: time{time}s",
                time = timer.get_elapsed_time()
            );
            let mut solver = if args.threads.get() > 1 {
                solvers::create_parallel_cabs(
                    tsptw,
                    parameters,
                    cabs_parameters,
                    args.threads.get(),
                )
            } else {
                solvers::create_cabs(tsptw, parameters, cabs_parameters)
            };
            io::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap()
        }
        SolverChoice::Astar => {
            println!("Preparing time: {time}s", time = timer.get_elapsed_time());
            let mut solver = solvers::create_astar(tsptw, parameters);
            io::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap()
        }
    };
    io::print_solution_statistics(&solution);

    if let Some(cost) = solution.cost {
        let transitions = solution
            .transitions
            .iter()
            .map(|t| format!("{t}"))
            .collect::<Vec<_>>()
            .join(" ");
        println!("Tour: {transitions}");

        if (args.minimize_makespan && instance.validate_makespan(&solution.transitions, cost))
            || instance.validate(&solution.transitions, cost)
        {
            println!("The solution is valid.");
        } else {
            println!("The solution is invalid.");
        }
    }
}
